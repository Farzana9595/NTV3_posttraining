"""Create a tiny sequence + label dataset from one NTv3 functional track.

This example targets ENCODE-backed tracks from functional_tracks_metadata.csv.
It resolves the ENCODE experiment to a BigWig signal file, reads only the
requested genomic windows remotely, fetches the matching reference sequence
from UCSC, and writes JSONL examples.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pybigtools
import requests


ENCODE_BASE_URL = "https://www.encodeproject.org"
UCSC_SEQUENCE_URL = "https://api.genome.ucsc.edu/getData/sequence"


def fetch_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(
        url,
        params=params,
        headers={"accept": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def find_track_metadata(metadata_csv: Path, track_id: str) -> dict:
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["file_id"] == track_id:
                return row
    raise ValueError(f"Track {track_id!r} was not found in {metadata_csv}")


def resolve_encode_bigwig(
    experiment_id: str,
    assembly: str,
    output_type: str,
) -> tuple[dict, str]:
    data = fetch_json(
        f"{ENCODE_BASE_URL}/search/",
        params={
            "type": "File",
            "dataset": f"/experiments/{experiment_id}/",
            "file_format": "bigWig",
            "format": "json",
            "frame": "object",
            "limit": "all",
        },
    )

    candidates = []
    for item in data.get("@graph", []):
        if item.get("status") != "released":
            continue
        if item.get("assembly") != assembly:
            continue
        if item.get("output_type") != output_type:
            continue
        if not item.get("href"):
            continue
        candidates.append(item)

    if not candidates:
        raise ValueError(
            f"No released {assembly} {output_type!r} BigWig found for {experiment_id}"
        )

    def rank(item: dict) -> tuple[int, int]:
        replicate_count = len(set(item.get("biological_replicates") or []))
        file_size = int(item.get("file_size") or 0)
        return (-replicate_count, file_size)

    selected = sorted(candidates, key=rank)[0]
    return selected, urljoin(ENCODE_BASE_URL, selected["href"])


def fetch_sequence(genome: str, chrom: str, start: int, end: int) -> str:
    data = fetch_json(
        UCSC_SEQUENCE_URL,
        params={"genome": genome, "chrom": chrom, "start": start, "end": end},
    )
    if "dna" not in data:
        raise ValueError(f"UCSC did not return DNA for {genome}:{chrom}:{start}-{end}")
    sequence = data["dna"].upper()
    expected = end - start
    if len(sequence) != expected:
        raise ValueError(f"Expected {expected} bases, got {len(sequence)}")
    return sequence


def read_bigwig_values(
    bigwig_url: str,
    chrom: str,
    start: int,
    end: int,
    nan_fill: float,
) -> np.ndarray:
    bigwig = pybigtools.open(bigwig_url)
    try:
        values = np.asarray(bigwig.values(chrom, start, end), dtype=np.float32)
    finally:
        bigwig.close()
    return np.nan_to_num(values, nan=nan_fill, posinf=nan_fill, neginf=nan_fill)


def summarize_values(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    summary = {
        "length": int(values.size),
        "finite_count": int(finite.size),
        "missing_or_invalid_count": int(values.size - finite.size),
    }
    if finite.size == 0:
        return summary

    quantiles = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])
    summary.update(
        {
            "min": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "max": float(quantiles[4]),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "sum": float(np.sum(finite)),
        }
    )
    return summary


def sequence_summary(sequence: str) -> dict:
    length = len(sequence)
    base_counts = {base: sequence.count(base) for base in "ACGTN"}
    gc_count = base_counts["G"] + base_counts["C"]
    return {
        "length": length,
        "base_counts": base_counts,
        "gc_fraction": gc_count / length if length else 0.0,
    }


def bin_values(values: np.ndarray, bin_size: int) -> np.ndarray:
    if bin_size <= 1:
        return values
    usable = (len(values) // bin_size) * bin_size
    if usable == 0:
        raise ValueError("bin_size is larger than the label vector")
    return values[:usable].reshape(-1, bin_size).mean(axis=1)


def bin_summary(values: np.ndarray, bin_size: int) -> list[dict]:
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    bins = []
    usable = (len(values) // bin_size) * bin_size
    for bin_start in range(0, usable, bin_size):
        chunk = values[bin_start : bin_start + bin_size]
        summary = summarize_values(chunk)
        summary["relative_start"] = bin_start
        summary["relative_end"] = bin_start + bin_size
        bins.append(summary)
    return bins


def records_for_window(bigwig: pybigtools.BBIRead, chrom: str, start: int, end: int) -> list[dict]:
    records = []
    for record_start, record_end, value in bigwig.records(chrom, start, end):
        clipped_start = max(int(record_start), start)
        clipped_end = min(int(record_end), end)
        records.append(
            {
                "start": int(record_start),
                "end": int(record_end),
                "clipped_start": clipped_start,
                "clipped_end": clipped_end,
                "relative_start": clipped_start - start,
                "relative_end": clipped_end - start,
                "covered_bases_in_window": clipped_end - clipped_start,
                "value": float(value),
            }
        )
    return records


def fetch_encode_file_metadata(accession: str) -> dict:
    return fetch_json(f"{ENCODE_BASE_URL}/files/{accession}/", params={"format": "json"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path(__file__).with_name("functional_tracks_metadata.csv"),
    )
    parser.add_argument("--track-id", default="ENCSR391NPE")
    parser.add_argument("--assembly", default="GRCh38")
    parser.add_argument("--ucsc-genome", default="hg38")
    parser.add_argument("--output-type", default="signal p-value")
    parser.add_argument("--chrom", default="chr1")
    parser.add_argument("--start", type=int, default=1_000_000)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--label-bin-size", type=int, default=32)
    parser.add_argument("--nan-fill", type=float, default=0.0)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(__file__).with_name("small_encode_sequence_label.jsonl"),
    )
    parser.add_argument(
        "--details-json",
        type=Path,
        default=Path(__file__).with_name("small_encode_sequence_label_details.json"),
    )
    args = parser.parse_args()

    metadata = find_track_metadata(args.metadata_csv, args.track_id)
    if metadata.get("dataset") != "encode_v3":
        raise ValueError(
            "This minimal example only resolves encode_v3 rows. "
            f"{args.track_id} is from dataset={metadata.get('dataset')!r}."
        )

    bigwig_file, bigwig_url = resolve_encode_bigwig(
        args.track_id,
        assembly=args.assembly,
        output_type=args.output_type,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.details_json.parent.mkdir(parents=True, exist_ok=True)

    detailed_samples = []
    bigwig = pybigtools.open(bigwig_url)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        try:
            bigwig_info = bigwig.info()
            chrom_sizes = bigwig.chroms()
            for i in range(args.num_windows):
                start = args.start + i * args.stride
                end = start + args.window_size
                sequence = fetch_sequence(args.ucsc_genome, args.chrom, start, end)
                raw_values = np.asarray(
                    bigwig.values(
                        args.chrom,
                        start,
                        end,
                        missing=np.nan,
                        oob=np.nan,
                    ),
                    dtype=np.float32,
                )
                per_base = np.nan_to_num(
                    raw_values,
                    nan=args.nan_fill,
                    posinf=args.nan_fill,
                    neginf=args.nan_fill,
                )
                binned = bin_values(per_base, args.label_bin_size)
                compact_record = {
                    "track_id": args.track_id,
                    "track_metadata": metadata,
                    "bigwig_accession": bigwig_file["accession"],
                    "bigwig_url": bigwig_url,
                    "assembly": args.assembly,
                    "genome": args.ucsc_genome,
                    "chrom": args.chrom,
                    "start": start,
                    "end": end,
                    "sequence": sequence,
                    "label_bin_size": args.label_bin_size,
                    "labels": [round(float(x), 6) for x in binned],
                }
                handle.write(json.dumps(compact_record) + "\n")

                records = records_for_window(bigwig, args.chrom, start, end)
                covered_bases = sum(item["covered_bases_in_window"] for item in records)
                detailed_samples.append(
                    {
                        "sample_index": i,
                        "coordinates": {
                            "assembly": args.assembly,
                            "genome": args.ucsc_genome,
                            "chrom": args.chrom,
                            "start_0_based_inclusive": start,
                            "end_0_based_exclusive": end,
                            "window_size": end - start,
                            "chrom_size": int(chrom_sizes[args.chrom]),
                        },
                        "sequence": sequence,
                        "sequence_summary": sequence_summary(sequence),
                        "bigwig_signal": {
                            "description": (
                                "Per-base values are the BigWig signal value at each base "
                                "in the window. Missing values are null here and filled with "
                                f"{args.nan_fill} in filled_per_base_values."
                            ),
                            "output_type": args.output_type,
                            "per_base_values": [
                                None if not np.isfinite(value) else float(value)
                                for value in raw_values
                            ],
                            "filled_per_base_values": [float(value) for value in per_base],
                            "per_base_summary": summarize_values(raw_values),
                            "records": records,
                            "record_count": len(records),
                            "record_covered_bases": covered_bases,
                            "record_coverage_fraction": covered_bases / (end - start),
                            "bin_size": args.label_bin_size,
                            "binned_mean_labels": [float(value) for value in binned],
                            "bin_summaries": bin_summary(per_base, args.label_bin_size),
                        },
                    }
                )
        finally:
            bigwig.close()

    details = {
        "track_id": args.track_id,
        "track_metadata_from_ntv3_index": metadata,
        "selected_encode_bigwig_file": bigwig_file,
        "selected_encode_bigwig_file_full_metadata": fetch_encode_file_metadata(
            bigwig_file["accession"]
        ),
        "bigwig_url": bigwig_url,
        "bigwig_global_info": bigwig_info,
        "sampling_parameters": {
            "chrom": args.chrom,
            "start": args.start,
            "window_size": args.window_size,
            "num_windows": args.num_windows,
            "stride": args.stride,
            "label_bin_size": args.label_bin_size,
            "nan_fill": args.nan_fill,
        },
        "samples": detailed_samples,
    }
    with args.details_json.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)

    print(f"Wrote {args.num_windows} examples to {args.output_jsonl}")
    print(f"Wrote detailed report to {args.details_json}")
    print(f"Selected BigWig: {bigwig_file['accession']} ({bigwig_url})")


if __name__ == "__main__":
    main()
