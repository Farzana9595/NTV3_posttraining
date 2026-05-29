"""Create tiny sequence + label samples from several NTv3 source corpora.

The NTv3 metadata file mixes multiple public archives. Some expose remotely
readable BigWig files; others expose sparse tag/CTSS text files. This script
creates small, separate examples for the sources that can be resolved without
re-running a full alignment pipeline:

- gtex: recount3 BigWig coverage
- deap: public DEAP BigWig coverage
- geo: processed SGA files from the EPD/MGA mirror
- fantom5: streamed FANTOM5 CTSS BED files

Rows from ncbi_rnaseq, ncbi_chrom_acc, and catlas are recorded in the manifest
as unresolved because the public rows point to raw accessions or unavailable
track downloads, not directly to small aligned signal tracks.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
from pathlib import Path
from urllib.parse import quote, urljoin

import numpy as np
import pybigtools
import requests


UCSC_SEQUENCE_URL = "https://api.genome.ucsc.edu/getData/sequence"
RECOUNT3_BASE = "http://duffel.rail.bio/recount3"


def fetch_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_text(url: str) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def stream_gzip_lines(url: str):
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as handle:
            for raw_line in handle:
                yield raw_line.decode("utf-8", errors="replace").strip()


def read_metadata_rows(metadata_csv: Path) -> list[dict]:
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_metadata_row(rows: list[dict], dataset: str, file_id: str | None = None) -> dict:
    for row in rows:
        if row["dataset"] != dataset:
            continue
        if file_id is not None and row["file_id"] != file_id:
            continue
        return row
    raise ValueError(f"No metadata row found for dataset={dataset!r}, file_id={file_id!r}")


def fetch_sequence(genome: str, chrom: str, start: int, end: int) -> str:
    data = fetch_json(
        UCSC_SEQUENCE_URL,
        params={"genome": genome, "chrom": chrom, "start": start, "end": end},
    )
    sequence = data.get("dna")
    if sequence is None:
        raise ValueError(f"UCSC did not return DNA for {genome}:{chrom}:{start}-{end}")
    return sequence.upper()


def sequence_summary(sequence: str) -> dict:
    base_counts = {base: sequence.count(base) for base in "ACGTN"}
    gc = base_counts["G"] + base_counts["C"]
    return {
        "length": len(sequence),
        "base_counts": base_counts,
        "gc_fraction": gc / len(sequence) if sequence else 0.0,
    }


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


def bin_values(values: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (len(values) // bin_size) * bin_size
    if usable == 0:
        raise ValueError("bin_size is larger than the label vector")
    return values[:usable].reshape(-1, bin_size).mean(axis=1)


def bin_summary(values: np.ndarray, bin_size: int) -> list[dict]:
    summaries = []
    usable = (len(values) // bin_size) * bin_size
    for start in range(0, usable, bin_size):
        chunk = values[start : start + bin_size]
        summary = summarize_values(chunk)
        summary["relative_start"] = start
        summary["relative_end"] = start + bin_size
        summaries.append(summary)
    return summaries


def bigwig_sample(
    *,
    corpus: str,
    row: dict,
    bigwig_url: str,
    genome: str,
    chrom: str,
    bigwig_chrom: str,
    start: int,
    window_size: int,
    num_windows: int,
    stride: int,
    label_bin_size: int,
    output_dir: Path,
    extra_source_metadata: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{corpus}_sequence_label.jsonl"
    details_path = output_dir / f"{corpus}_details.json"

    samples = []
    bigwig = pybigtools.open(bigwig_url)
    try:
        bigwig_info = bigwig.info()
        chrom_sizes = bigwig.chroms()
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for sample_index in range(num_windows):
                sample_start = start + sample_index * stride
                sample_end = sample_start + window_size
                sequence = fetch_sequence(genome, chrom, sample_start, sample_end)
                values = np.asarray(
                    bigwig.values(
                        bigwig_chrom,
                        sample_start,
                        sample_end,
                        missing=np.nan,
                        oob=np.nan,
                    ),
                    dtype=np.float32,
                )
                filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                binned = bin_values(filled, label_bin_size)
                records = [
                    {
                        "start": int(rec_start),
                        "end": int(rec_end),
                        "relative_start": max(int(rec_start), sample_start) - sample_start,
                        "relative_end": min(int(rec_end), sample_end) - sample_start,
                        "value": float(value),
                    }
                    for rec_start, rec_end, value in bigwig.records(
                        bigwig_chrom, sample_start, sample_end
                    )
                ]
                compact = {
                    "corpus": corpus,
                    "track_metadata": row,
                    "source_url": bigwig_url,
                    "genome": genome,
                    "chrom": chrom,
                    "start": sample_start,
                    "end": sample_end,
                    "sequence": sequence,
                    "label_bin_size": label_bin_size,
                    "labels": [round(float(value), 6) for value in binned],
                }
                handle.write(json.dumps(compact) + "\n")
                samples.append(
                    {
                        "sample_index": sample_index,
                        "coordinates": {
                            "genome": genome,
                            "chrom": chrom,
                            "bigwig_chrom": bigwig_chrom,
                            "start_0_based_inclusive": sample_start,
                            "end_0_based_exclusive": sample_end,
                            "window_size": window_size,
                            "chrom_size": int(chrom_sizes[bigwig_chrom]),
                        },
                        "sequence": sequence,
                        "sequence_summary": sequence_summary(sequence),
                        "signal": {
                            "per_base_values": [
                                None if not np.isfinite(value) else float(value)
                                for value in values
                            ],
                            "filled_per_base_values": [float(value) for value in filled],
                            "per_base_summary": summarize_values(values),
                            "records": records,
                            "record_count": len(records),
                            "label_bin_size": label_bin_size,
                            "binned_mean_labels": [float(value) for value in binned],
                            "bin_summaries": bin_summary(filled, label_bin_size),
                        },
                    }
                )
    finally:
        bigwig.close()

    details = {
        "corpus": corpus,
        "track_metadata_from_ntv3_index": row,
        "source_url": bigwig_url,
        "source_metadata": extra_source_metadata or {},
        "bigwig_global_info": bigwig_info,
        "samples": samples,
    }
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    return {
        "corpus": corpus,
        "status": "created",
        "jsonl": str(jsonl_path),
        "details": str(details_path),
        "sample_count": len(samples),
        "source_url": bigwig_url,
    }


def recount3_gtex_bigwig_url(sample_id: str, project: str) -> str:
    project_bucket = project[-2:]
    sample_no_suffix = sample_id.rsplit(".", 1)[0]
    sample_bucket = sample_no_suffix[-2:]
    encoded_sample = quote(sample_id, safe="")
    filename = f"gtex.base_sums.{project}_{encoded_sample}.ALL.bw"
    return (
        f"{RECOUNT3_BASE}/human/data_sources/gtex/base_sums/"
        f"{project_bucket}/{project}/{sample_bucket}/{filename}"
    )


def make_gtex(rows: list[dict], output_root: Path, args: argparse.Namespace) -> dict:
    row = first_metadata_row(rows, "gtex", "GTEX-132QS-2526-SM-62LFJ.1")
    project = "ADIPOSE_TISSUE"
    url = recount3_gtex_bigwig_url(row["file_id"], project)
    return bigwig_sample(
        corpus="gtex",
        row=row,
        bigwig_url=url,
        genome="hg38",
        chrom="chr1",
        bigwig_chrom="chr1",
        start=1_000_000,
        window_size=args.window_size,
        num_windows=args.num_windows,
        stride=args.stride,
        label_bin_size=args.label_bin_size,
        output_dir=output_root / "gtex",
        extra_source_metadata={
            "archive": "recount3",
            "project": project,
            "note": "Sample-level RNA-seq coverage BigWig from recount3.",
        },
    )


def make_deap(rows: list[dict], output_root: Path, args: argparse.Namespace) -> dict:
    row = first_metadata_row(rows, "deap", "window_16-18_cluster_6")
    url = (
        "https://shendure-web.gs.washington.edu/content/members/"
        "DEAP_website/public/RNA/update/inferred_time_cluster_bws/"
        "window_16-18_cluster_6.bw"
    )
    return bigwig_sample(
        corpus="deap",
        row=row,
        bigwig_url=url,
        genome="dm6",
        chrom="chr2L",
        bigwig_chrom="2L",
        start=100_000,
        window_size=args.window_size,
        num_windows=args.num_windows,
        stride=args.stride,
        label_bin_size=args.label_bin_size,
        output_dir=output_root / "deap",
        extra_source_metadata={
            "archive": "DEAP public processed data",
            "note": "Drosophila embryo RNA pseudobulk BigWig for one time-window cluster.",
        },
    )


def find_fantom5_ctss_url(sample_prefix: str) -> str:
    base = "https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/"
    html = fetch_text(base)
    hrefs = re.findall(r'href="([^"]+)"', html)
    for href in hrefs:
        if sample_prefix in href and href.endswith(".ctss.bed.gz"):
            return urljoin(base, href)
    raise ValueError(f"Could not find FANTOM5 CTSS BED for {sample_prefix}")


def parse_fantom5_ctss_row(line: str) -> dict:
    parts = line.split("\t")
    if len(parts) < 6:
        raise ValueError(f"Unexpected FANTOM5 CTSS row: {line!r}")
    return {
        "chrom": parts[0],
        "start": int(parts[1]),
        "end": int(parts[2]),
        "name": parts[3],
        "score": float(parts[4]),
        "strand": parts[5],
        "raw": line,
    }


def make_sparse_position_sample(
    *,
    corpus: str,
    row: dict,
    source_url: str,
    genome: str,
    chrom: str,
    start: int,
    window_size: int,
    num_windows: int,
    stride: int,
    label_bin_size: int,
    records: list[dict],
    output_dir: Path,
    source_metadata: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{corpus}_sequence_label.jsonl"
    details_path = output_dir / f"{corpus}_details.json"
    samples = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample_index in range(num_windows):
            sample_start = start + sample_index * stride
            sample_end = sample_start + window_size
            sequence = fetch_sequence(genome, chrom, sample_start, sample_end)
            values = np.zeros(window_size, dtype=np.float32)
            sample_records = []
            for record in records:
                if record["chrom"] != chrom:
                    continue
                if record["end"] <= sample_start or record["start"] >= sample_end:
                    continue
                clipped_start = max(record["start"], sample_start)
                clipped_end = min(record["end"], sample_end)
                values[clipped_start - sample_start : clipped_end - sample_start] += record[
                    "score"
                ]
                sample_record = dict(record)
                sample_record["relative_start"] = clipped_start - sample_start
                sample_record["relative_end"] = clipped_end - sample_start
                sample_records.append(sample_record)

            binned = bin_values(values, label_bin_size)
            compact = {
                "corpus": corpus,
                "track_metadata": row,
                "source_url": source_url,
                "genome": genome,
                "chrom": chrom,
                "start": sample_start,
                "end": sample_end,
                "sequence": sequence,
                "label_bin_size": label_bin_size,
                "labels": [round(float(value), 6) for value in binned],
            }
            handle.write(json.dumps(compact) + "\n")
            samples.append(
                {
                    "sample_index": sample_index,
                    "coordinates": {
                        "genome": genome,
                        "chrom": chrom,
                        "start_0_based_inclusive": sample_start,
                        "end_0_based_exclusive": sample_end,
                        "window_size": window_size,
                    },
                    "sequence": sequence,
                    "sequence_summary": sequence_summary(sequence),
                    "signal": {
                        "encoding": "sparse intervals expanded to per-base values",
                        "per_base_values": [float(value) for value in values],
                        "per_base_summary": summarize_values(values),
                        "records": sample_records,
                        "record_count": len(sample_records),
                        "label_bin_size": label_bin_size,
                        "binned_mean_labels": [float(value) for value in binned],
                        "bin_summaries": bin_summary(values, label_bin_size),
                    },
                }
            )

    details = {
        "corpus": corpus,
        "track_metadata_from_ntv3_index": row,
        "source_url": source_url,
        "source_metadata": source_metadata,
        "samples": samples,
    }
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    return {
        "corpus": corpus,
        "status": "created",
        "jsonl": str(jsonl_path),
        "details": str(details_path),
        "sample_count": len(samples),
        "source_url": source_url,
    }


def make_fantom5(rows: list[dict], output_root: Path, args: argparse.Namespace) -> dict:
    row = first_metadata_row(rows, "fantom5", "CNhs12331_P")
    source_url = find_fantom5_ctss_url("CNhs12331")
    selected_strand = "+"
    records = []
    for line in stream_gzip_lines(source_url):
        if not line or line.startswith("#"):
            continue
        record = parse_fantom5_ctss_row(line)
        if record["chrom"] != "chr1":
            continue
        if record["strand"] != selected_strand:
            continue
        records.append(record)
        if len(records) >= 1000:
            break

    if not records:
        raise ValueError("No FANTOM5 records were found for chr1 + strand")

    start = max(0, records[0]["start"] - 128)
    return make_sparse_position_sample(
        corpus="fantom5",
        row=row,
        source_url=source_url,
        genome="hg19",
        chrom="chr1",
        start=start,
        window_size=args.window_size,
        num_windows=args.num_windows,
        stride=args.stride,
        label_bin_size=args.label_bin_size,
        records=records,
        output_dir=output_root / "fantom5",
        source_metadata={
            "archive": "FANTOM5",
            "format": "CTSS BED gzip",
            "selected_strand": selected_strand,
            "note": "CTSS tag counts are sparse genomic intervals.",
        },
    )


def sga_seqname_to_ucsc(seqname: str) -> str:
    mapping = {
        "NC_000001.10": "chr1",
        "NC_000002.11": "chr2",
        "NC_000003.11": "chr3",
        "NC_000004.11": "chr4",
        "NC_000005.9": "chr5",
        "NC_000006.11": "chr6",
        "NC_000007.13": "chr7",
        "NC_000008.10": "chr8",
        "NC_000009.11": "chr9",
        "NC_000010.10": "chr10",
        "NC_000011.9": "chr11",
        "NC_000012.11": "chr12",
        "NC_000013.10": "chr13",
        "NC_000014.8": "chr14",
        "NC_000015.9": "chr15",
        "NC_000016.9": "chr16",
        "NC_000017.10": "chr17",
        "NC_000018.9": "chr18",
        "NC_000019.9": "chr19",
        "NC_000020.10": "chr20",
        "NC_000021.8": "chr21",
        "NC_000022.10": "chr22",
        "NC_000023.10": "chrX",
        "NC_000024.9": "chrY",
    }
    return mapping.get(seqname, seqname)


def parse_sga_row(line: str) -> dict:
    parts = line.split("\t")
    if len(parts) < 5:
        raise ValueError(f"Unexpected SGA row: {line!r}")
    ucsc_chrom = sga_seqname_to_ucsc(parts[0])
    position_1_based = int(parts[2])
    start = position_1_based - 1
    return {
        "seqname": parts[0],
        "feature": parts[1],
        "position_1_based": position_1_based,
        "chrom": ucsc_chrom,
        "start": start,
        "end": start + 1,
        "strand": parts[3],
        "score": float(parts[4]),
        "raw": line,
    }


def make_geo(rows: list[dict], output_root: Path, args: argparse.Namespace) -> dict:
    row = first_metadata_row(rows, "geo", "GSM1208709")
    source_url = (
        "https://epd.expasy.org/mga/hg19/yan13/"
        "GSM1208709_batch2_chrom1_LoVo_AEBP2_Goat_PassedQC.sga.gz"
    )
    records = []
    for line in stream_gzip_lines(source_url):
        if not line or line.startswith("#"):
            continue
        record = parse_sga_row(line)
        if record["chrom"] != "chr1":
            continue
        records.append(record)
        if len(records) >= 3000:
            break

    if not records:
        raise ValueError("No GEO SGA records were found for chr1")

    start = max(0, records[0]["start"] - 128)
    return make_sparse_position_sample(
        corpus="geo",
        row=row,
        source_url=source_url,
        genome="hg19",
        chrom="chr1",
        start=start,
        window_size=args.window_size,
        num_windows=args.num_windows,
        stride=args.stride,
        label_bin_size=args.label_bin_size,
        records=records,
        output_dir=output_root / "geo",
        source_metadata={
            "archive": "EPD/MGA mirror of GEO GSE49402",
            "format": "SGA gzip",
            "note": "SGA rows are sparse one-base ChIP-seq tag positions.",
        },
    )


def unresolved_entries(rows: list[dict]) -> list[dict]:
    unresolved = []
    for dataset, reason in {
        "ncbi_rnaseq": (
            "Rows are plant SRA/DRA experiment accessions. To create aligned labels, "
            "download FASTQ/SRA, align to the correct plant genome, then generate "
            "coverage BigWig files."
        ),
        "ncbi_chrom_acc": (
            "Rows are plant SRA experiment accessions for chromatin assays. Public "
            "accessions do not directly expose small aligned signal tracks."
        ),
        "catlas": (
            "The CATlas v2 resource pages list signal tracks as coming soon; I did "
            "not find stable public BigWig URLs for the kai* track IDs."
        ),
    }.items():
        try:
            row = first_metadata_row(rows, dataset)
        except ValueError:
            row = {}
        unresolved.append(
            {
                "corpus": dataset,
                "status": "not_created",
                "example_metadata_row": row,
                "reason": reason,
            }
        )
    return unresolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path(__file__).with_name("functional_tracks_metadata.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).with_name("corpus_samples"),
    )
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--label-bin-size", type=int, default=32)
    args = parser.parse_args()

    rows = read_metadata_rows(args.metadata_csv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, fn in [
        ("gtex", make_gtex),
        ("deap", make_deap),
        ("fantom5", make_fantom5),
        ("geo", make_geo),
    ]:
        try:
            result = fn(rows, args.output_root, args)
        except Exception as exc:
            result = {"corpus": name, "status": "failed", "error": repr(exc)}
        manifest.append(result)

    manifest.extend(unresolved_entries(rows))
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote manifest to {manifest_path}")
    for item in manifest:
        print(f"{item['corpus']}: {item['status']}")


if __name__ == "__main__":
    main()
