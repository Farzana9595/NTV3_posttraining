#!/usr/bin/env python3
"""Prepare and QC CyVerse/MaizeGDB RNA-seq BigWigs for training.

Outputs a clean line-organized directory with references, tracks,
chromosome-size files, track_metadata.csv, and qc_report.csv.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

import pybigtools


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NTV3_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = NTV3_ROOT / "data" / "cyverse_maizegdb_rnaseq_all"
DEFAULT_BIGWIG_MAP = DEFAULT_BASE_DIR / "manifests" / "bigwig_to_reference_map.csv"
DEFAULT_SELECTED_REFERENCES = DEFAULT_BASE_DIR / "manifests" / "reference_selected_manifest.csv"
DEFAULT_REFERENCE_SOURCE_DIR = DEFAULT_BASE_DIR / "reference_genomes"
DEFAULT_PREPARED_DIR = DEFAULT_BASE_DIR / "prepared"
PRIMARY_CHROMS = {f"chr{i}" for i in range(1, 11)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    try:
        tmp_path.replace(path)
    except PermissionError:
        fallback_path = path.with_name(path.stem + "_updated" + path.suffix)
        tmp_path.replace(fallback_path)
        print(f"WARNING: {path} is locked; wrote {fallback_path} instead", file=sys.stderr)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def link_or_copy(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if src.exists() and dest.stat().st_size == src.stat().st_size:
            return "exists"
        dest.unlink()
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def selected_reference_paths(rows: list[dict[str, str]], source_dir: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for row in rows:
        founder = row["founder"]
        source_path = source_dir / founder / row["reference_file_name"]
        out.setdefault(founder, {})
        if row["selection_role"] == "primary_genome_fasta":
            out[founder]["fasta_source"] = source_path
            out[founder]["reference_assembly"] = row["reference_file_name"].removesuffix(".fa.gz")
        elif row["selection_role"] == "primary_gene_model_gff3":
            out[founder]["gff3_source"] = source_path
    return out


def parse_fasta_chrom_sizes(fasta_gz: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    current_name: Optional[str] = None
    current_size = 0
    with gzip.open(fasta_gz, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                if current_name is not None:
                    sizes[current_name] = current_size
                current_name = line[1:].strip().split()[0]
                current_size = 0
            else:
                current_size += len(line.strip())
    if current_name is not None:
        sizes[current_name] = current_size
    return sizes


def write_chrom_sizes(path: Path, sizes: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chrom, size in sizes.items():
            handle.write(f"{chrom}\t{size}\n")


def inspect_bigwig(path: Path) -> tuple[bool, dict[str, int], dict[str, float], str]:
    try:
        bw = pybigtools.open(str(path))
        try:
            chroms = {str(k): int(v) for k, v in bw.chroms().items()}
            info = bw.info()
        finally:
            bw.close()
        summary = info.get("summary", {}) if isinstance(info, dict) else {}
        stats = {
            "bases_covered": float(summary.get("basesCovered") or 0),
            "min_signal": float(summary.get("min") or 0),
            "max_signal": float(summary.get("max") or 0),
            "mean_signal": float(summary.get("mean") or 0),
            "sum_signal": float(summary.get("sum") or 0),
        }
        return True, chroms, stats, ""
    except Exception as exc:  # noqa: BLE001
        return False, {}, {
            "bases_covered": 0.0,
            "min_signal": 0.0,
            "max_signal": 0.0,
            "mean_signal": 0.0,
            "sum_signal": 0.0,
        }, f"{type(exc).__name__}: {exc}"


def chromosome_match(reference: dict[str, int], bigwig: dict[str, int]) -> tuple[str, str, int, int, int]:
    ref_keys = set(reference)
    bw_keys = set(bigwig)
    missing = sorted(ref_keys - bw_keys)
    extra = sorted(bw_keys - ref_keys)
    mismatched = sorted(chrom for chrom in ref_keys & bw_keys if reference[chrom] != bigwig[chrom])
    primary_mismatched = sorted(
        chrom for chrom in PRIMARY_CHROMS
        if chrom in reference and chrom in bigwig and reference[chrom] != bigwig[chrom]
    )
    missing_primary = sorted(chrom for chrom in PRIMARY_CHROMS if chrom in reference and chrom not in bigwig)
    extra_primary = sorted(chrom for chrom in PRIMARY_CHROMS if chrom in bigwig and chrom not in reference)

    if not missing and not extra and not mismatched:
        return "pass_exact", "", 0, 0, 0
    if not mismatched and not extra and not missing_primary:
        reason = f"compatible_subset: missing {len(missing)} reference chromosome/scaffold(s) from BigWig"
        return "fail_compatible_subset", reason, len(missing), len(extra), len(mismatched)
    reason_parts = []
    if missing:
        reason_parts.append(f"missing_in_bigwig={len(missing)}")
    if extra:
        reason_parts.append(f"extra_in_bigwig={len(extra)}")
    if mismatched:
        preview = ";".join(f"{c}:{reference[c]}!={bigwig[c]}" for c in mismatched[:5])
        reason_parts.append(f"size_mismatch={len(mismatched)}[{preview}]")
    if primary_mismatched or missing_primary or extra_primary:
        reason_parts.append("primary_chromosome_problem")
    return "fail_mismatch", "; ".join(reason_parts), len(missing), len(extra), len(mismatched)


def main_chromosome_match(reference: dict[str, int], bigwig: dict[str, int]) -> tuple[str, str]:
    expected = [chrom for chrom in sorted(PRIMARY_CHROMS, key=lambda c: int(c[3:])) if chrom in reference]
    missing = [chrom for chrom in expected if chrom not in bigwig]
    mismatched = [
        chrom for chrom in expected
        if chrom in bigwig and bigwig[chrom] != reference[chrom]
    ]
    if not missing and not mismatched and len(expected) == 10:
        return "pass", ""
    details = []
    if missing:
        details.append("missing=" + ";".join(missing))
    if mismatched:
        details.append(
            "size_mismatch=" + ";".join(
                f"{chrom}:ref={reference[chrom]},bw={bigwig.get(chrom, '')}"
                for chrom in mismatched
            )
        )
    if len(expected) != 10:
        details.append(f"reference_expected_main_count={len(expected)}")
    return "fail", "; ".join(details)


def signal_status(bigwig_opens: bool, stats: dict[str, float]) -> tuple[str, str]:
    if not bigwig_opens:
        return "fail", "BigWig did not open"
    if stats["bases_covered"] <= 0:
        return "fail", "BigWig has no bases covered"
    if stats["max_signal"] <= 0 or stats["sum_signal"] <= 0:
        return "fail", "BigWig has no non-zero signal"
    if stats["min_signal"] < 0:
        return "fail", "BigWig has negative signal values"
    return "pass", ""


def parse_track_fields(file_name: str) -> dict[str, str]:
    stem = file_name.removesuffix(".bw").removesuffix(".bigwig")
    stem = stem.replace(".Aligned.sortedByCoord.out", "")
    parts = stem.split("_")
    maize_line = parts[0] if parts else ""
    descriptor = "_".join(parts[1:])
    treatment = "none"
    development_stage = "unknown"
    tissue = "unknown"

    if descriptor.startswith("8DAS_root"):
        development_stage = "8DAS"
        tissue = "root"
    elif descriptor == "root":
        tissue = "root"
    elif descriptor == "leaf":
        tissue = "leaf"
    elif descriptor == "seedling_root":
        development_stage = "seedling"
        tissue = "root"
    elif descriptor == "immature_tassel":
        development_stage = "immature"
        tissue = "tassel"
    elif descriptor == "immature_unpollinated_ear_tip":
        development_stage = "immature"
        treatment = "unpollinated"
        tissue = "ear_tip"
    elif descriptor.startswith("whole_seed_") and "days_after_pollination" in descriptor:
        tissue = "whole_seed"
        development_stage = descriptor.removeprefix("whole_seed_")
    else:
        tissue = descriptor or "unknown"

    return {
        "maize_line": maize_line,
        "tissue": tissue,
        "development_stage": development_stage,
        "treatment": treatment,
        "normalization": "unknown_source_bigwig_signal",
    }


def track_id_from_name(file_name: str) -> str:
    stem = file_name.removesuffix(".bw").removesuffix(".bigwig")
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def prepare(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, Path]]]:
    bigwig_rows = read_csv(args.bigwig_reference_map)
    reference_rows = read_csv(args.selected_references)
    references = selected_reference_paths(reference_rows, args.reference_source_dir)
    prepared_refs = args.prepared_dir / "references"
    prepared_tracks = args.prepared_dir / "tracks"
    prepared_reports = args.prepared_dir / "reports"
    chrom_sizes_by_line: dict[str, dict[str, int]] = {}
    prepared_reference_paths: dict[str, dict[str, Path]] = {}

    for line, paths in sorted(references.items()):
        fasta_source = paths.get("fasta_source")
        gff3_source = paths.get("gff3_source")
        if not fasta_source or not fasta_source.exists():
            raise FileNotFoundError(f"Missing FASTA for {line}: {fasta_source}")
        if not gff3_source or not gff3_source.exists():
            raise FileNotFoundError(f"Missing GFF3 for {line}: {gff3_source}")

        line_ref_dir = prepared_refs / line
        fasta_dest = line_ref_dir / f"{line}.fa.gz"
        gff3_dest = line_ref_dir / f"{line}.gff3.gz"
        link_or_copy(fasta_source, fasta_dest)
        link_or_copy(gff3_source, gff3_dest)
        chrom_sizes_path = line_ref_dir / f"{line}.chrom.sizes"
        chrom_sizes = parse_fasta_chrom_sizes(fasta_dest)
        write_chrom_sizes(chrom_sizes_path, chrom_sizes)
        chrom_sizes_by_line[line] = chrom_sizes
        prepared_reference_paths[line] = {
            "reference_fasta": fasta_dest,
            "reference_gff3": gff3_dest,
            "chrom_sizes": chrom_sizes_path,
            "reference_assembly": paths["reference_assembly"],
        }

    metadata_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    for row in bigwig_rows:
        line = row["founder"]
        source_bw = resolve_path(row["local_path"])
        if not source_bw.exists():
            raise FileNotFoundError(f"Missing BigWig: {source_bw}")
        track_dir = prepared_tracks / line
        track_dest = track_dir / source_bw.name
        link_or_copy(source_bw, track_dest)

        bigwig_opens, bw_chroms, stats, open_error = inspect_bigwig(track_dest)
        ref_chroms = chrom_sizes_by_line[line]
        chrom_status, chrom_reason, missing_count, extra_count, mismatch_count = chromosome_match(ref_chroms, bw_chroms)
        main_chrom_status, main_chrom_reason = main_chromosome_match(ref_chroms, bw_chroms)
        sig_status, sig_reason = signal_status(bigwig_opens, stats)
        parsed = parse_track_fields(track_dest.name)

        practical_chrom_status = "pass_main_chromosomes" if main_chrom_status == "pass" else "fail_main_chromosomes"

        exclude_reasons = []
        if main_chrom_status != "pass":
            exclude_reasons.append(main_chrom_reason or practical_chrom_status)
        if sig_status != "pass":
            exclude_reasons.append(sig_reason)
        if parsed["tissue"] == "unknown":
            exclude_reasons.append("metadata tissue could not be parsed")
        if row.get("reference_match_for_workbook_srx") != "exact_srx":
            # Keep this as provenance only. These public MaizeGDB tracks are usable
            # as a NAM RNA-seq corpus, even though they are not exact workbook SRX files.
            pass

        use_for_training = (
            bigwig_opens
            and main_chrom_status == "pass"
            and sig_status == "pass"
            and parsed["tissue"] != "unknown"
        )
        exclude_reason = "; ".join(reason for reason in exclude_reasons if reason) if not use_for_training else ""

        metadata_rows.append({
            "track_id": track_id_from_name(track_dest.name),
            "species": "zea_mays",
            "maize_line": line,
            "reference_assembly": prepared_reference_paths[line]["reference_assembly"],
            "reference_fasta": str(prepared_reference_paths[line]["reference_fasta"]),
            "reference_gff3": str(prepared_reference_paths[line]["reference_gff3"]),
            "bigwig_file": str(track_dest),
            "assay": "RNA-seq",
            "tissue": parsed["tissue"],
            "development_stage": parsed["development_stage"],
            "treatment": parsed["treatment"],
            "normalization": parsed["normalization"],
            "chromosome_match_status": practical_chrom_status,
            "full_chromosome_match_status": chrom_status,
            "main_chromosome_match_status": main_chrom_status,
            "signal_status": sig_status,
            "use_for_training": str(use_for_training).lower(),
            "exclude_reason": exclude_reason,
        })

        qc_rows.append({
            "bigwig_file": str(track_dest),
            "maize_line": line,
            "expected_reference": prepared_reference_paths[line]["reference_assembly"],
            "bigwig_opens": str(bigwig_opens).lower(),
            "chromosome_match_status": practical_chrom_status,
            "full_chromosome_match_status": chrom_status,
            "main_chromosome_match_status": main_chrom_status,
            "main_chromosome_exclude_reason": main_chrom_reason,
            "n_bigwig_chromosomes": len(bw_chroms),
            "n_reference_chromosomes": len(ref_chroms),
            "missing_reference_chromosomes_in_bigwig": missing_count,
            "extra_bigwig_chromosomes": extra_count,
            "size_mismatched_chromosomes": mismatch_count,
            "bases_covered": int(stats["bases_covered"]),
            "min_signal": stats["min_signal"],
            "max_signal": stats["max_signal"],
            "mean_signal": stats["mean_signal"],
            "sum_signal": stats["sum_signal"],
            "signal_status": sig_status,
            "use_for_training": str(use_for_training).lower(),
            "exclude_reason": exclude_reason,
            "open_error": open_error,
        })

    track_metadata_fields = [
        "track_id",
        "species",
        "maize_line",
        "reference_assembly",
        "reference_fasta",
        "reference_gff3",
        "bigwig_file",
        "assay",
        "tissue",
        "development_stage",
        "treatment",
        "normalization",
        "chromosome_match_status",
        "full_chromosome_match_status",
        "main_chromosome_match_status",
        "signal_status",
        "use_for_training",
        "exclude_reason",
    ]
    qc_fields = [
        "bigwig_file",
        "maize_line",
        "expected_reference",
        "bigwig_opens",
        "chromosome_match_status",
        "full_chromosome_match_status",
        "main_chromosome_match_status",
        "main_chromosome_exclude_reason",
        "n_bigwig_chromosomes",
        "n_reference_chromosomes",
        "missing_reference_chromosomes_in_bigwig",
        "extra_bigwig_chromosomes",
        "size_mismatched_chromosomes",
        "bases_covered",
        "min_signal",
        "max_signal",
        "mean_signal",
        "sum_signal",
        "signal_status",
        "use_for_training",
        "exclude_reason",
        "open_error",
    ]
    write_csv(args.prepared_dir / "track_metadata.csv", metadata_rows, track_metadata_fields)
    write_csv(args.prepared_dir / "qc_report.csv", qc_rows, qc_fields)
    write_csv(prepared_reports / "track_metadata.csv", metadata_rows, track_metadata_fields)
    write_csv(prepared_reports / "qc_report.csv", qc_rows, qc_fields)
    return metadata_rows, qc_rows, prepared_reference_paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare organized references/tracks and QC reports.")
    parser.add_argument("--bigwig-reference-map", type=Path, default=DEFAULT_BIGWIG_MAP)
    parser.add_argument("--selected-references", type=Path, default=DEFAULT_SELECTED_REFERENCES)
    parser.add_argument("--reference-source-dir", type=Path, default=DEFAULT_REFERENCE_SOURCE_DIR)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metadata_rows, qc_rows, prepared_reference_paths = prepare(args)
    pass_count = sum(1 for row in metadata_rows if row["use_for_training"] == "true")
    print(f"Prepared references: {len(prepared_reference_paths)} maize lines")
    print(f"Prepared tracks: {len(metadata_rows)} BigWigs")
    print(f"Training-pass tracks: {pass_count}")
    print(f"Training-fail tracks: {len(metadata_rows) - pass_count}")
    print(f"Wrote {args.prepared_dir / 'track_metadata.csv'}")
    print(f"Wrote {args.prepared_dir / 'qc_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
