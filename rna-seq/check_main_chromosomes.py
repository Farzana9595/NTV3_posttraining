#!/usr/bin/env python3
"""Check chr1..chr10 presence and size matches in prepared BigWigs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import pybigtools


NTV3_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED_DIR = NTV3_ROOT / "data" / "cyverse_maizegdb_rnaseq_tracks" / "prepared"
MAIN_CHROMS = [f"chr{i}" for i in range(1, 11)]


def read_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            name, size = line.rstrip("\n").split("\t")[:2]
            sizes[name] = int(size)
    return sizes


def read_bigwig_chroms(path: Path) -> dict[str, int]:
    bw = pybigtools.open(str(path))
    try:
        return {str(name): int(size) for name, size in bw.chroms().items()}
    finally:
        bw.close()


def check_track(bigwig_path: Path, prepared_dir: Path) -> dict[str, object]:
    maize_line = bigwig_path.parent.name
    reference_sizes = read_chrom_sizes(prepared_dir / "references" / maize_line / f"{maize_line}.chrom.sizes")
    bigwig_sizes = read_bigwig_chroms(bigwig_path)

    expected_main = [chrom for chrom in MAIN_CHROMS if chrom in reference_sizes]
    missing = [chrom for chrom in expected_main if chrom not in bigwig_sizes]
    size_mismatches = [
        chrom for chrom in expected_main
        if chrom in bigwig_sizes and bigwig_sizes[chrom] != reference_sizes[chrom]
    ]
    present = [chrom for chrom in expected_main if chrom in bigwig_sizes]
    status = "pass" if not missing and not size_mismatches and len(present) == len(expected_main) else "fail"

    mismatch_details = ";".join(
        f"{chrom}:ref={reference_sizes[chrom]},bw={bigwig_sizes.get(chrom, '')}"
        for chrom in size_mismatches
    )
    return {
        "bigwig_file": str(bigwig_path),
        "maize_line": maize_line,
        "main_chromosome_status": status,
        "expected_main_chromosomes": ";".join(expected_main),
        "present_main_chromosomes": ";".join(present),
        "missing_main_chromosomes": ";".join(missing),
        "size_mismatched_main_chromosomes": ";".join(size_mismatches),
        "size_mismatch_details": mismatch_details,
        "n_expected_main_chromosomes": len(expected_main),
        "n_present_main_chromosomes": len(present),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bigwig_file",
        "maize_line",
        "main_chromosome_status",
        "expected_main_chromosomes",
        "present_main_chromosomes",
        "missing_main_chromosomes",
        "size_mismatched_main_chromosomes",
        "size_mismatch_details",
        "n_expected_main_chromosomes",
        "n_present_main_chromosomes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check main chromosomes in prepared BigWig files.")
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREPARED_DIR / "main_chromosome_qc_report.csv")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tracks_dir = args.prepared_dir / "tracks"
    rows = [check_track(path, args.prepared_dir) for path in sorted(tracks_dir.rglob("*.bw"))]
    write_csv(args.output, rows)
    passing = sum(1 for row in rows if row["main_chromosome_status"] == "pass")
    print(f"Checked BigWigs: {len(rows)}")
    print(f"Main chromosome pass: {passing}")
    print(f"Main chromosome fail: {len(rows) - passing}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
