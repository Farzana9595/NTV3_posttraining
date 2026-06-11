#!/usr/bin/env python3
"""Create an S3 upload manifest for prepared CyVerse/MaizeGDB RNA-seq data.

This script does not upload anything. It records every prepared reference,
track, and report file with the destination S3 URI that a later upload step
would use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "instadepp_rna_seq"
DEFAULT_PREPARED_DIR = DEFAULT_BASE_DIR / "prepared"
DEFAULT_OUTPUT = DEFAULT_BASE_DIR / "manifests" / "s3_upload_manifest.csv"
DEFAULT_S3_ROOT = "s3://us.com.syngenta.mlx.nonprod/GenAI_Platform/Farzana/posttraining_data/instadepp_rna_seq/"


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - checksum for transfer audit, not security
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_type_for(path: Path, prepared_dir: Path) -> str:
    rel = path.relative_to(prepared_dir).as_posix()
    name = path.name.lower()
    if rel.startswith("references/") and name.endswith((".fa.gz", ".fasta.gz", ".fna.gz")):
        return "reference_fasta"
    if rel.startswith("references/") and name.endswith((".gff3.gz", ".gff.gz", ".gff3", ".gff")):
        return "reference_gff3"
    if rel.startswith("references/") and name.endswith(".chrom.sizes"):
        return "chromosome_sizes"
    if rel.startswith("tracks/") and name.endswith((".bw", ".bigwig")):
        return "rna_seq_bigwig"
    if name.endswith(".csv"):
        return "metadata_or_qc_csv"
    return "other"


def iter_files(prepared_dir: Path) -> list[Path]:
    return sorted(path for path in prepared_dir.rglob("*") if path.is_file())


def build_rows(prepared_dir: Path, s3_root: str, include_md5: bool) -> list[dict[str, object]]:
    s3_root = s3_root.rstrip("/")
    rows: list[dict[str, object]] = []
    for path in iter_files(prepared_dir):
        rel = path.relative_to(prepared_dir).as_posix()
        rows.append({
            "local_path": str(path),
            "relative_path": rel,
            "file_type": file_type_for(path, prepared_dir),
            "bytes": path.stat().st_size,
            "md5": md5_file(path) if include_md5 else "",
            "s3_uri": f"{s3_root}/{rel}",
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["local_path", "relative_path", "file_type", "bytes", "md5", "s3_uri"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a dry-run S3 upload manifest for prepared RNA-seq data.")
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--s3-root", default=DEFAULT_S3_ROOT,
                        help=f"Destination prefix (default: {DEFAULT_S3_ROOT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-md5", action="store_true", help="Compute MD5 checksums for every file")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.prepared_dir.exists():
        raise SystemExit(f"Prepared directory does not exist: {args.prepared_dir}")
    if not args.s3_root.startswith("s3://"):
        raise SystemExit("--s3-root must start with s3://")
    rows = build_rows(args.prepared_dir, args.s3_root, args.include_md5)
    write_csv(args.output, rows)
    total_bytes = sum(int(row["bytes"]) for row in rows)
    print(f"Prepared files listed: {len(rows)}")
    print(f"Total bytes: {total_bytes}")
    print(f"Wrote S3 upload manifest: {args.output}")
    print("No upload was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
