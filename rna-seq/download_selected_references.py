#!/usr/bin/env python3
"""Download selected MaizeGDB reference FASTA/GFF files."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional, Sequence

import requests


DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "instadepp_rna_seq"
DEFAULT_MANIFEST = DEFAULT_BASE_DIR / "manifests" / "reference_selected_manifest.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "reference_genomes"
DEFAULT_STATUS = DEFAULT_BASE_DIR / "manifests" / "reference_download_status.csv"


def read_rows(path: Path, roles: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if not roles or row.get("selection_role") in roles]


def download_one(session: requests.Session, url: str, local_path: Path, timeout: int, overwrite: bool) -> tuple[str, int, str]:
    if local_path.exists() and local_path.stat().st_size > 0 and not overwrite:
        return "skipped_existing", local_path.stat().st_size, ""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = local_path.with_suffix(local_path.suffix + ".part")
    try:
        total = 0
        with session.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with part_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    total += len(chunk)
        part_path.replace(local_path)
        return "downloaded", total, ""
    except Exception as exc:  # noqa: BLE001
        return "error", 0, f"{type(exc).__name__}: {exc}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected MaizeGDB reference files.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--roles", default="primary_genome_fasta,primary_gene_model_gff3", help="Comma-separated selection_role values to download")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List selected reference files but do not download")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    roles = {role.strip() for role in args.roles.split(",") if role.strip()}
    rows = read_rows(args.manifest, roles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.status_csv.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "maizegdb-reference-downloader/1.0"})

    print(f"Selected {len(rows)} reference file(s)")
    if args.dry_run:
        for row in rows:
            local_path = args.output_dir / row["founder"] / row["reference_file_name"]
            print(f"{row['reference_url']} -> {local_path}")
        print("No reference files were downloaded.")
        return 0

    with args.status_csv.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = ["timestamp", "founder", "selection_role", "status", "bytes", "local_path", "url", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if handle.tell() == 0:
            writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            local_path = args.output_dir / row["founder"] / row["reference_file_name"]
            url = row["reference_url"]
            print(f"[{idx}/{len(rows)}] {row['founder']} {row['reference_file_name']}")
            status, byte_count, error = download_one(session, url, local_path, args.timeout, args.overwrite)
            writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "founder": row["founder"],
                "selection_role": row["selection_role"],
                "status": status,
                "bytes": byte_count,
                "local_path": str(local_path),
                "url": url,
                "error": error,
            })
            handle.flush()
            if error:
                print(f"ERROR: {error}")
    print(f"Wrote status: {args.status_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
