#!/usr/bin/env python3
"""Upload prepared RNA-seq files listed in an S3 manifest.

Default mode is dry-run. Actual upload requires --execute.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse


DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "instadepp_rna_seq"
DEFAULT_MANIFEST = DEFAULT_BASE_DIR / "manifests" / "s3_upload_manifest.csv"
DEFAULT_STATUS = DEFAULT_BASE_DIR / "manifests" / "s3_upload_status.csv"
DEFAULT_S3_ROOT = "s3://us.com.syngenta.mlx.nonprod/GenAI_Platform/Farzana/posttraining_data/instadepp_rna_seq/"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def write_status(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "status", "local_path", "s3_uri", "bytes", "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute uploads from an S3 manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--execute", action="store_true", help="Actually upload files. Without this flag, only a dry-run status is written.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting destination keys when executing")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = read_manifest(args.manifest)
    status_rows: list[dict[str, object]] = []

    if not args.execute:
        for row in rows:
            status_rows.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "dry_run",
                "local_path": row["local_path"],
                "s3_uri": row["s3_uri"],
                "bytes": row.get("bytes", ""),
                "error": "",
            })
        write_status(args.status_csv, status_rows)
        print(f"Dry-run rows: {len(status_rows)}")
        print(f"Wrote dry-run upload status: {args.status_csv}")
        print("No upload was performed. Re-run with --execute to upload.")
        return 0

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise SystemExit("Missing package: boto3. Install it or use AWS CLI separately from the manifest.") from exc

    s3 = boto3.client("s3")
    for row in rows:
        local_path = Path(row["local_path"])
        s3_uri = row["s3_uri"]
        status = "uploaded"
        error = ""
        try:
            bucket, key = parse_s3_uri(s3_uri)
            if not args.overwrite:
                try:
                    s3.head_object(Bucket=bucket, Key=key)
                    status = "skipped_existing"
                except ClientError as exc:
                    code = exc.response.get("Error", {}).get("Code")
                    if code not in {"404", "NoSuchKey", "NotFound"}:
                        raise
            if status == "uploaded":
                s3.upload_file(str(local_path), bucket, key)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        status_rows.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "local_path": str(local_path),
            "s3_uri": s3_uri,
            "bytes": row.get("bytes", ""),
            "error": error,
        })
        if error:
            print(f"ERROR {s3_uri}: {error}")

    write_status(args.status_csv, status_rows)
    print(f"Wrote upload status: {args.status_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
