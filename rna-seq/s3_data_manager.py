#!/usr/bin/env python3
"""S3 Data Manager for RNA-seq Pipeline.

This script handles all S3 operations:
- Sync data between local and S3
- List S3 contents
- Download/upload files and directories
- Set up working directory from S3 or initialize fresh

Target S3: s3://us.com.syngenta.mlx.nonprod/GenAI_Platform/Farzana/posttraining_data/instadepp_rna_seq/
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

# S3 Configuration
DEFAULT_S3_ROOT = "s3://us.com.syngenta.mlx.nonprod/GenAI_Platform/Farzana/posttraining_data/instadepp_rna_seq/"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_ROOT = SCRIPT_DIR.parent / "data" / "instadepp_rna_seq"


def get_s3_client():
    """Get boto3 S3 client."""
    import boto3
    return boto3.client('s3')


def get_s3_resource():
    """Get boto3 S3 resource."""
    import boto3
    return boto3.resource('s3')


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse S3 URI into bucket and key."""
    parsed = urlparse(uri)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def list_s3_contents(s3_uri: str, recursive: bool = True) -> list[dict]:
    """List contents of an S3 path."""
    s3 = get_s3_client()
    bucket, prefix = parse_s3_uri(s3_uri)

    # Ensure prefix ends with / for directories
    if prefix and not prefix.endswith('/'):
        prefix = prefix + '/'

    contents = []
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            contents.append({
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'].isoformat(),
                's3_uri': f"s3://{bucket}/{obj['Key']}"
            })

    return contents


def sync_s3_to_local(s3_uri: str, local_path: Path, dry_run: bool = False) -> None:
    """Sync S3 directory to local using boto3."""
    import boto3
    from botocore.exceptions import ClientError

    local_path.mkdir(parents=True, exist_ok=True)
    bucket, prefix = parse_s3_uri(s3_uri)
    if not prefix.endswith('/'):
        prefix += '/'

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    downloaded = 0
    skipped = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'):  # Skip "directories"
                continue

            rel_path = key[len(prefix):] if key.startswith(prefix) else key
            local_file = local_path / rel_path

            # Skip if local file exists and is same size
            if local_file.exists() and local_file.stat().st_size == obj['Size']:
                skipped += 1
                continue

            if dry_run:
                print(f"[dry-run] Would download: {key} -> {local_file}")
            else:
                local_file.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading: {rel_path}")
                s3.download_file(bucket, key, str(local_file))
                downloaded += 1

    print(f"Downloaded: {downloaded}, Skipped: {skipped}")


def sync_local_to_s3(local_path: Path, s3_uri: str, dry_run: bool = False) -> None:
    """Sync local directory to S3 using boto3."""
    import boto3
    from botocore.exceptions import ClientError

    if not local_path.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    bucket, prefix = parse_s3_uri(s3_uri)
    if not prefix.endswith('/'):
        prefix += '/'

    s3 = boto3.client('s3')

    # Get existing S3 objects for comparison
    existing = {}
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            existing[obj['Key']] = obj['Size']

    uploaded = 0
    skipped = 0

    for local_file in local_path.rglob('*'):
        if not local_file.is_file():
            continue

        rel_path = local_file.relative_to(local_path).as_posix()
        s3_key = prefix + rel_path

        # Skip if S3 object exists and is same size
        if s3_key in existing and existing[s3_key] == local_file.stat().st_size:
            skipped += 1
            continue

        if dry_run:
            print(f"[dry-run] Would upload: {local_file} -> s3://{bucket}/{s3_key}")
        else:
            print(f"Uploading: {rel_path}")
            s3.upload_file(str(local_file), bucket, s3_key)
            uploaded += 1

    print(f"Uploaded: {uploaded}, Skipped: {skipped}")


def download_file(s3_uri: str, local_path: Path, overwrite: bool = False) -> bool:
    """Download a single file from S3."""
    if local_path.exists() and not overwrite:
        print(f"Skipping existing: {local_path}")
        return False

    s3 = get_s3_client()
    bucket, key = parse_s3_uri(s3_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {s3_uri} -> {local_path}")
    s3.download_file(bucket, key, str(local_path))
    return True


def upload_file(local_path: Path, s3_uri: str, overwrite: bool = False) -> bool:
    """Upload a single file to S3."""
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    s3 = get_s3_client()
    bucket, key = parse_s3_uri(s3_uri)

    # Check if exists
    if not overwrite:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"Skipping existing: {s3_uri}")
            return False
        except:
            pass

    print(f"Uploading: {local_path} -> {s3_uri}")
    s3.upload_file(str(local_path), bucket, key)
    return True


def init_s3_structure(s3_root: str) -> None:
    """Initialize the S3 folder structure for the pipeline."""
    s3 = get_s3_client()
    bucket, prefix = parse_s3_uri(s3_root)

    # Ensure prefix ends with /
    if not prefix.endswith('/'):
        prefix = prefix + '/'

    # Create marker files to establish directory structure
    folders = [
        "manifests/",
        "tracks_raw/",
        "reference_genomes/",
        "prepared/references/",
        "prepared/tracks/",
    ]

    for folder in folders:
        key = prefix + folder + ".gitkeep"
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=b'')
            print(f"Created: s3://{bucket}/{key}")
        except Exception as e:
            print(f"Warning: Could not create {key}: {e}")


def show_status(s3_root: str, local_root: Path) -> None:
    """Show current status of S3 and local data."""
    print(f"\n{'='*60}")
    print("RNA-seq Pipeline S3 Data Status")
    print(f"{'='*60}")
    print(f"\nS3 Root: {s3_root}")
    print(f"Local Root: {local_root}")

    print(f"\n--- S3 Contents ---")
    try:
        contents = list_s3_contents(s3_root)
        if contents:
            total_size = sum(c['size'] for c in contents)
            print(f"Total files: {len(contents)}")
            print(f"Total size: {total_size / (1024*1024*1024):.2f} GB")

            # Group by top-level folder
            by_folder = {}
            bucket, prefix = parse_s3_uri(s3_root)
            if not prefix.endswith('/'):
                prefix += '/'
            for c in contents:
                rel = c['key'][len(prefix):] if c['key'].startswith(prefix) else c['key']
                folder = rel.split('/')[0] if '/' in rel else '(root)'
                by_folder.setdefault(folder, []).append(c)

            for folder, files in sorted(by_folder.items()):
                folder_size = sum(f['size'] for f in files)
                print(f"  {folder}/: {len(files)} files, {folder_size / (1024*1024):.1f} MB")
        else:
            print("No files found in S3")
    except Exception as e:
        print(f"Error listing S3: {e}")

    print(f"\n--- Local Contents ---")
    if local_root.exists():
        local_files = list(local_root.rglob("*"))
        local_files = [f for f in local_files if f.is_file()]
        if local_files:
            total_size = sum(f.stat().st_size for f in local_files)
            print(f"Total files: {len(local_files)}")
            print(f"Total size: {total_size / (1024*1024*1024):.2f} GB")
        else:
            print("No files found locally")
    else:
        print("Local directory does not exist")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage S3 data for RNA-seq pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current status
  python s3_data_manager.py status
  
  # Initialize S3 folder structure
  python s3_data_manager.py init
  
  # Download everything from S3 to local
  python s3_data_manager.py pull
  
  # Upload everything from local to S3
  python s3_data_manager.py push
  
  # List S3 contents
  python s3_data_manager.py list
  
  # Custom S3 path
  python s3_data_manager.py status --s3-root s3://bucket/path/
"""
    )
    parser.add_argument("action", choices=["status", "init", "pull", "push", "list"],
                        help="Action to perform")
    parser.add_argument("--s3-root", default=DEFAULT_S3_ROOT,
                        help=f"S3 root path (default: {DEFAULT_S3_ROOT})")
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT,
                        help=f"Local root path (default: {DEFAULT_LOCAL_ROOT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without doing it")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Ensure S3 root ends with /
    s3_root = args.s3_root.rstrip('/') + '/'

    if args.action == "status":
        show_status(s3_root, args.local_root)

    elif args.action == "init":
        print(f"Initializing S3 structure at: {s3_root}")
        init_s3_structure(s3_root)
        print("Done!")

    elif args.action == "pull":
        print(f"Pulling from S3 to local...")
        print(f"  S3: {s3_root}")
        print(f"  Local: {args.local_root}")
        sync_s3_to_local(s3_root, args.local_root, args.dry_run)
        print("Done!")

    elif args.action == "push":
        print(f"Pushing from local to S3...")
        print(f"  Local: {args.local_root}")
        print(f"  S3: {s3_root}")
        sync_local_to_s3(args.local_root, s3_root, args.dry_run)
        print("Done!")

    elif args.action == "list":
        contents = list_s3_contents(s3_root)
        for c in contents:
            print(f"{c['size']:>12}  {c['last_modified'][:19]}  {c['s3_uri']}")
        print(f"\nTotal: {len(contents)} files")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



