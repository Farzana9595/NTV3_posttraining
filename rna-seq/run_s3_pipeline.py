#!/usr/bin/env python3
"""Run the full RNA-seq pipeline with S3 integration.

This wrapper script:
1. Optionally pulls existing data from S3 before running
2. Runs the cyverse_rnaseq_all pipeline
3. Uploads results to S3 after completion

Target S3: s3://us.com.syngenta.mlx.nonprod/GenAI_Platform/Farzana/posttraining_data/instadepp_rna_seq/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
NTV3_ROOT = SCRIPT_DIR.parent
DEFAULT_S3_ROOT = "s3://us.com.syngenta.mlx.nonprod/GenAI_Platform/Farzana/posttraining_data/instadepp_rna_seq/"
DEFAULT_DATA_ROOT = NTV3_ROOT / "data" / "instadepp_rna_seq"

# Excel file with space in name
WORKBOOK_PATH = SCRIPT_DIR / "data" / "ntv3_maize_posttraining_data_extract 1.xlsx"
# Also check for standard name
WORKBOOK_PATH_ALT = SCRIPT_DIR / "data" / "ntv3_maize_posttraining_data_extract.xlsx"


def find_workbook() -> Path:
    """Find the workbook file."""
    if WORKBOOK_PATH.exists():
        return WORKBOOK_PATH
    if WORKBOOK_PATH_ALT.exists():
        return WORKBOOK_PATH_ALT
    # Check current directory
    for name in ["ntv3_maize_posttraining_data_extract 1.xlsx",
                 "ntv3_maize_posttraining_data_extract.xlsx"]:
        p = Path(name)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find workbook. Checked:\n"
        f"  {WORKBOOK_PATH}\n"
        f"  {WORKBOOK_PATH_ALT}"
    )


def run_cmd(cmd: list[str], dry_run: bool = False) -> int:
    """Run a command and return exit code."""
    print(f"\n>>> {' '.join(cmd)}")
    if dry_run:
        print("(dry-run: not executed)")
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def sync_from_s3(s3_root: str, local_root: Path, dry_run: bool = False) -> int:
    """Pull data from S3 to local using boto3."""
    print(f"\n{'='*60}")
    print("Syncing from S3 to local...")
    print(f"{'='*60}")

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        return 1

    local_root.mkdir(parents=True, exist_ok=True)

    # Parse S3 URI
    from urllib.parse import urlparse
    parsed = urlparse(s3_root)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    if not prefix.endswith('/'):
        prefix += '/'

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    downloaded = 0
    skipped = 0

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):
                    continue

                rel_path = key[len(prefix):] if key.startswith(prefix) else key
                local_file = local_root / rel_path

                if local_file.exists() and local_file.stat().st_size == obj['Size']:
                    skipped += 1
                    continue

                if dry_run:
                    print(f"[dry-run] Would download: {rel_path}")
                else:
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Downloading: {rel_path}")
                    s3.download_file(bucket, key, str(local_file))
                    downloaded += 1
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            print(f"Bucket does not exist: {bucket}")
        else:
            print(f"S3 Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    print(f"Downloaded: {downloaded}, Skipped: {skipped}")
    return 0


def sync_to_s3(local_root: Path, s3_root: str, dry_run: bool = False) -> int:
    """Push data from local to S3 using boto3."""
    print(f"\n{'='*60}")
    print("Syncing from local to S3...")
    print(f"{'='*60}")

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        return 1

    if not local_root.exists():
        print(f"Warning: Local root does not exist: {local_root}")
        return 1

    # Parse S3 URI
    from urllib.parse import urlparse
    parsed = urlparse(s3_root)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    if not prefix.endswith('/'):
        prefix += '/'

    s3 = boto3.client('s3')

    # Get existing S3 objects
    existing = {}
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                existing[obj['Key']] = obj['Size']
    except ClientError:
        pass  # Bucket might be empty or prefix doesn't exist yet

    uploaded = 0
    skipped = 0

    for local_file in local_root.rglob('*'):
        if not local_file.is_file():
            continue

        rel_path = local_file.relative_to(local_root).as_posix()
        s3_key = prefix + rel_path

        if s3_key in existing and existing[s3_key] == local_file.stat().st_size:
            skipped += 1
            continue

        if dry_run:
            print(f"[dry-run] Would upload: {rel_path}")
        else:
            print(f"Uploading: {rel_path}")
            try:
                s3.upload_file(str(local_file), bucket, s3_key)
                uploaded += 1
            except Exception as e:
                print(f"Error uploading {rel_path}: {e}")

    print(f"Uploaded: {uploaded}, Skipped: {skipped}")
    return 0


def run_pipeline(
    workbook: Path,
    data_root: Path,
    s3_root: str,
    limit_tracks: Optional[int] = None,
    start_at: str = "scan",
    stop_after: str = "s3_manifest",
    resume: bool = False,
    dry_run_downloads: bool = False,
    dry_run_commands: bool = False,
    include_md5: bool = False,
) -> int:
    """Run the main pipeline."""
    print(f"\n{'='*60}")
    print("Running RNA-seq Pipeline...")
    print(f"{'='*60}")
    print(f"Workbook: {workbook}")
    print(f"Data root: {data_root}")
    print(f"S3 root: {s3_root}")

    pipeline_script = SCRIPT_DIR / "run_cyverse_rnaseq_all.py"

    if not pipeline_script.exists():
        print(f"ERROR: Pipeline script not found: {pipeline_script}")
        print("The main pipeline scripts may not be fully set up yet.")
        print("Available scripts in rna-seq/:")
        for f in SCRIPT_DIR.glob("*.py"):
            print(f"  - {f.name}")
        return 1

    cmd = [
        sys.executable,
        str(pipeline_script),
        "--workbook", str(workbook),
        "--data-root", str(data_root),
        "--start-at", start_at,
        "--stop-after", stop_after,
        "--s3-root", s3_root,
    ]

    if limit_tracks is not None:
        cmd.extend(["--limit-tracks", str(limit_tracks)])
    if resume:
        cmd.append("--resume")
    if dry_run_downloads:
        cmd.append("--dry-run-downloads")
    if dry_run_commands:
        cmd.append("--dry-run-commands")
    if include_md5:
        cmd.append("--include-md5")

    return run_cmd(cmd)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RNA-seq pipeline with S3 integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
S3 Target: {DEFAULT_S3_ROOT}

Examples:
  # ★ FULL PIPELINE: all RNA-seq BigWigs → audit → references → prepare → upload S3
  python run_s3_pipeline.py --full-rna-seq

  # Resume full pipeline if interrupted
  python run_s3_pipeline.py --full-rna-seq --resume

  # Smoke test (5 samples, matched pipeline)
  python run_s3_pipeline.py --smoke-test 5

  # Download all RNA-seq BigWig files from CyVerse (~1344 files, ~126 GB)
  python run_s3_pipeline.py --download-all-bigwig

  # Just push local data to S3
  python run_s3_pipeline.py --push-only
"""
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--smoke-test", type=int, metavar="N",
                           help="Run smoke test with N samples (matched pipeline)")
    mode_group.add_argument("--full", action="store_true",
                           help="Run full matched pipeline (small subset, strict scoring)")
    mode_group.add_argument("--full-rna-seq", action="store_true",
                           help="Run FULL pipeline for ALL 1344 RNA-seq BigWig files → audit → references → prepare → S3")
    mode_group.add_argument("--download-all-bigwig", action="store_true",
                           help="Download all RNA-seq BigWig files from CyVerse (~1344 files, ~126 GB)")
    mode_group.add_argument("--pull-only", action="store_true",
                           help="Only pull data from S3")
    mode_group.add_argument("--push-only", action="store_true",
                           help="Only push data to S3")
    mode_group.add_argument("--status", action="store_true",
                           help="Show S3 and local status")

    # S3/Local paths
    parser.add_argument("--s3-root", default=DEFAULT_S3_ROOT,
                        help=f"S3 root (default: {DEFAULT_S3_ROOT})")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"Local data root (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--workbook", type=Path, default=None,
                        help="Excel workbook path (auto-detected if not specified)")

    # Pipeline control
    parser.add_argument("--start-at", default="scan",
                        choices=["scan", "download_tracks", "audit_initial",
                                "locate_references", "download_references",
                                "audit_final", "prepare", "s3_manifest"],
                        help="Start at this step")
    parser.add_argument("--stop-after", default="s3_manifest",
                        choices=["scan", "download_tracks", "audit_initial",
                                "locate_references", "download_references",
                                "audit_final", "prepare", "s3_manifest"],
                        help="Stop after this step")
    parser.add_argument("--resume", action="store_true",
                        help="Skip steps whose output already exists")

    # Sync control
    parser.add_argument("--skip-pull", action="store_true",
                        help="Skip pulling from S3 before running")
    parser.add_argument("--skip-push", action="store_true",
                        help="Skip pushing to S3 after running")

    # Dry run
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without doing it")
    parser.add_argument("--dry-run-downloads", action="store_true",
                        help="Run pipeline but don't actually download files")

    # Extra options
    parser.add_argument("--include-md5", action="store_true",
                        help="Compute MD5 checksums in S3 manifest")

    return parser.parse_args(argv)


def run_full_rna_seq_pipeline(args: argparse.Namespace, s3_root: str) -> int:
    """Run the complete RNA-seq pipeline for all 1344 RNA-seq BigWig files.

    Steps:
      1. Scan CyVerse (or reuse existing scan)
      2. Download all RNA-seq BigWig files → writes compatible track_download_manifest.csv
      3. Audit BigWig headers → infer reference genomes
      4. Locate + download reference FASTA/GFF3
      5. Final audit
      6. Prepare training data (organized by line, QC reports)
      7. Create S3 manifest
      8. Push everything to S3
    """
    data_root = args.data_root
    manifests = data_root / "manifests"
    data_root.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    resume = args.resume
    dry_run = args.dry_run

    scan_xlsx   = manifests / "cyverse_rnaseq_scan.xlsx"
    track_mfst  = manifests / "track_download_manifest.csv"
    track_status= manifests / "track_download_status.csv"
    audit_csv   = manifests / "bigwig_reference_audit.csv"
    audit_md    = NTV3_ROOT / "docs" / "cyverse_maizegdb_bigwig_reference_audit.md"
    ref_mfst    = manifests / "reference_download_manifest.csv"
    sel_refs    = manifests / "reference_selected_manifest.csv"
    ref_dir     = data_root / "reference_genomes"
    ref_status  = manifests / "reference_download_status.csv"
    bw_ref_map  = manifests / "bigwig_to_reference_map.csv"
    prepared    = data_root / "prepared"
    s3_manifest = manifests / "s3_upload_manifest.csv"
    tracks_dir  = data_root / "tracks_raw"

    # Find workbook
    try:
        workbook = find_workbook() if not args.workbook else args.workbook
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    print(f"\n{'='*60}")
    print("FULL RNA-seq Pipeline (all 1344 BigWig files)")
    print(f"{'='*60}")
    print(f"  Workbook : {workbook}")
    print(f"  Data root: {data_root}")
    print(f"  S3 root  : {s3_root}")

    # ── STEP 1: Scan CyVerse ────────────────────────────────────────────────
    if resume and scan_xlsx.exists():
        print(f"\n[SKIP] scan — found {scan_xlsx}")
    else:
        print("\n[STEP 1/8] Scanning CyVerse MaizeGDB...")
        rc = run_cmd([
            sys.executable, str(SCRIPT_DIR / "cyverse_maizegdb_track_scanner.py"),
            "--input", str(workbook),
            "--output", str(scan_xlsx),
            "--sheet", "maize_func_tracks",
            "--assay-filter", "RNA-seq",
            "--species-filter", "zea_mays",
            "--dataset-filter", "ncbi_rnaseq",
            "--max-depth", "6",
            "--top-candidates-per-row", "20",
            "--min-score", "8",
        ], dry_run)
        if rc != 0:
            print("ERROR: Scan step failed")
            return rc

    # ── STEP 2: Download all RNA-seq BigWig files ───────────────────────────
    if resume and track_status.exists():
        print(f"\n[SKIP] download_tracks — found {track_status}")
    else:
        print("\n[STEP 2/8] Downloading all RNA-seq BigWig files (~1344 files, ~126 GB)...")
        cmd = [
            sys.executable, str(SCRIPT_DIR / "download_all_cyverse_bigwig.py"),
            "--scan-xlsx", str(scan_xlsx),
            "--output-dir", str(tracks_dir),
            "--manifest", str(track_mfst),
            "--status-csv", str(track_status),
        ]
        if dry_run:
            cmd.append("--dry-run")
        rc = run_cmd(cmd, dry_run)
        if rc != 0:
            print("ERROR: Download step failed")
            return rc

    # ── STEP 3: Audit BigWig headers ────────────────────────────────────────
    if resume and audit_csv.exists():
        print(f"\n[SKIP] audit_initial — found {audit_csv}")
    else:
        print("\n[STEP 3/8] Auditing BigWig chromosome headers...")
        rc = run_cmd([
            sys.executable, str(SCRIPT_DIR / "audit_cyverse_bigwig_references.py"),
            "--manifest", str(track_mfst),
            "--output-csv", str(audit_csv),
            "--output-md", str(audit_md),
            "--selected-references", str(sel_refs),
            "--bigwig-reference-map", str(bw_ref_map),
        ], dry_run)
        if rc != 0:
            print("ERROR: Audit step failed")
            return rc

    # ── STEP 4: Locate reference FASTA/GFF3 on MaizeGDB ────────────────────
    if resume and sel_refs.exists():
        print(f"\n[SKIP] locate_references — found {sel_refs}")
    else:
        print("\n[STEP 4/8] Locating MaizeGDB reference genomes...")
        rc = run_cmd([
            sys.executable, str(SCRIPT_DIR / "locate_maizegdb_nam_references.py"),
            "--audit", str(audit_csv),
            "--output", str(ref_mfst),
            "--selected-output", str(sel_refs),
        ], dry_run)
        if rc != 0:
            print("ERROR: Locate references step failed")
            return rc

    # ── STEP 5: Download reference FASTA/GFF3 ──────────────────────────────
    if resume and ref_status.exists():
        print(f"\n[SKIP] download_references — found {ref_status}")
    else:
        print("\n[STEP 5/8] Downloading reference genomes (FASTA + GFF3)...")
        cmd = [
            sys.executable, str(SCRIPT_DIR / "download_selected_references.py"),
            "--manifest", str(sel_refs),
            "--output-dir", str(ref_dir),
            "--status-csv", str(ref_status),
        ]
        if dry_run:
            cmd.append("--dry-run")
        rc = run_cmd(cmd, dry_run)
        if rc != 0:
            print("ERROR: Download references step failed")
            return rc

    # ── STEP 6: Final audit with reference map ──────────────────────────────
    print("\n[STEP 6/8] Running final audit with reference map...")
    rc = run_cmd([
        sys.executable, str(SCRIPT_DIR / "audit_cyverse_bigwig_references.py"),
        "--manifest", str(track_mfst),
        "--output-csv", str(audit_csv),
        "--output-md", str(audit_md),
        "--selected-references", str(sel_refs),
        "--bigwig-reference-map", str(bw_ref_map),
    ], dry_run)
    if rc != 0:
        print("WARNING: Final audit step failed, continuing...")

    # ── STEP 7: Prepare training data ───────────────────────────────────────
    if resume and (prepared / "qc_report.csv").exists():
        print(f"\n[SKIP] prepare — found {prepared / 'qc_report.csv'}")
    else:
        print("\n[STEP 7/8] Preparing training data (organize by line, QC reports)...")
        rc = run_cmd([
            sys.executable, str(SCRIPT_DIR / "prepare_cyverse_rnaseq_training_data.py"),
            "--bigwig-reference-map", str(bw_ref_map),
            "--selected-references", str(sel_refs),
            "--reference-source-dir", str(ref_dir),
            "--prepared-dir", str(prepared),
        ], dry_run)
        if rc != 0:
            print("ERROR: Prepare step failed")
            return rc

    # ── STEP 8: S3 manifest + push ──────────────────────────────────────────
    print("\n[STEP 8/8] Creating S3 manifest and uploading to S3...")
    rc = run_cmd([
        sys.executable, str(SCRIPT_DIR / "make_s3_upload_manifest.py"),
        "--prepared-dir", str(prepared),
        "--s3-root", s3_root,
        "--output", str(s3_manifest),
    ], dry_run)

    if not args.skip_push:
        push_rc = sync_to_s3(data_root, s3_root, dry_run)
        if push_rc != 0:
            print("WARNING: S3 push had issues")

    print(f"\n{'='*60}")
    print("Full RNA-seq Pipeline Complete!")
    print(f"{'='*60}")
    print(f"  BigWig tracks : {tracks_dir}")
    print(f"  References    : {ref_dir}")
    print(f"  Prepared data : {prepared}")
    print(f"  S3            : {s3_root}")
    return 0


def show_status(s3_root: str, local_root: Path) -> int:
    """Show status using s3_data_manager."""
    manager_script = SCRIPT_DIR / "s3_data_manager.py"
    if manager_script.exists():
        return run_cmd([
            sys.executable,
            str(manager_script),
            "status",
            "--s3-root", s3_root,
            "--local-root", str(local_root),
        ])
    else:
        print(f"S3 Root: {s3_root}")
        print(f"Local Root: {local_root}")
        print(f"Local exists: {local_root.exists()}")
        return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Ensure S3 root ends with /
    s3_root = args.s3_root.rstrip('/') + '/'

    print(f"{'='*60}")
    print("RNA-seq Pipeline with S3 Integration")
    print(f"{'='*60}")
    print(f"S3 Root: {s3_root}")
    print(f"Local Root: {args.data_root}")

    # Handle status
    if args.status:
        return show_status(s3_root, args.data_root)

    # Handle pull-only
    if args.pull_only:
        return sync_from_s3(s3_root, args.data_root, args.dry_run)

    # Handle push-only
    if args.push_only:
        return sync_to_s3(args.data_root, s3_root, args.dry_run)

    # Handle download-all-bigwig
    if args.download_all_bigwig:
        scan_xlsx = args.data_root / "manifests" / "cyverse_rnaseq_scan.xlsx"
        if not scan_xlsx.exists():
            print(f"ERROR: Scan file not found: {scan_xlsx}")
            print("Run the pipeline with --full first to generate the scan, or run:")
            print(f"  python {SCRIPT_DIR / 'cyverse_maizegdb_track_scanner.py'} --input <workbook> --output {scan_xlsx}")
            return 1
        
        download_script = SCRIPT_DIR / "download_all_cyverse_bigwig.py"
        cmd = [
            sys.executable,
            str(download_script),
            "--scan-xlsx", str(scan_xlsx),
            "--output-dir", str(args.data_root / "tracks_raw"),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        
        print(f"\n{'='*60}")
        print("Downloading ALL BigWig files from CyVerse...")
        print(f"{'='*60}")
        rc = run_cmd(cmd)
        
        if rc == 0 and not args.skip_push:
            sync_to_s3(args.data_root, s3_root, args.dry_run)
        
        return rc

    # Handle full-rna-seq: complete pipeline for all 1344 RNA-seq BigWig files
    if args.full_rna_seq:
        return run_full_rna_seq_pipeline(args, s3_root)

    # Find workbook
    if args.workbook:
        workbook = args.workbook
    else:
        try:
            workbook = find_workbook()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1

    print(f"Workbook: {workbook}")

    # Determine limit
    limit_tracks = args.smoke_test if args.smoke_test else None

    # Step 1: Pull from S3 (unless skipped)
    if not args.skip_pull:
        rc = sync_from_s3(s3_root, args.data_root, args.dry_run)
        if rc != 0 and not args.dry_run:
            print("Warning: S3 pull had issues, continuing anyway...")

    # Step 2: Run pipeline
    rc = run_pipeline(
        workbook=workbook,
        data_root=args.data_root,
        s3_root=s3_root,
        limit_tracks=limit_tracks,
        start_at=args.start_at,
        stop_after=args.stop_after,
        resume=args.resume,
        dry_run_downloads=args.dry_run_downloads or args.dry_run,
        dry_run_commands=args.dry_run,
        include_md5=args.include_md5,
    )

    if rc != 0:
        print(f"\nPipeline finished with return code: {rc}")
        if not args.skip_push:
            print("Pushing partial results to S3 anyway...")

    # Step 3: Push to S3 (unless skipped)
    if not args.skip_push:
        push_rc = sync_to_s3(args.data_root, s3_root, args.dry_run)
        if push_rc != 0:
            print("Warning: S3 push had issues")

    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Local data: {args.data_root}")
    print(f"S3 data: {s3_root}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())


