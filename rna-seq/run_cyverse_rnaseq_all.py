#!/usr/bin/env python3
"""Run the full CyVerse/MaizeGDB RNA-seq BigWig preparation pipeline.

Pipeline:
1. Read the NTv3 Excel RNA-seq rows and scan CyVerse/MaizeGDB.
2. Build and optionally download all selected BigWig candidates.
3. Audit BigWig headers and infer founder-specific references.
4. Locate and download matching MaizeGDB FASTA/GFF3 references.
5. Organize files by maize line and write QC/metadata reports.
6. Optionally create an S3 upload manifest. Upload is never performed here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
NTV3_ROOT = SCRIPT_DIR.parent
# Support both filename variations (with and without space)
DEFAULT_WORKBOOK = SCRIPT_DIR / "data" / "ntv3_maize_posttraining_data_extract 1.xlsx"
if not DEFAULT_WORKBOOK.exists():
    DEFAULT_WORKBOOK = SCRIPT_DIR / "data" / "ntv3_maize_posttraining_data_extract.xlsx"
DEFAULT_DATA_ROOT = NTV3_ROOT / "data" / "instadepp_rna_seq"
DEFAULT_S3_ROOT = "s3://us.com.syngenta.mlx.nonprod/GenAI_Platform/Farzana/posttraining_data/instadepp_rna_seq/"
STEPS = [
    "scan",
    "download_tracks",
    "audit_initial",
    "locate_references",
    "download_references",
    "audit_final",
    "prepare",
    "s3_manifest",
]


def script(name: str) -> Path:
    return SCRIPT_DIR / name


def pipeline_paths(data_root: Path) -> dict[str, Path]:
    manifests = data_root / "manifests"
    return {
        "scan_xlsx": manifests / "cyverse_rnaseq_scan.xlsx",
        "tracks_dir": data_root / "tracks_raw",
        "track_manifest": manifests / "track_download_manifest.csv",
        "track_status": manifests / "track_download_status.csv",
        "audit_csv": manifests / "bigwig_reference_audit.csv",
        "audit_md": NTV3_ROOT / "docs" / "cyverse_maizegdb_bigwig_reference_audit.md",
        "reference_manifest": manifests / "reference_download_manifest.csv",
        "selected_references": manifests / "reference_selected_manifest.csv",
        "reference_dir": data_root / "reference_genomes",
        "reference_status": manifests / "reference_download_status.csv",
        "bigwig_reference_map": manifests / "bigwig_to_reference_map.csv",
        "prepared_dir": data_root / "prepared",
        "s3_manifest": manifests / "s3_upload_manifest.csv",
    }


def step_range(start_at: str, stop_after: str) -> list[str]:
    start = STEPS.index(start_at)
    stop = STEPS.index(stop_after)
    if stop < start:
        raise SystemExit("--stop-after must be the same as or after --start-at")
    return STEPS[start:stop + 1]


def run_cmd(cmd: list[str], dry_run_commands: bool) -> None:
    print("\n" + " ".join(cmd))
    if dry_run_commands:
        return
    subprocess.run(cmd, check=True)


def maybe_skip(step: str, output: Path, resume: bool) -> bool:
    if resume and output.exists():
        print(f"Skipping {step}: found {output}")
        return True
    return False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all CyVerse/MaizeGDB RNA-seq download, reference, preparation, and QC steps.")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--start-at", choices=STEPS, default="scan")
    parser.add_argument("--stop-after", choices=STEPS, default="prepare")
    parser.add_argument("--resume", action="store_true", help="Skip steps whose primary output already exists")
    parser.add_argument("--dry-run-commands", action="store_true", help="Print commands without running them")
    parser.add_argument("--dry-run-downloads", action="store_true", help="Run scan/manifest steps but do not download tracks or references")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--scanner-min-score", type=int, default=8)
    parser.add_argument("--top-candidates-per-row", type=int, default=20)
    parser.add_argument("--download-min-score", type=int, default=15,
                        help="Minimum score for track downloads (default 15, was 38)")
    parser.add_argument("--confidence", default="exact_srx,exact_sample_accession,sample_name_metadata,study_plus_metadata,tissue_only")
    parser.add_argument("--limit-tracks", type=int, default=None, help="Limit BigWig downloads for a smoke test")
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--s3-root", default="", help="Optional destination prefix for an S3 upload manifest, e.g. s3://bucket/path")
    parser.add_argument("--include-md5", action="store_true", help="Compute MD5 checksums in the S3 manifest")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = pipeline_paths(args.data_root)
    args.data_root.mkdir(parents=True, exist_ok=True)
    paths["track_manifest"].parent.mkdir(parents=True, exist_ok=True)

    for step_name in step_range(args.start_at, args.stop_after):
        if step_name == "scan":
            if maybe_skip(step_name, paths["scan_xlsx"], args.resume):
                continue
            run_cmd([
                sys.executable,
                str(script("cyverse_maizegdb_track_scanner.py")),
                "--input", str(args.workbook),
                "--output", str(paths["scan_xlsx"]),
                "--sheet", "maize_func_tracks",
                "--assay-filter", "RNA-seq",
                "--species-filter", "zea_mays",
                "--dataset-filter", "ncbi_rnaseq",
                "--max-depth", str(args.max_depth),
                "--top-candidates-per-row", str(args.top_candidates_per_row),
                "--min-score", str(args.scanner_min_score),
            ], args.dry_run_commands)

        elif step_name == "download_tracks":
            if maybe_skip(step_name, paths["track_status"], args.resume):
                continue
            cmd = [
                sys.executable,
                str(script("download_cyverse_track_files.py")),
                "--input", str(paths["scan_xlsx"]),
                "--output-dir", str(paths["tracks_dir"]),
                "--manifest", str(paths["track_manifest"]),
                "--status-csv", str(paths["track_status"]),
                "--file-types", "BigWig",
                "--confidence", args.confidence,
                "--min-score", str(args.download_min_score),
            ]
            if args.limit_tracks is not None:
                cmd.extend(["--limit", str(args.limit_tracks)])
            if args.overwrite_downloads:
                cmd.append("--overwrite")
            if args.dry_run_downloads:
                cmd.append("--dry-run")
            run_cmd(cmd, args.dry_run_commands)

        elif step_name == "audit_initial":
            if maybe_skip(step_name, paths["audit_csv"], args.resume):
                continue
            run_cmd([
                sys.executable,
                str(script("audit_cyverse_bigwig_references.py")),
                "--manifest", str(paths["track_manifest"]),
                "--output-csv", str(paths["audit_csv"]),
                "--output-md", str(paths["audit_md"]),
                "--selected-references", str(paths["selected_references"]),
                "--bigwig-reference-map", str(paths["bigwig_reference_map"]),
            ], args.dry_run_commands)

        elif step_name == "locate_references":
            if maybe_skip(step_name, paths["selected_references"], args.resume):
                continue
            run_cmd([
                sys.executable,
                str(script("locate_maizegdb_nam_references.py")),
                "--audit", str(paths["audit_csv"]),
                "--output", str(paths["reference_manifest"]),
                "--selected-output", str(paths["selected_references"]),
            ], args.dry_run_commands)

        elif step_name == "download_references":
            if maybe_skip(step_name, paths["reference_status"], args.resume):
                continue
            cmd = [
                sys.executable,
                str(script("download_selected_references.py")),
                "--manifest", str(paths["selected_references"]),
                "--output-dir", str(paths["reference_dir"]),
                "--status-csv", str(paths["reference_status"]),
            ]
            if args.overwrite_downloads:
                cmd.append("--overwrite")
            if args.dry_run_downloads:
                cmd.append("--dry-run")
            run_cmd(cmd, args.dry_run_commands)

        elif step_name == "audit_final":
            run_cmd([
                sys.executable,
                str(script("audit_cyverse_bigwig_references.py")),
                "--manifest", str(paths["track_manifest"]),
                "--output-csv", str(paths["audit_csv"]),
                "--output-md", str(paths["audit_md"]),
                "--selected-references", str(paths["selected_references"]),
                "--bigwig-reference-map", str(paths["bigwig_reference_map"]),
            ], args.dry_run_commands)

        elif step_name == "prepare":
            if maybe_skip(step_name, paths["prepared_dir"] / "qc_report.csv", args.resume):
                continue
            run_cmd([
                sys.executable,
                str(script("prepare_cyverse_rnaseq_training_data.py")),
                "--bigwig-reference-map", str(paths["bigwig_reference_map"]),
                "--selected-references", str(paths["selected_references"]),
                "--reference-source-dir", str(paths["reference_dir"]),
                "--prepared-dir", str(paths["prepared_dir"]),
            ], args.dry_run_commands)

        elif step_name == "s3_manifest":
            if not args.s3_root:
                print("Skipping s3_manifest: provide --s3-root to create one")
                continue
            cmd = [
                sys.executable,
                str(script("make_s3_upload_manifest.py")),
                "--prepared-dir", str(paths["prepared_dir"]),
                "--s3-root", args.s3_root,
                "--output", str(paths["s3_manifest"]),
            ]
            if args.include_md5:
                cmd.append("--include-md5")
            run_cmd(cmd, args.dry_run_commands)

    print("\nPipeline command sequence complete.")
    print(f"Data root: {args.data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
