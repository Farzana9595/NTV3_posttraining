# CyVerse/MaizeGDB RNA-seq All-Sample Pipeline

This workflow starts from `maize_rnaseq_check/ntv3_maize_posttraining_data_extract.xlsx`, filters the `maize_func_tracks` sheet to Zea mays `RNA-seq` rows from `ncbi_rnaseq`, searches CyVerse/MaizeGDB for BigWig tracks, downloads the selected tracks, downloads matching MaizeGDB founder references, and prepares training-ready metadata/QC reports.

Important: the public MaizeGDB tracks found in CyVerse are NAM/B73 browser tracks. They are not exact SRX/SRR workbook files unless the scanner reports exact accession evidence. The QC step therefore treats them as a public MaizeGDB/NAM RNA-seq corpus and requires the BigWig main chromosomes to match the corresponding founder FASTA.

## Output Layout

Default all-sample output root:

```text
NTv3_post_training_data/data/cyverse_maizegdb_rnaseq_all/
  manifests/
    cyverse_rnaseq_scan.xlsx
    track_download_manifest.csv
    track_download_status.csv
    bigwig_reference_audit.csv
    bigwig_to_reference_map.csv
    reference_download_manifest.csv
    reference_selected_manifest.csv
    reference_download_status.csv
    s3_upload_manifest.csv
  tracks_raw/
  reference_genomes/
  prepared/
    references/<maize_line>/<maize_line>.fa.gz
    references/<maize_line>/<maize_line>.gff3.gz
    references/<maize_line>/<maize_line>.chrom.sizes
    tracks/<maize_line>/*.bw
    track_metadata.csv
    qc_report.csv
```

## Full Run

This downloads selected BigWig tracks and matching FASTA/GFF3 references, then prepares QC outputs:

```powershell
python NTv3_post_training_data\maize_rnaseq_check\run_cyverse_rnaseq_all.py
```

Useful options:

```powershell
python NTv3_post_training_data\maize_rnaseq_check\run_cyverse_rnaseq_all.py `
  --download-min-score 38 `
  --confidence exact_srx,exact_sample_accession,sample_name_metadata,study_plus_metadata `
  --max-depth 6 `
  --top-candidates-per-row 20
```

## Smoke Test

Run only a small subset first:

```powershell
python NTv3_post_training_data\maize_rnaseq_check\run_cyverse_rnaseq_all.py `
  --limit-tracks 30
```

Resume later without repeating completed outputs:

```powershell
python NTv3_post_training_data\maize_rnaseq_check\run_cyverse_rnaseq_all.py --resume
```

## Step-by-Step Control

Stop after scanning and manifest creation:

```powershell
python NTv3_post_training_data\maize_rnaseq_check\run_cyverse_rnaseq_all.py --stop-after download_tracks --dry-run-downloads
```

Start from reference discovery after tracks are already downloaded and audited:

```powershell
python NTv3_post_training_data\maize_rnaseq_check\run_cyverse_rnaseq_all.py --start-at locate_references
```

## S3 Manifest Only

Create an upload manifest without uploading:

```powershell
python NTv3_post_training_data\maize_rnaseq_check\run_cyverse_rnaseq_all.py `
  --start-at s3_manifest `
  --stop-after s3_manifest `
  --s3-root s3://YOUR_BUCKET/YOUR_PREFIX/cyverse_maizegdb_rnaseq_all
```

Or run the manifest script directly:

```powershell
python NTv3_post_training_data\maize_rnaseq_check\make_s3_upload_manifest.py `
  --prepared-dir NTv3_post_training_data\data\cyverse_maizegdb_rnaseq_all\prepared `
  --s3-root s3://YOUR_BUCKET/YOUR_PREFIX/cyverse_maizegdb_rnaseq_all
```

Dry-run the upload manifest:

```powershell
python NTv3_post_training_data\maize_rnaseq_check\upload_s3_manifest.py
```

Actual S3 upload requires `--execute`; do not use it until the manifest and QC reports are reviewed.

## QC Rules

A track is marked usable only when:

- the BigWig opens;
- the matching founder FASTA exists;
- main chromosomes `chr1` through `chr10` in the BigWig match the FASTA names and sizes;
- the BigWig has covered bases and non-zero, non-negative signal;
- basic tissue metadata can be parsed from the MaizeGDB filename.

Scaffolds and extra reference contigs may be present in the FASTA but absent from a BigWig. That is acceptable when the ten main chromosomes match.
