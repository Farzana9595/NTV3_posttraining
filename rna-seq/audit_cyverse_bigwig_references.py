#!/usr/bin/env python3
"""Audit downloaded CyVerse/MaizeGDB RNA-seq BigWig reference assemblies.

The script reads a download manifest, inspects BigWig chromosome headers with
pybigtools, and writes a CSV plus Markdown summary with inferred reference
genome/version evidence.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

import pybigtools


DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "cyverse_maizegdb_rnaseq_all"
DEFAULT_MANIFEST = DEFAULT_BASE_DIR / "manifests" / "track_download_manifest.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_BASE_DIR / "manifests" / "bigwig_reference_audit.csv"
DEFAULT_OUTPUT_MD = Path(__file__).resolve().parents[1] / "docs" / "cyverse_maizegdb_bigwig_reference_audit.md"
DEFAULT_SELECTED_REFERENCES = DEFAULT_BASE_DIR / "manifests" / "reference_selected_manifest.csv"
DEFAULT_BIGWIG_REFERENCE_MAP = DEFAULT_BASE_DIR / "manifests" / "bigwig_to_reference_map.csv"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def founder_from_file_name(file_name: str) -> str:
    match = re.match(r"([A-Za-z0-9]+)_", file_name)
    return match.group(1) if match else ""


def genome_from_url(url: str, file_name: str) -> tuple[str, str, str]:
    """Return inferred assembly, coordinate basis, and evidence."""
    lower_url = url.lower()
    if "nam_project_jbrowse_and_analyses/nam_rna-seq" in lower_url:
        founder = founder_from_file_name(file_name)
        if "nam_consortium_2020" in lower_url:
            return (
                f"NAM founder {founder} reference assembly, NAM v5.0 family" if founder else "NAM founder reference assemblies, NAM v5.0 family",
                "founder-specific NAM coordinates",
                "CyVerse path NAM_PROJECT_JBROWSE_AND_ANALYSES/NAM_RNA-seq/NAM_Consortium_2020 plus founder prefix in filename",
            )
        if "lin_2017" in lower_url or "diepenbrock_2016" in lower_url:
            return (
                f"NAM founder {founder} reference assembly, NAM v5.0 family",
                "founder-specific NAM coordinates",
                "CyVerse path NAM_PROJECT_JBROWSE_AND_ANALYSES/NAM_RNA-seq plus founder prefix in filename",
            )
        return (
            "NAM founder reference assemblies, NAM v5.0 family",
            "founder-specific NAM coordinates",
            "CyVerse path NAM_PROJECT_JBROWSE_AND_ANALYSES/NAM_RNA-seq",
        )
    if "b73v5_jbrowse_and_analyses/b73v5_rna-seq" in lower_url:
        return (
            "Zea mays B73 RefGen_v5 / Zm-B73-REFERENCE-NAM-5.0",
            "B73v5 coordinates",
            "CyVerse path B73v5_JBROWSE_AND_ANALYSES/B73v5_RNA-seq",
        )
    if "b73_refgen_v4" in lower_url or "agpv4" in lower_url:
        return ("B73 RefGen_v4 / AGPv4", "B73v4 coordinates", "URL contains B73_RefGen_v4 or AGPv4")
    return ("unknown", "unknown", "No explicit assembly marker in URL")


def chrom_pattern(chroms: dict[str, int]) -> str:
    names = list(chroms)
    if all(re.fullmatch(r"chr[1-9]|chr10", name) for name in names[:10]):
        return "chr1..chr10"
    if all(re.fullmatch(r"[1-9]|10", name) for name in names[:10]):
        return "1..10"
    if names and all(name.startswith("Zm") for name in names[:10]):
        return "Zm-prefixed contigs"
    return ";".join(names[:10])


def inspect_bigwig(path: Path) -> tuple[int, int, str, str]:
    bw = pybigtools.open(str(path))
    try:
        chroms = bw.chroms()
        info = bw.info()
    finally:
        bw.close()
    total_bp = sum(int(size) for size in chroms.values())
    first_chroms = ";".join(f"{name}:{size}" for name, size in list(chroms.items())[:12])
    return len(chroms), total_bp, chrom_pattern(chroms), first_chroms + f"; basesCovered={info.get('basesCovered', '')}"


def audit_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        local_path = Path(row["local_path"])
        assembly, coordinate_basis, evidence = genome_from_url(row["matched_file_url"], row["matched_file_name"])
        chrom_count = total_bp = 0
        chroms = error = ""
        if local_path.exists():
            try:
                chrom_count, total_bp, pattern, chroms = inspect_bigwig(local_path)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                pattern = "unreadable"
        else:
            error = "local_path_not_found"
            pattern = "missing"
        out.append({
            "file_id": row.get("file_id", ""),
            "source_excel_row": row.get("source_excel_row", ""),
            "founder": founder_from_file_name(row.get("matched_file_name", "")),
            "matched_file_name": row.get("matched_file_name", ""),
            "matched_file_url": row.get("matched_file_url", ""),
            "local_path": row.get("local_path", ""),
            "inferred_reference_genome": assembly,
            "coordinate_basis": coordinate_basis,
            "reference_evidence": evidence,
            "chromosome_count": str(chrom_count),
            "total_chromosome_bp": str(total_bp),
            "chromosome_name_pattern": pattern,
            "first_chromosomes": chroms,
            "reference_match_for_workbook_srx": "no_exact_srx_match",
            "use_with_ntv3_jhax7": "requires_liftover_or_realignment",
            "error": error,
        })
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_selected_references(path: Path) -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = defaultdict(dict)
    if not path.exists():
        return refs
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            founder = row["founder"]
            if row["selection_role"] == "primary_genome_fasta":
                refs[founder]["reference_fasta_name"] = row["reference_file_name"]
                refs[founder]["reference_fasta_url"] = row["reference_url"]
            elif row["selection_role"] == "primary_gene_model_gff3":
                refs[founder]["reference_gff3_name"] = row["reference_file_name"]
                refs[founder]["reference_gff3_url"] = row["reference_url"]
    return refs


def write_bigwig_reference_map(
    path: Path,
    audit_rows: list[dict[str, str]],
    selected_references_path: Path,
) -> None:
    refs = read_selected_references(selected_references_path)
    fieldnames = [
        "file_id",
        "source_excel_row",
        "matched_file_name",
        "founder",
        "inferred_reference_genome",
        "coordinate_basis",
        "chromosome_count",
        "total_chromosome_bp",
        "reference_fasta_name",
        "reference_fasta_url",
        "reference_gff3_name",
        "reference_gff3_url",
        "local_path",
        "matched_file_url",
        "reference_match_for_workbook_srx",
        "use_with_ntv3_jhax7",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit_rows:
            ref = refs.get(row["founder"], {})
            writer.writerow({key: row.get(key, ref.get(key, "")) for key in fieldnames})


def write_md(path: Path, rows: list[dict[str, str]], csv_path: Path, selected_refs_path: Path, map_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assembly_counts = Counter(row["inferred_reference_genome"] for row in rows)
    basis_counts = Counter(row["coordinate_basis"] for row in rows)
    pattern_counts = Counter(row["chromosome_name_pattern"] for row in rows)
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if "NAM_Consortium_2020" in row["matched_file_url"]:
            family = "NAM_Consortium_2020"
        elif "Lin_2017" in row["matched_file_url"]:
            family = "Lin_2017"
        elif "Diepenbrock_2016" in row["matched_file_url"]:
            family = "Diepenbrock_2016"
        else:
            family = "other"
        by_family[family].append(row)

    lines = [
        "# CyVerse MaizeGDB BigWig Reference Audit",
        "",
        "This audit covers the RNA-seq BigWig files downloaded from CyVerse/MaizeGDB.",
        "",
        "## Conclusion",
        "",
        "The downloaded BigWigs are not exact SRX/SRR-matched workbook tracks. They are MaizeGDB/NAM RNA-seq tracks. Their CyVerse paths place them under `NAM_PROJECT_JBROWSE_AND_ANALYSES/NAM_RNA-seq`, and their filenames use NAM founder prefixes such as B73, B97, CML103, CML228, CML247, CML277, and CML322.",
        "",
        "Use these tracks with their matching NAM/B73 reference coordinates. Do not mix them directly with the SOW4/JHAX7 reference mentioned in the maize RNA-seq pipeline unless you realign the reads or perform an explicit liftover.",
        "",
        f"Detailed CSV: `{csv_path.as_posix()}`",
        f"Selected FASTA/GFF manifest: `{selected_refs_path.as_posix()}`",
        f"BigWig-to-reference map: `{map_path.as_posix()}`",
        "",
        "## Summary",
        "",
        f"- BigWigs audited: {len(rows)}",
        f"- Read errors: {sum(1 for row in rows if row['error'])}",
        f"- Coordinate basis: {dict(basis_counts)}",
        f"- Chromosome name patterns: {dict(pattern_counts)}",
        "",
        "## Inferred Reference Genomes",
        "",
    ]
    for assembly, count in assembly_counts.most_common():
        lines.append(f"- {assembly}: {count}")
    if selected_refs_path.exists():
        lines.extend(["", "## Selected Reference Downloads", ""])
        with selected_refs_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                lines.append(f"- {row['founder']} {row['selection_role']}: `{row['reference_file_name']}`")
    lines.extend(["", "## Groups", ""])
    for family, family_rows in sorted(by_family.items()):
        lines.append(f"### {family}")
        lines.append("")
        for row in family_rows:
            lines.append(
                f"- `{row['matched_file_name']}` -> {row['inferred_reference_genome']} "
                f"({row['coordinate_basis']}; {row['chromosome_name_pattern']})"
            )
        lines.append("")
    lines.extend([
        "## Reference Matching Rule",
        "",
        "- Files from `B73v5_JBROWSE_AND_ANALYSES/B73v5_RNA-seq` should be paired with B73 RefGen_v5 / `Zm-B73-REFERENCE-NAM-5.0`.",
        "- Files from `NAM_PROJECT_JBROWSE_AND_ANALYSES/NAM_RNA-seq` should be treated as NAM founder-reference tracks. Founder-prefixed files such as `B97_*` and `CML103_*` are not interchangeable with B73 sequence without coordinate conversion.",
        "- The local SOW4 note says the target NTv3 maize work used in-house JHAX7 FASTA/GFF. These public CyVerse BigWigs should therefore be kept as a separate public MaizeGDB/NAM corpus unless they are lifted over or regenerated on JHAX7.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reference assemblies for downloaded CyVerse BigWigs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--selected-references", type=Path, default=DEFAULT_SELECTED_REFERENCES)
    parser.add_argument("--bigwig-reference-map", type=Path, default=DEFAULT_BIGWIG_REFERENCE_MAP)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = read_manifest(args.manifest)
    audited = audit_rows(rows)
    write_csv(args.output_csv, audited)
    write_bigwig_reference_map(args.bigwig_reference_map, audited, args.selected_references)
    write_md(args.output_md, audited, args.output_csv, args.selected_references, args.bigwig_reference_map)
    print(f"Audited {len(audited)} BigWig rows")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.bigwig_reference_map}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
