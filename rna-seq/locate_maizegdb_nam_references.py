#!/usr/bin/env python3
"""Locate MaizeGDB NAM reference FASTA/GFF files for CyVerse BigWigs."""

from __future__ import annotations

import argparse
import csv
import html.parser
import re
from collections import deque
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urljoin

import requests


DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "cyverse_maizegdb_rnaseq_all"
DEFAULT_AUDIT = DEFAULT_BASE_DIR / "manifests" / "bigwig_reference_audit.csv"
DEFAULT_OUTPUT = DEFAULT_BASE_DIR / "manifests" / "reference_download_manifest.csv"
DEFAULT_SELECTED_OUTPUT = DEFAULT_BASE_DIR / "manifests" / "reference_selected_manifest.csv"
SEARCH_ROOTS = [
    "https://download.maizegdb.org/Genomes/",
    "https://download.maizegdb.org/All_assembly_sequence/",
    "https://download.maizegdb.org/All_gene_model_GFF/",
    "https://download.maizegdb.org/All_gene_model_genomic/",
    "https://download.maizegdb.org/",
]
REFERENCE_EXTENSIONS = (
    ".fa",
    ".fasta",
    ".fna",
    ".fa.gz",
    ".fasta.gz",
    ".fna.gz",
    ".gff",
    ".gff3",
    ".gff.gz",
    ".gff3.gz",
)


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self.hrefs.append(href)


def read_founders(path: Path) -> list[str]:
    founders: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            file_name = row.get("matched_file_name", "")
            match = re.match(r"([A-Za-z0-9]+)_", file_name)
            if match:
                founders.add(match.group(1))
    return sorted(founders)


def wanted_tokens(founder: str) -> list[str]:
    return [
        f"Zm-{founder}-REFERENCE-NAM",
        f"Zm_{founder}_REFERENCE_NAM",
        f"{founder}-REFERENCE-NAM",
        f"{founder}_REFERENCE_NAM",
        f"{founder}.fa",
        f"{founder}.fasta",
        f"{founder}.gff",
        f"{founder}.gff3",
    ]


def parse_links(text: str, base_url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(text)
    links: list[str] = []
    for href in parser.hrefs:
        if href.startswith("?") or href.startswith("#") or href in {"../", "/"}:
            continue
        links.append(urljoin(base_url, href))
    return links


def crawl_references(founders: Sequence[str], max_depth: int = 3) -> list[dict[str, str]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "maizegdb-nam-reference-locator/1.0"})
    founder_tokens = {founder: wanted_tokens(founder) for founder in founders}
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    queue = deque((root, 0) for root in SEARCH_ROOTS)

    while queue:
        url, depth = queue.popleft()
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        try:
            response = session.get(url, timeout=60)
        except requests.RequestException:
            continue
        if response.status_code != 200:
            continue
        links = parse_links(response.text, url)
        for link in links:
            basename = link.rstrip("/").rsplit("/", 1)[-1]
            basename_l = basename.lower()
            link_l = link.lower()
            if link.endswith("/"):
                if any(token.lower() in link_l for tokens in founder_tokens.values() for token in tokens):
                    queue.append((link, depth + 1))
                elif depth < 1 and any(part in link_l for part in ["genome", "assembly", "annotation", "gene_model", "nam"]):
                    queue.append((link, depth + 1))
                continue
            if not basename_l.endswith(REFERENCE_EXTENSIONS):
                continue
            for founder, tokens in founder_tokens.items():
                if any(token.lower() in link_l for token in tokens):
                    kind = "annotation_gff" if ".gff" in basename_l else "genome_fasta"
                    found.append({
                        "founder": founder,
                        "reference_file_type": kind,
                        "reference_url": link,
                        "reference_file_name": basename,
                    })
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in found:
        unique[(row["founder"], row["reference_file_type"], row["reference_url"])] = row
    return sorted(unique.values(), key=lambda r: (r["founder"], r["reference_file_type"], r["reference_url"]))


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["founder", "reference_file_type", "reference_file_name", "reference_url"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_primary_references(founders: Sequence[str], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    by_founder: dict[str, list[dict[str, str]]] = {founder: [] for founder in founders}
    for row in rows:
        by_founder.setdefault(row["founder"], []).append(row)

    for founder in founders:
        founder_rows = by_founder.get(founder, [])
        fasta = [
            row for row in founder_rows
            if row["reference_file_type"] == "genome_fasta"
            and "/All_assembly_sequence/" in row["reference_url"]
            and row["reference_file_name"].startswith(f"Zm-{founder}-REFERENCE-NAM-")
            and row["reference_file_name"].endswith(".fa.gz")
            and "_" not in row["reference_file_name"]
        ]
        gff = [
            row for row in founder_rows
            if row["reference_file_type"] == "annotation_gff"
            and "/All_gene_model_GFF/" in row["reference_url"]
            and row["reference_file_name"].startswith(f"Zm-{founder}-REFERENCE-NAM-")
            and row["reference_file_name"].endswith(".gff3.gz")
            and ".nc." not in row["reference_file_name"]
            and ".TE." not in row["reference_file_name"]
        ]
        for row in fasta[:1]:
            out = dict(row)
            out["selection_role"] = "primary_genome_fasta"
            selected.append(out)
        for row in gff[:1]:
            out = dict(row)
            out["selection_role"] = "primary_gene_model_gff3"
            selected.append(out)
    return selected


def write_selected_output(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["founder", "selection_role", "reference_file_type", "reference_file_name", "reference_url"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate matching MaizeGDB NAM reference FASTA/GFF files.")
    parser.add_argument("--audit", "--manifest", dest="audit", type=Path, default=DEFAULT_AUDIT, help="CSV containing matched_file_name values; audit or download manifest both work")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selected-output", type=Path, default=DEFAULT_SELECTED_OUTPUT)
    parser.add_argument("--max-depth", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    founders = read_founders(args.audit)
    rows = crawl_references(founders, max_depth=args.max_depth)
    write_output(args.output, rows)
    selected = select_primary_references(founders, rows)
    write_selected_output(args.selected_output, selected)
    print(f"Founders: {', '.join(founders)}")
    print(f"Reference rows: {len(rows)}")
    print(f"Selected primary reference rows: {len(selected)}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.selected_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
