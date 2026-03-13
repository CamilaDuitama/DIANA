#!/usr/bin/env python3
"""
06_fetch_logan_hit_sra_metadata.py
===================================
Fetch SRA run metadata from ENA API for a list of run accessions.

Uses EBI ENA portal API (fields=all) which returns far richer metadata than
NCBI efetch runinfo: sample attributes, library info, organism, geo_loc, etc.

Intended use: characterise the source SRA runs of the best Logan hits for
BLAST-unannotated unitigs (produced by 03_annotate_features.py / Logan search).

Usage:
    python scripts/feature_analysis/06_fetch_logan_hit_sra_metadata.py \\
        <accessions_file> <output_tsv>
    # e.g.:
    python scripts/feature_analysis/06_fetch_logan_hit_sra_metadata.py \\
        /tmp/best_hit_non_diana.txt \\
        results/logan_search/best_hit_sra_metadata.tsv
"""

import sys
import time
import requests
from pathlib import Path
from typing import Optional

SLEEP_SEC = 0.3   # rate-limit: be polite to EBI
ENA_URL   = "https://www.ebi.ac.uk/ena/portal/api/filereport"

def fetch_one(accession: str) -> Optional[dict]:
    """
    Fetch all ENA metadata fields for a single run accession.
    Returns a dict or None on failure.
    """
    params = {
        "accession": accession,
        "result":    "read_run",
        "fields":    "all",
    }
    try:
        r = requests.get(ENA_URL, params=params, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return None
        header = lines[0].split("\t")
        vals   = lines[1].split("\t")
        vals  += [""] * (len(header) - len(vals))
        return dict(zip(header, vals[:len(header)]))
    except Exception as e:
        print(f"  WARNING [{accession}]: {e}", flush=True)
        return None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    acc_file = Path(sys.argv[1])
    output   = Path(sys.argv[2])

    if not acc_file.exists():
        print(f"ERROR: accession file not found: {acc_file}", file=sys.stderr)
        sys.exit(1)

    accessions = [l.strip() for l in acc_file.read_text().splitlines() if l.strip()]
    total      = len(accessions)
    print(f"Fetching ENA metadata for {total} accessions -> {output}")
    print(f"Sleep: {SLEEP_SEC}s per request | URL: {ENA_URL}")

    output.parent.mkdir(parents=True, exist_ok=True)

    all_rows    = []
    failed_accs = []

    for i, acc in enumerate(accessions, 1):
        print(f"[{i}/{total}] {acc}...", end=" ", flush=True)
        row = fetch_one(acc)
        if row:
            all_rows.append(row)
            print("✓", flush=True)
        else:
            failed_accs.append(acc)
            print("✗", flush=True)
        time.sleep(SLEEP_SEC)

        time.sleep(SLEEP_SEC)

    if not all_rows:
        print("ERROR: No data fetched.", file=sys.stderr)
        sys.exit(1)

    # Collect union of all column names (preserve first-seen order)
    all_cols = list(dict.fromkeys(k for row in all_rows for k in row))

    # Write TSV
    with open(output, "w") as f:
        f.write("\t".join(all_cols) + "\n")
        for row in all_rows:
            f.write("\t".join(row.get(c, "") for c in all_cols) + "\n")

    succeeded = total - len(failed_accs)
    print(f"\nDone. Fetched: {succeeded}/{total} | Failed: {len(failed_accs)}")
    print(f"Output: {output}  ({len(all_rows)} rows, {len(all_cols)} columns)")

    if failed_accs:
        failed_path = output.with_suffix(".failed.txt")
        failed_path.write_text("\n".join(failed_accs) + "\n")
        print(f"Failed accessions: {failed_path}")


if __name__ == "__main__":
    main()
