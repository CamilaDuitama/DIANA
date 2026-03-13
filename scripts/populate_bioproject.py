"""
Populate the BioProject column in paper/metadata/{train,test,validation}_metadata.tsv
using NCBI eutils, which accepts both SRR and ERR run accessions in batches of 200.
Only rows where BioProject does not already start with 'PRJ' are queried.
"""

import time
import requests
import pandas as pd
from io import StringIO

METADATA_FILES = {
    "train": "paper/metadata/train_metadata.tsv",
    "test": "paper/metadata/test_metadata.tsv",
    "val": "paper/metadata/validation_metadata.tsv",
}
BATCH_SIZE = 200
NCBI_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_bioproject_ncbi(run_accessions: list[str]) -> dict[str, str]:
    """Query NCBI eutils runinfo for a batch of run accessions. Returns {run: bioproject}."""
    ids = ",".join(run_accessions)
    params = {"db": "sra", "id": ids, "rettype": "runinfo", "retmode": "text"}
    r = requests.get(NCBI_URL, params=params, timeout=60)
    r.raise_for_status()
    text = r.text.strip()
    if not text:
        return {}
    try:
        df = pd.read_csv(StringIO(text))
    except Exception:
        return {}
    if "Run" not in df.columns or "BioProject" not in df.columns:
        return {}
    result = {}
    for _, row in df.iterrows():
        run = str(row["Run"]).strip()
        bp = str(row["BioProject"]).strip()
        if run and bp and bp.startswith("PRJ"):
            result[run] = bp
    return result


def collect_missing_runs(files: dict) -> dict[str, str]:
    """Return {run_accession: bioproject} for all runs missing BioProject across all files."""
    missing_runs = set()
    for name, path in files.items():
        df = pd.read_csv(path, sep="\t", low_memory=False)
        needs_fill = ~df["BioProject"].astype(str).str.startswith("PRJ")
        runs = df.loc[needs_fill, "Run_accession"].dropna().unique()
        missing_runs.update(runs)
        print(f"  {name}: {needs_fill.sum()} rows missing BioProject ({len(runs)} unique runs)")
    return sorted(missing_runs)


def batch_fetch(runs: list[str]) -> dict[str, str]:
    """Fetch BioProject for all runs in batches."""
    mapping = {}
    n_batches = (len(runs) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(runs), BATCH_SIZE):
        batch = runs[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Fetching batch {batch_num}/{n_batches} ({len(batch)} accessions)...", end=" ", flush=True)
        try:
            result = fetch_bioproject_ncbi(batch)
            mapping.update(result)
            print(f"got {len(result)} BioProjects")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.4)  # be polite to NCBI
    return mapping


def update_files(files: dict, mapping: dict[str, str]):
    """Fill BioProject column in each file using the mapping, save in place."""
    for name, path in files.items():
        df = pd.read_csv(path, sep="\t", low_memory=False)
        needs_fill = ~df["BioProject"].astype(str).str.startswith("PRJ")
        before = needs_fill.sum()
        df.loc[needs_fill, "BioProject"] = df.loc[needs_fill, "Run_accession"].map(mapping)
        after = (~df["BioProject"].astype(str).str.startswith("PRJ")).sum()
        df.to_csv(path, sep="\t", index=False)
        print(f"  {name}: filled {before - after}/{before} missing → {after} still missing")


if __name__ == "__main__":
    print("=== Step 1: Collecting missing runs ===")
    missing_runs = collect_missing_runs(METADATA_FILES)
    print(f"  Total unique runs to query: {len(missing_runs)}\n")

    print("=== Step 2: Fetching BioProject from NCBI ===")
    mapping = batch_fetch(missing_runs)
    print(f"  Resolved {len(mapping)}/{len(missing_runs)} accessions\n")

    print("=== Step 3: Updating metadata files ===")
    update_files(METADATA_FILES, mapping)
    print("\nDone.")
