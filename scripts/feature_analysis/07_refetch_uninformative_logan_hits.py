#!/usr/bin/env python3
"""
Iteratively refetch ENA metadata for uninformative Logan best hits.

For unitigs whose best Logan hit resolves to a generic, uninformative label
(scientific_name in UNINFORMATIVE below), this script walks further down the
ranked per-query hit list until it finds an informative SRA source, or exhausts
all hits for that query.

Strategy:
  1. Load all existing ENA metadata into a cache.
  2. Identify uninformative labels (unidentified, metagenome, viral metagenome, ...)
  3. Stream the Logan TSV.gz: for each query whose best hit is uninformative,
     collect up to MAX_CANDIDATES subsequent hits (bitscore-descending order)
     that are not DIANA samples and not cached-uninformative.
  4. Fetch ENA metadata for any candidate not yet in cache.
  5. Per query, select the FIRST candidate whose scientific_name is informative.
  6. Write resolved TSV (one row per original best-hit accession, substituted
     where a better annotation was found).

INPUTS:
    data/metadata/logan_best_hit_sra_metadata.tsv          -- original best-hit metadata
    data/metadata/logan_fallback_sra_metadata.tsv          -- previously fetched fallbacks (v1)
    results/logan_search/genomic_bacteria/all_results.tsv.gz
    data/diana_samples.fof

OUTPUTS:
    data/metadata/logan_fallback_sra_metadata_v2.tsv       -- all newly fetched metadata
    data/metadata/logan_best_hit_sra_metadata_resolved.tsv -- final merged table

USAGE:
    python scripts/feature_analysis/07_refetch_uninformative_logan_hits.py
"""

import csv
import gzip
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional

import requests

# ── paths ────────────────────────────────────────────────────────────────────
LOGAN_FILE      = Path("results/logan_search/genomic_bacteria/all_results.tsv.gz")
METADATA_TSV    = Path("data/metadata/logan_best_hit_sra_metadata.tsv")
FALLBACK_TSV_V1 = Path("data/metadata/logan_fallback_sra_metadata.tsv")
DIANA_FOF       = Path("data/diana_samples.fof")
FALLBACK_TSV_V2 = Path("data/metadata/logan_fallback_sra_metadata_v2.tsv")
RESOLVED_TSV    = Path("data/metadata/logan_best_hit_sra_metadata_resolved.tsv")
LOG_FILE        = Path("logs/refetch_uninformative_logan.log")

# Labels treated as uninformative (lower-case comparison)
UNINFORMATIVE = {
    "unidentified", "metagenome", "viral metagenome",
    "unknown", "", "na", "n/a",
    "not available", "not applicable", "unidentified virus",
}

MAX_CANDIDATES = 20   # max candidates to collect per query (memory guard)
ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"
SLEEP   = 0.3


# ── helpers ──────────────────────────────────────────────────────────────────

def is_uninformative(meta: Optional[dict]) -> bool:
    if meta is None:
        return True
    return (meta.get("scientific_name") or "").strip().lower() in UNINFORMATIVE


def fetch_ena(accession: str) -> Optional[dict]:
    try:
        r = requests.get(
            ENA_API,
            params={"accession": accession, "result": "read_run", "fields": "all"},
            timeout=30,
        )
        r.raise_for_status()
        lines = r.text.strip().splitlines()
        if len(lines) < 2:
            return None
        rows = list(csv.DictReader(lines, delimiter="\t"))
        return rows[0] if rows else None
    except Exception:
        return None


def load_metadata_file(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["run_accession"]: r for r in csv.DictReader(f, delimiter="\t")}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w")

    def lp(msg: str) -> None:
        print(msg); log.write(msg + "\n"); log.flush()

    # 1. Build metadata cache from all existing files ──────────────────────────
    lp("Loading existing ENA metadata cache ...")
    cache: dict = {}
    cache.update(load_metadata_file(METADATA_TSV))
    cache.update(load_metadata_file(FALLBACK_TSV_V1))
    cache.update(load_metadata_file(FALLBACK_TSV_V2))  # picks up progress from previous runs
    lp(f"  Cache size: {len(cache)} accessions")

    with open(METADATA_TSV) as f:
        primary_fieldnames = csv.DictReader(f, delimiter="\t").fieldnames

    # 2. DIANA accessions ──────────────────────────────────────────────────────
    diana: set = set()
    if DIANA_FOF.exists():
        with open(DIANA_FOF) as f:
            for line in f:
                acc = line.strip().split()[-1]
                if acc:
                    diana.add(acc)
    lp(f"  DIANA accessions: {len(diana)}")

    # 3. Identify uninformative original best-hit accessions ───────────────────
    orig_meta = load_metadata_file(METADATA_TSV)
    uninf_best_hits: set = {
        acc for acc, row in orig_meta.items() if is_uninformative(row)
    }
    lp(f"  Uninformative original best-hit accessions: {len(uninf_best_hits)}")

    skip_set = uninf_best_hits | diana

    # 4. Stream Logan: collect up to MAX_CANDIDATES per uninformative query ─────
    lp("\nStreaming Logan file to collect candidates ...")

    candidates: dict = defaultdict(list)
    current_query: Optional[str] = None
    current_is_uninf = False

    with gzip.open(LOGAN_FILE, "rt") as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            query   = parts[0]
            sgenome = parts[3]

            if query != current_query:
                current_query = query
                current_is_uninf = sgenome in uninf_best_hits

            if not current_is_uninf:
                continue
            if sgenome in skip_set:
                continue

            cands = candidates[query]
            if len(cands) >= MAX_CANDIDATES:
                continue
            if sgenome in cands:
                continue
            # Skip if cached and uninformative
            if sgenome in cache and is_uninformative(cache[sgenome]):
                skip_set.add(sgenome)
                continue
            cands.append(sgenome)

    lp(f"  Queries needing resolution: {len(candidates)}")
    unique_candidates = {sg for clist in candidates.values() for sg in clist}
    lp(f"  Unique candidate sgenomes:  {len(unique_candidates)}")

    # 5. Fetch metadata for candidates not in cache ────────────────────────────
    to_fetch = sorted(unique_candidates - set(cache.keys()))
    lp(f"  New accessions to fetch:    {len(to_fetch)}")

    # Open v2 fallback file for incremental append (safe on restart)
    v2_exists = FALLBACK_TSV_V2.exists()
    v2_writer = None
    v2_file = None
    newly_fetched_count = 0

    for i, acc in enumerate(to_fetch, 1):
        meta = fetch_ena(acc)
        if meta:
            cache[acc] = meta
            newly_fetched_count += 1
            # Lazy-init writer on first successful fetch
            if v2_writer is None:
                v2_file = open(FALLBACK_TSV_V2, "a", newline="")
                v2_writer = csv.DictWriter(v2_file, fieldnames=list(meta.keys()),
                                           delimiter="\t", extrasaction="ignore")
                if not v2_exists:
                    v2_writer.writeheader()
            v2_writer.writerow(meta)
            v2_file.flush()
            sci = meta.get("scientific_name", "?")
            if i % 100 == 0 or i <= 5:
                lp(f"  [{i}/{len(to_fetch)}] {acc} -> {sci}")
        else:
            if i <= 5:
                lp(f"  [{i}/{len(to_fetch)}] {acc} -> FAILED")
        time.sleep(SLEEP)

    if v2_file:
        v2_file.close()
    lp(f"  Successfully fetched: {newly_fetched_count}")

    # 6. Re-scan Logan to get orig sgenome per query ───────────────────────────
    lp("\nRe-scanning Logan for orig sgenome per query ...")
    query_to_orig: dict = {}
    with gzip.open(LOGAN_FILE, "rt") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            query, sgenome = parts[0], parts[3]
            if query not in query_to_orig:
                query_to_orig[query] = sgenome

    # 7. Resolve: per query, first informative candidate ───────────────────────
    resolved_sgenome: dict = {}
    for query, clist in candidates.items():
        for sg in clist:
            meta = cache.get(sg)
            if meta and not is_uninformative(meta):
                resolved_sgenome[query] = sg
                break

    lp(f"  Queries resolved to informative hit:        {len(resolved_sgenome)}")
    lp(f"  Queries still uninformative (no better hit): {len(candidates) - len(resolved_sgenome)}")

    # For each uninf orig sgenome, pick the replacement most commonly chosen
    orig_to_rep_counter: dict = defaultdict(Counter)
    for query, rep_sg in resolved_sgenome.items():
        orig_sg = query_to_orig.get(query)
        if orig_sg and orig_sg in uninf_best_hits:
            orig_to_rep_counter[orig_sg][rep_sg] += 1

    orig_to_best_rep: dict = {
        orig: ctr.most_common(1)[0][0]
        for orig, ctr in orig_to_rep_counter.items()
    }
    lp(f"  Orig->replacement mappings: {len(orig_to_best_rep)}")

    # 8. Build resolved rows ───────────────────────────────────────────────────
    resolved_rows = []
    for acc, row in orig_meta.items():
        if acc in uninf_best_hits and acc in orig_to_best_rep:
            rep_acc = orig_to_best_rep[acc]
            rep_meta = cache.get(rep_acc, {})
            r = dict(row)
            for col in primary_fieldnames:
                if col in rep_meta:
                    r[col] = rep_meta[col]
            r["run_accession"] = acc
            r["resolved_from"] = f"fallback_v2:{rep_acc}"
        elif acc in uninf_best_hits:
            r = dict(row)
            r["resolved_from"] = f"unresolved:{acc}"
        else:
            r = dict(row)
            r["resolved_from"] = f"original:{acc}"
        resolved_rows.append(r)

    res_cols = list(primary_fieldnames) + ["resolved_from"]
    with open(RESOLVED_TSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=res_cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(resolved_rows)
    lp(f"\nSaved resolved metadata -> {RESOLVED_TSV}  ({len(resolved_rows)} rows)")

    # 9. Summary ───────────────────────────────────────────────────────────────
    sci_counts = Counter(r.get("scientific_name", "").strip() for r in resolved_rows)
    still_uninf = sum(v for k, v in sci_counts.items() if k.lower() in UNINFORMATIVE)
    lp(f"\nFinal uninformative entries remaining: {still_uninf}/{len(resolved_rows)}")
    lp(f"\nTop 20 scientific names in resolved table:")
    for sci, cnt in sci_counts.most_common(20):
        lp(f"  {cnt:4d}  {sci}")

    lp("\nDone.")
    log.close()


if __name__ == "__main__":
    main()
