#!/usr/bin/env python3
"""
Regenerate blast_summary.json using taxonkit for proper kingdom assignment.

For each feature:
 - best_hits: single hit with highest bitscore (for pident stats)
 - best annotated hit (non-N/A taxid, highest bitscore) used for kingdom via taxonkit

Writes:
 - results/feature_analysis/all_features_blast/blast_summary.json  (updated)
"""
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

BLAST_FILE  = Path("results/feature_analysis/all_features_blast/blast_results.txt")
SUMMARY_OUT = Path("results/feature_analysis/all_features_blast/blast_summary.json")
TOTAL_FEATURES = 107480

# ── 1. Parse BLAST results ──────────────────────────────────────────────────
print("Parsing BLAST results …")
best_hits      = {}   # query_id -> best hit (any, for pident)
best_taxid_hit = {}   # query_id -> best hit with a valid taxid (for kingdom)

with open(BLAST_FILE) as fh:
    for line in fh:
        if not line.strip():
            continue
        f = line.rstrip('\n').split('\t')
        if len(f) < 14:
            continue
        qid      = f[0]
        bitscore = float(f[11])
        taxid    = f[12].strip()

        hit = dict(qid=qid, pident=float(f[2]), evalue=float(f[10]),
                   bitscore=bitscore, taxid=taxid, desc=f[13])

        if qid not in best_hits or bitscore > best_hits[qid]['bitscore']:
            best_hits[qid] = hit

        if taxid and taxid not in ('N/A', '0', ''):
            if qid not in best_taxid_hit or bitscore > best_taxid_hit[qid]['bitscore']:
                best_taxid_hit[qid] = hit

hits          = list(best_hits.values())
taxid_hits    = list(best_taxid_hit.values())
n_with_taxid  = len(taxid_hits)
n_no_taxid    = len(hits) - n_with_taxid
print(f"  Best hits: {len(hits):,}")
print(f"  With valid taxid: {n_with_taxid:,}")
print(f"  Without taxid (no NCBI record): {n_no_taxid:,}")

# ── 2. Run taxonkit on unique taxids ────────────────────────────────────────
unique_taxids = sorted({h['taxid'] for h in taxid_hits})
print(f"\nRunning taxonkit on {len(unique_taxids):,} unique taxids …")

# Write taxids to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
    tf.write('\n'.join(unique_taxids) + '\n')
    tmp_path = tf.name

try:
    cmd = (
        f'bash -c "module load taxonkit/ 2>/dev/null && '
        f'cat {tmp_path} | taxonkit lineage -i 1 -j 16 2>/dev/null"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not result.stdout.strip():
        print("  ERROR: taxonkit failed or returned empty output")
        print("  stderr:", result.stderr[:200])
        raise SystemExit(1)
finally:
    os.unlink(tmp_path)

# ── 3. Parse taxonkit output → kingdom ──────────────────────────────────────
KINGDOM_MAP = {
    'Bacteria':  'Bacteria',
    'Archaea':   'Archaea',
    'Eukaryota': 'Eukaryota',
    'Viruses':   'Viruses',
}

taxid_to_kingdom = {}
for line in result.stdout.strip().split('\n'):
    if not line:
        continue
    parts = line.split('\t')
    if len(parts) < 2:
        continue
    taxid   = parts[0].strip()
    lineage = parts[1].strip()
    kingdom = 'Unknown'
    for k in KINGDOM_MAP:
        if k in lineage:
            kingdom = KINGDOM_MAP[k]
            break
    taxid_to_kingdom[taxid] = kingdom

print(f"  Resolved {len(taxid_to_kingdom):,} taxids")
kingdom_dist = Counter(taxid_to_kingdom.values())
print(f"  Kingdom distribution among taxids: {dict(kingdom_dist)}")

# ── 4. Assign kingdom to each feature ───────────────────────────────────────
kingdom_counts: Counter = Counter()
for h in taxid_hits:
    k = taxid_to_kingdom.get(h['taxid'], 'Unknown')
    kingdom_counts[k] += 1

# Features with no valid taxid at all
kingdom_counts['Unknown'] += n_no_taxid

print(f"\nFinal kingdom counts (per feature, best taxid hit):")
for k, v in sorted(kingdom_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:,}")

# ── 5. pident stats (from best overall hit) ──────────────────────────────────
pident_ranges = {
    '>=95%':  sum(1 for h in hits if h['pident'] >= 95),
    '90-95%': sum(1 for h in hits if 90 <= h['pident'] < 95),
    '80-90%': sum(1 for h in hits if 80 <= h['pident'] < 90),
    '<80%':   sum(1 for h in hits if h['pident'] < 80),
}

# ── 6. Genera from taxid hits (with known kingdom) ──────────────────────────
known_kingdom_hits = [h for h in taxid_hits
                      if taxid_to_kingdom.get(h['taxid'], 'Unknown') != 'Unknown']
genera = []
for h in known_kingdom_hits:
    words = h['desc'].split()
    if words:
        g = words[0].capitalize()
        if len(g) > 2:
            genera.append(g)
top_genera = dict(Counter(genera).most_common(20))

# ── 7. Write updated summary JSON ───────────────────────────────────────────
summary = {
    "total_features":             TOTAL_FEATURES,
    "features_with_blast_hits":   len(hits),
    "features_with_valid_taxid":  n_with_taxid,
    "hit_rate_percent":           round(100 * len(hits) / TOTAL_FEATURES, 2),
    "blast_statistics": {
        "total_hits":              len(hits),
        "unique_queries_with_hits": len(hits),
        "pident_ranges":           pident_ranges,
        "note": "One best hit per feature (highest bitscore); pident from best overall hit"
    },
    "taxonomy": {
        "top_genera":                   top_genera,
        "kingdom_counts":               dict(kingdom_counts),
        "features_with_known_kingdom":  len(known_kingdom_hits),
        "unique_genera":                len(set(genera)),
        "total_assigned":               len(genera),
        "note": "Kingdom from taxonkit lineage on best-taxid hit per feature; "
                "Unknown includes features with no valid taxid or unresolved lineage"
    },
    "blast_results_file": str(BLAST_FILE)
}

with open(SUMMARY_OUT, 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f"\nWritten to {SUMMARY_OUT}")
