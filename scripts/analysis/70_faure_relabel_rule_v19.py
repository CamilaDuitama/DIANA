#!/usr/bin/env python3
"""The metadata-relabeling rule of Faure et al. 2026 (Section 4.6, tag groups of Sup. Table 2),
recomputed for our runs from their released filtered matrix and metadata, and put beside the
AncientMetagenomeDir labels (our true labels) and DIANA's predictions.

Rule, as published: among a run's stored neighbours (itself excluded), keep the most similar run
of each BioProject; drop those with similarity < 0.1 and those tagged "metagenome", "human
metagenome" or "Homo sapiens"; if at least 4 remain and at least 80 % of their tags fall in one
tag group other than the run's own, propose that group's representative.

DIANA's prediction for a run always comes from a model that never saw the run's study: the
dev-fold prediction for training runs, the held-out prediction for held-out runs (argmax, no
no-call). Tags are compared with our fields through the stated mapping (MAP_* below).

Writes results/sra_sketch_overlap_v19/faure_rule_runs.tsv and prints the comparisons.

    ./env/bin/python scripts/analysis/70_faure_relabel_rule_v19.py <neighbour_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sra_sketch_overlap_v19"
FIELDS = ["community_type", "sample_host", "material"]
# Sup. Table 2: group representative -> additional tags (tags not listed form their own group)
GROUPS = {
    "general metagenome": ["human metagenome", "metagenome", "Homo sapiens"],
    "human blood metagenome": ["blood metagenome"],
    "human gut metagenome": ["gut metagenome", "feces metagenome", "human feces metagenome"],
    "human lung metagenome": ["lung metagenome", "respiratory tract metagenome", "upper respiratory tract metagenome", "human tracheal metagenome"],
    "human oral metagenome": ["oral metagenome", "human saliva metagenome"],
    "human skin metagenome": ["skin metagenome"],
    "human urinary tract metagenome": ["urine metagenome"],
    "human vaginal metagenome": ["vaginal metagenome"],
    "hypersaline metagenome": ["hypersaline lake metagenome", "salt marsh metagenome", "salt lake metagenome"],
    "sludge metagenome": ["activated sludge metagenome", "bioreactor metagenome", "bioreactor sludge metagenome", "anaerobic digester metagenome"],
    "unidentified": ["unidentified", "uncultured bacterium", "uncultured virus", "uncultured crAssphage"],
    "bovine gut metagenome": ["bovine metagenome"], "canine gut metagenome": ["canine metagenome"],
    "fish gut metagenome": ["fish metagenome"], "insect gut metagenome": ["insect metagenome"],
    "mouse gut metagenome": ["mouse metagenome", "Mus musculus", "Rattus norvegicus"],
    "pig gut metagenome": ["pig metagenome"], "sheep gut metagenome": ["sheep metagenome"],
    "dust metagenome": ["indoor metagenome"],
    "freshwater metagenome": ["lake water metagenome", "riverine metagenome", "drinking water metagenome", "groundwater metagenome"],
    "hot springs metagenome": ["hydrothermal vent metagenome"],
    "marine metagenome": ["seawater metagenome", "marine plankton metagenome"],
    "manure metagenome": ["poultry litter metagenome"],
    "sediment metagenome": ["freshwater sediment metagenome", "marine sediment metagenome"],
    "soil metagenome": ["rhizosphere metagenome", "root metagenome", "terrestrial metagenome"],
    "viral metagenome": ["human viral metagenome", "aquatic viral metagenome", "phage metagenome"],
}
GROUP = {t: rep for rep, tags in GROUPS.items() for t in [rep, *tags]}
GENERIC = {"metagenome", "human metagenome", "Homo sapiens"}
RULE = {"min_similarity": 0.1, "min_neighbours": 4, "min_share": 0.8}
# tag -> value it implies for our fields (None: says nothing about that field)
MAP_COMMUNITY = {"human oral metagenome": "oral", "oral metagenome": "oral", "human gut metagenome": "gut", "gut metagenome": "gut"}
MAP_MATERIAL = {"sediment metagenome": {"sediment", "lake sediment", "marine sediment"}, "marine sediment metagenome": {"marine sediment"},
                "soil metagenome": {"soil"}, "permafrost metagenome": {"permafrost"}}
HUMAN_TAGS = {"human metagenome", "human oral metagenome", "human gut metagenome", "Homo sapiens"}
SPECIES = {"Homo sapiens", "Gorilla beringei graueri", "Gorilla gorilla gorilla", "Homo sapiens neanderthalensis",
           "Gorilla beringei beringei", "Pan troglodytes schweinfurthii", "Pan troglodytes verus", "Pan troglodytes ellioti",
           "Alouatta palliata", "Betula pendula"}


def implied(tag, field):
    """The value (or set of values) a tag implies for one of our fields, or None."""
    if not isinstance(tag, str):
        return None
    if field == "community_type":
        return MAP_COMMUNITY.get(tag)
    if field == "material":
        return MAP_MATERIAL.get(tag)
    if field == "sample_host":
        return "Homo sapiens" if tag in HUMAN_TAGS else (tag if tag in SPECIES else None)
    return None


def agrees(imp, value):
    if imp is None or not isinstance(value, str):
        return None
    return value in imp if isinstance(imp, set) else value == imp


def propose(nb: pd.DataFrame, own_tag: str) -> tuple:
    nb = nb.sort_values("Jaccard", ascending=False).drop_duplicates("bioproject")
    nb = nb[(nb.Jaccard >= RULE["min_similarity"]) & ~nb.organism.isin(GENERIC)]
    if len(nb) < RULE["min_neighbours"]:
        return len(nb), None, np.nan
    g = nb.organism.map(lambda t: GROUP.get(t, t)).value_counts()
    top, share = g.index[0], g.iloc[0] / len(nb)
    own = GROUP.get(own_tag, own_tag)
    return len(nb), (top if share >= RULE["min_share"] and top != own else None), share


def diana_predictions() -> pd.DataFrame:
    parts = {}
    for f in FIELDS + ["feature"]:
        tabs = []
        for p in (ROOT / f"results/devfold_probs_v19/probs_DIANA_{f}.tsv", ROOT / f"results/heldout_v19/diana/probs_DIANA_{f}.tsv"):
            d = pd.read_csv(p, sep="\t").set_index("Run_accession")
            pc = [c for c in d.columns if c.startswith("p_")]
            tabs.append(pd.DataFrame({f"diana_{f}": d[pc].idxmax(axis=1).str[2:], f"diana_{f}_p": d[pc].max(axis=1)}))
        parts[f] = pd.concat(tabs)
    return pd.concat(parts.values(), axis=1)


def main(ndir: Path) -> None:
    ours = pd.concat([pd.read_csv(ROOT / f"data/splits_v19/{f}", sep="\t", dtype=str).assign(split=s)
                      for f, s in (("train_metadata.tsv", "train"), ("test_metadata.tsv", "held-out"))]).replace("", np.nan).set_index("Run_accession")
    meta = pq.read_table(ROOT / "data/external/metagenome_metadata.parquet", columns=["acc", "bioproject", "organism"]).to_pandas().set_index("acc")
    rows = []
    for f in sorted(ndir.glob("*_n.tsv")):
        run = f.name[:-len("_n.tsv")]
        nb = pd.read_csv(f, sep="\t", dtype={"ID": str})
        nb = nb[nb.ID != run].rename(columns={"ID": "acc"}).join(meta, on="acc")
        n, prop, share = propose(nb, meta.organism.get(run))
        rows.append({"Run_accession": run, "sra_tag": meta.organism.get(run), "faure_proposal": prop,
                     "faure_selected_neighbours": n, "faure_top_share": share})
    d = pd.DataFrame(rows).set_index("Run_accession").join(ours[["split", "archive_project", *FIELDS, "feature"]]).join(diana_predictions())
    d.to_csv(OUT / "faure_rule_runs.tsv", sep="\t")
    print(f"runs: {len(d)} | rule could assess (>= {RULE['min_neighbours']} neighbours): {int((d.faure_selected_neighbours >= RULE['min_neighbours']).sum())} "
          f"| relabel proposed: {int(d.faure_proposal.notna().sum())}")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
    if d.faure_proposal.notna().any():
        p = d[d.faure_proposal.notna()]
        print("\nproposed relabelings (SRA tag -> proposal), with AMD labels and DIANA predictions:")
        print(p.groupby(["sra_tag", "faure_proposal", "community_type", "diana_community_type", "sample_host", "diana_sample_host",
                         "material", "diana_material"], dropna=False).size().to_string())
    print("\nwhere DIANA disagrees with AMD: does the SRA tag (or the rule's proposal, when there is one) side with DIANA or with AMD?")
    out = []
    for fld in FIELDS:
        dis = d[d[fld].notna() & d[f"diana_{fld}"].notna() & (d[fld] != d[f"diana_{fld}"])]
        for src in ("sra_tag", "faure_proposal"):
            imp = dis[src].map(lambda t: implied(t, fld))
            withd = [agrees(i, v) for i, v in zip(imp, dis[f"diana_{fld}"])]
            witha = [agrees(i, v) for i, v in zip(imp, dis[fld])]
            out.append({"field": fld, "source": src, "DIANA != AMD": len(dis),
                        "tag says something about the field": int(sum(w is not None for w in withd)),
                        "DIANA only": int(sum(w is True and a is False for w, a in zip(withd, witha))),
                        "AMD only": int(sum(w is False and a is True for w, a in zip(withd, witha))),
                        "both": int(sum(w is True and a is True for w, a in zip(withd, witha))),
                        "neither": int(sum(w is False and a is False for w, a in zip(withd, witha)))})
    print(pd.DataFrame(out).to_string(index=False))
    for fld in FIELDS:
        dis = d[d[fld].notna() & d[f"diana_{fld}"].notna() & (d[fld] != d[f"diana_{fld}"])]
        s = dis[[agrees(implied(t, fld), v) is True and agrees(implied(t, fld), a) is False
                 for t, v, a in zip(dis.sra_tag, dis[f"diana_{fld}"], dis[fld])]]
        if len(s):
            print(f"\n{fld}: runs where the SRA tag sides with DIANA only:")
            print(s.groupby(["archive_project", "sra_tag", fld, f"diana_{fld}"]).size().to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
