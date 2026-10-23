#!/usr/bin/env python3
"""v19 C4: pick an arm's configuration from its five re-scored trials.

The winner is the highest five-seed smoothed criterion; its argmax is the arm's epoch
count. A winner whose argmax lies within 5 epochs of the cap has not peaked: the five
candidates are re-scored with the cap doubled and nothing is written (exit code 3).
Output: <out-dir>/<arm>.json, a training config with the epoch count and provenance.

    ./env/bin/python scripts/training/32_select_v19.py --arm feature \\
        --search-dir results/search_v19/feature --out-dir configs/v19_final
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--search-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    files = sorted((a.search_dir / "rescored").glob("rank*_cap*.json"))
    caps = sorted({int(f.stem.split("cap")[1]) for f in files})
    if not caps:
        raise SystemExit("no re-scored candidates")
    cap = caps[-1]
    cands = [json.load(open(f)) for f in files if f.stem.endswith(f"cap{cap}")]
    if sorted(c["rank"] for c in cands) != list(range(5)):
        raise SystemExit(f"cap {cap}: re-scored ranks {sorted(c['rank'] for c in cands)}, need 0-4")
    for c in sorted(cands, key=lambda c: -c["selection"]["criterion_smoothed"]):
        log.info("rank %d trial %d: %.4f at %d epochs (peaked %s)", c["rank"], c["trial"],
                 c["selection"]["criterion_smoothed"], c["selection"]["epochs"], c["selection"]["peaked"])
    win = max(cands, key=lambda c: c["selection"]["criterion_smoothed"])
    if not win["selection"]["peaked"]:
        log.warning("winner (rank %d) did not peak below the cap %d: re-score all five with --cap %d",
                    win["rank"], cap, 2 * cap)
        return 3
    cfg = dict(win["config"])
    cfg["epochs"] = int(win["selection"]["epochs"])
    cfg["_provenance"] = {"protocol": "PROTOCOLS.md v19, C3-C4", "search_trial": win["trial"], "rank": win["rank"],
                          "five_seed_criterion_smoothed": win["selection"]["criterion_smoothed"],
                          "cap": cap, "selection": win["selection"],
                          "candidates": {c["rank"]: c["selection"]["criterion_smoothed"] for c in cands}}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{a.arm}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    json.dump(cfg, open(out, "w"), indent=1)
    log.info("%s: trial %d, %d epochs -> %s", a.arm, win["trial"], cfg["epochs"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
