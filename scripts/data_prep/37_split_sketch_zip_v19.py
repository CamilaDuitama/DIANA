#!/usr/bin/env python3
"""Repackage our sketch zip as one sourmash zip per run, the input layout of the authors'
`project_everything` (it unzips each file and reads every signatures/*.gz in it, naming the
vector after the file name up to the first dot). Only the sketch files are copied, byte for
byte; nothing is recomputed.

    ./env/bin/python scripts/data_prep/37_split_sketch_zip_v19.py <out_dir>
"""
from __future__ import annotations

import csv
import io
import logging
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKETCHES = ROOT / "data/sketches_v9/unitigs_split_v9.zip"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refusing to overwrite")
    with zipfile.ZipFile(SKETCHES) as z:
        text = z.read("SOURMASH-MANIFEST.csv").decode()
        rows = list(csv.DictReader(io.StringIO("\n".join(l for l in text.splitlines() if not l.startswith("#")))))
        names = [r["name"] for r in rows]
        if len(set(names)) != len(names):
            raise SystemExit("a run has more than one signature in the manifest")
        for r in rows:
            if r["ksize"] != "31":
                raise SystemExit(f"{r['name']}: ksize {r['ksize']}")
            with zipfile.ZipFile(out / f"{r['name']}.sig.zip", "w", zipfile.ZIP_STORED) as o:
                o.writestr(r["internal_location"], z.read(r["internal_location"]))
    log.info("%d per-run zips in %s", len(rows), out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
