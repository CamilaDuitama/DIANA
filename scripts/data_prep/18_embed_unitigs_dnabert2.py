#!/usr/bin/env python3
"""S5a/S5b: one frozen DNABERT-2 description per unitig, cached to disk.

Why a cached table rather than a reader inside the model
-------------------------------------------------------
S4 computed each unitig's input-layer weights with a small CNN trained alongside the
model, which meant rebuilding all 110,202 vectors at every optimiser step: 9 GB of
activations, a chunked gradient replay, a reader learning rate to get wrong, and a
collapse to near-identical cards. None of that exists here. The 110,202 shared unitig
sequences never change, so a frozen model describes them **once**, the table goes to disk,
and training is a lookup.

Three things this script had to work around, all found in S5a on 2026-09-16 and recorded
so nobody rediscovers them:

1. `transformers` 5.17.0 cannot load DNABERT-2 at all. Its remote code reads
   `config.pad_token_id`, which 5.x no longer sets, and it cannot survive 5.x's
   meta-device initialisation (`low_cpu_mem_usage=False`, `device_map=None` and an
   explicit dtype all fail with "Tensor on device meta is not on the expected device").
   The environment was moved to `transformers==4.44.2`, which is safe because nothing
   else in the repo imports transformers.
2. Its model class declares the **standard** `transformers.BertConfig` while the repo
   ships its own, so `AutoModel.from_pretrained` refuses the pair. Passing the standard
   config explicitly is the fix.
3. Its attention is a Triton flash-attention kernel that asserts `q.is_cuda`, so this
   **cannot run on CPU**. A GPU is required even for a 10-sequence probe.
4. On a GPU that kernel then fails to compile: it calls `tl.dot(..., trans_b=True)` and
   Triton 3, which ships with torch 2.5.1, has removed that argument. Its own code has an
   escape hatch: `bert_layers.py:161` takes a pure-PyTorch attention path whenever
   `attention_probs_dropout_prob` is non-zero. Setting it to 0.1 routes around Triton, and
   because the model runs under `eval()` the dropout is a no-op, so the result is
   arithmetically the same attention, merely unfused.

Orientation is handled by canonicalising, not by averaging
---------------------------------------------------------
kmtricks and ggcat work in canonical k-mers, so which of a unitig's two orientations ggcat
wrote out is arbitrary and carries no biology. A description must not depend on it.

The first version embedded both orientations and averaged them. That is lossy: measured on
10 unitigs, a unitig sits at raw cosine 0.687 from its own reverse complement while two
completely unrelated unitigs sit at 0.641, and raw cosines here are dominated by a shared
direction whose norm is 2.29 against a typical deviation of 1.00. So the two orientations
are close to unrelated vectors and averaging them blurs the description rather than
combining two views of one object.

`--orientation canonical`, the default, instead embeds the lexicographically smaller of the
sequence and its reverse complement. Same immunity to ggcat's coin flip, nothing averaged
away, and one forward pass instead of two. `--orientation average` reproduces the old
behaviour for comparison and writes to a different file.

Pooling and the length rule, both fixed before the full run
-----------------------------------------------------------
A sequence becomes one vector by averaging the hidden states over its real tokens, with
padding and special tokens excluded. DNABERT-2's context is 512 tokens and its BPE
compresses roughly fivefold, so a unitig beyond about 2.5 kb does not fit; only ~1 % of
ours exceed 966 bp. Such a unitig is split into `WINDOW_BP` windows, each embedded, and
the windows averaged. Sequences are processed in length-sorted batches so padding stays
small.

    # S5a, needs a GPU
    python scripts/data_prep/18_embed_unitigs_dnabert2.py --probe 10
    # S5b
    python scripts/data_prep/18_embed_unitigs_dnabert2.py
    python scripts/data_prep/18_embed_unitigs_dnabert2.py --shuffled
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = ROOT / "data/sequences_v9"
MODEL = "zhihan1996/DNABERT-2-117M"
BASES = np.array(list("ACGTN"), dtype="U1")
WINDOW_BP = 2000          # comfortably under 512 BPE tokens; only ~1 % of unitigs need it
COMPLEMENT = str.maketrans("ACGT", "TGCA")
MAX_TOKENS_PER_BATCH = 16384


def load_model(device: str):
    """DNABERT-2 with both loading workarounds applied. See the module docstring."""
    from transformers import AutoModel, AutoTokenizer
    from transformers.models.bert.configuration_bert import BertConfig

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    cfg = BertConfig.from_pretrained(MODEL)     # standard config, not the repo's own
    # Non-zero attention dropout is what routes bert_layers.py:161 away from the Triton
    # kernel, which no longer compiles. eval() makes the dropout inert, so this changes
    # which code path runs and not what it computes.
    cfg.attention_probs_dropout_prob = 0.1
    model = AutoModel.from_pretrained(MODEL, config=cfg, trust_remote_code=True)
    model = model.to(device).eval()
    logger.info("loaded %s: %s, %s parameters, on %s", MODEL, type(model).__name__,
                f"{sum(p.numel() for p in model.parameters()):,}", device)
    return tok, model


def read_sequences(shuffled: bool) -> list[str]:
    """Unitig sequences in matrix-column order, decoded from the cached base codes.

    Read from the `.npy` rather than the fasta for both arms, because the shuffled control
    only exists in that form and because `17_unitig_sequence_tensor.py` already asserted
    that order against `unitigs.frac.mat` row by row.
    """
    suffix = ".shuffled" if shuffled else ""
    flat = np.load(SEQ_DIR / f"unitig_bases{suffix}.npy")
    offsets = np.load(SEQ_DIR / "unitig_offsets.npy")
    chars = BASES[flat]
    return ["".join(chars[offsets[i]:offsets[i + 1]]) for i in range(len(offsets) - 1)]


@torch.no_grad()
def embed_batch(tok, model, seqs: list[str], device: str) -> torch.Tensor:
    """Mean over real tokens, special tokens and padding excluded."""
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    h = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
    mask = enc["attention_mask"].clone()
    for special in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id):
        if special is not None:
            mask[enc["input_ids"] == special] = 0
    m = mask.unsqueeze(-1).to(h.dtype)
    return ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu()


def windows(seq: str) -> list[str]:
    return [seq[i:i + WINDOW_BP] for i in range(0, len(seq), WINDOW_BP)] or [seq]


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def canonical(seq: str) -> str:
    """The lexicographically smaller of a sequence and its reverse complement.

    kmtricks and ggcat work in canonical k-mers, so which of the two orientations ggcat
    writes out for a unitig is arbitrary and carries no biology. A description must not
    depend on that coin flip.

    Averaging the two orientations' embeddings was the first attempt and it is lossy:
    DNABERT-2 does not recognise them as related. Measured on 10 unitigs, a unitig sits at
    raw cosine 0.687 from its own reverse complement while two unrelated unitigs sit at
    0.641, and raw cosines in this embedding are dominated by a shared direction whose norm
    is 2.29 against a typical deviation of 1.00. So averaging blends a description with a
    near-unrelated one instead of combining two views of the same thing.

    Choosing one orientation by a fixed rule gives the identical invariance and blends
    nothing away: the same unitig yields the same string whichever way ggcat wrote it.
    """
    rc = reverse_complement(seq)
    return seq if seq <= rc else rc


def embed_all(tok, model, seqs: list[str], device: str,
              orientation: str = "canonical") -> np.ndarray:
    """Every sequence, in length-sorted batches, long ones windowed and averaged.

    `orientation="canonical"` embeds one fixed orientation per unitig, chosen by
    :func:`canonical`. `orientation="average"` reproduces the original lossy behaviour and
    exists only so the two can be compared on the same code path.
    """
    if orientation == "canonical":
        seqs = [canonical(s) for s in seqs]
    order = np.argsort([len(s) for s in seqs], kind="stable")
    dim = model.config.hidden_size
    table = np.zeros((len(seqs), dim), dtype=np.float32)
    batch, batch_idx, done, t0 = [], [], 0, time.time()

    def flush():
        nonlocal batch, batch_idx, done
        if not batch:
            return
        fwd = embed_batch(tok, model, batch, device)
        if orientation == "average":
            rev = embed_batch(tok, model, [reverse_complement(s) for s in batch], device)
            fwd = (fwd + rev) / 2
        table[batch_idx] = fwd.numpy()
        done += len(batch)
        if done % 20000 < len(batch):
            logger.info("%d / %d unitigs, %.0f s elapsed", done, len(seqs), time.time() - t0)
        batch, batch_idx = [], []

    for i in order:
        s = seqs[i]
        if len(s) > WINDOW_BP:
            flush()
            w = windows(s)
            parts = embed_batch(tok, model, w, device)
            if orientation == "average":
                parts_rc = embed_batch(tok, model, [reverse_complement(x) for x in w], device)
                parts = (parts + parts_rc) / 2
            table[i] = parts.mean(0).numpy()
            done += 1
            continue
        # keep the padded batch under a token budget: BPE gives roughly 5 bases per token
        longest = max([len(s)] + [len(x) for x in batch])
        if batch and (len(batch) + 1) * (longest // 5 + 8) > MAX_TOKENS_PER_BATCH:
            flush()
        batch.append(s)
        batch_idx.append(int(i))
    flush()
    logger.info("embedded %d unitigs in %.0f s", len(seqs), time.time() - t0)
    return table


def report_table(table: np.ndarray) -> dict:
    """The acceptance check: a table of near-identical rows cannot work downstream.

    S4 burned 40 fits because nothing looked at whether its per-unitig vectors actually
    differed from each other. One minute of arithmetic here answers that.
    """
    x = table - table.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    ev = (s ** 2) / (s ** 2).sum()
    n = table / np.clip(np.linalg.norm(table, axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(n), size=min(2000, len(n)), replace=False)
    sub = n[pick]
    cos = (sub @ sub.T)[~np.eye(len(sub), dtype=bool)]
    return {"pc1_variance_share": float(ev[0]),
            "pc10_variance_share": float(ev[:10].sum()),
            "n_components_for_90pct": int(np.searchsorted(np.cumsum(ev), 0.90) + 1),
            "cosine_between_unitigs_mean": float(cos.mean()),
            "cosine_between_unitigs_p01": float(np.percentile(cos, 1)),
            "cosine_between_unitigs_p99": float(np.percentile(cos, 99))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=int, default=0,
                    help="S5a: embed only this many unitigs and report the checks")
    ap.add_argument("--shuffled", action="store_true",
                    help="use the within-unitig shuffled bases (the S5 control table)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--orientation", choices=["canonical", "average"], default="canonical",
                    help="How to make the description independent of the arbitrary "
                         "orientation ggcat wrote. 'canonical' embeds one fixed orientation "
                         "per unitig and blends nothing away; 'average' is the original lossy "
                         "behaviour, kept only for comparison. Tables are written to "
                         "different files so neither overwrites the other.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("DNABERT-2's Triton attention asserts q.is_cuda; a GPU is required")

    tok, model = load_model(args.device)
    seqs = read_sequences(args.shuffled)
    logger.info("%d sequences, %d bases, shuffled=%s", len(seqs),
                sum(len(s) for s in seqs), args.shuffled)

    if args.probe:
        sub = seqs[:args.probe]
        emb = embed_batch(tok, model, sub, args.device)
        logger.info("probe: %d unitigs of lengths %s -> embedding %s",
                    len(sub), [len(s) for s in sub], tuple(emb.shape))
        comp = str.maketrans("ACGT", "TGCA")
        rc = embed_batch(tok, model, [s.translate(comp)[::-1] for s in sub], args.device)
        cos_rc = torch.nn.functional.cosine_similarity(emb, rc, dim=1)
        logger.info("cosine(unitig, its reverse complement): mean %.4f min %.4f",
                    cos_rc.mean(), cos_rc.min())
        n = torch.nn.functional.normalize(emb, dim=1)
        off = (n @ n.T)[~torch.eye(len(n), dtype=bool)]
        logger.info("cosine between different unitigs:       mean %.4f min %.4f max %.4f",
                    off.mean(), off.min(), off.max())
        return

    table = embed_all(tok, model, seqs, args.device, args.orientation)
    suffix = ".shuffled" if args.shuffled else ""
    tag = "" if args.orientation == "average" else ".canon"
    out = SEQ_DIR / f"unitig_dnabert2{tag}{suffix}.npy"
    if out.exists():
        logger.error("%s already exists; refusing to overwrite a description table, since a "
                     "stale table silently propagates into every matrix built from it", out)
        return 1
    np.save(out, table)
    checks = report_table(table)
    checks.update({"model": MODEL, "n_unitigs": len(seqs), "dim": int(table.shape[1]),
                   "shuffled": bool(args.shuffled), "window_bp": WINDOW_BP,
                   "orientation": "forward and reverse complement embedded and averaged",
                   "pooling": "mean over real tokens, special tokens and padding excluded"})
    (SEQ_DIR / f"unitig_dnabert2{suffix}.json").write_text(json.dumps(checks, indent=2))
    logger.info("wrote %s  %s", out, table.shape)
    for k, v in checks.items():
        if isinstance(v, float):
            logger.info("  %-32s %.4f", k, v)


if __name__ == "__main__":
    main()
