"""S4: compute each unitig's input-layer weights from its bases instead of storing them.

What this replaces
------------------
``MultiTaskMLP.backbone[0]`` is ``Linear(n_unitigs, H0)``. Column ``u`` of its weight is a
private vector of ``H0`` numbers belonging to unitig ``u``, and the layer computes::

    h = sum_u  fraction_u * w_u  +  b

With 110,202 unitigs that is 14.1 M free numbers at H0 = 128 and 56.4 M at H0 = 512, all
fitted to 2,716 samples, and a unitig present in 11 samples has its vector fitted to those
11 samples alone. :class:`SequenceInputLayer` computes ``w_u`` from the unitig's bases with
one small CNN shared by every unitig, so the same vector comes from a function fitted to
all 17.5 M bases in the corpus and similar sequences get similar vectors for free.

Everything after this layer is untouched: BatchNorm, activation, dropout, the remaining
hidden layers and the task heads are byte-identical to the fraction arm.

Two gates, not one
------------------
A weighted sum alone is the operation that has already tied in 36 of 37 representation
comparisons, so each unitig gets two vectors::

    h = sum_u [ fraction_u * A_u  +  present_u * B_u ]  +  b

which lets the model express "this sequence matters only when the unitig is complete".
The reader therefore emits ``2 * H0`` numbers per unitig. Presence rather than abundance
for the second gate: abundance is byte-identical to the fraction for 828 of 922 held-out
runs, so it would duplicate the first.

Memory, which is the binding constraint
---------------------------------------
Every sample sums over all 110,202 unitigs, so all 110,202 vectors must exist at every
step, and storing one conv layer's activations for them is 17.5 M x 96 x 4 B = 6.7 GB.
Three layers plus backward does not fit anywhere comfortably, so:

1. the vectors are built in **length-sorted chunks** under ``no_grad`` into a cache that is
   deliberately *not* a parameter (padding all unitigs to the 40,205 bp maximum would
   process 4.43 billion positions against 17,498,477 real ones, a 253x waste, so chunks
   are cut by padded size);
2. the gradient that autograd accumulates on that cache is pushed back into the reader by
   **replaying the same chunks with gradients enabled**, one chunk at a time.

Peak memory is then set by the chunk, not by the corpus, and the result is exact for the
steps covered by one refresh. The cache is rebuilt every ``refresh_every`` optimiser
steps.

``refresh_every`` defaults to 1, and the reason is a failure worth recording. It was first
set to 8 to save compute, on the reasoning that the interval only trades staleness against
time. That was wrong: the reader is updated once per rebuild, so the interval *is* the
reader's update budget. At batch 256 on 1,840 training runs an epoch is 8 steps, so
``refresh_every=8`` gave the reader one update per epoch, and with early stopping picking
epoch 9 the trunk received nine Adam updates at a learning rate tuned for a 56 M-parameter
table. The real and shuffled arms then produced byte-identical predictions and both
collapsed to the majority class, because neither had trained. At 1 the reader is updated
on every step, like every other parameter in the model.

The reader has no dropout by design: a stochastic reader would freeze one dropout sample
into the cache for the whole refresh interval.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

N_SYMBOLS = 5  # A, C, G, T and one shared index for everything else


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each position, for a (B, C, L) tensor.

    Each position is normalised using only its own channels, so a padded position can
    never influence a real one and a unitig's card does not depend on which other unitigs
    share its chunk. Normalising over the length axis instead (GroupNorm, InstanceNorm)
    pulls the zero padding into the statistics and breaks exactly that property.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class UnitigReader(nn.Module):
    """1D CNN over bases, shared across unitigs, ending in a masked mean over positions.

    The mean is what makes the output length-independent: a 61 bp unitig and a 40,205 bp
    unitig both come out as ``out_dim`` numbers.

    Normalisation is per position over channels (:class:`ChannelLayerNorm`), which is the
    only choice that survives chunking. ``BatchNorm`` would take statistics over a chunk,
    and chunks are length-sorted, so those are statistics over one narrow length band that
    change with the chunk plan. ``GroupNorm`` and ``InstanceNorm`` pool over the length
    axis, which means the zero padding enters the statistics and a unitig's card then
    depends on how long the longest unitig in its chunk happened to be. Both were tried;
    ``tests/test_unitig_encoder.py`` fails on either.
    """

    def __init__(self,
                 out_dim: int,
                 embed_dim: int = 32,
                 channels: Sequence[int] = (96, 96),
                 kernel: int = 9,
                 activation: str = "relu") -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"kernel must be odd so padding is symmetric, got {kernel}")
        act = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]

        self.embed = nn.Embedding(N_SYMBOLS, embed_dim)
        blocks: List[nn.Module] = []
        prev = embed_dim
        for width in channels:
            blocks.append(nn.Sequential(
                nn.Conv1d(prev, width, kernel_size=kernel, padding=kernel // 2),
                ChannelLayerNorm(width),
                act(),
            ))
            prev = width
        self.blocks = nn.ModuleList(blocks)
        self.project = nn.Linear(prev, out_dim)
        self.kernel = kernel

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``tokens`` (B, L) int64, ``mask`` (B, L) bool -> (B, out_dim).

        Padded positions are zeroed after every convolution. Without that, a same-padded
        kernel of width 9 lets pad positions bleed 4 positions into every real one, and
        the amount of bleed would depend on the chunk a unitig happened to land in.
        """
        m = mask.unsqueeze(1)  # (B, 1, L)
        h = self.embed(tokens).transpose(1, 2) * m  # (B, embed, L)
        for block in self.blocks:
            h = block(h) * m
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        pooled = h.sum(dim=2) / lengths  # (B, C)
        return self.project(pooled)


def plan_chunks(lengths: np.ndarray, max_padded: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Length-sorted unitig order plus ``(start, stop)`` slices of that order.

    A chunk is padded to its own longest member, so cutting on ``rows * max_len`` keeps
    every chunk's padded size under ``max_padded`` regardless of how skewed the tail is.
    """
    order = np.argsort(lengths, kind="stable")
    chunks: List[Tuple[int, int]] = []
    start = 0
    for i in range(len(order)):
        longest = int(lengths[order[i]])
        if (i + 1 - start) * longest > max_padded and i > start:
            chunks.append((start, i))
            start = i
    chunks.append((start, len(order)))
    return order, chunks


class SequenceInputLayer(nn.Module):
    """Drop-in replacement for ``Linear(n_unitigs, out_dim)`` reading weights from DNA.

    Args:
        bases: flat ``uint8`` base codes for the whole vocabulary, in matrix column order.
        offsets: ``n_unitigs + 1`` entries; unitig ``u`` occupies ``bases[offsets[u]:offsets[u+1]]``.
        out_dim: ``H0``, the first hidden width of the model being replaced. Not a free
            choice: it must equal the width the fraction arm used, or the arms differ in
            more than the input layer.
        refresh_every: optimiser steps between cache rebuilds, and therefore also the
            steps between reader updates. **Leave this at 1** unless memory or time forces
            otherwise; see the module docstring for what 8 did.
        max_padded_per_chunk: padded positions per chunk, the knob that sets peak memory.
    """

    def __init__(self,
                 bases: np.ndarray,
                 offsets: np.ndarray,
                 out_dim: int,
                 refresh_every: int = 1,
                 max_padded_per_chunk: int = 1_000_000,
                 reader: Optional[UnitigReader] = None,
                 **reader_kwargs) -> None:
        super().__init__()
        if bases.dtype != np.uint8:
            raise ValueError(f"bases must be uint8, got {bases.dtype}")
        if int(offsets[-1]) != len(bases):
            raise ValueError(f"offsets end at {int(offsets[-1])} but bases has {len(bases)} entries")

        self.n_unitigs = len(offsets) - 1
        self.out_dim = out_dim
        self.refresh_every = int(refresh_every)

        # persistent=False keeps 17.5 MB of bases out of every checkpoint while still
        # following the module to whichever device .to() moves it to.
        self.register_buffer("bases", torch.from_numpy(bases.astype(np.int64)), persistent=False)
        self.register_buffer("offsets", torch.from_numpy(offsets.astype(np.int64)), persistent=False)

        lengths = np.diff(offsets)
        order, chunks = plan_chunks(lengths, max_padded_per_chunk)
        self.register_buffer("order", torch.from_numpy(order.astype(np.int64)), persistent=False)
        self._chunks = chunks
        self.register_buffer("lengths", torch.from_numpy(lengths.astype(np.int64)), persistent=False)

        self.reader = reader if reader is not None else UnitigReader(out_dim=2 * out_dim, **reader_kwargs)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self._match_linear_init_scale()

        # Not parameters and not buffers: derived state. Recomputed on demand, never saved.
        self._cards: Optional[torch.Tensor] = None
        self._grad_accum: Optional[torch.Tensor] = None
        self._step = 0
        self._reader_updates = 0

    # ---------------------------------------------------------------- initialisation

    def _match_linear_init_scale(self) -> None:
        """Start with cards the size ``Linear(n_unitigs, out_dim)`` would have started with.

        Torch initialises that layer to U(-1/sqrt(n_unitigs), +1/sqrt(n_unitigs)), i.e. an
        entry standard deviation of 1/sqrt(3 * n_unitigs) = 0.0017 at 110,202 unitigs. The
        reader's projection would otherwise emit entries around 1/sqrt(channels) = 0.1,
        some 60x larger, and since a sample sums over roughly 3,190 present unitigs the two
        arms would enter the first BatchNorm at completely different scales. That is a
        difference in effective learning rate, not in representation, and it would
        contaminate the comparison this arm exists to make.
        """
        target = 1.0 / np.sqrt(3.0 * self.n_unitigs)
        with torch.no_grad():
            observed = self.reader.project.weight.std().item()
            self.reader.project.weight.mul_(target / observed)
            self.reader.project.bias.zero_()
            # The presence half starts at exactly zero, so at step 0 this layer computes
            # what the fraction-only Linear computes and nothing else. Without it the
            # second gate is the louder of the two from the start: it sums ones where the
            # first sums fractions averaging about 0.16, and none of the four v9 configs
            # uses BatchNorm, so there is nothing downstream to absorb the difference. The
            # gate is not dead, because its gradient is the presence vector, which is not
            # zero; it grows only if it earns the gradient.
            self.reader.project.weight[self.out_dim:].zero_()
        logger.info("reader projection rescaled to entry sd %.2e (Linear(%d, %d) default); "
                    "presence gate zero-initialised", target, self.n_unitigs, self.out_dim)

    # ---------------------------------------------------------------- card computation

    def _chunk_tensors(self, start: int, stop: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Padded ``(tokens, mask, index)`` for one slice of the length-sorted order.

        Gathered in one indexing operation rather than a row loop: a loop would issue
        110,202 small copies per cache rebuild, which on a GPU costs more than the
        convolutions it is feeding.
        """
        idx = self.order[start:stop]
        lens = self.lengths[idx]
        width = int(lens.max().item())
        pos = torch.arange(width, device=self.bases.device)
        mask = pos.unsqueeze(0) < lens.unsqueeze(1)
        flat = (self.offsets[idx].unsqueeze(1) + pos.unsqueeze(0)).clamp_(max=len(self.bases) - 1)
        tokens = torch.where(mask, self.bases[flat], torch.zeros((), dtype=torch.long,
                                                                 device=self.bases.device))
        return tokens, mask, idx

    @torch.no_grad()
    def _build_cards(self) -> torch.Tensor:
        """All ``n_unitigs x 2*out_dim`` cards, chunk by chunk, no autograd graph kept."""
        cards = torch.empty((self.n_unitigs, 2 * self.out_dim),
                            dtype=self.bias.dtype, device=self.bias.device)
        for start, stop in self._chunks:
            tokens, mask, idx = self._chunk_tensors(start, stop)
            cards[idx] = self.reader(tokens, mask)
        return cards

    def _harvest(self) -> None:
        """Move the gradient autograd put on the cache into the persistent accumulator."""
        if self._cards is not None and self._cards.grad is not None:
            if self._grad_accum is None:
                self._grad_accum = torch.zeros_like(self._cards)
            self._grad_accum += self._cards.grad
            self._cards.grad = None

    def _flush_to_reader(self) -> None:
        """Replay the chunks with gradients and push the accumulated gradient into the reader.

        This is the step that keeps peak memory bounded: the activations for one chunk
        exist at a time, and ``torch.autograd.backward`` on that chunk's cards with the
        matching slice of the accumulated gradient adds exactly the contribution the
        un-chunked computation would have added.

        Called from inside ``forward`` during training, so the resulting ``.grad`` on the
        reader's parameters is written after the trainer's ``zero_grad()`` and consumed by
        the ``step()`` that follows.
        """
        if self._grad_accum is None:
            return
        self._reader_updates += 1
        # Logged sparsely so an inert reader is visible in the job log: a fit that ends with
        # a two-digit count has not trained the reader, whatever its loss curve says.
        if self._reader_updates in (1, 10) or self._reader_updates % 250 == 0:
            logger.info("reader update %d (step %d)", self._reader_updates, self._step)
        for start, stop in self._chunks:
            tokens, mask, idx = self._chunk_tensors(start, stop)
            with torch.enable_grad():
                out = self.reader(tokens, mask)
            torch.autograd.backward(out, grad_tensors=self._grad_accum[idx])
        self._grad_accum = None

    def refresh(self, flush: bool = True) -> None:
        """Push pending gradient into the reader, then rebuild the cache from it."""
        self._harvest()
        if flush:
            self._flush_to_reader()
        else:
            self._grad_accum = None
        cards = self._build_cards()
        self._cards = cards.requires_grad_(self.training)

    def train(self, mode: bool = True) -> "SequenceInputLayer":
        """Invalidate the cache on any mode switch.

        Cards built during training are up to ``refresh_every`` steps behind the reader.
        Evaluating on stale cards would score a model that is not the one being trained,
        which matters because early stopping reads that score. Any pending gradient is
        harvested first so nothing is lost.
        """
        if self.training != mode:
            self._harvest()
            self._cards = None
        return super().train(mode)

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (B, n_unitigs) fractions -> (B, out_dim)."""
        if x.shape[1] != self.n_unitigs:
            raise ValueError(f"expected {self.n_unitigs} input columns, got {x.shape[1]}")

        if self.training:
            if self._cards is None or self._step % self.refresh_every == 0:
                self.refresh()
            self._step += 1
        elif self._cards is None:
            with torch.no_grad():
                self._cards = self._build_cards()

        a, b = self._cards[:, :self.out_dim], self._cards[:, self.out_dim:]
        present = (x > 0).to(x.dtype)
        return x @ a + present @ b + self.bias

    def extra_repr(self) -> str:
        return (f"n_unitigs={self.n_unitigs}, out_dim={self.out_dim}, "
                f"refresh_every={self.refresh_every}, chunks={len(self._chunks)}, "
                f"reader_updates={self._reader_updates}")


def attach_sequence_encoder(model: nn.Module, cfg: dict, root=None) -> nn.Module:
    """Swap ``model.backbone[0]`` for a :class:`SequenceInputLayer` built from ``cfg``.

    ``cfg`` is the ``sequence_encoder`` block of a training config. The replaced layer's
    ``out_features`` fixes the card length, so the width is taken from the model rather
    than restated in the config, where it could drift.

    Must be called before the optimiser is constructed (the reader's parameters have to be
    in ``model.parameters()``) and before ``load_state_dict`` (the key names change).
    """
    from pathlib import Path

    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    first = model.backbone[0]
    if not isinstance(first, nn.Linear):
        raise TypeError(f"backbone[0] is {type(first).__name__}, expected Linear")

    bases = np.load(root / cfg["bases"])
    offsets = np.load(root / cfg["offsets"])
    if len(offsets) - 1 != first.in_features:
        raise ValueError(f"{len(offsets) - 1} unitigs in {cfg['bases']} but the model "
                         f"expects {first.in_features} input columns")

    layer = SequenceInputLayer(
        bases=bases,
        offsets=offsets,
        out_dim=first.out_features,
        refresh_every=cfg.get("refresh_every", 1),
        max_padded_per_chunk=cfg.get("max_padded_per_chunk", 1_000_000),
        embed_dim=cfg.get("embed_dim", 32),
        channels=cfg.get("channels", (96, 96)),
        kernel=cfg.get("kernel", 9),
        activation=cfg.get("reader_activation", "relu"),
    )
    replaced = sum(p.numel() for p in first.parameters())
    added = sum(p.numel() for p in layer.parameters())
    model.backbone[0] = layer
    logger.info("sequence encoder attached: %s free numbers -> %s shared (%.1fx fewer), %s",
                f"{replaced:,}", f"{added:,}", replaced / added, cfg.get("bases"))
    return model
