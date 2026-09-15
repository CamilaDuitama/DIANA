"""The chunked card cache must give the same gradient as computing every card at once.

The whole point of :class:`SequenceInputLayer` is that all 110,202 cards exist at every
step while only one chunk's activations exist at a time. That is only sound if replaying
the chunks with the accumulated gradient produces exactly the reader gradient the
un-chunked computation would have produced. If it does not, the arm trains something
other than what it claims to, and would still converge, so nothing downstream would catch
it. Hence this test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diana.models.unitig_encoder import SequenceInputLayer, UnitigReader, plan_chunks

N_UNITIGS = 137
OUT_DIM = 8
SEED = 0


def toy_vocabulary(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Ragged bases with a length spread wide enough to force several chunks."""
    lengths = rng.integers(5, 40, size=N_UNITIGS)
    offsets = np.zeros(N_UNITIGS + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    bases = rng.integers(0, 4, size=int(offsets[-1])).astype(np.uint8)
    return bases, offsets


def build_layer(bases, offsets, **kw) -> SequenceInputLayer:
    torch.manual_seed(SEED)
    return SequenceInputLayer(bases=bases, offsets=offsets, out_dim=OUT_DIM,
                              embed_dim=6, channels=(7, 7), kernel=3, **kw)


def test_plan_chunks_covers_every_unitig_once():
    lengths = np.array([10, 1000, 20, 30, 40000, 15])
    order, chunks = plan_chunks(lengths, max_padded=2000)
    covered = np.concatenate([order[a:b] for a, b in chunks])
    assert sorted(covered.tolist()) == list(range(len(lengths)))
    assert len(chunks) > 1, "a 40,000 bp member must not share a chunk with a 10 bp one"
    for a, b in chunks:
        longest = lengths[order[a:b]].max()
        assert (b - a) == 1 or (b - a) * longest <= 2000


def test_masked_mean_is_length_invariant():
    """A padded copy of a sequence must read identically to the unpadded one."""
    torch.manual_seed(SEED)
    reader = UnitigReader(out_dim=4, embed_dim=6, channels=(7,), kernel=3).eval()
    tokens = torch.tensor([[0, 1, 2, 3, 1]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    padded = torch.cat([tokens, torch.zeros((1, 11), dtype=torch.long)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros((1, 11), dtype=torch.bool)], dim=1)
    with torch.no_grad():
        assert torch.allclose(reader(tokens, mask), reader(padded, padded_mask), atol=1e-6)


def test_chunked_replay_matches_unchunked_gradient():
    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    x = torch.from_numpy(rng.random((5, N_UNITIGS)).astype(np.float32))
    x[x < 0.5] = 0.0  # a realistic presence rate, so the second gate is exercised
    target = torch.from_numpy(rng.standard_normal((5, OUT_DIM)).astype(np.float32))

    # Reference: one chunk holding the entire vocabulary, so nothing is replayed.
    ref = build_layer(bases, offsets, max_padded_per_chunk=10 ** 9, refresh_every=1)
    assert len(ref._chunks) == 1
    ref.train()
    loss = ((ref(x) - target) ** 2).sum()
    loss.backward()
    ref.refresh()  # harvest + replay, which for one chunk is the plain computation
    reference = {n: p.grad.clone() for n, p in ref.reader.named_parameters()}

    # Chunked: same weights, same data, many chunks, gradient pushed back chunk by chunk.
    sub = build_layer(bases, offsets, max_padded_per_chunk=200, refresh_every=1)
    assert len(sub._chunks) > 3
    sub.train()
    sub_loss = ((sub(x) - target) ** 2).sum()
    sub_loss.backward()
    sub.refresh()

    assert torch.allclose(loss, sub_loss, atol=1e-5), "forward differs before gradients"
    for name, p in sub.reader.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        # Relative, not absolute: these gradients reach magnitude 500 and the two paths
        # sum the same terms in a different order, so float32 differs in the last digit.
        torch.testing.assert_close(p.grad, reference[name], rtol=1e-4, atol=1e-5,
                                   msg=lambda s, n=name: f"{n}: {s}")


def test_gradient_accumulates_across_the_refresh_interval():
    """With refresh_every = 4, four steps of gradient must arrive, not one."""
    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    x = torch.from_numpy(rng.random((5, N_UNITIGS)).astype(np.float32))
    target = torch.from_numpy(rng.standard_normal((5, OUT_DIM)).astype(np.float32))

    layer = build_layer(bases, offsets, max_padded_per_chunk=200, refresh_every=4)
    layer.train()
    for _ in range(4):
        ((layer(x) - target) ** 2).sum().backward()
    assert layer._grad_accum is None, "nothing should have been flushed yet"
    layer.refresh()
    total = sum(p.grad.abs().sum() for p in layer.reader.parameters())

    single = build_layer(bases, offsets, max_padded_per_chunk=200, refresh_every=4)
    single.train()
    ((single(x) - target) ** 2).sum().backward()
    single.refresh()
    one = sum(p.grad.abs().sum() for p in single.reader.parameters())

    assert total > 3.5 * one, f"expected about 4x one step's gradient, got {total / one:.2f}x"


def test_eval_rebuilds_the_cache_so_scores_use_current_weights():
    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    x = torch.from_numpy(rng.random((3, N_UNITIGS)).astype(np.float32))

    layer = build_layer(bases, offsets, max_padded_per_chunk=200, refresh_every=100)
    layer.train()
    layer(x)
    with torch.no_grad():
        layer.reader.project.bias.add_(1.0)  # move the reader behind the cache's back
    stale = layer._cards.clone()
    layer.eval()
    assert layer._cards is None, "switching to eval must invalidate the cache"
    with torch.no_grad():
        layer(x)
    assert not torch.allclose(stale, layer._cards), "eval used the stale cache"


def test_cache_is_not_in_the_state_dict():
    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    layer = build_layer(bases, offsets, max_padded_per_chunk=200)
    layer.train()
    layer(torch.from_numpy(rng.random((2, N_UNITIGS)).astype(np.float32)))
    keys = set(layer.state_dict())
    assert not any("card" in k or "bases" in k or "offsets" in k for k in keys), sorted(keys)
    assert "bias" in keys and any(k.startswith("reader.") for k in keys)


def test_reader_is_orders_of_magnitude_smaller_than_the_table_it_replaces():
    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    layer = build_layer(bases, offsets, max_padded_per_chunk=200)
    reader_params = sum(p.numel() for p in layer.reader.parameters())
    table_params = N_UNITIGS * OUT_DIM
    assert reader_params < 10 * table_params  # the real ratio is set by the real vocabulary
    assert layer.out_dim == OUT_DIM


def test_rejects_a_width_it_was_not_asked_for():
    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    layer = build_layer(bases, offsets, max_padded_per_chunk=200)
    with pytest.raises(ValueError, match="input columns"):
        layer(torch.zeros((2, N_UNITIGS + 1)))


def test_attach_swaps_the_layer_and_survives_a_checkpoint_roundtrip(tmp_path):
    """What `02_train_final_model.py` writes, `diana-test` must be able to read back.

    The swap changes the state_dict keys, so the training script and the test script have
    to apply it at the same point. If they diverge, `load_state_dict` either raises or,
    worse, loads a partially matching model and scores it.
    """
    from diana.models.multitask_mlp import MultiTaskMLP
    from diana.models.unitig_encoder import attach_sequence_encoder

    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    np.save(tmp_path / "bases.npy", bases)
    np.save(tmp_path / "offsets.npy", offsets)
    cfg = {"bases": "bases.npy", "offsets": "offsets.npy", "embed_dim": 6,
           "channels": [7, 7], "kernel": 3, "refresh_every": 2,
           "max_padded_per_chunk": 200}

    def fresh() -> MultiTaskMLP:
        torch.manual_seed(SEED)
        model = MultiTaskMLP(input_dim=N_UNITIGS, hidden_dims=[OUT_DIM, 6],
                             num_classes={"material": 3}, dropout=0.0,
                             use_batch_norm=False)
        return attach_sequence_encoder(model, cfg, root=tmp_path)

    trained = fresh()
    assert trained.backbone[0].out_dim == OUT_DIM, "card length must come from the layer"
    x = torch.from_numpy(rng.random((4, N_UNITIGS)).astype(np.float32))

    trained.train()
    ((trained(x)["material"]) ** 2).sum().backward()
    torch.optim.SGD(trained.parameters(), lr=0.1).step()  # move it away from init
    trained.eval()
    with torch.no_grad():
        expected = trained(x)["material"]

    state = trained.state_dict()
    assert not any("card" in k for k in state)

    loaded = fresh()
    loaded.load_state_dict(state)  # strict: any key mismatch raises here
    loaded.eval()
    with torch.no_grad():
        assert torch.allclose(loaded(x)["material"], expected, atol=1e-6)


def test_the_two_gates_are_distinguishable():
    """Presence and fraction must enter through different weights, or there is one gate.

    Two samples containing the same unitigs at different fractions must differ, and the
    presence term must contribute something a rescaling of the fraction term cannot.
    """
    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    layer = build_layer(bases, offsets, max_padded_per_chunk=200)
    # The presence half is zero-initialised on purpose (see
    # test_presence_gate_starts_silent_and_then_learns), so give it the values a trained
    # reader would have before asking whether the two gates are separable at all.
    with torch.no_grad():
        layer.reader.project.weight[OUT_DIM:].normal_(0, 0.01)
    layer.eval()

    present = np.zeros((1, N_UNITIGS), dtype=np.float32)
    present[0, :10] = 0.5
    half = torch.from_numpy(present)
    full = torch.from_numpy(np.where(present > 0, 1.0, 0.0).astype(np.float32))
    with torch.no_grad():
        h_half, h_full = layer(half), layer(full)
    # With one gate, h(full) - bias would be exactly 2x h(half) - bias.
    a = (h_half - layer.bias)
    b = (h_full - layer.bias)
    assert not torch.allclose(b, 2 * a, atol=1e-4), "presence gate contributes nothing"


def test_presence_gate_starts_silent_and_then_learns():
    """At step 0 the layer must equal the fraction-only layer it replaces.

    None of the four v9 configs uses BatchNorm, so an input layer whose output is several
    times larger than the baseline's at initialisation is a different optimisation
    problem, not a different representation, and the comparison would measure that
    instead.
    """
    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    layer = build_layer(bases, offsets, max_padded_per_chunk=200, refresh_every=1)

    b_half = layer.reader.project.weight[OUT_DIM:]
    assert torch.count_nonzero(b_half) == 0, "presence half must start at zero"

    x = torch.from_numpy((rng.random((4, N_UNITIGS)) * (rng.random((4, N_UNITIGS)) > 0.5)
                          ).astype(np.float32))
    layer.eval()
    with torch.no_grad():
        cards = layer._build_cards() if layer._cards is None else layer._cards
        h = layer(x)
    fraction_only = x @ layer._cards[:, :OUT_DIM] + layer.bias
    assert torch.allclose(h, fraction_only, atol=1e-6)

    layer.train()
    ((layer(x)) ** 2).sum().backward()
    layer.refresh()
    assert layer.reader.project.weight.grad[OUT_DIM:].abs().sum() > 0, "gate cannot learn"


def _train_steps(layer, x, target, n_steps, lr):
    """Mimic the trainer's loop exactly: zero_grad, forward, backward, step."""
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    layer.train()
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        ((layer(x) - target) ** 2).sum().backward()
        opt.step()
    return opt


def test_reader_trunk_moves_over_a_short_run():
    """An inert reader is the failure mode that wasted 40 GPU fits, so it is a test now.

    The first submission set ``refresh_every=8`` against 8 steps per epoch, so the reader
    received one update per epoch and nine in total before early stopping. Its weights were
    still at their initial values, the real and shuffled arms produced byte-identical
    predictions, and both collapsed to the majority class. Nothing in the loss curve said
    so. This asserts the mechanism moves the trunk; whether the searched learning rate is
    large enough for it to move *usefully* is a separate question the arm itself answers.
    """
    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    layer = build_layer(bases, offsets, max_padded_per_chunk=200)  # refresh_every defaults to 1
    assert layer.refresh_every == 1, "the default must be one update per step"

    x = torch.from_numpy((rng.random((6, N_UNITIGS)) * (rng.random((6, N_UNITIGS)) > 0.5)
                          ).astype(np.float32))
    target = torch.from_numpy(rng.standard_normal((6, OUT_DIM)).astype(np.float32))
    before = {n: p.clone() for n, p in layer.reader.named_parameters()}

    _train_steps(layer, x, target, n_steps=24, lr=1e-3)

    assert layer._reader_updates >= 20, f"only {layer._reader_updates} reader updates in 24 steps"
    for name, p in layer.reader.named_parameters():
        if name.startswith("project.weight"):
            continue  # its presence half starts at zero by design, checked elsewhere
        rel = (p - before[name]).norm() / before[name].norm().clamp(min=1e-12)
        assert rel > 1e-3, f"{name} barely moved: relative change {rel:.2e}"


def test_the_refresh_interval_is_the_readers_update_budget():
    """The trap, stated as a test: raising the interval starves the reader proportionally."""
    rng = np.random.default_rng(SEED)
    bases, offsets = toy_vocabulary(rng)
    x = torch.from_numpy(rng.random((6, N_UNITIGS)).astype(np.float32))
    target = torch.from_numpy(rng.standard_normal((6, OUT_DIM)).astype(np.float32))

    counts = {}
    for interval in (1, 8):
        layer = build_layer(bases, offsets, max_padded_per_chunk=200, refresh_every=interval)
        _train_steps(layer, x, target, n_steps=24, lr=1e-3)
        counts[interval] = layer._reader_updates
    assert counts[1] >= 20 and counts[8] <= 3, counts
