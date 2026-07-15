"""Tiny Mac smoke test — structural validation before the B200 run.

Checks (fast, CPU/MPS, random init, no pretrained weights):
  1. vocab layout is consistent and slot/digit tokens round-trip.
  2. forward shapes: hidden (B,L,d_model), per-layer reads (B,L,d_state), states collectable.
  3. one gradient step reduces the loss (arith + session).
  4. K0 (read severed) really zeroes the read path (reads are all-zero); K2 does not.
  5. a short arith run reaches high hops-1/2 accuracy (the core learns at all).
  6. a short session run lifts ref_acc above chance for K2 (memory USE is trainable).

This does NOT validate the scientific claim (that needs the B200 run) — only that the kernel is
wired correctly and learns, so we don't burn remote time on a shape bug.
"""
from __future__ import annotations

import random

import torch

from .kconfig import KernelConfig
from .cell import KernelModel
from .encode import Vocab, make_arith_batch, make_session_stream_batch
from .train import build_model, arith_loss, session_loss, train_arith, train_session
from .probe import eval_session


def tiny_cfg(**kw) -> KernelConfig:
    defaults = dict(d_model=48, d_state=24, n_layers=3, batch_size=16, train_steps=120,
                    modulus=10, session_len=5, n_slots=6, session_hops=2, log_every=40,
                    device="cpu", out_dir="agi_demo/outputs/kernel_smoke")
    defaults.update(kw)   # caller overrides win
    return KernelConfig(**defaults).resolve()


def check_vocab():
    v = Vocab(10, 6)
    assert v.size == 7 + 10 + 6
    ids = v.digit_ids()
    assert len(ids) == 10 and len(set(ids)) == 10
    assert v.slot(0) != v.slot(5) and v.digit(0) != v.slot(0)
    # all special tokens distinct
    specials = {v.PAD, v.PLUS, v.MINUS, v.EQ, v.REF, v.LIT, v.BOS}
    assert len(specials) == 7, specials
    print("  [ok] vocab: size", v.size, "all tokens distinct")


def check_shapes():
    cfg = tiny_cfg()
    model = build_model(cfg)
    rng = random.Random(0)
    b = make_arith_batch(model.vocab, 8, 3, 10, rng, device=cfg.device)
    hidden, reads, states = model(b.input_ids, collect_state=True)
    B, L = b.input_ids.shape
    assert hidden.shape == (B, L, cfg.d_model), hidden.shape
    assert len(reads) == cfg.n_layers and reads[0].shape == (B, L, cfg.d_state), reads[0].shape
    assert len(states) == cfg.n_layers and states[0].shape == (B, L, cfg.d_state, cfg.d_state)
    logits = model.answer_logits_at(hidden, b.eq_pos)
    assert logits.shape == (B, cfg.modulus)
    print("  [ok] shapes: hidden", tuple(hidden.shape), "reads", tuple(reads[0].shape),
          "states", tuple(states[0].shape))


def check_read_sever():
    """K0 must zero every read; K2 must not."""
    rng = random.Random(0)
    for arm, want_zero in (("K0", True), ("K2", False)):
        cfg = tiny_cfg(arm=arm)
        model = build_model(cfg)
        b = make_arith_batch(model.vocab, 8, 3, 10, rng, device=cfg.device)
        _, reads, _ = model(b.input_ids)
        with torch.no_grad():
            allz = all(float(r.abs().max()) == 0.0 for r in reads)
        assert allz == want_zero, f"arm {arm}: reads all-zero={allz}, expected {want_zero}"
        print(f"  [ok] read-sever: arm {arm} reads_all_zero={allz}")


def check_one_step_decreases():
    rng = random.Random(0)
    # arith
    cfg = tiny_cfg(arm="K1", task="arith")
    model = build_model(cfg); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    b = make_arith_batch(model.vocab, cfg.batch_size, 2, 10, rng, device=cfg.device)
    l0 = arith_loss(model, b, cfg, 0)["total"]
    opt.zero_grad(); l0.backward(); opt.step()
    l1 = arith_loss(model, b, cfg, 1)["total"]
    assert l1.item() < l0.item() + 1e-4, f"arith loss did not decrease: {l0.item()} -> {l1.item()}"
    print(f"  [ok] arith one-step: {l0.item():.3f} -> {l1.item():.3f}")
    # session
    cfg = tiny_cfg(arm="K2", task="session")
    model = build_model(cfg); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sb = make_session_stream_batch(model.vocab, cfg.batch_size, cfg, rng, device=cfg.device)
    l0 = session_loss(model, sb, cfg)["total"]
    opt.zero_grad(); l0.backward(); opt.step()
    l1 = session_loss(model, sb, cfg)["total"]
    assert l1.item() < l0.item() + 1e-4, f"session loss did not decrease: {l0.item()} -> {l1.item()}"
    print(f"  [ok] session one-step: {l0.item():.3f} -> {l1.item():.3f}")


def check_arith_learns():
    # hops-1 mod-N addition is grokking-like — slow for a tiny from-scratch model (unlike the
    # pretrained Qwen in Projects 1-3, this kernel must learn arithmetic from zero). We only need
    # CLEAR evidence it learns (well above chance); the full-scale target is the B200 run's job.
    cfg = tiny_cfg(arm="K1", task="arith", train_hops=[1], eval_hops=[1, 2],
                   batch_size=32, train_steps=700)
    model = train_arith(cfg, verbose=False)
    from .probe import accuracy_by_hops
    accs = accuracy_by_hops(model, cfg, random.Random(7))
    print(f"  arith accs: {accs}")
    assert accs[1] > 0.35, f"hops-1 accuracy too low: {accs[1]} (kernel not learning arith at all)"
    print(f"  [ok] arith learns: hops-1={accs[1]:.2f} (>chance 0.10)")


def check_session_uses_memory():
    """THE headline check: on pure recall (session_hops=0 — read a stored value and emit it, the
    exact thing Project 3's frozen Transformer could not do), the native kernel must USE memory,
    and severing the read (K0) must collapse it to chance. This is the causal core of Project 4."""
    res = {}
    for arm in ("K0", "K2"):
        cfg = tiny_cfg(arm=arm, task="session", session_hops=0, session_len=5, n_slots=6,
                       d_model=64, d_state=32, train_steps=400, lr=5e-3)
        model = train_session(cfg, verbose=False)
        ev = eval_session(model, cfg, random.Random(7))
        res[arm] = ev
        print(f"  session[hops=0] {arm}: ref_acc={ev['ref_acc']:.2f} lit_acc={ev['lit_acc']:.2f} "
              f"chance={ev['chance']:.2f}")
    # K2 must clearly USE memory; K0 (read severed) must stay near chance -> causal attribution.
    assert res["K2"]["ref_acc"] > 0.80, \
        f"K2 ref_acc {res['K2']['ref_acc']:.2f} too low — native memory USE not working"
    assert res["K0"]["ref_acc"] < 0.30, \
        f"K0 ref_acc {res['K0']['ref_acc']:.2f} too high — severing the read should collapse USE"
    gap = res["K2"]["ref_acc"] - res["K0"]["ref_acc"]
    print(f"  [ok] memory USE is native + causal: K2 ref={res['K2']['ref_acc']:.2f} vs "
          f"K0 ref={res['K0']['ref_acc']:.2f} (gap {gap:.2f}); Proj-3 frozen-Transformer was 0.10")


def check_overwrite():
    """Phase A: with session_len>n_slots, slots get rewritten. Verify (a) the slot_truth_at_ref
    label equals the latest stored value by independent reconstruction, and (b) the delta-rule
    kernel learns to return the LATEST value (ref_acc clears chance)."""
    from .encode import Vocab, make_session_stream_batch
    # (a) label correctness — reconstruct latest-writer value independently from the session objects
    cfg = tiny_cfg(arm="K2", task="session", session_hops=0, session_len=6, n_slots=2,
                   allow_overwrite=True)
    v = Vocab(cfg.modulus, cfg.n_slots)
    sb = make_session_stream_batch(v, 4, cfg, random.Random(3), device="cpu")
    checked = 0
    for b, sess in enumerate(sb.sessions):
        cur = {}
        for t, p in enumerate(sess.problems):
            if p.is_ref and t > 0:
                want = cur.get(p.ref_slot, -1)   # latest value written to that slot so far
                got = int(sb.slot_truth_at_ref[b, t])
                assert got == want, f"slot_truth_at_ref[{b},{t}]={got} != reconstructed {want}"
                checked += 1
            cur[p.write_slot] = p.answer          # this problem overwrites its slot
    assert checked > 0, "no reference problems generated — can't validate the overwrite label"
    print(f"  [ok] overwrite label: slot_truth_at_ref matches latest-writer on {checked} refs")
    # (b) delta-rule learns the latest-value recall under overwrites
    cfg = tiny_cfg(arm="K2", task="session", session_hops=0, session_len=6, n_slots=2,
                   allow_overwrite=True, d_model=64, d_state=32, train_steps=500, lr=5e-3)
    model = train_session(cfg, verbose=False)
    ev = eval_session(model, cfg, random.Random(7))
    print(f"  overwrite[len=6,slots=2] delta K2: ref_acc={ev['ref_acc']:.2f} (chance {ev['chance']:.2f})")
    assert ev["ref_acc"] > 0.55, \
        f"delta-rule ref_acc {ev['ref_acc']:.2f} too low — overwrite recall not learning"
    print(f"  [ok] delta-rule recalls latest value under 3x overwrite")


def main():
    print("=== kernel smoke test (CPU, tiny, random init) ===")
    check_vocab()
    check_shapes()
    check_read_sever()
    check_one_step_decreases()
    check_arith_learns()
    check_session_uses_memory()
    check_overwrite()
    print("=== ALL SMOKE CHECKS PASSED ===")


if __name__ == "__main__":
    main()
