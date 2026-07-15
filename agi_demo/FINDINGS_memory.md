# Findings — Project 3: Short-Term Latent Memory (addressable slots on a frozen base)

Frozen Qwen2.5-3B + LoRA + Coconut latent loop, **plus** an attention-addressed latent memory
(`LatentMemory`): M slots, parallel attention read, gated write. Task: **sessions of linked
problems** — literal problems (`3 +4 =`) write their answer to a slot; reference problems
(`R0 +2 =`) must READ slot 0's value and compute from it (the value is not in the prompt text, so a
correct reference answer *requires* using memory). Every slot's ground-truth value is known, so we
can supervise and probe it. Arms: **M0** no memory, **M1** memory + answer CE, **M2** memory +
write-supervision + read-address-supervision. Read fed to the base two ways: raw hidden inject
(every latent step) and **decode-and-re-embed** (read → digit distribution → frozen token embedding).

## TL;DR — a clean negative result (and why it was expected)

**Short-term latent memory is writable, addressable, and probe-readable — but the frozen base
cannot USE it, and unfreezing to fix that destroys the base instead.**

Diagnosed link by link (all measured on 3B after training):

| link | metric | result | chance | verdict |
|------|--------|--------|--------|---------|
| **write** — is the value in the slot? | slot-probe | **0.62** | 0.10 | ✅ stored & observable |
| **address** — attend to the right slot? | argmax attn == ref_slot | **0.72** | 0.125 | ✅ addressing works |
| **read** — does the read vector carry the value? | read-probe | **0.69** | 0.10 | ✅ value is read out |
| **use** — answer the reference problem? | ref_acc | **0.10** | 0.10 | ❌ **cannot use it** |

The first three links all work; the value is written to the correct slot, retrieved, and linearly
decodable by an external probe. Yet the model answers reference problems at chance. **Observability
≠ usability**: a value can be stored, addressed, and probe-decoded and still be unusable by the same
frozen model end-to-end.

## Attribution: does unfreezing the base rescue usage? No.

Unfroze the last N base transformer blocks (N ∈ {0,2,4,6}), M2, lr_unfrozen=2e-5:

| unfreeze N | ref_acc | lit_acc | slot-probe | note |
|-----------|---------|---------|-----------|------|
| 0 (frozen) | 0.09 | 1.00 | 0.62 | clean baseline: memory unused |
| 2 | 0.09 | 0.09 | 0.09 | **diverged** (loss 1.6→3.7 at step ~300); ref flat ~0.12 even in the stable window |
| 4 | ~0.12 | 0.62 (mid) | — | ref still chance; unstable |
| 6 | ~0.09 | 0.44 (mid) | — | ref still chance; unstable |

Two facts: (a) **no unfreeze setting lifts ref_acc above chance** — even in unfreeze_2's stable
phase (steps 50-250, lit=1.0) ref stayed ~0.12; (b) unfreezing **destabilizes training** (mid-run
loss blow-up) and, when it collapses, takes the write/probe down with it (slot-probe 0.62→0.09).

## Why this is expected (not a bug — a structural result)

On a small, frozen pretrained Transformer, bolting on a memory and then unfreezing a few layers to
teach it to *use* that memory is effectively **a larger, cruder fine-tune**: it damages the base's
pretrained competence (lit_acc collapses) while also sacrificing the RNN-like stability of the
external memory module — worst of both worlds, so divergence and non-usage are the expected outcome.
The Transformer's "frozenness" is structural: it was never trained to consume an injected
latent-space value as an operand, and a LoRA-scale nudge (or a destabilizing partial unfreeze)
cannot install that new interface.

## Implication for the research vision

This precisely confirms the crux flagged in the vision (`memory/agi-research-vision.md`): the split
between **② short-term memory** and the frozen long-term base is not free — **a value being
observable in memory does not make it usable by the model**. Retrofitting dynamic memory onto a
frozen pretrained Transformer hits a wall. The forward path is not "unfreeze more" but to **build a
small base that natively supports hierarchical / dynamic memory** (memory-as-operand designed in
from the start), rather than bolting it onto an architecture whose frozenness is load-bearing.

## What DID work (kept for reuse)
- Session task with exact per-slot ground truth (`task.py: make_session`).
- Parallel attention-addressed `LatentMemory` (read/write, near-identity write init) — the
  addressing and write both learn cleanly (0.72 / 0.62 probes).
- The link-by-link diagnostic method (`diag_read.py`) that localized the failure to the "use" step
  — this is the reusable methodology.

## Files
- `memory.py`, `task.py` (make_session), `model.py` (forward_problem + decode-re-embed),
  `train.py` (session_loss + address/write supervision), `probe.py` (eval_session, probe_memory),
  `diag_read.py` (the link-by-link diagnostic), `run_unfreeze_sweep.sh`
- results: `outputs/3b_unfreeze_pulled/unfreeze_{0,2,4,6}/`

## Reproduce
```bash
# session memory arms:
python -m agi_demo.run --session --arm M2 --local-model-dir <3B> --dtype bfloat16 --steps 600
# link-by-link diagnostic (address vs read vs use):
python -m agi_demo.diag_read
# unfreeze attribution sweep:
bash agi_demo/run_unfreeze_sweep.sh
```

## Honest limitations
- Single seed per unfreeze setting; unfreeze_4/6 stopped mid-run (chance-level already clear).
- Only one read interface family tried (hidden-inject + decode-re-embed); a cross-attention-into-
  every-layer read might differ, but that approaches re-architecting the base.
- Divergence might be tamable with heavier regularization, but the in-stable-window ref≈chance
  already indicates unfreezing wouldn't rescue *usage* even if made stable.
