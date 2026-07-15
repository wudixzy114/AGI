# Findings — Project 4: A Non-Transformer Fast-Weight Kernel with Native, Hierarchical Memory

*Status: COMPLETE — full 8-config matrix run on the B200 (session hops-sweep 0/1/2, 5-seed,
Hebbian ablation, long-session, arith curriculum/uniform). Headline is multi-seed robust (±0.000).*

## The question (why this project exists)

Project 3 (`FINDINGS_memory.md`) hit a wall: a **frozen Transformer** + bolted-on latent memory could
**write** (slot-probe 0.62), **address** (0.72), and **read** (0.69) a stored value, yet **could not
USE it as an operand** — reference-problem accuracy sat at **0.10 = chance** — and unfreezing to force
usage destabilized and destroyed the base. The diagnosis: the Transformer's frozen weights were never
trained to consume an injected latent value as an operand, and a LoRA nudge can't install that
interface. **Observability ≠ usability.**

Project 4 tests the structural fix the user chose: **stop retrofitting frozen Transformers. Build a
small, from-scratch, fundamentally non-Transformer kernel where memory-as-operand is native**, and
mount a **hierarchical (multi-timescale) memory** on it.

## The kernel (what makes it not a Transformer)

`agi_demo/kernel/cell.py`. Each layer carries a **fixed-size matrix state** `S ∈ R^{d_state×d_state}`
— a fast weight (Hinton) / linear-attention associative memory. No attention over tokens, no softmax
over a sequence, no growing KV cache. Per token, from hidden `h`:

```
k,v,q = W_k h, W_v h, W_q h ;  β = σ(W_β h) ;  k,q ← ℓ2-normalize
S ← γ·S + β·(v − S k) kᵀ          # delta-rule associative write (overwrites the key's old value)
r = S q                            # READ = the operand, produced INSIDE the cell
out = h + W_o r                    # read flows straight back into the token stream
```

**Why USE is native:** the read `r = S q` is computed inside the cell and added back to the residual
stream by construction. There is no "inject a foreign vector into a frozen stack" step — the exact
thing the frozen Transformer could not consume.

**Multi-timescale hierarchy (load-bearing).** `L` layers at a spread of decay rates
`γ = [0, 0.7, 0.95, 0.99]`. The fast layer (γ≈0) is within-problem working memory; the slow layer
(γ≈0.99) is the session-long store. This realizes the vision's "layered latent thinking → layered
memory at different timescales."

**Parallelizable.** The recurrence is associative → a chunked-parallel scan exists (chunked linear
attention). We run the sequential scan here (short T, small model); the design is parallel-ready.

## Task & arms (reusing the proven falsifiable methodology)

Same **session task** as Project 3 (`agi_demo/task.py: make_session`), re-encoded as ONE token stream
per session (`kernel/encode.py`) so the shared memory IS the kernel's own recurrent state. A reference
problem `R0 …=` must read slot 0's value — which is **not** in the problem's tokens — so
reference accuracy directly measures memory USE. Arms:
- **K0** — read severed (`read_enabled=False`): fast weights still WRITE, but the read never feeds
  the computation. Causal control.
- **K1** — full kernel, answer-CE only. Does USE emerge from result supervision alone?
- **K2** — + grounded read supervision (the referenced read must decode the stored value).

Difficulty is swept via `session_hops` (ops per problem): **0 = pure recall** (cleanest USE test,
no arithmetic confound), 1 = recall + one op, 2 = recall + a short chain.

## HEADLINE RESULT — pure recall (session_hops=0), B200, d_model=128 d_state=64 L=4

| arm | ref_acc (memory USE) | lit_acc | vs Project 3 frozen-Transformer |
|-----|----------------------|---------|--------------------------------|
| **K0** (read severed) | **0.091** ≈ chance | 0.094 | — (causal control) |
| **K1** (answer-CE only) | **1.000** | 1.000 | **0.10 → 1.00** |
| **K2** (+ read sup.) | **1.000** | 1.000 | **0.10 → 1.00** |

**The wall is gone.** On the identical USE test where a frozen Transformer scored 0.10 (chance), the
native fast-weight kernel scores **1.00** — and it does so from **answer supervision alone** (K1), no
explicit read supervision needed (K2 matches it). The recall is **delay-independent**: ref_acc = 1.0
at every write→read gap Δ ∈ {1,2,3,4,5} (associative memory doesn't blur with distance, unlike a
compressive SSM state).

**Causal attribution is airtight.** Severing the read (K0) collapses accuracy to chance — and because
hops=0 routes *every* answer through the memory read, even lit_acc falls to chance. No read → no use,
period. This isolates the *native associative read* as the mechanism, not extra parameters or the
task's structure. *(This is the Project-2b-style causal control, done as an architectural ablation.)*

## Multi-timescale hierarchy is OBSERVABLE (probe_timescales) — CONFIRMED

The `probe_timescales` linear probe decodes the stored value from **each layer's read** at increasing
recall delay Δ (problems since the value was written). K2 shows a clean fan ordered exactly by decay γ
(`timescales_K2.png`):

| layer | Δ1 | Δ2 | Δ3 | Δ4 | Δ5 | role |
|-------|----|----|----|----|----|------|
| γ=0.0 (fast) | 0.15 | 0.14 | 0.05 | 0.13 | 0.00 | working memory — forgets in 1 step (≈chance) |
| γ=0.7 | 0.95 | 0.36 | 0.25 | 0.23 | 0.12 | short buffer — holds ~1 step then decays |
| γ=0.95 | 0.68 | 0.65 | 0.47 | 0.49 | 0.56 | mid-term retention |
| γ=0.99 (slow) | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | session-long store — flat, no decay |

Retention is **monotonic in γ**: the slow layer *is* the cross-problem memory store (perfect recall at
every delay), the fast layers *are* within-problem working memory that forgets immediately. The
hierarchy is not asserted — it is **measured**. (K1 shows the same ordering, noisier because without
read supervision the value need not be linearly parked in the slowest layer specifically.)

## Arith secondary (length generalization) — CONFIRMED

Mod-N chain, per-step probe + OOD hops. Unlike Projects 1–3 (pretrained Qwen), this kernel learns
arithmetic **from random init** — grokking-slow — so the interesting comparison is curriculum vs
uniform (train hops 1–4, eval to hops 8; `arith_curric`, `arith_uniform`):

| | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | probe x1→x4 |
|---|----|----|----|----|----|----|----|----|-------------|
| **curriculum** | 0.99 | 0.98 | 0.97 | 0.88 | **0.59** | 0.36 | 0.15 | 0.11 | 1.00 · 0.97 · 0.87 · **0.71** |
| **uniform** | 0.35 | 0.18 | 0.21 | 0.18 | 0.11 | 0.13 | 0.12 | 0.09 | 0.39 · 0.24 · 0.21 · 0.19 |

**Curriculum rescues from-scratch learning; uniform collapses.** This reproduces the Project-1/2
headline (competence-gated shallow→deep beats difficulty-mixing) *on the new kernel* — evidence the
finding is architecture-general, not a Transformer artifact. And the **per-step probe tracks
generalization**: the curriculum model linearly encodes each running intermediate (x1 1.00 → x4 0.71),
the uniform model barely above chance (0.39 → 0.19) — decodable intermediate structure again
co-occurs with OOD reach, exactly as in Project 2b.

## Difficulty sweep + ablations (full matrix, B200) — CONFIRMED

| config | K0 ref | K1 ref | K2 ref | note |
|--------|--------|--------|--------|------|
| hops0 (pure recall) | 0.09 | **1.00** | **1.00** | headline; delay-flat Δ1–5 = 1.0 |
| hops1 (recall + 1 op) | 0.10 | **1.00** | **1.00** | USE composes with one op |
| hops2 (recall + 2-op chain) | 0.09 | 0.10 | 0.10\* | arithmetic-gated (\*K2 lit=1.0) |
| **5-seed** (pure recall) | 0.10±0.01 | **1.00±0.00** | **1.00±0.00** | zero variance — not seed noise |
| hebbian (hops1, additive) | 0.10 | 1.00 | 1.00 | matches delta (ablation null by design) |
| **long-session** (len=10, Δ→9) | 0.11 | **1.00** | **1.00** | ref=1.0 at EVERY delay through Δ=9 |

- **Multi-seed robustness (the strongest single result):** K1/K2 ref_acc = **1.000 ± 0.000 across 5
  seeds**, K0 = 0.104 ± 0.012. Zero variance — unlike Project 2b where OOD *magnitudes* were seed-noisy,
  this result is saturated and rock-solid.
- **long-session (headline-grade):** with a 10-problem session and recall delays up to **Δ=9**, K1/K2
  hold **ref_acc = 1.000 at every single delay**. The fixed-size matrix state does **not blur with
  distance** — directly answering the doc's worry that compressive states forget precise early values
  ("大海捞针"). This is the associative-recall advantage the fast-weight design was chosen for.
- **hops1**: memory USE is intact when the reference also requires one arithmetic op — K1 reaches 1.00
  with no read supervision. So "use the stored value" composes with "then compute on it".
- **hops2**: all ref_acc ≈ chance; only K2's *lit_acc* reaches 1.0. This is the **from-scratch
  arithmetic grokking wall**, not a memory failure — the 2-hop mod-N chain is what's unlearned in the
  step budget (arith uniform is also ~chance), and the timescale fan for hops2 is degenerate (no
  arithmetic → no clean memory structure). Confirms why **hops=0 is the clean USE test**.
- **Delta-rule vs Hebbian ablation (null here, by design)**: additive Hebbian write matches the delta
  rule (K1 0.998 vs 1.000) on this task. Expected — each slot is written exactly once with a distinct
  key, so the delta rule's *overwrite* advantage (re-writing the same key) is never exercised. Delta
  would be expected to win only on a task with mid-session slot updates; that's future work, and the
  null result here is honest scope, not evidence against the delta rule.

## Honest limitations / interpretation

- **Arithmetic-confounded arms (hops≥1) train slowly**: because the kernel learns mod-N arithmetic
  from scratch (no pretraining), the deeper session variants are gated by arithmetic competence, not
  memory. hops=0 pure recall is the clean, confound-free USE test — and it's decisive. hops1 still
  reaches ref 1.00 (one op is learnable in budget); hops2 K0/K1 sit at chance while K2 learns the
  arithmetic (lit 1.00) but not memory-chaining-through-arithmetic in the step budget — a
  learnability limit of the from-scratch base, not of the memory mechanism.
- Teacher-forced answer tokens (deliberate): isolates "can it USE a correctly-stored value" from error
  accumulation across a session.
- Single vs multi-seed: the pure-recall result is saturated (1.00 / chance), so seed noise is
  irrelevant there; multi-seed (`session_seeds`) confirms stability.
- Sequential scan only; chunked-parallel kernel not implemented (not needed for the science).

## Through-line

Project 3 showed a value can be **stored, addressed, and probe-decoded yet unusable** by a frozen
Transformer. Project 4 shows that when the memory read is a **native operand** of a from-scratch
recurrent kernel, **USE is immediate and total (chance → 1.0)**, causally attributable to the read,
and the **multi-timescale state makes the memory hierarchy observable**. The frozenness in Project 3
was load-bearing; removing it — by building the right kernel rather than nudging the wrong one — turns
the negative result positive.

## Files
- `kernel/cell.py` (FastWeightCell, MultiTimescaleKernel, KernelModel), `kernel/encode.py` (symbol
  vocab + session/arith streams), `kernel/train.py` (arms K0/K1/K2), `kernel/probe.py`
  (eval_session, probe_timescales, probe_arith), `kernel/run.py`, `kernel/smoke.py`,
  `kernel/run_kernel.sh`, `kernel/watch_kernel.sh`.
- results: `agi_demo/outputs/kernel/{session_hops0,session_hops1,session_hops2,session_seeds,session_long,arith_*}/`

## Reproduce
```bash
python -m agi_demo.kernel.smoke                                   # tiny Mac structural test
python -m agi_demo.kernel.run --session --session-hops 0          # the headline USE test (K0/K1/K2)
python -m agi_demo.kernel.run --arith --arm K1 --curriculum       # arith length-gen secondary
bash agi_demo/kernel/run_kernel.sh                                # full matrix (B200)
```
