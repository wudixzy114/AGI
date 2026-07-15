# Findings — Project 5 Phase A: Memory-Overwrite (delta-rule vs Hebbian)

*Status: COMPLETE — 14-config matrix on the B200 (delta/Hebbian × K0/K1/K2 at 3× overwrite, plus an
overwrite-depth sweep n_slots ∈ {2,3,4,6,12} at session_len=12, all 3-seed except K0). The headline is
a **nuanced, honest result**: delta-rule's advantage is real but appears only at the deepest overwrite
and manifests as **robustness (zero variance), not mean accuracy**.*

## The question (why this phase exists)

Project 4 closed with a **deliberate null**: delta-rule write `S ← γS + β(v−Sk)kᵀ` tied additive-Hebbian
`S ← γS + βvkᵀ` (1.000 vs 0.998), because in that task **each slot was written exactly once with a
distinct key** — the delta rule's *overwrite* advantage (re-writing the same key erases its old
association instead of superposing) was never exercised. Phase A builds the task that exercises it:
**the same slot is rewritten mid-session, and a query must return its LATEST value.** Prediction:
delta holds ref≈1.0; Hebbian degrades because writing `v_old⊗k` then `v_new⊗k` without erasure leaves
`S k ≈ v_old+v_new`, a superposition the readout should struggle to resolve.

## Setup

Pure recall (session_hops=0, no arithmetic confound), `session_len=12`, sweeping `n_slots` so each slot
is overwritten `12/n_slots` times (1×…6×). Kernel unchanged from Project 4 (d_model=128, d_state=64,
4 layers γ=[0,0.7,0.95,0.99]). The only code additions: `allow_overwrite` (relax the 1:1 slot↔problem
assertion), a `slot_truth_at_ref` label = the slot's **latest** stored value (the correct
read-supervision/probe target under overwrites), and last-write-time delay binning. Write rule toggled
by `--no-delta`. Arms K0 (read severed) / K1 (answer-CE) / K2 (+read supervision) as in Project 4.

## HEADLINE — the overwrite-depth sweep (`overwrite_sweep.png`)

Reference accuracy (memory USE, mean ± std over 3 seeds), K1 arm:

| overwrite | n_slots | **delta-rule** | **Hebbian** |
|-----------|---------|----------------|-------------|
| 1× | 12 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 2× | 6 | 1.000 ± 0.000 | 0.998 ± 0.002 |
| 3× | 4 | 1.000 ± 0.000 | 0.999 ± 0.002 |
| 4× | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **6×** | **2** | **1.000 ± 0.000** | **0.910 ± 0.153** |

**The separation is real but appears only under maximum overwrite pressure (6×), and it is a
*variance* effect, not a mean collapse.** delta-rule is unconditionally perfect and **seed-invariant
(std = 0.000 at every depth)**. Hebbian matches it everywhere except the deepest overwrite, where it
drops to 0.910 with a large ±0.153 spread.

## The mechanism, seen per-seed (`seed_spread.png`)

At 6× overwrite the three seeds tell the story:
- **delta-rule:** `[1.000, 1.000, 1.000]` — invariant.
- **Hebbian:** `[0.998, 0.733, 1.000]` — **one seed collapses to 0.733; the other two are fine.**

And that collapsed seed's recall **decays with delay** — its delay-binned accuracy is `{Δ1: 0.77,
Δ2: 0.69}` vs the healthy seeds' flat `{Δ1: 1.0, Δ2: 1.0}`. That delay-decay is the *predicted*
superposition signature: the older the last-write-to-read gap, the more stale summed values bleed
through a Hebbian slot. So the mechanism is exactly the hypothesized one — **but it only bites
stochastically**, when a Hebbian run happens to land in a regime where the superposed values aren't
linearly separable.

## Honest interpretation — why Hebbian is stronger than expected

At d_state=64 with only 10 value classes, a sum of a few one-hot-ish values `v_old+v_new+…` usually
**remains linearly separable**, so the readout recovers the latest value even from a superposition,
most of the time. The delta rule's erase term removes the superposition entirely, which is why it has
**zero variance** — it is not "more accurate on average" here, it is *robust*: it never lands in the
bad regime. So the correct claim is narrow and defensible:

> **Under repeated overwrites, the delta rule's erase term buys robustness (seed-invariant recall),
> and Hebbian's advantage-gap opens only at high overwrite multiplicity where its superposition
> occasionally fails to stay separable.** At this scale the gap is 1.000 vs 0.910±0.153 (6×).

This is a weaker separation than Phase A predicted, and we report it as such. It would sharpen by
(a) **shrinking d_state** (less room for superposition to stay separable), (b) **more value classes /
continuous values**, or (c) **deeper overwrite** (session_len ≫ n_slots). Those are the natural
follow-ups if a stronger delta-vs-Hebbian result is wanted.

## Causal control (K0) — unchanged from Project 4, confirms the read is load-bearing

`ovw_delta_K0` and `ovw_hebb_K0` both sit at **ref 0.096 ≈ chance** (read severed → memory unusable),
and are identical delta-vs-Hebbian (with no read, the write rule cannot matter). So any K1/K2 signal is
attributable to the read consuming the stored state — the write rule only governs *what is stored*.

## K2 (read supervision) — no help here
K1 and K2 are statistically identical (both ~1.0), because pure recall already grounds the read via the
answer CE; the extra read-supervision term is redundant when there's no arithmetic to distract the
model. (Consistent with Project 4's hops=0.)

## What this validates for later phases

The **`slot_truth_at_ref` "latest-writer" machinery** — the label, the read-supervision target, and the
last-write delay binning — is now built and verified correct (smoke test reconstructs it independently;
recall reaches 1.0 under 3× overwrite). Phase B (sleep consolidation) needs exactly this "current value
of a slot after arbitrary writes" bookkeeping, so Phase A also de-risks it.

## Honest limitations
- Separation is a variance effect at one depth (6×), not a broad mean gap — reported as such, not
  oversold. Sharper separation needs smaller d_state / more classes / deeper overwrite (future work).
- 3 seeds; the 0.910±0.153 is exactly the kind of seed-noisy magnitude Project 2b warned about —
  the *robustness* claim (delta std=0) is the seed-solid part; the exact Hebbian mean at 6× is not.
- Pure recall only (session_hops=0), by design (confound-free). Overwrite + arithmetic is untested.
- Infrastructure detour: the run was delayed by a scheduler/OOM saga (CPU RAM, not GPU, was the binding
  constraint on the notebook) — see `POSTMORTEM_p5a_scheduler.md`; final runs used `run_capped.py`
  (concurrency capped on `MemAvailable`).

## Files
- code: `kernel/kconfig.py` (allow_overwrite), `kernel/encode.py` (slot_truth_at_ref),
  `kernel/train.py` + `kernel/probe.py` (latest-value target + last-write delay), `kernel/smoke.py`
  (check_overwrite), `scheduler/jobs_p5a.json`, `scheduler/run_capped.py`.
- results: `agi_demo/outputs/p5a_pulled/` (14 configs + `overwrite_sweep.png`, `seed_spread.png`).

## Reproduce
```bash
python -m agi_demo.kernel.smoke     # includes check_overwrite (label correctness + delta learns)
# full matrix on the B200, CPU-RAM-capped concurrency:
python3 agi_demo/scheduler/run_capped.py agi_demo/scheduler/jobs_p5a.json \
    --max-concurrent 5 --min-free-mb 6000
```
