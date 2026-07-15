# Findings — Project 2: Scaling to 3B (Qwen2.5-3B-Instruct, B200)

> **⚠️ CORRECTION (see `FINDINGS_causal.md`).** Result #1 below ("scale reversal — at 3B plain
> latent generalizes *worse* than baseline") was based on a SINGLE seed and **does not survive
> multi-seed testing**. With 3-seed error bars, A (0.63±0.20) ≈ B (0.65±0.07) at hops-7 — the
> 0.84-vs-0.60 gap was seed noise. Treat the single-seed OOD magnitudes below as illustrative, not
> robust. The causal-ablation + probe mechanism (Result #2/#3) DID hold up and was strengthened.

Follow-up to the 0.5B study (`FINDINGS.md`), run on an internal NVIDIA B200 with the internal-hub
model `/media/cfs/9n-das-admin/llm_models/Qwen2.5-3B-Instruct` (3.09B params, frozen), bf16.
Deeper task: train hops 1-6, eval to 8 (hops 7-8 are out-of-distribution depth). 3000 steps/arm,
batch 256. Same design: frozen base + LoRA + Coconut continuous-thought loop; arms A (result-only),
B (latent + answer CE), C (latent + process CE).

## TL;DR — three headline results

1. **Scale reversal.** At 0.5B, latent thinking (B) *beat* the direct baseline (A) on OOD depth.
   At 3B it **reverses**: plain latent (B) generalizes *worse* than A (hops-7: 0.60 vs 0.84).
   A strong base solves the training range directly and, left to answer-CE alone, the latent
   channel learns a non-generalizing **shortcut** (encode only the final answer).

2. **Process supervision is what makes latent pay off at scale (过程 > 结果).** Only C
   (latent + process) beats A on OOD (hops-7: **0.90** vs 0.84). The probe gives the causal
   chain: process supervision → the whole intermediate chain stays linearly decodable →
   generalization. Without it (B), only the endpoint is decodable and OOD collapses.

3. **The universal law across all arms & both model sizes:**
   **OOD generalization ⇔ how much of the intermediate reasoning chain is linearly decodable.**
   This single relationship explains every result below.

## Master results table (verified from metrics.json)

Answer exact-match accuracy by hop count (train 1-6, **7-8 = OOD**):

| config | arm | params | 1-4 | 5 | 6 | **7** | **8** |
|--------|-----|--------|-----|---|---|-------|-------|
| curriculum | A result-only | 3.7M | ~1.00 | 1.00 | 0.99 | 0.84 | 0.36 |
| curriculum | B latent | 20.5M | ~1.00 | 1.00 | 0.98 | 0.60 | 0.15 |
| curriculum | **C latent+process** | 20.5M | ~1.00 | 1.00 | 0.99 | **0.90** | 0.34 |
| residual+uniform | B latent | 20.5M | ~1.00 | 0.99 | 0.89 | 0.51 | 0.19 |
| residual+uniform | C latent+process | 20.5M | ~1.00 | 0.99 | 0.96 | 0.44 | 0.17 |
| curriculum | A′ matched-SFT (r=32) | 14.7M | 1.00 | 0.99 | 0.98 | 0.38 | 0.03 |

Linear probe — can thought-vector i decode intermediate x_i? (hops=6, chance 0.10):

| config·arm | x1 | x2 | x3 | x4 | x5 | x6 | → hops-7 acc |
|-----------|----|----|----|----|----|----|----|
| curriculum·B | 0.17 | 0.11 | 0.09 | 0.12 | 0.14 | 0.98 | 0.60 |
| **curriculum·C** | 1.00 | 1.00 | 0.99 | 0.86 | 0.65 | 0.98 | **0.90** |
| residual·B | 0.13 | 0.12 | 0.10 | 0.10 | 0.18 | 0.95 | 0.51 |
| residual·C | 1.00 | 0.98 | 0.43 | 0.44 | 0.21 | 0.94 | 0.44 |

The correlation is unmistakable: chains that stay decodable through the middle (curriculum·C)
generalize; chains decodable only at the endpoint (all B arms) or that collapse in the middle
(residual·C) do not.

## Answers to the three questions that motivated Project 2

**Q1 — Would residual help, and why does curriculum fix the near-zero-init failure?**
Confirmed and refined. The 0.5B collapse under uniform training was caused by **recursively
composing an initially-untrained transform**: `thought_proj` inits near-zero, so the K-step latent
chain starts far from identity and gradients vanish/explode through the depth. Two independent
fixes:
- **Residual** (`t = h_last + proj(h_last)`): starts the loop at Coconut's identity → gradients
  flow from step 0. At 3B, residual+uniform learns the whole training range (hops 1-6: 0.89-1.00)
  with **no curriculum** — vindicating the "pass step-1's data through" intuition.
- **Curriculum** (shallow→deep): masters each depth before composing deeper.

BUT they are **not interchangeable**. Residual fixes *trainability*; only curriculum produces a
*clean, decodable chain*. Compare arm C: curriculum keeps x1-x5 decodable (0.65 at x5) → OOD 0.90;
residual+uniform lets the middle collapse (x3=0.43, x5=0.21) → OOD 0.44. **Curriculum > residual
for chain quality / generalization**, even though both make training succeed.

**Q2 — Same compute as plain SFT: does the "external fine-tune" actually beat it?**
Yes, and the benefit is **not just parameters**. Matched-capacity SFT (A′, LoRA r=32, 14.7M, no
latent) generalizes *worse* than both small-SFT and latent+process: hops-7 A′=0.38 vs A(r=8)=0.84
vs C=0.90. Adding LoRA rank actually **hurt OOD** (overfits the training depths). So C's advantage
comes from latent test-time computation + process supervision, not from the extra trainable params.
(Note: A′=14.7M is 4× A but not an exact match to C=20.5M; the direction is nonetheless clear.)

**Q3 — Combine Result CE + Process CE?**
They are already combined in arm C (`total = answer_CE + λ·process_CE`), and at 3B that
combination is the single best generalizer. Project 2 did not sweep λ/schedule further; the
open question (a λ that keeps C's in-dist strength while matching A's hops-8) remains.

## Interpretation vs the original thesis

- **高维解耦 (思考二): confirmed, with a caveat.** The model genuinely encodes reasoning steps in
  continuous latent vectors (probe). But a *capable* base will shortcut to encoding only the answer
  unless the process is supervised — decoupling is real but must be *induced*, not assumed.
- **过程 > 结果 (价值观一): strongly confirmed, and cleaner at 3B than 0.5B.** Result-only training
  of a strong model yields a non-generalizing shortcut; process supervision forces genuine
  step-by-step latent computation, which is what extrapolates. Direct causal chain observed.
- **变 vs 不变 (问题二): the growable module is trainable-but-fragile.** Trainability is rescued by
  curriculum OR residual; generalization quality needs curriculum. The "变" channel only surpasses
  the frozen "不变" baseline when both trained well AND process-supervised.

## Reproduce (on the B200 notebook)

```bash
cd /media/cfs/xiezongyu.1/AGI
bash agi_demo/run_3b_matrix.sh   # runs all 6 configs sequentially to agi_demo/outputs/3b/
```
Model: `/media/cfs/9n-das-admin/llm_models/Qwen2.5-3B-Instruct`. Env: /opt/conda (py3.10,
torch 2.9.1+cu129, transformers 5.13.1, peft 0.19.1). Remote has NO public internet — deps
installed via internal mirror `http://mirrors.jd.local/pypi/web/simple/`.

Results (pulled locally): `agi_demo/outputs/3b_pulled/{curriculum,q1_residual_uniform_B,
q1_residual_uniform_C,q2_matched_sft}/`.
