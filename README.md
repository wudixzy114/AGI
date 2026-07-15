# AGI — Latent Reasoning, Self-Evolution, and Hierarchical Memory

A falsifiable research program probing an alternative to "freeze the weights after training."
The thesis (see [`docs/thesis_memory-and-dimensionality_20260713.md`](docs/thesis_memory-and-dimensionality_20260713.md))
is a single system in three parts:

1. **Latent-space deep reasoning** — think in continuous space, not by decoding tokens (Coconut-style).
2. **"留白" + iterative self-evolution** — the model keeps changing after training; a stability–plasticity split (frozen base + plastic modules).
3. **Short-term memory via hierarchical latent** — layered thinking → layered memory at different timescales.

Unifying frame: a **memory hierarchy with consolidation** — working memory/compute → short-term memory → consolidation (short→long) → frozen base as long-term memory.

Every experiment uses synthetic mod-N multi-hop arithmetic where **every intermediate step has exact ground truth**, so a linear probe can measure whether the latent chain stays decodable. That observability is the backbone of every result below.

---

## The research program (Project 1 → 5)

| # | Question | Verdict | Writeup |
|---|----------|---------|---------|
| **1** | Can a frozen 0.5B + LoRA reason in latent space? | Latent arms collapse under uniform difficulty but **beat baseline OOD under a competence-gated curriculum**. The "变" module is fragile to train, not weak. | [`agi_demo/FINDINGS.md`](agi_demo/FINDINGS.md) |
| **2** | Does it hold at 3B scale? | Process supervision is what makes latent pay off; **OOD generalization ⇔ intermediate-chain decodability**. | [`agi_demo/FINDINGS_3B.md`](agi_demo/FINDINGS_3B.md) |
| **2b** | Causal ablation + robustness | Varying *only* which latent steps get process supervision moves OOD **monotonically** (none < deep < shallow < full), and the probe mirrors it — supervision *causes* decodability. Cross-task (perm) holds; magnitudes are seed-noisy. | [`agi_demo/FINDINGS_causal.md`](agi_demo/FINDINGS_causal.md) |
| **3** | Bolt external memory onto the frozen Transformer? | **Clean negative:** write ✓, address ✓, read ✓, but **USE = chance**. A frozen Transformer was never trained to consume an injected latent as an operand; partial unfreezing destabilizes it. Observability ≠ usability. | [`agi_demo/FINDINGS_memory.md`](agi_demo/FINDINGS_memory.md) |
| **4** | Build a kernel where memory-as-operand is *native*? | A from-scratch **fast-weight kernel** (matrix state `S`, delta-rule write, in-cell read) turns Project 3's `USE=0.10` into **`ref_acc=1.00`**, with a **measured** timescale hierarchy ordered by decay γ. | [`agi_demo/FINDINGS_kernel.md`](agi_demo/FINDINGS_kernel.md) |
| **5** | Consolidation loop on a more realistic task | **In progress** on the B200 (overwrite task → sleep consolidation → bAbI-style entity tracking → moderate scale-up). | — |

---

## Layout

```
AGI/
├── agi_demo/                  # the single Python package for the whole program
│   ├── config.py model.py task.py train.py run.py probe.py plot.py
│   │                          #   Projects 1–2b: frozen Qwen + LoRA + Coconut loop
│   ├── memory.py diag_read.py #   Project 3: addressable latent memory slots
│   ├── kernel/                #   Project 4: from-scratch non-Transformer fast-weight kernel
│   ├── dashboard/             # data-driven live training dashboard (stdlib + vendored Chart.js)
│   ├── scheduler/             # resource-aware GPU job scheduler (packs a matrix by measured VRAM)
│   ├── outputs/               # committed results: metrics.json, logs, figures (no checkpoints)
│   └── FINDINGS*.md           # one writeup per project
├── docs/
│   ├── thesis_memory-and-dimensionality_20260713.md   # the originating thesis
│   └── architecture.html                              # architecture overview
├── requirements.txt
├── archives/                  # per-project result bundles (git-ignored; see below)
└── models/  .venv/            # local weights + venv (git-ignored, ~2.3 GB)
```

## Running

Everything is invoked as a module from the repo root (import paths depend on the `agi_demo` package name):

```bash
# Projects 1–2b (Qwen latent reasoning) — always use --curriculum for latent arms
python -m agi_demo.run --curriculum --out-dir agi_demo/outputs/<name>

# Project 4 (fast-weight kernel) — arms K0/K1/K2, --session / --arith
python -m agi_demo.kernel.run --session --out-dir agi_demo/outputs/<name>

# Live dashboard (watches a remote or local run dir; renders to /tmp, opens a browser)
bash agi_demo/dashboard/dashboard.sh <host> <run_dir> <interval_s>

# Resource-aware scheduler (edit the jobs_*.json matrix, not the code)
bash agi_demo/scheduler/submit.sh <host> agi_demo/scheduler/jobs_p5a.json <max_concurrent>
```

## Remote environment (九数 / 9N)

Training runs on a B200 notebook (`ea-ssh` host `ea-main2`): 183 GB, **no public internet**, model
hub at `/media/cfs/9n-das-admin/llm_models/`, pip via `http://mirrors.jd.local/pypi/web/simple/`,
code + results synced under `/media/cfs/xiezongyu.1/AGI/`. The dashboard and scheduler SSH into this
host; local scripts only pull state.

## Archives

`archives/` holds per-project result bundles named `agi_<proj>_<topic>_<YYYYMMDD>.zip`
(`agi_proj1-2b_latent-reasoning_*`, `agi_proj3_memory_*`, `agi_proj4_kernel_*`). They are a
convenience for sharing and are **git-ignored** — the results themselves are tracked under
`agi_demo/outputs/`, so committing the zips too would only duplicate them.
