"""Figures for the kernel study.

  plot_session_arms   : ref_acc / lit_acc bars per arm, with the Project-3 frozen-Transformer
                        ref_acc=0.10 drawn as a reference line (the wall we're trying to clear).
  plot_timescales     : per-layer probe-acc vs recall delay Δ — the multi-timescale hierarchy.
  plot_arith          : accuracy vs hops (OOD) for the arith secondary run.
"""
from __future__ import annotations

from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_LABEL = {"K0": "K0: read severed (control)", "K1": "K1: fast-weight, answer-CE",
             "K2": "K2: + grounded read sup."}
ARM_COLOR = {"K0": "#b0b0b0", "K1": "#2b6cb0", "K2": "#c05621"}
PROJECT3_REF = 0.10   # frozen-Transformer ref_acc from FINDINGS_memory.md (chance)


def plot_session_arms(metrics: Dict, out_path: str):
    arms = [a for a in ("K0", "K1", "K2") if a in metrics["arms"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=140)
    x = range(len(arms)); w = 0.36
    ref = [metrics["arms"][a]["ref_acc"] for a in arms]
    lit = [metrics["arms"][a]["lit_acc"] for a in arms]
    ax.bar([i - w / 2 for i in x], ref, w, label="ref_acc (memory USE)",
           color=[ARM_COLOR[a] for a in arms])
    ax.bar([i + w / 2 for i in x], lit, w, label="lit_acc (no memory needed)",
           color=[ARM_COLOR[a] for a in arms], alpha=0.45)
    chance = metrics["arms"][arms[0]]["chance"]
    ax.axhline(chance, ls=":", c="red", alpha=0.7, label=f"chance ({chance:.2f})")
    ax.axhline(PROJECT3_REF, ls="--", c="purple", alpha=0.7,
               label=f"Proj-3 frozen-Transformer ref ({PROJECT3_REF:.2f})")
    ax.set_xticks(list(x)); ax.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=8, rotation=8)
    ax.set_ylabel("accuracy"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Can a native fast-weight kernel USE stored memory as an operand?")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_timescales(ts: Dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=140)
    cmap = plt.get_cmap("viridis")
    layers = ts["layers"]
    for i, layer in enumerate(layers):
        abd = layer["acc_by_delay"]
        if not abd:
            continue
        ds = sorted(abd); ys = [abd[d] for d in ds]
        color = cmap(i / max(1, len(layers) - 1))
        ax.plot(ds, ys, marker="o", color=color, label=f"layer {i} (γ={layer['gamma']})")
    ax.axhline(ts["chance"], ls=":", c="red", alpha=0.7, label=f"chance ({ts['chance']:.2f})")
    ax.set_xlabel("recall delay Δ (problems since the value was stored)")
    ax.set_ylabel("linear-probe accuracy for the stored value")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Multi-timescale memory: fast layers forget, slow layers hold")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_arith(metrics: Dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    for arm in metrics["arms"]:
        accs = metrics["arms"][arm].get("acc_by_hops")
        if not accs:
            continue
        hops = sorted(int(h) for h in accs); ys = [accs[str(h)] for h in hops]
        ax.plot(hops, ys, marker="o", label=arm)
    tmax = max(metrics["config"]["train_hops"])
    ax.axvline(tmax + 0.5, ls="--", c="k", alpha=0.4)
    ax.axhline(1.0 / metrics["config"]["modulus"], ls=":", c="red", alpha=0.5, label="chance")
    ax.set_xlabel("hop count"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Kernel arith: length generalization"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
