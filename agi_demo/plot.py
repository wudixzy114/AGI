"""Two figures: answer accuracy vs hops (A/B/C), and per-step probe accuracy."""
from __future__ import annotations

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARM_LABEL = {
    "A": "A: result-only (no latent)",
    "B": "B: latent thinking (result CE)",
    "C": "C: latent + process CE",
}
ARM_COLOR = {"A": "#b0b0b0", "B": "#2b6cb0", "C": "#c05621"}


def plot_acc_vs_hops(metrics: Dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    for arm in ("A", "B", "C"):
        if arm not in metrics["arms"]:
            continue
        accs = metrics["arms"][arm]["acc_by_hops"]
        hops = sorted(int(h) for h in accs)
        ys = [accs[str(h)] for h in hops]
        ax.plot(hops, ys, marker="o", label=ARM_LABEL[arm], color=ARM_COLOR[arm])
    train_max = max(metrics["config"]["train_hops"])
    ax.axvline(train_max + 0.5, ls="--", c="k", alpha=0.4)
    ax.text(train_max + 0.55, 0.05, "train | OOD ->", fontsize=8, alpha=0.6)
    ax.axhline(1.0 / metrics["config"]["modulus"], ls=":", c="red", alpha=0.5, label="chance")
    ax.set_xlabel("hop count (reasoning depth)")
    ax.set_ylabel("answer exact-match accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Does latent 'thinking' resist accuracy decay with depth?")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_probe(metrics: Dict, out_path: str, probe_hop: int):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    for arm in ("B", "C"):
        if arm not in metrics["arms"]:
            continue
        pr = metrics["arms"][arm].get("probe")
        if not pr:
            continue
        accs = pr["per_step_acc"]
        steps = list(range(1, len(accs) + 1))
        ax.plot(steps, accs, marker="s", label=ARM_LABEL[arm], color=ARM_COLOR[arm])
        chance = pr["chance"]
    ax.axhline(1.0 / metrics["config"]["modulus"], ls=":", c="red", alpha=0.6, label="chance")
    ax.set_xlabel(f"thought-vector index i  (probe target = intermediate x_i, hops={probe_hop})")
    ax.set_ylabel("linear-probe accuracy for x_i")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Do the latent thoughts linearly encode each reasoning step?")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
