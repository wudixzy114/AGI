"""Figures for the causal-ablation + robustness study (Project 2b, 3B).

fig_causal_ablation : OOD accuracy vs which latent steps are process-supervised
                      (none < deep-half < shallow-half < all)  — the causal core.
fig_causal_probe    : per-step probe for the 4 ablation configs — decodability follows
                      exactly which steps were supervised (mirror pattern).
fig_seed_errorbars  : 3-seed mean±std for A/B/C at each hop — shows A≈B (earlier single-seed
                      'reversal' was noise) and C highest but within ~1σ.
fig_perm_crosstask  : perm (lookup) task — A,B collapse to chance at OOD, C holds 3x chance.
"""
import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "outputs", "3b_causal_pulled")
OUT = os.path.join(HERE, "outputs", "figures_causal")
os.makedirs(OUT, exist_ok=True)


def acc_line_from_log(path):
    """Parse the '[arm X] acc_by_hops: 1:.. 2:..' line from a run.log."""
    import re
    txt = open(path).read()
    m = re.findall(r"acc_by_hops:\s*([0-9:.,\s]+)", txt)
    if not m:
        return {}
    d = {}
    for tok in m[-1].replace(",", " ").split():
        if ":" in tok:
            h, a = tok.split(":"); d[int(h)] = float(a)
    return d


def load(cfg):
    mj = os.path.join(P, cfg, "metrics.json")
    if os.path.exists(mj):
        d = json.load(open(mj)); arm = list(d["arms"])[0]; return d["arms"][arm]
    # fallback: parse run.log (causal_none had no metrics.json)
    lg = os.path.join(P, cfg, "run.log")
    return {"acc_by_hops": {str(k): v for k, v in acc_line_from_log(lg).items()}, "probe": None}


def hop_acc(v, h):
    return v["acc_by_hops"].get(str(h))


# ---------- fig 1: causal ablation (OOD hops-7 bar) ----------
abl = [
    ("none\n(=arm B)", "causal_none", "#8a8a8a"),
    ("deep half\n(x4-x6)", "causal_second_half", "#dd8452"),
    ("shallow half\n(x1-x3)", "causal_first_half", "#c05621"),
    ("full chain\n(x1-x6)", "causal_all", "#7b341e"),
]
fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
vals7 = [hop_acc(load(c), 7) for _, c, _ in abl]
vals8 = [hop_acc(load(c), 8) for _, c, _ in abl]
xs = range(len(abl)); w = 0.38
ax.bar([i - w/2 for i in xs], vals7, w, color=[c for *_, c in abl], label="hops-7 (OOD +1)")
ax.bar([i + w/2 for i in xs], vals8, w, color=[c for *_, c in abl], alpha=0.5, label="hops-8 (OOD +2)")
for i, v in enumerate(vals7): ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(vals8): ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, alpha=0.7)
ax.axhline(0.1, ls=":", c="red", alpha=0.6, label="chance")
ax.set_xticks(list(xs)); ax.set_xticklabels([n for n, *_ in abl], fontsize=9)
ax.set_ylabel("out-of-distribution answer accuracy"); ax.set_ylim(0, 1.0)
ax.set_xlabel("which latent steps receive process supervision")
ax.set_title("Causal ablation: OOD generalization rises monotonically\nwith how much of the reasoning chain is supervised")
ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_causal_ablation.png")); plt.close(fig)

# ---------- fig 2: causal probe mirror ----------
fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
probe_cfgs = [
    ("supervise x1-x3 (shallow)", "causal_first_half", "#c05621"),
    ("supervise x4-x6 (deep)", "causal_second_half", "#2b6cb0"),
    ("supervise all", "causal_all", "#7b341e"),
]
for lab, c, col in probe_cfgs:
    pr = load(c).get("probe")
    if pr:
        ys = pr["per_step_acc"]; ax.plot(range(1, len(ys) + 1), ys, marker="s", color=col, label=lab)
ax.axhline(0.1, ls=":", c="red", alpha=0.6, label="chance")
ax.set_xlabel("thought-vector index i  (probe target = intermediate x_i, hops=6)")
ax.set_ylabel("linear-probe accuracy for x_i"); ax.set_ylim(-0.02, 1.05)
ax.set_title("Probe mirrors supervision: exactly the supervised steps become decodable\n(supervision CAUSES decodability, which drives generalization)")
ax.legend(fontsize=8); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_causal_probe.png")); plt.close(fig)

# ---------- fig 3: multi-seed error bars A/B/C ----------
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
seed_cfgs = [("A: result-only", "seeds_A", "#8a8a8a"),
             ("B: latent", "seeds_B", "#2b6cb0"),
             ("C: latent+process", "seeds_C", "#c05621")]
for lab, c, col in seed_cfgs:
    v = load(c); a = v["acc_by_hops"]; sd = v.get("acc_by_hops_std", {})
    hs = sorted(int(h) for h in a)
    ys = [a[str(h)] for h in hs]; es = [sd.get(str(h), 0) for h in hs]
    ax.errorbar(hs, ys, yerr=es, marker="o", color=col, label=lab, capsize=3, elinewidth=1.2)
ax.axvline(6.5, ls="--", c="k", alpha=0.4); ax.axhline(0.1, ls=":", c="red", alpha=0.5)
ax.annotate("train | OOD ->", (6.55, 0.03), fontsize=8, alpha=0.6)
ax.set_xlabel("hop count (reasoning depth)"); ax.set_ylabel("answer accuracy (3-seed mean ± std)")
ax.set_ylim(-0.02, 1.05)
ax.set_title("With error bars: A≈B at OOD (single-seed 'reversal' was noise);\nC highest but overlaps within ~1σ — magnitudes need more seeds")
ax.legend(fontsize=9, loc="lower left"); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_seed_errorbars.png")); plt.close(fig)

# ---------- fig 4: perm cross-task ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
perm = [("A: result-only", "perm_A", "#8a8a8a"), ("B: latent", "perm_B", "#2b6cb0"),
        ("C: latent+process", "perm_C", "#c05621")]
for lab, c, col in perm:
    v = load(c); a = v["acc_by_hops"]; hs = sorted(int(h) for h in a)
    axes[0].plot(hs, [a[str(h)] for h in hs], marker="o", color=col, label=lab)
axes[0].axvline(6.5, ls="--", c="k", alpha=0.4); axes[0].axhline(0.1, ls=":", c="red", alpha=0.5, label="chance")
axes[0].annotate("train | OOD ->", (6.55, 0.03), fontsize=8, alpha=0.6)
axes[0].set_xlabel("hop count"); axes[0].set_ylabel("answer accuracy"); axes[0].set_ylim(-0.02, 1.05)
axes[0].set_title("perm (multi-hop lookup): accuracy vs depth"); axes[0].legend(fontsize=8, loc="lower left"); axes[0].grid(alpha=0.2)
# OOD bar
h7 = [load(c)["acc_by_hops"].get("7") for _, c, _ in perm]
axes[1].bar([p[0] for p in perm], h7, color=[p[2] for p in perm])
for i, v in enumerate(h7): axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
axes[1].axhline(0.1, ls=":", c="red", alpha=0.6, label="chance")
axes[1].set_ylabel("OOD accuracy (hops-7)"); axes[1].set_ylim(0, 0.5)
axes[1].set_title("On lookup, only process supervision (C)\nbeats chance OOD — A,B collapse")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2, axis="y")
plt.setp(axes[1].get_xticklabels(), fontsize=8)
fig.suptitle("Cross-task confirmation: where the baseline hits chance, process supervision still extrapolates", fontsize=11, y=1.01)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_perm_crosstask.png")); plt.close(fig)

print("wrote figures to", OUT)
for f in sorted(os.listdir(OUT)): print("  ", f)
