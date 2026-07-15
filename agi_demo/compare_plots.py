"""Generate cross-model / cross-config comparison figures from ALL_RESULTS.csv.

Figures produced (English labels — paper-portable; Chinese narrative lives in README_zh.md):
  fig1_scale_reversal      0.5B-B-vs-A  next to  3B-B-vs-A : latent helps small, hurts large
  fig2_process_ood_3b      3B A/B/C accuracy vs depth: only C (process) beats baseline OOD
  fig3_probe_mechanism_3b  3B probe per-step B vs C: chain-decodability = the mechanism
  fig4_matched_sft         3B OOD bars: C vs A vs A'(r=32) — latent's edge isn't parameters
  fig5_curriculum_vs_resid 3B arm C: curriculum vs residual+uniform, acc + probe side by side
  fig6_decodability_law    scatter: mean mid-chain probe  vs  OOD accuracy, all latent arms
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "outputs", "ALL_RESULTS.csv")
OUT = os.path.join(HERE, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

C = {"A": "#8a8a8a", "B": "#2b6cb0", "C": "#c05621", "Aprime": "#6b46c1"}
LAB = {"A": "A: result-only", "B": "B: latent (result CE)", "C": "C: latent + process CE"}


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def accs(row, upto=8):
    xs, ys = [], []
    for h in range(1, upto + 1):
        v = row.get(f"acc_h{h}", "")
        if v not in ("", None):
            xs.append(h); ys.append(float(v))
    return xs, ys


def probes(row, upto=6):
    xs, ys = [], []
    for i in range(1, upto + 1):
        v = row.get(f"probe_x{i}", "")
        if v not in ("", None):
            xs.append(i); ys.append(float(v))
    return xs, ys


def get(rows, model, config, arm):
    for r in rows:
        if r["model"] == model and r["config"] == config and r["arm"] == arm:
            return r
    return None


def main():
    rows = load()
    M05, M3 = "Qwen2.5-0.5B", "Qwen2.5-3B-Instruct"

    # ---- fig1: scale reversal (two panels) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
    for ax, (model, cfg, trainmax, title) in zip(
        axes,
        [(M05, "curriculum", 4, "0.5B (train hops 1-4)"),
         (M3, "curriculum", 6, "3B (train hops 1-6)")]):
        for arm in ("A", "B"):
            r = get(rows, model, cfg, arm)
            if r:
                xs, ys = accs(r); ax.plot(xs, ys, marker="o", color=C[arm], label=LAB[arm])
        ax.axvline(trainmax + 0.5, ls="--", c="k", alpha=0.4)
        ax.axhline(0.1, ls=":", c="red", alpha=0.5)
        ax.set_title(title); ax.set_xlabel("hop count (reasoning depth)")
        ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.2)
        ax.annotate("train | OOD", (trainmax + 0.55, 0.02), fontsize=8, alpha=0.6)
    axes[0].set_ylabel("answer exact-match accuracy")
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Scale reversal: latent thinking helps the small model, hurts the large one (OOD)",
                 fontsize=12, y=1.00)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_scale_reversal.png")); plt.close(fig)

    # ---- fig2: 3B process helps OOD (A/B/C) ----
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    for arm in ("A", "B", "C"):
        r = get(rows, M3, "curriculum", arm)
        xs, ys = accs(r); ax.plot(xs, ys, marker="o", color=C[arm], label=LAB[arm])
    ax.axvline(6.5, ls="--", c="k", alpha=0.4); ax.axhline(0.1, ls=":", c="red", alpha=0.5, label="chance")
    ax.annotate("train | OOD ->", (6.55, 0.04), fontsize=8, alpha=0.6)
    ax.set_title("3B: only latent+process (C) beats the baseline out-of-distribution")
    ax.set_xlabel("hop count (reasoning depth)"); ax.set_ylabel("answer exact-match accuracy")
    ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.2); ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_process_ood_3b.png")); plt.close(fig)

    # ---- fig3: 3B probe mechanism (B vs C) ----
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    for arm in ("B", "C"):
        r = get(rows, M3, "curriculum", arm)
        xs, ys = probes(r); ax.plot(xs, ys, marker="s", color=C[arm], label=LAB[arm])
    ax.axhline(0.1, ls=":", c="red", alpha=0.6, label="chance")
    ax.set_title("Mechanism: process supervision keeps the whole reasoning chain decodable")
    ax.set_xlabel("thought-vector index i  (probe target = intermediate x_i, 3B hops=6)")
    ax.set_ylabel("linear-probe accuracy for x_i")
    ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.2); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_probe_mechanism_3b.png")); plt.close(fig)

    # ---- fig4: matched-SFT bars (OOD hops-7) ----
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    bars = [
        ("C\nlatent+process\n(20.5M)", get(rows, M3, "curriculum", "C"), C["C"]),
        ("A\nresult-only\n(3.7M)", get(rows, M3, "curriculum", "A"), C["A"]),
        ("A'\nmatched-SFT r=32\n(14.7M)", get(rows, M3, "q2_matched_sft", "A"), C["Aprime"]),
    ]
    labels = [b[0] for b in bars]
    h7 = [float(b[1]["acc_h7"]) for b in bars]
    h8 = [float(b[1]["acc_h8"]) for b in bars]
    x = range(len(bars)); w = 0.38
    ax.bar([i - w/2 for i in x], h7, w, color=[b[2] for b in bars], label="hops-7 (OOD +1)")
    ax.bar([i + w/2 for i in x], h8, w, color=[b[2] for b in bars], alpha=0.5, label="hops-8 (OOD +2)")
    for i, v in enumerate(h7): ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(h8): ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("OOD answer accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("Latent's edge is not just parameters:\nmatched-capacity SFT (A') generalizes worse than plain A")
    ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_matched_sft.png")); plt.close(fig)

    # ---- fig5: curriculum vs residual (arm C) acc + probe ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
    cur = get(rows, M3, "curriculum", "C"); res = get(rows, M3, "q1_residual_uniform_C", "C")
    xs, ys = accs(cur); axes[0].plot(xs, ys, marker="o", color="#c05621", label="curriculum")
    xs, ys = accs(res); axes[0].plot(xs, ys, marker="o", color="#38a169", label="residual + uniform")
    axes[0].axvline(6.5, ls="--", c="k", alpha=0.4); axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_title("Arm C accuracy vs depth"); axes[0].set_xlabel("hop count")
    axes[0].set_ylabel("answer accuracy"); axes[0].grid(alpha=0.2); axes[0].legend(fontsize=9)
    xs, ys = probes(cur); axes[1].plot(xs, ys, marker="s", color="#c05621", label="curriculum")
    xs, ys = probes(res); axes[1].plot(xs, ys, marker="s", color="#38a169", label="residual + uniform")
    axes[1].axhline(0.1, ls=":", c="red", alpha=0.6); axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_title("Arm C probe: chain decodability"); axes[1].set_xlabel("thought-vector index i")
    axes[1].set_ylabel("probe accuracy"); axes[1].grid(alpha=0.2); axes[1].legend(fontsize=9)
    fig.suptitle("Curriculum ≠ residual: both train, but only curriculum keeps the mid-chain decodable → better OOD",
                 fontsize=11, y=1.00)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_curriculum_vs_residual.png")); plt.close(fig)

    # NOTE: a 6th figure (scatter of mean mid-chain decodability vs OOD accuracy) was tried and
    # dropped: with only 4 comparable 3B latent arms the points do NOT form a clean monotonic
    # relationship (a MEAN over steps hides that what matters is *contiguous* decodability through
    # the deep steps, not the average). The per-step probe curves (fig3, fig5-right) show that
    # mechanism honestly, so we don't ship a scatter that overclaims a scalar "law".

    print("wrote figures to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
