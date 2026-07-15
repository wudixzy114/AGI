"""Orchestrate the kernel study: train arms, eval, probe, dump metrics.json + figures.

Usage:
  python -m agi_demo.kernel.run --quick                       # fast Mac sanity (all session arms)
  python -m agi_demo.kernel.run --session --arm K2            # one session arm
  python -m agi_demo.kernel.run --arith --arm K1 --curriculum # arith secondary run
  python -m agi_demo.kernel.run --session --seeds 0,1,2       # multi-seed ref_acc mean±std

Session arms K0/K1/K2 are the headline memory-USE study; arith arm is the length-generalization
secondary. Figures + metrics land in --out-dir (default agi_demo/outputs/kernel).
"""
from __future__ import annotations

import argparse
import json
import os
import random

from .kconfig import KernelConfig
from . import kplot


def _mean_std(vals):
    n = len(vals); m = sum(vals) / n
    if n < 2:
        return m, 0.0
    return m, (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5


# arm -> (short label, what it tests). Kept here (not in the dashboard) so meaning travels WITH
# the data: the dashboard reads these strings, never a hardcoded map, so new arms self-describe.
ARM_INFO = {
    "K0": ("K0 · read severed", "causal control — fast weights write but the read never feeds compute"),
    "K1": ("K1 · fast-weight memory", "answer supervision only — does USE emerge on its own?"),
    "K2": ("K2 · + grounded read", "adds read supervision (referenced read must decode the stored value)"),
}


def _describe(cfg, mode: str, seeds) -> dict:
    """Human title/subtitle for this config, derived from its OWN params (no hardcoded names)."""
    seedtxt = f"{len(seeds)} seeds" if seeds and len(seeds) > 1 else "1 seed"
    if mode == "session":
        kind = ("pure recall (no arithmetic)" if cfg.session_hops == 0
                else f"recall + {cfg.session_hops}-hop chain")
        title = f"Session memory · {kind}"
        subtitle = (f"session_len={cfg.session_len}, n_slots={cfg.n_slots}, "
                    f"{'delta-rule' if cfg.delta_rule else 'Hebbian'} write · {seedtxt}")
        metric = "ref_lit"          # tells the dashboard: draw ref/lit arm bars
    else:
        title = f"Arithmetic · mod-{cfg.modulus} chain"
        subtitle = (f"train hops {min(cfg.train_hops)}–{max(cfg.train_hops)}, "
                    f"{'curriculum' if cfg.curriculum else 'uniform'} · {seedtxt}")
        metric = "acc_by_hops"      # tells the dashboard: draw accuracy-vs-depth lines
    return {"title": title, "subtitle": subtitle, "metric": metric,
            "kernel": f"d_model={cfg.d_model}, d_state={cfg.d_state}, {cfg.n_layers} layers "
                      f"(γ={cfg.decays})", "arm_info": ARM_INFO}


def run_session_arm(cfg: KernelConfig) -> dict:
    from .train import train_session
    from .probe import eval_session, probe_timescales
    model = train_session(cfg, verbose=True)
    rng = random.Random(cfg.seed + 1234)
    ev = eval_session(model, cfg, rng)
    print(f"[arm {cfg.arm}] lit_acc={ev['lit_acc']:.3f} ref_acc={ev['ref_acc']:.3f} "
          f"(chance {ev['chance']:.2f}) delay={ {d: round(a,2) for d,a in ev['delay_acc'].items()} }")
    result = {"trainable_params": model.num_trainable(),
              "lit_acc": ev["lit_acc"], "ref_acc": ev["ref_acc"], "chance": ev["chance"],
              "delay_acc": {str(d): a for d, a in ev["delay_acc"].items()}, "timescales": None}
    # timescale hierarchy probe (only meaningful when reads are enabled)
    if model.read_enabled:
        ts = probe_timescales(model, cfg, rng)
        result["timescales"] = ts
        for layer in ts["layers"]:
            abd = layer["acc_by_delay"]
            if abd:
                ds = sorted(abd)
                print(f"    layer γ={layer['gamma']}: " +
                      ", ".join(f"Δ{d}:{abd[d]:.2f}" for d in ds))
    del model
    return result


def run_arith_arm(cfg: KernelConfig) -> dict:
    from .train import train_arith
    from .probe import accuracy_by_hops, probe_arith
    model = train_arith(cfg, verbose=True)
    rng = random.Random(cfg.seed + 1234)
    accs = accuracy_by_hops(model, cfg, rng)
    print(f"[arm {cfg.arm}] acc_by_hops: " + ", ".join(f"{h}:{a:.2f}" for h, a in accs.items()))
    result = {"trainable_params": model.num_trainable(),
              "acc_by_hops": {str(h): a for h, a in accs.items()}, "probe": None}
    probe_hop = max(cfg.train_hops)
    pr = probe_arith(model, cfg, probe_hop, rng)
    result["probe"] = pr; result["probe_hop"] = probe_hop
    print(f"[arm {cfg.arm}] probe(hops={probe_hop}) per-step: " +
          ", ".join(f"x{i+1}:{a:.2f}" for i, a in enumerate(pr["per_step_acc"])) +
          f"  (chance {pr['chance']:.2f})")
    del model
    return result


def apply_quick(c: KernelConfig):
    c.train_steps = 150; c.eval_examples_per_hop = 128; c.probe_examples = 256
    c.eval_sessions = 128; c.probe_sessions = 150; c.eval_hops = [1, 2, 3, 4, 5]; c.log_every = 30


def build_cfg(args) -> KernelConfig:
    c = KernelConfig()
    if args.d_model: c.d_model = args.d_model
    if args.d_state: c.d_state = args.d_state
    if args.n_layers: c.n_layers = args.n_layers
    if args.decays: c.decays = [float(x) for x in args.decays.split(",")]
    if args.out_dir: c.out_dir = args.out_dir
    if args.train_hops: c.train_hops = [int(x) for x in args.train_hops.split(",")]
    if args.eval_hops: c.eval_hops = [int(x) for x in args.eval_hops.split(",")]
    if args.batch_size: c.batch_size = args.batch_size
    if args.session_len: c.session_len = args.session_len
    if args.session_hops is not None: c.session_hops = args.session_hops
    if args.n_slots: c.n_slots = args.n_slots
    if args.device: c.device = args.device
    if args.allow_overwrite: c.allow_overwrite = True   # must precede resolve() (it gates the assert)
    c = c.resolve()
    if args.quick: apply_quick(c)
    if args.steps: c.train_steps = args.steps
    if args.lr: c.lr = args.lr
    if args.curriculum: c.curriculum = True
    if args.curriculum_patience is not None: c.curriculum_patience_steps = args.curriculum_patience
    if args.no_delta: c.delta_rule = False
    if args.task: c.task = args.task
    if args.seed is not None: c.seed = args.seed
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["K0", "K1", "K2"], default=None)
    ap.add_argument("--session", action="store_true", help="run the memory-USE study (K0/K1/K2)")
    ap.add_argument("--arith", action="store_true", help="run the arith length-gen secondary")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--d-state", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--decays", type=str, default=None, help="comma list, e.g. 0,0.7,0.95,0.99")
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--curriculum-patience", type=int, default=None,
                    help="force-advance a curriculum level after this many steps")
    ap.add_argument("--no-delta", action="store_true", help="ablation: additive Hebbian write")
    ap.add_argument("--task", type=str, default=None, choices=["arith", "perm"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None, help="comma list -> mean±std ref_acc")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--session-len", type=int, default=None)
    ap.add_argument("--session-hops", type=int, default=None,
                    help="ops per problem in a session; 0 = pure recall (cleanest memory-USE test)")
    ap.add_argument("--n-slots", type=int, default=None)
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="Phase A: permit session_len>n_slots so slots are rewritten mid-session "
                         "(the delta-rule vs Hebbian overwrite contest)")
    ap.add_argument("--train-hops", type=str, default=None)
    ap.add_argument("--eval-hops", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    base = build_cfg(args)
    arith_mode = args.arith and not args.session
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    if arith_mode:
        arms = [args.arm] if args.arm else ["K1"]   # arith uses one arm (full kernel) by default
    else:
        arms = [args.arm] if args.arm else ["K0", "K1", "K2"]

    mode = "arith" if arith_mode else "session"
    metrics = {"config": {k: getattr(base, k) for k in
                          ["d_model", "d_state", "n_layers", "decays", "delta_rule", "modulus",
                           "task", "train_hops", "eval_hops", "session_len", "n_slots", "ref_prob",
                           "session_hops", "train_steps", "batch_size", "lr", "device"]},
               "mode": mode, "seeds": seeds or [base.seed],
               # self-describing block so the dashboard auto-titles this config from its own params,
               # never from a hardcoded name map. Any param change is reflected automatically.
               "dashboard": _describe(base, mode, seeds),
               "arms": {}}

    for arm in arms:
        print(f"\n{'='*60}\nARM {arm}\n{'='*60}")
        if arith_mode:
            if seeds:
                per = []
                for s in seeds:
                    c = build_cfg(args); c.arm = arm; c.seed = s
                    print(f"\n----- {arm} seed {s} -----"); per.append(run_arith_arm(c))
                hops = sorted(per[0]["acc_by_hops"], key=int)
                metrics["arms"][arm] = {
                    "trainable_params": per[0]["trainable_params"],
                    "acc_by_hops": {h: _mean_std([p["acc_by_hops"][h] for p in per])[0] for h in hops},
                    "acc_by_hops_std": {h: _mean_std([p["acc_by_hops"][h] for p in per])[1] for h in hops},
                    "probe": per[0]["probe"], "probe_hop": per[0].get("probe_hop"),
                    "per_seed_acc": [p["acc_by_hops"] for p in per]}
            else:
                c = build_cfg(args); c.arm = arm
                metrics["arms"][arm] = run_arith_arm(c)
        else:
            if seeds:
                per = []
                for s in seeds:
                    c = build_cfg(args); c.arm = arm; c.seed = s
                    print(f"\n----- {arm} seed {s} -----"); per.append(run_session_arm(c))
                ra = _mean_std([p["ref_acc"] for p in per]); la = _mean_std([p["lit_acc"] for p in per])
                metrics["arms"][arm] = {
                    "trainable_params": per[0]["trainable_params"],
                    "ref_acc": ra[0], "ref_acc_std": ra[1], "lit_acc": la[0], "lit_acc_std": la[1],
                    "chance": per[0]["chance"], "per_seed_ref": [p["ref_acc"] for p in per],
                    "timescales": per[0]["timescales"]}
                print(f"[arm {arm}] {len(seeds)}-seed ref_acc={ra[0]:.3f}±{ra[1]:.3f}")
            else:
                c = build_cfg(args); c.arm = arm
                metrics["arms"][arm] = run_session_arm(c)

    out_json = os.path.join(base.out_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {out_json}")

    # figures
    if arith_mode:
        kplot.plot_arith(metrics, os.path.join(base.out_dir, "arith_acc_vs_hops.png"))
    else:
        kplot.plot_session_arms(metrics, os.path.join(base.out_dir, "session_arms.png"))
        for arm in arms:
            ts = metrics["arms"].get(arm, {}).get("timescales")
            if ts:
                kplot.plot_timescales(ts, os.path.join(base.out_dir, f"timescales_{arm}.png"))
    print(f"wrote figures to {base.out_dir}/")
    _summarize(metrics)


def _summarize(metrics: dict):
    print(f"\n{'='*60}\nSUMMARY ({metrics['mode']})\n{'='*60}")
    if metrics["mode"] == "session":
        for arm in ("K0", "K1", "K2"):
            a = metrics["arms"].get(arm)
            if not a:
                continue
            line = f"arm {arm}: ref_acc={a['ref_acc']:.2f}"
            if a.get("ref_acc_std"):
                line += f"±{a['ref_acc_std']:.2f}"
            line += f" lit_acc={a['lit_acc']:.2f} (chance {a['chance']:.2f})"
            print(line)
        print("Reading: K1/K2 ref_acc >> chance ⇒ the native kernel USES memory (Proj-3 got 0.10). "
              "K0 (read severed) ≈ chance ⇒ it's the associative READ that matters.")
    else:
        for arm, a in metrics["arms"].items():
            ood = [h for h in metrics["config"]["eval_hops"] if h > max(metrics["config"]["train_hops"])]
            oa = [round(a["acc_by_hops"].get(str(h), 0), 2) for h in ood]
            line = f"arm {arm}: params={a['trainable_params']:,} OOD(hops {ood})={oa}"
            if a.get("probe"):
                line += f" | probe last-step={a['probe']['per_step_acc'][-1]:.2f}"
            print(line)


if __name__ == "__main__":
    main()
