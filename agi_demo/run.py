"""Orchestrate arms A/B/C: train, evaluate, probe, dump metrics.json, plot.

Usage:
  python -m agi_demo.run --quick            # fast sanity: few steps, few examples
  python -m agi_demo.run                     # full run (all three arms)
  python -m agi_demo.run --arm B             # a single arm
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict

from .config import Config
from .train import train_arm
from .probe import accuracy_by_hops, linear_probe
from . import plot


def run_session_arm(cfg: Config) -> dict:
    """Project 3: train on the session task, then eval reference-vs-literal accuracy,
    write→read delay, and the memory-slot probe (observability)."""
    from .train import train_session
    from .probe import eval_session, probe_memory
    model = train_session(cfg, verbose=True)
    rng = random.Random(cfg.seed + 1234)
    ev = eval_session(model, cfg, rng)
    print(f"[arm {cfg.arm}] lit_acc={ev['lit_acc']:.3f} ref_acc={ev['ref_acc']:.3f} "
          f"(chance {ev['chance']:.2f}) delay_acc={ {d: round(a,2) for d,a in ev['delay_acc'].items()} }")
    result = {
        "trainable_params": model.num_trainable(),
        "lit_acc": ev["lit_acc"], "ref_acc": ev["ref_acc"], "chance": ev["chance"],
        "delay_acc": {str(d): a for d, a in ev["delay_acc"].items()},
        "memory_probe": None,
    }
    if model.memory is not None:
        pm = probe_memory(model, cfg, rng)
        result["memory_probe"] = pm
        print(f"[arm {cfg.arm}] memory-slot probe: {pm['slot_probe_acc']:.3f} (chance {pm['chance']:.2f})")
    del model
    return result


def run_arm(cfg: Config) -> dict:
    model = train_arm(cfg, verbose=True)
    rng = random.Random(cfg.seed + 1234)
    accs = accuracy_by_hops(model, cfg, rng)
    print(f"[arm {cfg.arm}] acc_by_hops: " + ", ".join(f"{h}:{a:.2f}" for h, a in accs.items()))

    result = {
        "trainable_params": model.num_trainable(),
        "acc_by_hops": {str(h): a for h, a in accs.items()},
        "probe": None,
    }
    if cfg.arm in ("B", "C"):
        probe_hop = max(cfg.train_hops)  # probe at deepest in-distribution depth
        pr = linear_probe(model, cfg, probe_hop, rng)
        result["probe"] = pr
        result["probe_hop"] = probe_hop
        print(f"[arm {cfg.arm}] probe(hops={probe_hop}) per-step: "
              + ", ".join(f"x{i+1}:{a:.2f}" for i, a in enumerate(pr["per_step_acc"]))
              + f"  (chance={pr['chance']:.2f})")
    del model
    return result


def _mean_std(vals):
    n = len(vals); m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, var ** 0.5


def run_arm_seeds(build_cfg, arm: str, seeds: list) -> dict:
    """Run one arm over several seeds; aggregate per-hop acc and per-step probe as mean±std.
    acc_by_hops holds the MEAN (so existing plots/summaries keep working); *_std and per-seed
    raw values are stored alongside for error bars and honesty."""
    per_seed = []
    for s in seeds:
        cfg = build_cfg(); cfg.arm = arm; cfg.seed = s
        print(f"\n----- arm {arm} seed {s} -----")
        per_seed.append(run_arm(cfg))
    hops = sorted(per_seed[0]["acc_by_hops"].keys(), key=int)
    acc_mean, acc_std = {}, {}
    for h in hops:
        vals = [r["acc_by_hops"][h] for r in per_seed]
        m, sd = _mean_std(vals); acc_mean[h] = m; acc_std[h] = sd
    out = {
        "trainable_params": per_seed[0]["trainable_params"],
        "acc_by_hops": acc_mean, "acc_by_hops_std": acc_std,
        "seeds": seeds, "per_seed_acc": [r["acc_by_hops"] for r in per_seed],
        "probe": None,
    }
    if per_seed[0].get("probe"):
        K = len(per_seed[0]["probe"]["per_step_acc"])
        pmean, pstd = [], []
        for i in range(K):
            vals = [r["probe"]["per_step_acc"][i] for r in per_seed]
            m, sd = _mean_std(vals); pmean.append(m); pstd.append(sd)
        out["probe"] = {"hops": per_seed[0]["probe"]["hops"], "per_step_acc": pmean,
                        "per_step_std": pstd, "chance": per_seed[0]["probe"]["chance"]}
        out["probe_hop"] = per_seed[0].get("probe_hop")
    print(f"[arm {arm}] {len(seeds)}-seed mean acc_by_hops: "
          + ", ".join(f"{h}:{acc_mean[h]:.2f}±{acc_std[h]:.2f}" for h in hops))
    return out


def apply_quick(cfg: Config):
    cfg.train_steps = 120
    cfg.eval_examples_per_hop = 96
    cfg.probe_examples_per_hop = 128
    cfg.eval_hops = [1, 2, 3, 4, 5]
    cfg.log_every = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C", "M0", "M1", "M2"], default=None,
                    help="run a single arm; A/B/C for chain tasks, M0/M1/M2 for session/memory")
    ap.add_argument("--session", action="store_true",
                    help="Project 3: run the session/short-term-memory study (arms M0/M1/M2)")
    ap.add_argument("--quick", action="store_true", help="fast sanity settings")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--model", type=str, default=None, help="override model id (e.g. for B100)")
    ap.add_argument("--curriculum", action="store_true",
                    help="competence-gated shallow->deep training (fix for difficulty-mixing collapse)")
    ap.add_argument("--residual", action="store_true",
                    help="Q1: residual thought loop t = h_last + proj(h_last)")
    ap.add_argument("--lora-r", type=int, default=None,
                    help="Q2: override LoRA rank (r=32 ~matches B's trainable params for a no-latent SFT control)")
    ap.add_argument("--curriculum-patience", type=int, default=None,
                    help="force-advance a curriculum level after this many steps")
    ap.add_argument("--process-warmup", type=int, default=None,
                    help="steps over which arm-C process loss ramps in")
    ap.add_argument("--process-steps", type=str, default=None,
                    help="① causal ablation: which latent steps get process CE "
                         "('all'|'first_half'|'second_half'|'1,3,5')")
    ap.add_argument("--task", type=str, default=None, choices=["arith", "perm"],
                    help="② task family: arith (mod-N chain) or perm (multi-hop lookup)")
    ap.add_argument("--seed", type=int, default=None, help="single seed override")
    ap.add_argument("--seeds", type=str, default=None,
                    help="② multi-seed: comma list, e.g. 0,1,2 -> report mean±std per hop")
    ap.add_argument("--unfreeze-last-n", type=int, default=None,
                    help="attribution: unfreeze the last N base transformer blocks (0 = frozen)")
    ap.add_argument("--out-dir", type=str, default=None, help="override output dir")
    ap.add_argument("--local-model-dir", type=str, default=None,
                    help="path to a local weights folder (e.g. internal hub model)")
    ap.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16", "float16"],
                    help="model dtype (bfloat16 for B200/CUDA; float32 for MPS)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--train-hops", type=str, default=None, help="comma list, e.g. 1,2,3,4,5,6")
    ap.add_argument("--eval-hops", type=str, default=None, help="comma list, e.g. 1,2,3,4,5,6,7,8")
    args = ap.parse_args()

    def build_cfg() -> Config:
        c = Config()
        # --- pre-resolve overrides (affect which weights get loaded) ---
        if args.local_model_dir:
            c.local_model_dir = args.local_model_dir
        if args.dtype:
            c.dtype = args.dtype
        if args.out_dir:
            c.out_dir = args.out_dir
        if args.train_hops:
            c.train_hops = [int(x) for x in args.train_hops.split(",")]
        if args.eval_hops:
            c.eval_hops = [int(x) for x in args.eval_hops.split(",")]
        if args.batch_size:
            c.batch_size = args.batch_size
        c = c.resolve()
        # --- post-resolve overrides ---
        if args.quick:
            apply_quick(c)
        if args.steps:
            c.train_steps = args.steps
        if args.model:
            c.model_id = args.model
        if args.curriculum:
            c.curriculum = True
        if args.residual:
            c.residual_thought = True
        if args.lora_r is not None:
            c.lora_r = args.lora_r
        if args.curriculum_patience is not None:
            c.curriculum_patience_steps = args.curriculum_patience
        if args.process_warmup is not None:
            c.process_warmup_steps = args.process_warmup
        if args.process_steps is not None:
            c.process_steps = args.process_steps
        if args.task is not None:
            c.task = args.task
        if args.seed is not None:
            c.seed = args.seed
        if args.unfreeze_last_n is not None:
            c.unfreeze_last_n = args.unfreeze_last_n
        # session/memory arms: M0 = no memory (baseline), M1/M2 = memory on
        if args.session or (args.arm in ("M0", "M1", "M2")):
            c.task = "session"
            c.use_memory = (args.arm in ("M1", "M2"))
        return c

    base = build_cfg()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    session_mode = args.session or (args.arm in ("M0", "M1", "M2"))

    if session_mode:
        arms = [args.arm] if args.arm in ("M0", "M1", "M2") else ["M0", "M1", "M2"]
    else:
        arms = [args.arm] if args.arm else ["A", "B", "C"]
    metrics = {
        "config": {
            "model_id": base.model_id, "device": base.device, "modulus": base.modulus,
            "train_hops": base.train_hops, "eval_hops": base.eval_hops,
            "train_steps": base.train_steps, "batch_size": base.batch_size,
            "curriculum": base.curriculum, "residual_thought": base.residual_thought,
            "lora_r": base.lora_r, "dtype": base.dtype,
            "task": ("session" if session_mode else base.task), "process_steps": base.process_steps,
            "session_len": base.session_len, "n_slots": base.n_slots, "ref_prob": base.ref_prob,
            "seeds": seeds if seeds else [base.seed],
        },
        "arms": {},
    }

    if session_mode:
        for arm in arms:
            print(f"\n{'='*60}\nARM {arm}\n{'='*60}")
            def build_arm_cfg(a=arm):
                c = build_cfg(); c.arm = a
                c.use_memory = (a in ("M1", "M2")); c.task = "session"
                return c
            if seeds:
                # aggregate ref_acc / lit_acc / memory_probe across seeds
                per = []
                for s in seeds:
                    c = build_arm_cfg(); c.seed = s
                    print(f"\n----- {arm} seed {s} -----")
                    per.append(run_session_arm(c))
                def ms(key):
                    vals = [p[key] for p in per if p.get(key) is not None]
                    return _mean_std(vals) if vals else (None, None)
                ra = ms("ref_acc"); la = ms("lit_acc")
                mp_vals = [p["memory_probe"]["slot_probe_acc"] for p in per if p.get("memory_probe")]
                agg = {"trainable_params": per[0]["trainable_params"], "seeds": seeds,
                       "ref_acc": ra[0], "ref_acc_std": ra[1], "lit_acc": la[0], "lit_acc_std": la[1],
                       "chance": per[0]["chance"], "per_seed_ref": [p["ref_acc"] for p in per]}
                if mp_vals:
                    m, sd = _mean_std(mp_vals); agg["memory_probe"] = {"slot_probe_acc": m, "std": sd, "chance": per[0]["chance"]}
                print(f"[arm {arm}] {len(seeds)}-seed ref_acc={ra[0]:.3f}±{ra[1]:.3f}" +
                      (f" | mem_probe={agg['memory_probe']['slot_probe_acc']:.3f}" if mp_vals else ""))
                metrics["arms"][arm] = agg
            else:
                metrics["arms"][arm] = run_session_arm(build_arm_cfg())
        out_json = os.path.join(base.out_dir, "metrics.json")
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nwrote {out_json}")
        _summarize_session(metrics)
        return

    for arm in arms:
        print(f"\n{'='*60}\nARM {arm}\n{'='*60}")
        if seeds:
            metrics["arms"][arm] = run_arm_seeds(build_cfg, arm, seeds)
        else:
            cfg = build_cfg(); cfg.arm = arm
            metrics["arms"][arm] = run_arm(cfg)

    out_json = os.path.join(base.out_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {out_json}")

    plot.plot_acc_vs_hops(metrics, os.path.join(base.out_dir, "acc_vs_hops.png"))
    probe_hop = max(base.train_hops)
    if any(metrics["arms"].get(a, {}).get("probe") for a in ("B", "C")):
        plot.plot_probe(metrics, os.path.join(base.out_dir, "probe_per_step.png"), probe_hop)
    print(f"wrote figures to {base.out_dir}/")
    _summarize(metrics)


def _summarize_session(metrics: dict):
    print(f"\n{'='*60}\nSESSION SUMMARY (short-term memory)\n{'='*60}")
    ch = metrics["config"]["modulus"]
    for arm in ("M0", "M1", "M2"):
        a = metrics["arms"].get(arm)
        if not a:
            continue
        line = f"arm {arm}: ref_acc={a['ref_acc']:.2f}"
        if a.get("ref_acc_std") is not None:
            line += f"±{a['ref_acc_std']:.2f}"
        line += f" lit_acc={a['lit_acc']:.2f} (chance {1/ch:.2f})"
        if a.get("memory_probe"):
            line += f" | mem-probe={a['memory_probe']['slot_probe_acc']:.2f}"
        print(line)
    print("Reading: ref_acc >> chance ⇒ memory read works; mem-probe >> chance ⇒ slots observably "
          "hold the value (the red line we must preserve).")


def _summarize(metrics: dict):
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    arms = metrics["arms"]
    mod = metrics["config"]["modulus"]
    ood = [h for h in metrics["config"]["eval_hops"] if h > max(metrics["config"]["train_hops"])]
    for arm in ("A", "B", "C"):
        if arm not in arms:
            continue
        a = arms[arm]
        ood_acc = [a["acc_by_hops"].get(str(h)) for h in ood]
        ood_acc = [x for x in ood_acc if x is not None]
        line = f"arm {arm}: params={a['trainable_params']:>9,}"
        if ood_acc:
            line += f" | OOD acc(hops {ood})={[round(x,2) for x in ood_acc]}"
        if a.get("probe"):
            ps = a["probe"]["per_step_acc"]
            line += f" | probe last-step={ps[-1]:.2f} (chance {1/mod:.2f})"
        print(line)


if __name__ == "__main__":
    main()
