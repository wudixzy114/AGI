"""Training loop — device-agnostic, one arm at a time.

Loss design (the corrected version of the doc's step 3):
  * Answer cross-entropy is ALWAYS the anchor -> guarantees the model actually learns
    to emit a correct answer (avoids the degenerate "output nothing" solution that a
    pure hidden-state-MSE objective collapses to).
  * Arm A: no latent steps; CE on answer decoded straight from "=".
  * Arm B: `hops` latent thinking steps, then CE on the answer. The "process" is rewarded
    only implicitly, through whether the latent chain leads to the right answer.
  * Arm C: B + an auxiliary CE that asks thought-vector i to predict intermediate x_i,
    via step_head. This is the explicit "process > result" supervision.

Each batch uses a single hop-count so the number of latent steps == hops for that batch.
"""
from __future__ import annotations

import random
from typing import Dict

import torch
import torch.nn.functional as F

from .config import Config
from .model import CoconutReasoner
from .task import make_batch, digit_token_ids, make_session_batch


def n_latent_for(cfg: Config, hops: int) -> int:
    # Arm A and the no-memory session baseline (M0) use no latent steps.
    if cfg.arm in ("A", "M0"):
        return 0
    return hops


def supervised_steps(cfg: Config, n_lat: int) -> set:
    """Which 0-indexed latent steps get process CE, given process_steps spec + chain length."""
    spec = cfg.process_steps
    if spec == "all":
        return set(range(n_lat))
    if spec == "first_half":
        return set(range(n_lat // 2))               # shallow half (steps encoding x1..)
    if spec == "second_half":
        return set(range(n_lat // 2, n_lat))        # deep half (.. up to x_last)
    # explicit 1-indexed list, e.g. "1,3,5"
    idx = {int(x) - 1 for x in spec.split(",") if x.strip()}
    return {i for i in idx if 0 <= i < n_lat}


def compute_loss(model: CoconutReasoner, batch, cfg: Config, step: int = 10**9) -> Dict[str, torch.Tensor]:
    n_lat = n_latent_for(cfg, batch.hops)
    answer_logits, thoughts = model(batch.input_ids, batch.attention_mask, n_lat)

    digit_logits = model.digit_logits(answer_logits)          # (B, N)
    ans_loss = F.cross_entropy(digit_logits, batch.answer_vals)

    proc_loss = torch.zeros((), device=digit_logits.device)
    if cfg.arm == "C" and thoughts:
        sup = supervised_steps(cfg, len(thoughts))
        n_sup = 0
        # thought i (0-indexed) should encode intermediate x_{i+1}; supervise only chosen steps
        for i, t in enumerate(thoughts):
            if i not in sup:
                continue
            logits_i = model.step_head(t)                     # (B, N)
            proc_loss = proc_loss + F.cross_entropy(logits_i, batch.inter_vals[:, i])
            n_sup += 1
        if n_sup:
            proc_loss = proc_loss / n_sup

    # Ramp the process weight in AFTER answer-CE has built the latent chain.
    # (Applying it from step 0 destabilizes: near-identical early thoughts get a large,
    #  conflicting gradient trying to predict distinct intermediates. Verified.)
    if cfg.arm == "C":
        ramp = min(1.0, step / max(1, cfg.process_warmup_steps))
        total = ans_loss + cfg.process_loss_weight * ramp * proc_loss
    else:
        total = ans_loss
    with torch.no_grad():
        acc = (digit_logits.argmax(-1) == batch.answer_vals).float().mean()
    return {"total": total, "ans": ans_loss.detach(), "proc": proc_loss.detach(), "acc": acc}


def session_loss(model: CoconutReasoner, sbatch, cfg: Config) -> Dict[str, torch.Tensor]:
    """Process a full session sequentially, sharing memory across problems.
    M0: no memory (baseline). M1: memory + answer CE. M2: + slot-write supervision.
    Returns total loss + overall acc + acc split by literal/reference problems."""
    dev = cfg.device
    B = sbatch.positions[0]["input_ids"].shape[0]
    if model.memory is not None:
        model.memory.reset(B, dev, next(model.parameters()).dtype)

    n_lat = 0 if cfg.arm == "M0" else cfg.session_hops
    total = torch.zeros((), device=dev)
    n_correct = n_tot = 0
    ref_correct = ref_tot = lit_correct = lit_tot = 0
    mem_loss_acc = torch.zeros((), device=dev)
    addr_loss_acc = torch.zeros((), device=dev)

    for t in range(sbatch.T):
        pos = sbatch.positions[t]
        logits, _, ans_hidden, addr_logits = model.forward_problem(
            pos["input_ids"], pos["attention_mask"], n_lat)
        dl = model.digit_logits(logits)                       # (B,N)
        total = total + F.cross_entropy(dl, pos["answer_vals"])

        # M2: address supervision — reference problems should attend to their ref_slot.
        if cfg.arm == "M2" and addr_logits is not None:
            ref = pos["is_ref"].bool()
            if ref.any():
                addr_loss_acc = addr_loss_acc + F.cross_entropy(
                    addr_logits[ref], pos["ref_slot"][ref])

        # write this problem's answer into its slot (M1/M2)
        if model.memory is not None:
            model.memory.write(ans_hidden, pos["write_slot"])
            # M2: supervise that the written slot decodes to the correct answer
            if cfg.arm == "M2":
                slot_vec = model.memory.slots[torch.arange(B, device=dev), pos["write_slot"]]  # (B,H)
                mem_logits = model.mem_slot_head(slot_vec)
                mem_loss_acc = mem_loss_acc + F.cross_entropy(mem_logits, pos["answer_vals"])

        with torch.no_grad():
            pred = dl.argmax(-1)
            corr = (pred == pos["answer_vals"])
            n_correct += corr.sum().item(); n_tot += B
            ref = pos["is_ref"].bool()
            ref_correct += corr[ref].sum().item(); ref_tot += ref.sum().item()
            lit_correct += corr[~ref].sum().item(); lit_tot += (~ref).sum().item()

    if cfg.arm == "M2":
        total = (total / sbatch.T
                 + cfg.mem_write_weight * (mem_loss_acc / sbatch.T)
                 + cfg.mem_write_weight * (addr_loss_acc / sbatch.T))
    else:
        total = total / sbatch.T
    return {
        "total": total, "acc": n_correct / max(1, n_tot),
        "ref_acc": ref_correct / max(1, ref_tot), "lit_acc": lit_correct / max(1, lit_tot),
        "mem_loss": (mem_loss_acc / sbatch.T).detach() if cfg.arm == "M2" else torch.zeros(()),
    }


def train_session(cfg: Config, verbose: bool = True) -> CoconutReasoner:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    model = CoconutReasoner(cfg); model.train()
    digit_ids = digit_token_ids(model.tokenizer, cfg.modulus)
    opt = torch.optim.AdamW(model.trainable_param_groups())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps)))
    if verbose:
        print(f"[arm {cfg.arm}] trainable: {model.num_trainable():,} | device={cfg.device} | "
              f"memory={'on' if model.memory is not None else 'off'} | session")
    ref_ema = 0.0
    for step in range(cfg.train_steps):
        sb = make_session_batch(model.tokenizer, cfg.batch_size, cfg, rng, device=cfg.device, digit_ids=digit_ids)
        out = session_loss(model, sb, cfg)
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in model.trainable_param_groups() for p in g["params"]], cfg.grad_clip)
        opt.step(); sched.step()
        ref_ema = 0.95 * ref_ema + 0.05 * out["ref_acc"]
        if verbose and (step + 1) % cfg.log_every == 0:
            print(f"  step {step+1:4d}/{cfg.train_steps} | loss={out['total'].item():.3f} "
                  f"| acc={out['acc']:.3f} ref={out['ref_acc']:.3f} lit={out['lit_acc']:.3f} "
                  f"mem={out['mem_loss'].item():.3f} | ref_ema={ref_ema:.3f}")
    model.eval()
    return model


def train_arm(cfg: Config, verbose: bool = True) -> CoconutReasoner:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    model = CoconutReasoner(cfg)
    model.train()

    digit_ids = digit_token_ids(model.tokenizer, cfg.modulus)
    opt = torch.optim.AdamW(model.trainable_param_groups())
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps))
    )

    if verbose:
        mode = "curriculum" if cfg.curriculum else "uniform"
        print(f"[arm {cfg.arm}] trainable params: {model.num_trainable():,} | device={cfg.device} | {mode}")

    sorted_hops = sorted(cfg.train_hops)
    # Curriculum state: index into sorted_hops of the deepest unlocked level.
    level = 0 if cfg.curriculum else len(sorted_hops) - 1
    steps_at_level = 0
    level_acc = 0.0          # EMA of accuracy AT the current deepest level (gates advancement)

    run_acc = 0.0
    for step in range(cfg.train_steps):
        if cfg.curriculum:
            # Sample from unlocked levels, biased toward the newest (hardest) unlocked one
            # so it keeps getting practice while easier levels are refreshed.
            unlocked = sorted_hops[: level + 1]
            hops = sorted_hops[level] if rng.random() < 0.5 else rng.choice(unlocked)
        else:
            hops = rng.choice(sorted_hops)

        batch = make_batch(
            model.tokenizer, cfg.batch_size, hops, cfg.modulus, rng,
            device=cfg.device, digit_ids=digit_ids, task=cfg.task,
        )
        out = compute_loss(model, batch, cfg, step=step)
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in model.trainable_param_groups() for p in g["params"]], cfg.grad_clip
        )
        opt.step()
        sched.step()

        run_acc = 0.98 * run_acc + 0.02 * out["acc"].item()
        steps_at_level += 1
        if cfg.curriculum and hops == sorted_hops[level]:
            level_acc = 0.9 * level_acc + 0.1 * out["acc"].item()

        # Advance the curriculum when competent at the current deepest level (or stuck too long).
        if cfg.curriculum and level < len(sorted_hops) - 1:
            competent = level_acc >= cfg.curriculum_threshold and steps_at_level >= cfg.curriculum_min_steps
            stuck = steps_at_level >= cfg.curriculum_patience_steps
            if competent or stuck:
                if verbose:
                    why = "competent" if competent else "patience"
                    print(f"  >> advance hops {sorted_hops[level]}->{sorted_hops[level+1]} "
                          f"at step {step+1} ({why}, level_acc={level_acc:.2f})")
                level += 1
                steps_at_level = 0
                level_acc = 0.0

        if verbose and (step + 1) % cfg.log_every == 0:
            extra = f" | maxhop={sorted_hops[level]} lvl_acc={level_acc:.2f}" if cfg.curriculum else ""
            print(
                f"  step {step+1:4d}/{cfg.train_steps} | hops={hops} "
                f"| loss={out['total'].item():.3f} ans={out['ans'].item():.3f} "
                f"proc={out['proc'].item():.3f} | ema_acc={run_acc:.3f}{extra}"
            )
    model.eval()
    return model
