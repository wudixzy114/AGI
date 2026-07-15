"""Training — from-scratch kernel, answer-CE is always the grounding anchor.

No latent loop, no injected vectors: the kernel consumes the whole token stream in ONE recurrent
scan, and "thinking"/"memory" are just its evolving fast-weight state. That is the point of a
non-Transformer core — the computation and the memory are the same object.

Arms:
  K0  read-severed control: cells write S but never read it (read_enabled=False). Grounding anchor
      only. Prediction: reference (memory-USE) accuracy collapses to chance.
  K1  full kernel, answer-CE only. Tests whether USE EMERGES from result supervision alone.
  K2  + grounded read supervision: at a reference problem's slot-query token, the slowest layer's
      memory read must decode (via mem_head) the referenced slot's stored value. This is the
      analogue of Project 3's read-probe, but the read now feeds the kernel's own computation.

Arith task (secondary): answer-CE at "=", optional process CE (arm-C style) asking the hidden state
at each op position to decode the running intermediate — reusing the process>result idea.
"""
from __future__ import annotations

import random
from typing import Dict, List

import torch
import torch.nn.functional as F

from .kconfig import KernelConfig
from .cell import KernelModel
from .encode import Vocab, make_arith_batch, make_session_stream_batch


def build_model(cfg: KernelConfig) -> KernelModel:
    torch.manual_seed(cfg.seed)
    vocab = Vocab(cfg.modulus, cfg.n_slots)
    return KernelModel(cfg, vocab)


# ---------------------------------------------------------------------------
# Arith
# ---------------------------------------------------------------------------
def arith_loss(model: KernelModel, batch, cfg: KernelConfig, step: int) -> Dict:
    hidden, _, _ = model(batch.input_ids)
    ans_logits = model.answer_logits_at(hidden, batch.eq_pos)         # (B,N)
    loss = F.cross_entropy(ans_logits, batch.answer_vals)
    proc = torch.zeros((), device=hidden.device)
    if cfg.process_loss_weight > 0 and batch.inter_pos:
        # hidden at each op-magnitude position should decode the running intermediate x_i
        n = 0
        for i, p in enumerate(batch.inter_pos):
            logits_i = model.readout(hidden[:, p])
            proc = proc + F.cross_entropy(logits_i, batch.inter_vals[:, i]); n += 1
        proc = proc / max(1, n)
        ramp = min(1.0, step / max(1, cfg.process_warmup_steps))
        loss = loss + cfg.process_loss_weight * ramp * proc
    with torch.no_grad():
        acc = (ans_logits.argmax(-1) == batch.answer_vals).float().mean()
    return {"total": loss, "acc": acc, "proc": proc.detach()}


# ---------------------------------------------------------------------------
# Session (the memory-USE test)
# ---------------------------------------------------------------------------
def session_loss(model: KernelModel, sb, cfg: KernelConfig) -> Dict:
    # one scan over the whole session stream; collect state only if we need read supervision
    hidden, reads, _ = model(sb.input_ids, collect_state=False)
    B = sb.input_ids.shape[0]
    dev = sb.input_ids.device
    total = torch.zeros((), device=dev)
    n_correct = n_tot = ref_c = ref_n = lit_c = lit_n = 0
    slow_reads = reads[-1]                        # (B,L,s) slowest layer = session-long store

    read_sup = torch.zeros((), device=dev)
    n_read_sup = 0
    for t in range(sb.T):
        eq = sb.eq_pos[t]
        logits = model.answer_logits_at(hidden, eq)               # (B,N)
        total = total + F.cross_entropy(logits, sb.answer_vals[:, t])
        # K2 grounded read supervision: at reference problems, the memory read at the slot-query
        # token must decode the referenced value (= this problem's answer, since a reference reads
        # a slot then applies its own ops; the STORED value is what was written there earlier).
        if cfg.arm == "K2":
            rp = sb.refslot_pos[t]
            if rp >= 0:
                ref_mask = sb.is_ref[:, t].bool()
                if ref_mask.any():
                    read_vec = slow_reads[:, rp][ref_mask]        # (n_ref, s)
                    # referenced slot's CURRENT stored value (latest writer) — correct under
                    # overwrites, where the writer is not simply problem #ref_slot.
                    stored_val = sb.slot_truth_at_ref[:, t][ref_mask]
                    read_sup = read_sup + F.cross_entropy(model.mem_head(read_vec), stored_val)
                    n_read_sup += 1
        with torch.no_grad():
            pred = logits.argmax(-1)
            corr = (pred == sb.answer_vals[:, t])
            n_correct += corr.sum().item(); n_tot += B
            ref = sb.is_ref[:, t].bool()
            ref_c += corr[ref].sum().item(); ref_n += ref.sum().item()
            lit_c += corr[~ref].sum().item(); lit_n += (~ref).sum().item()

    total = total / sb.T
    if cfg.arm == "K2" and n_read_sup:
        total = total + cfg.mem_write_weight * (read_sup / n_read_sup)
    return {"total": total, "acc": n_correct / max(1, n_tot),
            "ref_acc": ref_c / max(1, ref_n), "lit_acc": lit_c / max(1, lit_n),
            "read_sup": (read_sup / max(1, n_read_sup)).detach()}


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------
def _opt_sched(model, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps)))
    return opt, sched


def train_session(cfg: KernelConfig, verbose: bool = True) -> KernelModel:
    rng = random.Random(cfg.seed)
    model = build_model(cfg); model.train()
    opt, sched = _opt_sched(model, cfg)
    if verbose:
        print(f"[arm {cfg.arm}] trainable={model.num_trainable():,} device={cfg.device} "
              f"decays={cfg.decays} read={'ON' if model.read_enabled else 'SEVERED'} | session")
    ref_ema = 0.0
    for step in range(cfg.train_steps):
        sb = make_session_stream_batch(model.vocab, cfg.batch_size, cfg, rng, device=cfg.device)
        out = session_loss(model, sb, cfg)
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        ref_ema = 0.95 * ref_ema + 0.05 * out["ref_acc"]
        if verbose and (step + 1) % cfg.log_every == 0:
            print(f"  step {step+1:4d}/{cfg.train_steps} loss={out['total'].item():.3f} "
                  f"acc={out['acc']:.3f} ref={out['ref_acc']:.3f} lit={out['lit_acc']:.3f} "
                  f"read_sup={out['read_sup'].item():.3f} | ref_ema={ref_ema:.3f}")
    model.eval()
    return model


def train_arith(cfg: KernelConfig, verbose: bool = True) -> KernelModel:
    rng = random.Random(cfg.seed)
    model = build_model(cfg); model.train()
    opt, sched = _opt_sched(model, cfg)
    if verbose:
        mode = "curriculum" if cfg.curriculum else "uniform"
        print(f"[arm {cfg.arm}] trainable={model.num_trainable():,} device={cfg.device} "
              f"decays={cfg.decays} | arith {mode}")
    sorted_hops = sorted(cfg.train_hops)
    level = 0 if cfg.curriculum else len(sorted_hops) - 1
    steps_at_level = 0; level_acc = 0.0; run_acc = 0.0
    for step in range(cfg.train_steps):
        if cfg.curriculum:
            unlocked = sorted_hops[: level + 1]
            hops = sorted_hops[level] if rng.random() < 0.5 else rng.choice(unlocked)
        else:
            hops = rng.choice(sorted_hops)
        batch = make_arith_batch(model.vocab, cfg.batch_size, hops, cfg.modulus, rng,
                                 device=cfg.device, task=cfg.task)
        out = arith_loss(model, batch, cfg, step)
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        run_acc = 0.98 * run_acc + 0.02 * out["acc"].item()
        steps_at_level += 1
        if cfg.curriculum and hops == sorted_hops[level]:
            level_acc = 0.9 * level_acc + 0.1 * out["acc"].item()
        if cfg.curriculum and level < len(sorted_hops) - 1:
            competent = level_acc >= cfg.curriculum_threshold and steps_at_level >= cfg.curriculum_min_steps
            stuck = steps_at_level >= cfg.curriculum_patience_steps
            if competent or stuck:
                if verbose:
                    print(f"  >> advance hops {sorted_hops[level]}->{sorted_hops[level+1]} "
                          f"at step {step+1} ({'competent' if competent else 'patience'}, lvl_acc={level_acc:.2f})")
                level += 1; steps_at_level = 0; level_acc = 0.0
        if verbose and (step + 1) % cfg.log_every == 0:
            extra = f" maxhop={sorted_hops[level]} lvl={level_acc:.2f}" if cfg.curriculum else ""
            print(f"  step {step+1:4d}/{cfg.train_steps} hops={hops} loss={out['total'].item():.3f} "
                  f"proc={out['proc'].item():.3f} | ema_acc={run_acc:.3f}{extra}")
    model.eval()
    return model
