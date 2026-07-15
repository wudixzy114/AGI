"""Evaluation + linear probes — the observability methodology, ported to the kernel.

Three measurements (all sklearn linear probes on FROZEN activations, mirroring agi_demo/probe.py):

  1) eval_session       : reference-vs-literal accuracy (+ delay binning). ref_acc is the headline
                          memory-USE metric — the exact number that was 0.10 (chance) on the frozen
                          Transformer in Project 3.
  2) probe_arith        : per-op-step decodability of the running intermediate from the hidden
                          stream (did the recurrent state do multi-step reasoning?).
  3) probe_timescales   : THE hierarchy figure. For each layer ℓ, at a slot-write moment snapshot
                          that layer's state, then re-read the SAME key after Δ more tokens and ask
                          a linear probe to decode the stored value from the delayed read. Fast
                          layers (γ≈0) lose it quickly; slow layers (γ≈0.99) hold it. Plotting
                          probe-acc vs Δ per layer SHOWS the multi-timescale memory.
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch

from .kconfig import KernelConfig
from .cell import KernelModel
from .encode import Vocab, make_arith_batch, make_session_stream_batch


def _fit_probe(X, y, cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    n = len(y); nt = max(1, int(n * cfg.probe_test_frac))
    idx = np.arange(n); np.random.RandomState(cfg.seed).shuffle(idx)
    tr, te = idx[nt:], idx[:nt]
    sc = StandardScaler().fit(X[tr])
    # n_jobs=1 (NOT -1): the probe runs inside training jobs that are launched together, so they hit
    # the seed-boundary probe phase in lockstep. n_jobs=-1 fork-bombs joblib workers across every job
    # simultaneously → the notebook's process/resource cap culls them, killing jobs between seeds
    # (P5A: 11/12 jobs died at the seed-0→1 boundary; see POSTMORTEM). lbfgs barely parallelizes
    # anyway, so single-process costs almost nothing and is collision-proof.
    clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=1).fit(sc.transform(X[tr]), y[tr])
    return float(clf.score(sc.transform(X[te]), y[te]))


# ---------------------------------------------------------------------------
# 1) session accuracy (headline)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_session(model: KernelModel, cfg: KernelConfig, rng: random.Random) -> Dict:
    model.eval()
    lit_c = lit_n = ref_c = ref_n = 0
    delay_c: Dict[int, int] = {}; delay_n: Dict[int, int] = {}
    remaining = cfg.eval_sessions
    while remaining > 0:
        bs = min(cfg.batch_size, remaining)
        sb = make_session_stream_batch(model.vocab, bs, cfg, rng, device=cfg.device)
        hidden, _, _ = model(sb.input_ids)
        for t in range(sb.T):
            logits = model.answer_logits_at(hidden, sb.eq_pos[t])
            pred = logits.argmax(-1)
            corr = (pred == sb.answer_vals[:, t]).cpu()
            ref = sb.is_ref[:, t].bool().cpu(); refslot = sb.ref_slot[:, t].cpu()
            wslot = sb.write_slot.cpu()
            lit_c += corr[~ref].sum().item(); lit_n += (~ref).sum().item()
            ref_c += corr[ref].sum().item(); ref_n += ref.sum().item()
            for b in range(bs):
                if ref[b]:
                    k = int(refslot[b])
                    last_write = max((tp for tp in range(t) if int(wslot[b, tp]) == k), default=-1)
                    d = t - last_write          # problems since slot k was LAST written (overwrite-safe)
                    delay_c[d] = delay_c.get(d, 0) + int(corr[b]); delay_n[d] = delay_n.get(d, 0) + 1
        remaining -= bs
    delay_acc = {d: delay_c[d] / delay_n[d] for d in sorted(delay_n)}
    return {"lit_acc": lit_c / max(1, lit_n), "ref_acc": ref_c / max(1, ref_n),
            "chance": 1.0 / cfg.modulus, "delay_acc": delay_acc}


# ---------------------------------------------------------------------------
# 2) arith per-step probe
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect_arith(model, cfg, hops, rng):
    perX: List[List[np.ndarray]] = [[] for _ in range(hops)]
    ys: List[np.ndarray] = []
    remaining = cfg.probe_examples
    while remaining > 0:
        bs = min(cfg.batch_size, remaining)
        batch = make_arith_batch(model.vocab, bs, hops, cfg.modulus, rng,
                                 device=cfg.device, task=cfg.task)
        hidden, _, _ = model(batch.input_ids)
        for i, p in enumerate(batch.inter_pos):
            perX[i].append(hidden[:, p].float().cpu().numpy())
        ys.append(batch.inter_vals.cpu().numpy())
        remaining -= bs
    X = [np.concatenate(step, 0) for step in perX]
    y = np.concatenate(ys, 0)
    return X, y


def probe_arith(model: KernelModel, cfg: KernelConfig, hops: int, rng: random.Random) -> Dict:
    X, y = _collect_arith(model, cfg, hops, rng)
    per_step = [_fit_probe(X[i], y[:, i], cfg) for i in range(hops)]
    return {"hops": hops, "per_step_acc": per_step, "chance": 1.0 / cfg.modulus}


@torch.no_grad()
def accuracy_by_hops(model: KernelModel, cfg: KernelConfig, rng: random.Random) -> Dict[int, float]:
    model.eval()
    accs = {}
    for hops in cfg.eval_hops:
        correct = total = 0
        remaining = cfg.eval_examples_per_hop
        while remaining > 0:
            bs = min(cfg.batch_size, remaining)
            batch = make_arith_batch(model.vocab, bs, hops, cfg.modulus, rng,
                                     device=cfg.device, task=cfg.task)
            hidden, _, _ = model(batch.input_ids)
            pred = model.answer_logits_at(hidden, batch.eq_pos).argmax(-1)
            correct += (pred == batch.answer_vals).sum().item(); total += bs
            remaining -= bs
        accs[hops] = correct / total
    return accs


# ---------------------------------------------------------------------------
# 3) multi-timescale hierarchy probe (the key new figure)
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe_timescales(model: KernelModel, cfg: KernelConfig, rng: random.Random,
                     max_delay: int = None) -> Dict:
    """For each layer, decode a slot's stored value from that layer's memory read at increasing
    delay Δ (in problems) after the value was written. Returns per-layer acc-vs-delay curves.

    Method: run sessions; the value written by problem k lives in the state after its answer token.
    We then probe, at every LATER problem t>k that references slot k, the layer's read vector at the
    slot-query token (delay = t-k). We bucket probe examples by (layer, delay) and fit one linear
    probe per bucket to predict the stored value. Fast layers should decay with Δ; slow layers hold.
    """
    model.eval()
    if max_delay is None:
        max_delay = cfg.session_len - 1
    n_layers = cfg.n_layers
    # buckets[layer][delay] -> (list of read vectors, list of stored values)
    buckets = [[([], []) for _ in range(max_delay + 1)] for _ in range(n_layers)]
    remaining = cfg.probe_sessions
    while remaining > 0:
        bs = min(cfg.batch_size, remaining)
        sb = make_session_stream_batch(model.vocab, bs, cfg, rng, device=cfg.device)
        _, reads, _ = model(sb.input_ids, collect_state=False)     # reads[ℓ] = (B,L,s)
        for t in range(sb.T):
            rp = sb.refslot_pos[t]
            if rp < 0:
                continue
            ref = sb.is_ref[:, t].bool()
            if not ref.any():
                continue
            ref_slot_idx = sb.ref_slot[:, t]                       # (B,)
            for b in range(bs):
                if not ref[b]:
                    continue
                k = int(ref_slot_idx[b])
                # delay = problems since slot k was LAST written (correct under overwrites, where
                # slot k is not necessarily written at problem #k). Scan write_slot history < t.
                last_write = max((tp for tp in range(t) if int(sb.write_slot[b, tp]) == k), default=-1)
                if last_write < 0:
                    continue
                d = t - last_write
                if d < 0 or d > max_delay:
                    continue
                stored = int(sb.slot_truth_at_ref[b, t])           # latest stored value of slot k
                for ell in range(n_layers):
                    vec = reads[ell][b, rp].float().cpu().numpy()
                    buckets[ell][d][0].append(vec); buckets[ell][d][1].append(stored)
        remaining -= bs
    curves = []
    for ell in range(n_layers):
        acc_by_delay = {}
        for d in range(max_delay + 1):
            Xs, ys = buckets[ell][d]
            if len(set(ys)) < 2 or len(ys) < 40:      # need class variety + enough samples
                continue
            acc_by_delay[d] = _fit_probe(np.stack(Xs), np.array(ys), cfg)
        curves.append({"gamma": cfg.decays[ell], "acc_by_delay": acc_by_delay})
    return {"chance": 1.0 / cfg.modulus, "layers": curves}
