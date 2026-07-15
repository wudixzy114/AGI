"""Evaluation + the linear probe — the "soul" of the experiment.

Two measurements:
  1) answer accuracy vs hop-count (exact match on the digit), per arm.
  2) linear probe: freeze the model, collect thought vector i for many examples of a
     fixed hop-count, fit a logistic regression to predict the true intermediate x_{i+1}.
     High per-step probe accuracy == the model linearly encodes the step-i result in its
     continuous thought, i.e. it is doing multi-step reasoning in latent space.

     The killer result: if arm B (never told the intermediates) is probeable, the
     step structure EMERGED on its own from only answer supervision.
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch

from .config import Config
from .model import CoconutReasoner
from .task import make_batch, digit_token_ids, make_session_batch
from .train import n_latent_for


@torch.no_grad()
def accuracy_by_hops(model: CoconutReasoner, cfg: Config, rng: random.Random) -> Dict[int, float]:
    model.eval()
    digit_ids = digit_token_ids(model.tokenizer, cfg.modulus)
    accs: Dict[int, float] = {}
    for hops in cfg.eval_hops:
        n_lat = n_latent_for(cfg, hops)
        correct = total = 0
        remaining = cfg.eval_examples_per_hop
        while remaining > 0:
            bs = min(cfg.batch_size, remaining)
            batch = make_batch(model.tokenizer, bs, hops, cfg.modulus, rng,
                               device=cfg.device, digit_ids=digit_ids, task=cfg.task)
            logits, _ = model(batch.input_ids, batch.attention_mask, n_lat)
            pred = model.digit_logits(logits).argmax(-1)
            correct += (pred == batch.answer_vals).sum().item()
            total += bs
            remaining -= bs
        accs[hops] = correct / total
    return accs


@torch.no_grad()
def collect_thoughts(model: CoconutReasoner, cfg: Config, hops: int, rng: random.Random):
    """Returns thoughts (n_steps, N_examples, H) and intermediates (N_examples, hops)."""
    model.eval()
    digit_ids = digit_token_ids(model.tokenizer, cfg.modulus)
    per_step: List[List[np.ndarray]] = [[] for _ in range(hops)]
    inters: List[np.ndarray] = []
    remaining = cfg.probe_examples_per_hop
    while remaining > 0:
        bs = min(cfg.batch_size, remaining)
        batch = make_batch(model.tokenizer, bs, hops, cfg.modulus, rng,
                           device=cfg.device, digit_ids=digit_ids, task=cfg.task)
        _, thoughts = model(batch.input_ids, batch.attention_mask, hops)
        for i, t in enumerate(thoughts):
            per_step[i].append(t.float().cpu().numpy())
        inters.append(batch.inter_vals.cpu().numpy())
        remaining -= bs
    thoughts_arr = np.stack([np.concatenate(step, axis=0) for step in per_step], axis=0)
    inter_arr = np.concatenate(inters, axis=0)
    return thoughts_arr, inter_arr


def linear_probe(model: CoconutReasoner, cfg: Config, hops: int, rng: random.Random) -> Dict:
    """Per-step probe accuracy at a given hop-count. Returns dict with per-step + chance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    thoughts, inter = collect_thoughts(model, cfg, hops, rng)  # (S,N,H),(N,hops)
    n = inter.shape[0]
    n_test = max(1, int(n * cfg.probe_test_frac))
    idx = np.arange(n)
    np.random.RandomState(cfg.seed).shuffle(idx)
    tr, te = idx[n_test:], idx[:n_test]

    per_step_acc = []
    for i in range(hops):
        X = thoughts[i]
        y = inter[:, i]
        sc = StandardScaler().fit(X[tr])
        # Faster solver: 2048-dim features -> lbfgs at max_iter=2000 pins a CPU core for a long
        # time. Cap iterations (probe accuracy is essentially unchanged) so the CPU-bound probe
        # phase doesn't stall the GPU pipeline between configs. n_jobs uses all cores.
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        acc = float(clf.score(sc.transform(X[te]), y[te]))
        per_step_acc.append(acc)
    return {"hops": hops, "per_step_acc": per_step_acc, "chance": 1.0 / cfg.modulus}


# ---- Project 3: session evaluation + memory-slot probe ----
@torch.no_grad()
def eval_session(model: CoconutReasoner, cfg: Config, rng: random.Random, n_sessions: int = 512) -> Dict:
    """Accuracy split by literal vs reference problems (reference = requires memory read),
    plus write→read delay: reference accuracy binned by how many problems since the slot was written."""
    from .train import session_loss
    model.eval()
    digit_ids = digit_token_ids(model.tokenizer, cfg.modulus)
    n_lat = 0 if cfg.arm == "M0" else cfg.session_hops
    lit_c = lit_n = ref_c = ref_n = 0
    delay_c: Dict[int, int] = {}; delay_n: Dict[int, int] = {}
    remaining = n_sessions
    while remaining > 0:
        bs = min(cfg.batch_size, remaining)
        sb = make_session_batch(model.tokenizer, bs, cfg, rng, device=cfg.device, digit_ids=digit_ids)
        if model.memory is not None:
            model.memory.reset(bs, cfg.device, next(model.parameters()).dtype)
        for t in range(sb.T):
            pos = sb.positions[t]
            logits, _, ans_hidden, _ = model.forward_problem(pos["input_ids"], pos["attention_mask"], n_lat)
            pred = model.digit_logits(logits).argmax(-1)
            if model.memory is not None:
                model.memory.write(ans_hidden, pos["write_slot"])
            corr = (pred == pos["answer_vals"]).cpu()
            ref = pos["is_ref"].bool().cpu(); refslot = pos["ref_slot"].cpu()
            lit_c += corr[~ref].sum().item(); lit_n += (~ref).sum().item()
            ref_c += corr[ref].sum().item(); ref_n += ref.sum().item()
            # delay = t - (problem index that wrote the referenced slot) = t - ref_slot (slot k written at problem k)
            for b in range(bs):
                if ref[b]:
                    d = t - int(refslot[b])
                    delay_c[d] = delay_c.get(d, 0) + int(corr[b]); delay_n[d] = delay_n.get(d, 0) + 1
        remaining -= bs
    delay_acc = {d: delay_c[d] / delay_n[d] for d in sorted(delay_n)}
    return {"lit_acc": lit_c / max(1, lit_n), "ref_acc": ref_c / max(1, ref_n),
            "chance": 1.0 / cfg.modulus, "delay_acc": delay_acc}


@torch.no_grad()
def _collect_slots(model, cfg, rng, n_sessions):
    """Run sessions; after each problem, snapshot the just-written slot vector + its ground-truth value."""
    digit_ids = digit_token_ids(model.tokenizer, cfg.modulus)
    n_lat = 0 if cfg.arm == "M0" else cfg.session_hops
    X, y = [], []
    remaining = n_sessions
    while remaining > 0:
        bs = min(cfg.batch_size, remaining)
        sb = make_session_batch(model.tokenizer, bs, cfg, rng, device=cfg.device, digit_ids=digit_ids)
        model.memory.reset(bs, cfg.device, next(model.parameters()).dtype)
        import torch as _t
        for t in range(sb.T):
            pos = sb.positions[t]
            _, _, ans_hidden, _ = model.forward_problem(pos["input_ids"], pos["attention_mask"], n_lat)
            model.memory.write(ans_hidden, pos["write_slot"])
            slot_vec = model.memory.slots[_t.arange(bs, device=cfg.device), pos["write_slot"]]
            X.append(slot_vec.float().cpu().numpy()); y.append(pos["answer_vals"].cpu().numpy())
        remaining -= bs
    return np.concatenate(X, 0), np.concatenate(y, 0)


def probe_memory(model: CoconutReasoner, cfg: Config, rng: random.Random, n_sessions: int = 400) -> Dict:
    """Can a linear probe decode a slot's ground-truth value from its stored vector?
    This is the OBSERVABILITY red line — if memory holds the value, the probe recovers it."""
    if model.memory is None:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = _collect_slots(model, cfg, rng, n_sessions)
    n = len(y); nt = max(1, int(n * cfg.probe_test_frac))
    idx = np.arange(n); np.random.RandomState(cfg.seed).shuffle(idx)
    tr, te = idx[nt:], idx[:nt]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=200, C=1.0).fit(sc.transform(X[tr]), y[tr])
    return {"slot_probe_acc": float(clf.score(sc.transform(X[te]), y[te])), "chance": 1.0 / cfg.modulus}

