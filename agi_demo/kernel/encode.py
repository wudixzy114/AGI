"""Token-stream encoders for the kernel — a tiny symbol vocabulary, no pretrained tokenizer.

Two streams, both built from the *structured* generators in agi_demo/task.py (we use the fields,
never parse the prompt string):

  arith  : [BOS, start, (sign,mag)*hops, EQ]                 -> predict answer at EQ
  session: [BOS] + per-problem[ SLOT_w, HEAD(2), ops, EQ, ans ]  (one stream per session)

The session stream is the whole point of Project 4: the T linked problems share ONE scan, so the
"memory" that a reference problem must read is the kernel's own recurrent fast-weight state — not a
bolted-on module. Because a correct reference answer needs a value that is NOT in the current
problem's tokens (only in memory), ref accuracy directly measures memory USE.

Uniform length by construction: every problem is [SLOT_w] + head(2) + ops(2*hops) + [EQ] + [ans],
where head is [LIT,start] or [REF,SLOT_ref] — both 2 tokens. So a batch of same-config sessions
tokenizes to identical length with NO padding (padding would pollute the recurrent scan).

The answer token after EQ is TEACHER-FORCED (the ground-truth digit) in both train and eval. This
isolates the scientific question — "can the kernel USE a correctly-stored value as an operand?" —
from error accumulation (a wrong early answer corrupting a slot for later references). Noted as a
deliberate choice in FINDINGS.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import torch

from ..task import make_example, make_example_perm, make_session


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class Vocab:
    """Compact symbol vocab. Layout: PAD, digits 0..N-1, PLUS, MINUS, EQ, REF, LIT, BOS, SLOT_0..."""

    def __init__(self, modulus: int, n_slots: int):
        self.modulus = modulus
        self.n_slots = n_slots
        self.PAD = 0
        # digits 0..N-1 -> 1..N
        self._digit0 = 1
        self.PLUS = 1 + modulus
        self.MINUS = 2 + modulus
        self.EQ = 3 + modulus
        self.REF = 4 + modulus
        self.LIT = 5 + modulus
        self.BOS = 6 + modulus
        self._slot0 = 7 + modulus
        self.size = 7 + modulus + n_slots

    def digit(self, d: int) -> int:
        assert 0 <= d < self.modulus
        return self._digit0 + d

    def slot(self, s: int) -> int:
        assert 0 <= s < self.n_slots, f"slot {s} out of range (n_slots={self.n_slots})"
        return self._slot0 + s

    def digit_ids(self) -> List[int]:
        return [self.digit(d) for d in range(self.modulus)]


def _encode_ops(vocab: Vocab, ops: List[int]) -> List[int]:
    toks: List[int] = []
    for d in ops:  # signed delta
        toks.append(vocab.PLUS if d >= 0 else vocab.MINUS)
        toks.append(vocab.digit(abs(d)))
    return toks


# ---------------------------------------------------------------------------
# Arith stream (mod-N chain) — single hop-count per batch, uniform length
# ---------------------------------------------------------------------------
@dataclass
class ArithBatch:
    input_ids: torch.Tensor      # (B, L)
    eq_pos: int                  # readout position (predict answer at EQ)
    inter_pos: List[int]         # positions of each op's magnitude digit (running sum x_i decided)
    answer_vals: torch.Tensor    # (B,)
    inter_vals: torch.Tensor     # (B, hops)
    hops: int


def make_arith_batch(vocab: Vocab, batch_size: int, hops: int, modulus: int,
                     rng: random.Random, device: str = "cpu", task: str = "arith") -> ArithBatch:
    gen = make_example_perm if task == "perm" else make_example
    exs = [gen(hops, modulus, rng) for _ in range(batch_size)]
    ids = []
    inter_pos: List[int] = []
    for j, e in enumerate(exs):
        toks = [vocab.BOS, vocab.digit(e.x0)]
        ip = []
        for d in e.ops:
            toks.append(vocab.PLUS if d >= 0 else vocab.MINUS)
            toks.append(vocab.digit(abs(d)))
            ip.append(len(toks) - 1)          # position of the magnitude digit just appended
        eq_pos = len(toks)
        toks.append(vocab.EQ)
        ids.append(toks)
        if j == 0:
            inter_pos = ip                    # identical across the batch (uniform structure)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    answer_vals = torch.tensor([e.answer for e in exs], dtype=torch.long, device=device)
    inter_vals = torch.tensor([e.intermediates for e in exs], dtype=torch.long, device=device)
    return ArithBatch(input_ids=input_ids, eq_pos=eq_pos, inter_pos=inter_pos,
                      answer_vals=answer_vals, inter_vals=inter_vals, hops=hops)


# ---------------------------------------------------------------------------
# Session stream — the memory-USE test. One stream per session, uniform length.
# ---------------------------------------------------------------------------
@dataclass
class SessionStreamBatch:
    input_ids: torch.Tensor      # (B, L)
    # per problem t (len session_len): tensors over the batch
    eq_pos: List[int]            # readout position of problem t (predict its answer)
    ans_pos: List[int]           # position of the teacher-forced answer token (= write position)
    refslot_pos: List[int]       # position of the SLOT token after REF (query address), -1 if literal
    answer_vals: torch.Tensor    # (B, T)
    write_slot: torch.Tensor     # (B, T)
    is_ref: torch.Tensor         # (B, T) bool
    ref_slot: torch.Tensor       # (B, T) int (-1 literal)
    slot_truth_at_ref: torch.Tensor  # (B, T) latest stored value of the referenced slot when read
                                     # (-1 at literal positions). Correct read-sup/probe target under
                                     # overwrites; equals answer_vals[ref_slot] in the 1:1 regime.
    T: int
    n_slots: int
    sessions: list = None            # the underlying Session objects (for eval/probe reconstruction)


def make_session_stream_batch(vocab: Vocab, batch_size: int, cfg, rng: random.Random,
                              device: str = "cpu") -> SessionStreamBatch:
    modulus = cfg.modulus
    sessions = [make_session(cfg.session_len, cfg.n_slots, modulus, rng,
                             hops=cfg.session_hops, ref_prob=cfg.ref_prob)
                for _ in range(batch_size)]
    T = cfg.session_len
    ids: List[List[int]] = []
    eq_pos = ans_pos = refslot_pos = None
    ans_vals, wslots, isref, refslots = [], [], [], []
    struth = []
    for sess in sessions:
        toks: List[int] = [vocab.BOS]
        ep, ap, rp = [], [], []
        for p in sess.problems:
            toks.append(vocab.slot(p.write_slot))
            if p.is_ref:
                toks.append(vocab.REF)
                toks.append(vocab.slot(p.ref_slot))
            else:
                start = (p.answer - sum(p.ops)) % modulus     # recover literal start digit
                toks.append(vocab.LIT)
                toks.append(vocab.digit(start))
            # head-argument token (REF's slot, or LIT's digit) sits at THIS fixed layout position
            # for every problem — record it unconditionally; is_ref gates its use downstream.
            rp.append(len(toks) - 1)
            toks.extend(_encode_ops(vocab, p.ops))
            ep.append(len(toks))
            toks.append(vocab.EQ)
            ap.append(len(toks))
            toks.append(vocab.digit(p.answer))
        ids.append(toks)
        if eq_pos is None:
            eq_pos, ans_pos, refslot_pos = ep, ap, rp     # identical layout across the batch
        ans_vals.append([p.answer for p in sess.problems])
        wslots.append([p.write_slot for p in sess.problems])
        isref.append([1 if p.is_ref else 0 for p in sess.problems])
        refslots.append([p.ref_slot for p in sess.problems])
        # latest stored value of the referenced slot AT THE MOMENT it is read = slot content after
        # the previous problem (slot_after[t-1][ref_slot]); -1 for literals or t=0. Correct under
        # overwrites, where the writer is not simply problem #ref_slot.
        row = []
        for t, p in enumerate(sess.problems):
            if p.is_ref and t > 0:
                row.append(sess.slot_after[t - 1][p.ref_slot])
            else:
                row.append(-1)
        struth.append(row)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    return SessionStreamBatch(
        input_ids=input_ids, eq_pos=eq_pos, ans_pos=ans_pos, refslot_pos=refslot_pos,
        answer_vals=torch.tensor(ans_vals, dtype=torch.long, device=device),
        write_slot=torch.tensor(wslots, dtype=torch.long, device=device),
        is_ref=torch.tensor(isref, dtype=torch.long, device=device),
        ref_slot=torch.tensor(refslots, dtype=torch.long, device=device),
        slot_truth_at_ref=torch.tensor(struth, dtype=torch.long, device=device),
        T=T, n_slots=cfg.n_slots, sessions=sessions)


if __name__ == "__main__":
    v = Vocab(10, 6)
    print("vocab size", v.size)
    rng = random.Random(0)
    b = make_arith_batch(v, 4, 3, 10, rng)
    print("arith ids", b.input_ids[0].tolist(), "eq_pos", b.eq_pos, "inter_pos", b.inter_pos)

    class _C:
        modulus = 10; n_slots = 6; session_len = 6; ref_prob = 0.6; session_hops = 2
    sb = make_session_stream_batch(v, 2, _C, rng)
    print("session L", sb.input_ids.shape, "eq_pos", sb.eq_pos, "ans_pos", sb.ans_pos,
          "refslot_pos", sb.refslot_pos)
    print("is_ref[0]", sb.is_ref[0].tolist(), "ref_slot[0]", sb.ref_slot[0].tolist())
