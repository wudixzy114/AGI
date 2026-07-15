"""Mod-N multi-hop arithmetic chains — a task whose *process* has exact ground truth.

A chain looks like:  x0=3, then +4, -7, +2 (each mod 10)  ->  answer = ((3+4-7+2) % 10) = 2
rendered as the prompt string:  "3 +4 -7 +2 ="   (answer token follows)

Why this task:
  * Every intermediate value x_1..x_K is known exactly -> we can supervise or probe
    the "process", not just the final answer.
  * All values are single digits (mod 10) -> single-token answers, clean 10-class probe.
  * Solving hop K genuinely requires chaining K sequential updates; there is no shortcut,
    so more hops = harder, which is what lets latent "thinking steps" show their value.
  * With single-digit start + single-digit constants, every prompt of a given hop-count
    tokenizes to the SAME length -> uniform batches, no padding, clean latent append.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Example:
    prompt: str
    x0: int
    ops: List[int]          # signed deltas, e.g. [+4, -7, +2]
    intermediates: List[int]  # x_1..x_K  (running mod-N results)
    answer: int             # == intermediates[-1]

    @property
    def hops(self) -> int:
        return len(self.ops)


def make_example(hops: int, modulus: int, rng: random.Random) -> Example:
    x0 = rng.randrange(modulus)
    ops: List[int] = []
    inter: List[int] = []
    x = x0
    for _ in range(hops):
        c = rng.randrange(1, min(10, modulus))   # nonzero single-digit delta
        sign = rng.choice((1, -1))
        delta = sign * c
        x = (x + delta) % modulus
        ops.append(delta)
        inter.append(x)
    parts = [str(x0)] + [f"{'+' if d >= 0 else '-'}{abs(d)}" for d in ops] + ["="]
    prompt = " ".join(parts)
    return Example(prompt=prompt, x0=x0, ops=ops, intermediates=inter, answer=x)


# --- Second task family: multi-hop permutation (lookup) ---
# A fixed set of single-digit unary maps g0..g{M-1} (each a permutation of 0..N-1). The prompt
# picks a start digit and a sequence of map-ids; the model must apply them in order. Structurally
# different from arithmetic (memorized lookup, not a rule), same clean per-step ground truth.
# Uses letters a,b,c... as map tokens (single-token in Qwen). Deterministic given (modulus, seed).
_PERM_CACHE = {}

def _perm_maps(modulus: int, n_maps: int = 3, seed: int = 12345):
    key = (modulus, n_maps, seed)
    if key not in _PERM_CACHE:
        r = random.Random(seed)
        maps = []
        for _ in range(n_maps):
            p = list(range(modulus)); r.shuffle(p)
            maps.append(p)
        _PERM_CACHE[key] = maps
    return _PERM_CACHE[key]


def make_example_perm(hops: int, modulus: int, rng: random.Random, n_maps: int = 3) -> Example:
    maps = _perm_maps(modulus, n_maps)
    letters = "abcdefgh"[:n_maps]
    x0 = rng.randrange(modulus)
    ops: List[int] = []       # here ops store the map-id used at each step
    inter: List[int] = []
    x = x0
    for _ in range(hops):
        m = rng.randrange(n_maps)
        x = maps[m][x]
        ops.append(m)
        inter.append(x)
    parts = [str(x0)] + [letters[m] for m in ops] + ["="]
    prompt = " ".join(parts)
    return Example(prompt=prompt, x0=x0, ops=ops, intermediates=inter, answer=x)


def digit_token_ids(tokenizer, modulus: int) -> List[int]:
    """Map each digit 0..modulus-1 to its single token id (asserts single-token)."""
    ids = []
    for d in range(modulus):
        enc = tokenizer.encode(str(d), add_special_tokens=False)
        assert len(enc) == 1, f"digit {d} is not single-token: {enc}"
        ids.append(enc[0])
    return ids


# --- Project 3: session task — linked problems that share a latent memory ---
# A session is T problems solved in order, sharing M memory slots. Problem t either:
#   * starts from a literal digit  "d (±c)* ="   -> answer written to slot t, OR
#   * REFERENCES an earlier slot   "@k (±c)* ="  -> answer = (slot_k value (±c)*) mod N.
# The referenced value is NOT in the current problem's text, so a correct answer to a
# reference problem REQUIRES reading it from memory (context-recompute cannot do it).
# We track, per problem: the answer, and the value each slot should hold after it.
@dataclass
class SessionProblem:
    prompt: str
    is_ref: bool
    ref_slot: int          # which slot it reads (-1 if literal-start)
    ops: List[int]
    answer: int            # single digit, also the value written to this problem's slot
    write_slot: int        # slot index this problem's answer is written to


@dataclass
class Session:
    problems: List[SessionProblem]
    slot_after: List[List[int]]   # slot_after[t][s] = value in slot s AFTER problem t (or -1 if empty)
    n_slots: int

    @property
    def length(self) -> int:
        return len(self.problems)


def make_session(session_len: int, n_slots: int, modulus: int, rng: random.Random,
                 hops: int = 2, ref_prob: float = 0.5, ref_token: str = "R") -> Session:
    """Generate one session. Problem t writes its answer to slot t (t < n_slots).
    A reference problem reads a uniformly-random earlier written slot."""
    problems: List[SessionProblem] = []
    slots = [-1] * n_slots          # current slot values (-1 = empty)
    slot_after: List[List[int]] = []
    for t in range(session_len):
        written = [s for s in range(t) if s < n_slots and slots[s] >= 0]
        is_ref = bool(written) and rng.random() < ref_prob
        ops = []
        if is_ref:
            k = rng.choice(written)
            x = slots[k]
            parts = [f"{ref_token}{k}"]
        else:
            k = -1
            x = rng.randrange(modulus)
            parts = [str(x)]
        for _ in range(hops):
            c = rng.randrange(1, min(10, modulus))
            sign = rng.choice((1, -1))
            x = (x + sign * c) % modulus
            ops.append(sign * c)
            parts.append(f"{'+' if sign > 0 else '-'}{c}")
        parts.append("=")
        wslot = t if t < n_slots else (t % n_slots)
        problems.append(SessionProblem(prompt=" ".join(parts), is_ref=is_ref, ref_slot=k,
                                       ops=ops, answer=x, write_slot=wslot))
        slots[wslot] = x
        slot_after.append(list(slots))
    return Session(problems=problems, slot_after=slot_after, n_slots=n_slots)


def session_ref_token_id(tokenizer, ref_token: str = "R") -> int:
    enc = tokenizer.encode(ref_token, add_special_tokens=False)
    return enc[0]  # may be multi-token in theory; R is single-token in Qwen


@dataclass
class Batch:
    input_ids: torch.Tensor        # (B, L)  uniform length, no padding
    attention_mask: torch.Tensor   # (B, L)  all ones
    answer_ids: torch.Tensor       # (B,) token id of the answer digit
    answer_vals: torch.Tensor      # (B,) integer answer 0..N-1
    inter_vals: torch.Tensor       # (B, K) integer intermediates x_1..x_K
    hops: int
    examples: List[Example]


def make_batch(
    tokenizer,
    batch_size: int,
    hops: int,
    modulus: int,
    rng: random.Random,
    device: str = "cpu",
    digit_ids: Optional[List[int]] = None,
    task: str = "arith",
) -> Batch:
    if digit_ids is None:
        digit_ids = digit_token_ids(tokenizer, modulus)
    gen = make_example_perm if task == "perm" else make_example
    exs = [gen(hops, modulus, rng) for _ in range(batch_size)]

    enc = [tokenizer.encode(e.prompt, add_special_tokens=False) for e in exs]
    lengths = {len(x) for x in enc}
    # Uniform hop-count + single-digit values should give identical lengths.
    # If a tokenizer merges some "+N" oddly, fall back to left-trim to min length
    # (keeps the trailing "=" aligned, which is what the latent loop appends after).
    if len(lengths) != 1:
        L = min(lengths)
        enc = [x[-L:] for x in enc]

    input_ids = torch.tensor(enc, dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    answer_ids = torch.tensor([digit_ids[e.answer] for e in exs], dtype=torch.long, device=device)
    answer_vals = torch.tensor([e.answer for e in exs], dtype=torch.long, device=device)
    inter_vals = torch.tensor([e.intermediates for e in exs], dtype=torch.long, device=device)
    return Batch(
        input_ids=input_ids,
        attention_mask=attn,
        answer_ids=answer_ids,
        answer_vals=answer_vals,
        inter_vals=inter_vals,
        hops=hops,
        examples=exs,
    )


@dataclass
class SessionBatch:
    """B parallel sessions. For each problem position t (0..T-1), a left-padded tensor view
    across the batch, plus per-position labels for answer / write-slot / ref-slot / slot-truth."""
    sessions: List[Session]
    T: int
    n_slots: int
    # per position t: dict with input_ids (B,Lt), attention_mask (B,Lt), answer_vals (B,),
    #                 answer_ids (B,), write_slot (B,), is_ref (B,), ref_slot (B,),
    #                 slot_truth (B, n_slots)  [value each slot SHOULD hold after problem t; -1 empty]
    positions: List[dict]


def make_session_batch(tokenizer, batch_size: int, cfg, rng: random.Random,
                       device: str = "cpu", digit_ids: Optional[List[int]] = None) -> SessionBatch:
    if digit_ids is None:
        digit_ids = digit_token_ids(tokenizer, cfg.modulus)
    sess = [make_session(cfg.session_len, cfg.n_slots, cfg.modulus, rng,
                         hops=cfg.session_hops, ref_prob=cfg.ref_prob) for _ in range(batch_size)]
    T = cfg.session_len
    positions = []
    for t in range(T):
        encs = [tokenizer.encode(s.problems[t].prompt, add_special_tokens=False) for s in sess]
        L = max(len(e) for e in encs)
        # left-pad (pad id 0); mask marks real tokens. Trailing "=" stays rightmost so the
        # latent loop / answer decode reads the correct final position.
        ids, masks = [], []
        for e in encs:
            pad = L - len(e)
            ids.append([0] * pad + e)
            masks.append([0] * pad + [1] * len(e))
        pos = {
            "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
            "answer_vals": torch.tensor([s.problems[t].answer for s in sess], dtype=torch.long, device=device),
            "answer_ids": torch.tensor([digit_ids[s.problems[t].answer] for s in sess], dtype=torch.long, device=device),
            "write_slot": torch.tensor([s.problems[t].write_slot for s in sess], dtype=torch.long, device=device),
            "is_ref": torch.tensor([1 if s.problems[t].is_ref else 0 for s in sess], dtype=torch.long, device=device),
            "ref_slot": torch.tensor([s.problems[t].ref_slot for s in sess], dtype=torch.long, device=device),
            "slot_truth": torch.tensor([s.slot_after[t] for s in sess], dtype=torch.long, device=device),
        }
        positions.append(pos)
    return SessionBatch(sessions=sess, T=T, n_slots=cfg.n_slots, positions=positions)


if __name__ == "__main__":
    rng = random.Random(0)
    for h in (1, 3, 5):
        e = make_example(h, 10, rng)
        print(f"hops={h}: {e.prompt:<28} inter={e.intermediates} answer={e.answer}")
    print("\n--- session ---")
    s = make_session(6, 8, 10, rng, hops=2, ref_prob=0.6)
    for t, p in enumerate(s.problems):
        print(f"  t={t} {'REF' if p.is_ref else 'LIT'} {p.prompt:<16} ans={p.answer} ->slot{p.write_slot} | slots={s.slot_after[t]}")

