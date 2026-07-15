"""Configuration for Project 4 — the from-scratch fast-weight kernel.

Unlike agi_demo/config.py, there is NO pretrained model here: this kernel is built and trained
from random init. So the config carries the kernel's own dims (d_model, d_state, n_layers, decays)
plus the shared task knobs. resolve() only picks a device and makes the out dir.

Defaults are tiny (Mac smoke). On the B200, bump d_model/d_state/steps via CLI flags.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_decays(n_layers: int) -> List[float]:
    """A spread of per-layer memory decay rates γ_ℓ, fast (recent-only) → slow (session-long).

    The spread is the whole point of the 'multi-timescale hierarchy': the fast layer (γ≈0) is
    within-problem working memory, the slow layer (γ≈0.99) is the cross-problem store. For odd
    layer counts we interpolate on a log-ish grid between ~0 and ~0.99.
    """
    if n_layers == 1:
        return [0.9]
    anchors = [0.0, 0.7, 0.95, 0.99]
    if n_layers <= len(anchors):
        # take a spread subset preserving the fast/slow endpoints
        idx = [round(i * (len(anchors) - 1) / (n_layers - 1)) for i in range(n_layers)]
        return [anchors[i] for i in idx]
    # more layers than anchors: linear ramp 0 → 0.99
    return [round(0.99 * i / (n_layers - 1), 4) for i in range(n_layers)]


@dataclass
class KernelConfig:
    # --- kernel dims (the from-scratch "变" — everything is trainable, nothing frozen) ---
    d_model: int = 64          # token/hidden width
    d_state: int = 32          # fast-weight matrix is d_state x d_state per layer
    n_layers: int = 3
    decays: List[float] = field(default_factory=list)   # per-layer γ; [] -> default_decays(n_layers)
    ffn_mult: float = 2.0
    delta_rule: bool = True    # True: DeltaNet overwrite; False: additive Hebbian write

    # --- task (reuses agi_demo/task.py generators) ---
    task: str = "session"      # "arith" (mod-N chain) or "session" (linked, the USE test)
    modulus: int = 10
    train_hops: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    eval_hops: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # session task knobs (mirror config.py so make_session works unchanged)
    n_slots: int = 6
    session_len: int = 6
    ref_prob: float = 0.5
    session_hops: int = 2
    # Project 5 Phase A: allow session_len > n_slots so slots are REWRITTEN mid-session (make_session
    # already reuses slot t%n_slots and a reference reads the latest value). This is what exercises the
    # delta-rule's overwrite/erase advantage over additive Hebbian. Off by default (Proj-4 configs
    # keep the 1:1 slot<->problem mapping and its assertion).
    allow_overwrite: bool = False

    # --- arm: K0 (read-severed control) | K1 (memory, answer-CE only) | K2 (+ grounded write/read) ---
    arm: str = "K2"
    mem_write_weight: float = 0.5   # K2: weight on slot-write CE (memory stores the value)
    mem_addr_weight: float = 0.5    # K2: weight on address loss (reference query -> right slot key)
    process_loss_weight: float = 0.1   # arith arm-C style: CE on intermediate running-sum states
    process_warmup_steps: int = 200

    # --- training ---
    batch_size: int = 32
    train_steps: int = 400
    lr: float = 3e-3            # from-scratch small net trains fast; higher LR than the LoRA runs
    warmup_steps: int = 30
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    log_every: int = 50
    seed: int = 0

    # curriculum (shallow -> deep), same competence-gated logic as train_arm
    curriculum: bool = False
    curriculum_threshold: float = 0.85
    curriculum_min_steps: int = 80
    curriculum_patience_steps: int = 400

    # --- eval / probe ---
    eval_examples_per_hop: int = 512
    probe_examples: int = 2000
    probe_test_frac: float = 0.4
    eval_sessions: int = 512
    probe_sessions: int = 400

    device: str = ""
    out_dir: str = "agi_demo/outputs/kernel"

    def resolve(self) -> "KernelConfig":
        if not self.device:
            self.device = get_device()
        if not self.decays:
            self.decays = default_decays(self.n_layers)
        assert len(self.decays) == self.n_layers, \
            f"decays {self.decays} must have length n_layers={self.n_layers}"
        assert self.session_len <= self.n_slots or self.allow_overwrite, \
            (f"session_len ({self.session_len}) > n_slots ({self.n_slots}) means slots are rewritten. "
             "That's the Phase-A overwrite regime — pass allow_overwrite=True to opt in. With the 1:1 "
             "mapping (default) each problem writes a distinct slot, which read-supervision assumes.")
        os.makedirs(self.out_dir, exist_ok=True)
        return self

    @property
    def max_hops(self) -> int:
        return max(self.eval_hops + self.train_hops)
