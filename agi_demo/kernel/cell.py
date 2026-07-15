"""The kernel — a from-scratch, non-Transformer recurrent core with native fast-weight memory.

WHY THIS IS NOT A TRANSFORMER
  * No attention over tokens, no softmax over a sequence, no growing KV cache.
  * Each layer carries a FIXED-SIZE matrix state S ∈ R^{d_state × d_state} — a "fast weight"
    (Hinton) / linear-attention associative memory. The sequence is consumed by a recurrent scan
    that updates S with an outer product per token and reads it back as r = S q.

WHY MEMORY-AS-OPERAND IS NATIVE (the Project-3 wall, removed)
  The read r_t = S_t q_t is produced INSIDE the cell and fed straight into the cell's own output
  MLP. The stored value participates in the computation by construction — there is no "inject a
  foreign vector into a frozen stack" step, which is exactly what a frozen Transformer could not
  consume (FINDINGS_memory.md: write/address/read all worked, USE = chance).

MULTI-TIMESCALE HIERARCHY (load-bearing)
  Layer ℓ decays its state by γ_ℓ each token: S_t = γ_ℓ S_{t-1} + (write). A spread of γ across
  layers (fast≈0 … slow≈0.99) gives working memory (recent tokens) at the bottom and a session-long
  store at the top. The probe reads each layer's state to SHOW this split.

DELTA RULE vs HEBBIAN
  * delta_rule=True (DeltaNet, Schlag/Yang): before writing key k, remove its current association
    S k, so re-writing the same key OVERWRITES rather than accumulates — crucial for exact recall
    of a slot's *latest* value. Update: S ← γS + β (v − Sk) kᵀ.
  * delta_rule=False: plain additive Hebbian S ← γS + β v kᵀ (kept as an ablation).

PARALLELISM
  The scan is associative → a chunked parallel form exists (chunked linear attention). Here we run
  the simple sequential scan (short T, tiny model); the design is parallel-ready by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kconfig import KernelConfig


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class FastWeightCell(nn.Module):
    """One recurrent layer holding an associative matrix state S (d_state × d_state).

    Per token (given input h ∈ R^{d_model}):
        k, v, q = W_k h, W_v h, W_q h        (each in R^{d_state})
        β       = sigmoid(W_β h)             (scalar write strength, per token/batch)
        k, q    = ℓ2-normalize(k), ℓ2-normalize(q)   (stabilizes recall)
        S       = γ S + β (v − S k) kᵀ       (delta rule)   OR   γ S + β v kᵀ (hebbian)
        r       = S q                        (READ — the native operand)
        out     = h + W_o(r)                 (residual back into the token stream)
    Exposes the read r and (optionally) the state S so probes can observe memory.
    """

    def __init__(self, cfg: KernelConfig, gamma: float):
        super().__init__()
        d, s = cfg.d_model, cfg.d_state
        self.d_state = s
        self.gamma = gamma
        self.delta_rule = cfg.delta_rule
        self.k_proj = nn.Linear(d, s, bias=False)
        self.v_proj = nn.Linear(d, s, bias=False)
        self.q_proj = nn.Linear(d, s, bias=False)
        self.beta = nn.Linear(d, 1)
        self.o_proj = nn.Linear(s, d, bias=False)
        # near-identity start: value write and output read begin tiny so the untrained memory path
        # does not destabilize early training (the cold-start trick from prior projects).
        nn.init.normal_(self.v_proj.weight, std=1e-3)
        nn.init.normal_(self.o_proj.weight, std=1e-3)
        nn.init.zeros_(self.beta.bias)   # sigmoid(0)=0.5 initial write strength

    def forward(self, x: torch.Tensor, read_enabled: bool = True,
                collect_state: bool = False) -> Tuple[torch.Tensor, List, List]:
        """x: (B, L, d_model). Returns (out (B,L,d_model), reads[list of (B,d_state)],
        states[list of (B,d_state,d_state)] if collect_state else [])."""
        B, L, _ = x.shape
        s = self.d_state
        S = torch.zeros(B, s, s, device=x.device, dtype=x.dtype)  # blank fast-weight ("留白")
        reads, states = [], []
        outs = []
        k_all = F.normalize(self.k_proj(x), dim=-1)      # (B,L,s)
        v_all = self.v_proj(x)                           # (B,L,s)
        q_all = F.normalize(self.q_proj(x), dim=-1)      # (B,L,s)
        beta_all = torch.sigmoid(self.beta(x))           # (B,L,1)
        for t in range(L):
            k = k_all[:, t]                               # (B,s)
            v = v_all[:, t]
            q = q_all[:, t]
            b = beta_all[:, t]                            # (B,1)
            if self.delta_rule:
                Sk = torch.bmm(S, k.unsqueeze(-1)).squeeze(-1)     # (B,s)  current value at key k
                write_v = v - Sk
            else:
                write_v = v
            # outer product (b * write_v) kᵀ  -> (B,s,s)
            outer = (b * write_v).unsqueeze(-1) * k.unsqueeze(1)
            S = self.gamma * S + outer
            if read_enabled:
                r = torch.bmm(S, q.unsqueeze(-1)).squeeze(-1)      # (B,s)  READ = operand
            else:
                r = torch.zeros(B, s, device=x.device, dtype=x.dtype)  # K0 control: severed read
            reads.append(r)
            if collect_state:
                states.append(S)
            outs.append(x[:, t] + self.o_proj(r))
        out = torch.stack(outs, dim=1)                    # (B,L,d_model)
        return out, reads, states


class FFN(nn.Module):
    def __init__(self, d: int, mult: float):
        super().__init__()
        inner = int(d * mult)
        self.net = nn.Sequential(nn.Linear(d, inner), nn.GELU(), nn.Linear(inner, d))

    def forward(self, x):
        return self.net(x)


class MultiTimescaleKernel(nn.Module):
    """Embedding -> L (FastWeightCell + FFN) blocks at spread decay rates -> hidden stream.

    read_enabled=False severs EVERY layer's read (the K0 causal control): fast weights still WRITE
    (so the value is stored/probeable) but the model can never USE it as an operand.
    """

    def __init__(self, cfg: KernelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.cells = nn.ModuleList([FastWeightCell(cfg, gamma=g) for g in cfg.decays])
        self.norms1 = nn.ModuleList([RMSNorm(cfg.d_model) for _ in cfg.decays])
        self.norms2 = nn.ModuleList([RMSNorm(cfg.d_model) for _ in cfg.decays])
        self.ffns = nn.ModuleList([FFN(cfg.d_model, cfg.ffn_mult) for _ in cfg.decays])
        self.final_norm = RMSNorm(cfg.d_model)

    def forward(self, input_ids: torch.Tensor, read_enabled: bool = True,
                collect_state: bool = False):
        """Returns (hidden (B,L,d_model), per_layer_reads, per_layer_states).
        per_layer_reads[ℓ] is a (B,L,d_state) tensor; states likewise (B,L,d_state,d_state)."""
        x = self.embed(input_ids)
        layer_reads, layer_states = [], []
        for cell, n1, n2, ffn in zip(self.cells, self.norms1, self.norms2, self.ffns):
            h, reads, states = cell(n1(x), read_enabled=read_enabled, collect_state=collect_state)
            x = x + h                                     # residual around the memory cell
            x = x + ffn(n2(x))                            # position-wise FFN (nonlinearity)
            layer_reads.append(torch.stack(reads, dim=1))            # (B,L,s)
            if collect_state:
                layer_states.append(torch.stack(states, dim=1))     # (B,L,s,s)
        return self.final_norm(x), layer_reads, layer_states


class KernelModel(nn.Module):
    """Kernel + a digit readout head. The ONLY task-specific head; everything is trained from init."""

    def __init__(self, cfg: KernelConfig, vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.core = MultiTimescaleKernel(cfg, vocab.size)
        self.readout = nn.Linear(cfg.d_model, cfg.modulus)   # -> mod-N digit logits
        # a small head that decodes a memory read/state slice to a digit — used BOTH for K2
        # write/read supervision AND as the observability probe's trainable analogue (kept linear).
        self.mem_head = nn.Linear(cfg.d_state, cfg.modulus)
        self.to(cfg.device)

    @property
    def read_enabled(self) -> bool:
        return self.cfg.arm != "K0"

    def forward(self, input_ids, collect_state: bool = False):
        hidden, reads, states = self.core(input_ids, read_enabled=self.read_enabled,
                                          collect_state=collect_state)
        return hidden, reads, states

    def answer_logits_at(self, hidden: torch.Tensor, pos) -> torch.Tensor:
        """Digit logits at a readout position. pos is an int (same for the batch) or (B,) tensor."""
        if isinstance(pos, int):
            h = hidden[:, pos]
        else:
            h = hidden[torch.arange(hidden.shape[0], device=hidden.device), pos]
        return self.readout(h)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
