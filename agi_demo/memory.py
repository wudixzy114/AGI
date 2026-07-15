"""LatentMemory — addressable latent memory slots (fast-weights / Titans-style).

Design constraints (from the research vision's cruxes):
  * PARALLELIZABLE: read is attention over slots (a single softmax-weighted sum), NOT a serial
    scan. No recurrent dependency is added along the already-serial latent axis.
  * OBSERVABLE: slots live in hidden space (H-dim); a linear probe can decode what each slot
    holds. The session task gives ground-truth slot contents, so we can measure this directly.

Per session: `slots` is (B, M, H), zeroed at session start ("留白" — blank). Two ops:
  * read(query)  -> (B, H): softmax(q·slotsᵀ/√H) · slots. Purely functional, parallel.
  * write(value, slot_idx) -> updates slots via a learned gate. write_proj is near-zero-init so
    at start the write is ~identity-preserving (slots barely move) — same trick that fixed the
    latent-loop cold-start (avoid composing an untrained transform destructively from step 0).

The module is part of the trainable "留白" set; the base LM stays frozen.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentMemory(nn.Module):
    def __init__(self, hidden: int, n_slots: int):
        super().__init__()
        self.hidden = hidden
        self.n_slots = n_slots
        # read: project the query, attend over slots
        self.q_proj = nn.Linear(hidden, hidden)
        self.r_proj = nn.Linear(hidden, hidden)
        # write: project the value to store + a scalar gate per write
        self.v_proj = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden, 1)
        # near-identity start: writes begin gentle so the untrained memory path doesn't
        # destabilize training (mirrors thought_proj near-zero init).
        nn.init.normal_(self.v_proj.weight, std=1e-3); nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.gate.bias)   # sigmoid(0)=0.5 start; learns its own gate
        self.slots = None  # (B, M, H), set by reset()

    def reset(self, batch_size: int, device, dtype):
        """Clear memory at the start of a session (blank slate)."""
        self.slots = torch.zeros(batch_size, self.n_slots, self.hidden, device=device, dtype=dtype)

    def read(self, query: torch.Tensor, return_attn: bool = False):
        """query (B,H) -> read vector (B,H) via parallel attention over slots.
        If return_attn, also return the (B,M) attention weights (for address supervision)."""
        q = self.q_proj(query).unsqueeze(1)                       # (B,1,H)
        k = self.slots                                            # (B,M,H)
        logits = (q * k).sum(-1) / (self.hidden ** 0.5)           # (B,M)
        attn = torch.softmax(logits, dim=-1)                      # (B,M)
        read = (attn.unsqueeze(-1) * self.slots).sum(1)           # (B,H)
        out = self.r_proj(read)
        if return_attn:
            return out, logits, attn
        return out

    def write(self, value: torch.Tensor, slot_idx: torch.Tensor):
        """Write `value` (B,H) into slot `slot_idx` (B,) with a learned gate. In-place on slots."""
        B = value.shape[0]
        v = self.v_proj(value)                                    # (B,H)
        g = torch.sigmoid(self.gate(value))                       # (B,1)
        onehot = F.one_hot(slot_idx, self.n_slots).to(value.dtype)  # (B,M)
        # new slot content = (1-g)*old + g*v, applied only at slot_idx
        cur = (onehot.unsqueeze(-1) * self.slots).sum(1)          # (B,H) current value at target slot
        updated = (1 - g) * cur + g * v                           # (B,H)
        delta = onehot.unsqueeze(-1) * (updated - cur).unsqueeze(1)  # (B,M,H)
        self.slots = self.slots + delta

    def parameters_list(self):
        return list(self.parameters())
