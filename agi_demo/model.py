"""CoconutReasoner: frozen base + trainable "留白" modules + continuous-thought loop.

Design (matches the approved plan):
  * Base LM (Qwen2.5-0.5B) is fully frozen  -> the "不变" long-term memory.
  * Trainable "变" parts:
      - LoRA on attention projections (a plasticity knob on the frozen base).
      - thought_proj: an MLP mapping a hidden state -> the next input embedding.
        This is the learnable bridge that lets the model "think" in latent space.
      - step_head (arm C only): reads a thought vector -> predicts a mod-N digit,
        the explicit *process* supervision.

  * Continuous-thought loop (Coconut, Hao et al. 2024): after reading the prompt
    (which ends in "="), instead of decoding a token we take the last hidden state,
    map it through thought_proj, and feed it back as the next input embedding —
    for `n_latent` steps. Only after those latent steps do we decode the answer.
    The model reasons in a continuous space without collapsing to one-dim tokens.

  Three arms, one class:
    A  n_latent=0                -> answer decoded straight from "=".  Result-only baseline.
    B  n_latent=hops, ans CE     -> latent thinking, reward only via reaching the answer.
    C  B + process CE on thoughts-> explicit per-step supervision of intermediates.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .task import digit_token_ids


class ThoughtProj(nn.Module):
    """Hidden-state space -> next-input-embedding space."""

    def __init__(self, hidden: int, mult: float):
        super().__init__()
        inner = int(hidden * mult)
        self.net = nn.Sequential(
            nn.Linear(hidden, inner),
            nn.GELU(),
            nn.Linear(inner, hidden),
        )
        # start near-identity-ish but small, so early thoughts perturb gently
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class CoconutReasoner(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype)
        for p in base.parameters():
            p.requires_grad_(False)

        # LoRA (the "变" plasticity on the frozen base)
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.net = get_peft_model(base, lora)

        # Attribution: optionally re-enable grad on the last N base transformer blocks.
        self.unfrozen_params = []
        if getattr(cfg, "unfreeze_last_n", 0) > 0:
            layers = base.model.layers                     # Qwen2 decoder blocks
            for layer in layers[-cfg.unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
                    self.unfrozen_params.append(p)

        H = base.config.hidden_size
        self.hidden_size = H
        self.embed = self.net.get_input_embeddings()  # frozen embedding table

        self.thought_proj = ThoughtProj(H, cfg.thought_hidden_mult).to(dtype)
        self.step_head = nn.Linear(H, cfg.modulus).to(dtype) if cfg.arm == "C" else None

        # Project 3: optional short-term latent memory ("留白" short-term store)
        self.memory = None
        self.mem_slot_head = None
        if getattr(cfg, "use_memory", False):
            from .memory import LatentMemory
            self.memory = LatentMemory(H, cfg.n_slots).to(dtype)
            # decodes a slot vector -> a digit; used for M2 write-supervision AND as the
            # observability probe head (kept separate from step_head).
            self.mem_slot_head = nn.Linear(H, cfg.modulus).to(dtype)

        self.digit_ids = torch.tensor(digit_token_ids(self.tokenizer, cfg.modulus), dtype=torch.long)

        self.to(cfg.device)
        self.digit_ids = self.digit_ids.to(cfg.device)

    # ---- param groups for the optimizer (base stays frozen unless unfreeze_last_n>0) ----
    def trainable_param_groups(self):
        unfrozen_ids = {id(p) for p in self.unfrozen_params}
        # LoRA adapter params (exclude any unfrozen base params, handled in their own group)
        lora_params = [p for n, p in self.net.named_parameters()
                       if p.requires_grad and id(p) not in unfrozen_ids]
        head_params = list(self.thought_proj.parameters())
        if self.step_head is not None:
            head_params += list(self.step_head.parameters())
        if self.memory is not None:
            head_params += list(self.memory.parameters()) + list(self.mem_slot_head.parameters())
        groups = [{"params": lora_params, "lr": self.cfg.lr_lora}]
        has_heads = (self.cfg.arm != "A") or (self.memory is not None)
        if has_heads and head_params:
            groups.append({"params": head_params, "lr": self.cfg.lr_head})
        if self.unfrozen_params:  # unfrozen base layers get a MUCH gentler LR (they destabilize
            # training otherwise — full-precision base weights + bf16 are sensitive; verified).
            groups.append({"params": self.unfrozen_params, "lr": self.cfg.lr_unfrozen})
        return groups

    def num_trainable(self) -> int:
        return sum(p.numel() for grp in self.trainable_param_groups() for p in grp["params"])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_latent: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns (answer_logits over full vocab (B,V), thoughts list of (B,H) len n_latent)."""
        emb = self.embed(input_ids)  # (B, L, H)
        out = self.net(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past = out.past_key_values
        h_last = out.hidden_states[-1][:, -1, :]  # hidden at "=" position (B, H)

        thoughts: List[torch.Tensor] = []
        cur_mask = attention_mask
        for _ in range(n_latent):
            proj = self.thought_proj(h_last)
            t = (h_last + proj if self.cfg.residual_thought else proj).unsqueeze(1)  # (B, 1, H)
            cur_mask = torch.cat([cur_mask, torch.ones_like(cur_mask[:, :1])], dim=1)
            out = self.net(
                inputs_embeds=t,
                attention_mask=cur_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = out.past_key_values
            h_last = out.hidden_states[-1][:, -1, :]
            thoughts.append(h_last)  # h after this thinking step; ~ intermediate x_i

        answer_logits = out.logits[:, -1, :]  # next-token distribution -> the answer
        return answer_logits, thoughts

    def digit_logits(self, answer_logits: torch.Tensor) -> torch.Tensor:
        """Restrict full-vocab logits to the N digit tokens -> (B, N)."""
        return answer_logits.index_select(-1, self.digit_ids)

    # ---- Project 3: one problem within a session (reads/writes shared memory) ----
    def forward_problem(self, input_ids, attention_mask, n_latent: int):
        """Like forward(), but with memory: read a value from the slots and inject it. Two
        mechanisms for USABILITY (the read vector alone can't be consumed by the frozen base):
          ① return the address logits so training can supervise WHICH slot is read;
          ② decode-and-re-embed: turn the read into a soft digit embedding via mem_slot_head +
             the frozen token-embedding table, so the base receives it like a normal digit token.
        Returns (answer_logits, thoughts, answer_hidden, addr_logits)."""
        emb = self.embed(input_ids)                      # (B, L, H)
        mask = attention_mask
        inject = None                                    # what we add at each latent step
        addr_logits = None
        if self.memory is not None:
            query = (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            read, addr_logits, _ = self.memory.read(query, return_attn=True)   # (B,H),(B,M),(B,M)
            # ② decode-and-re-embed: read -> digit distribution -> soft token embedding (frozen table)
            digit_probs = torch.softmax(self.mem_slot_head(read), dim=-1)      # (B,N)
            digit_emb = self.embed(self.digit_ids)                            # (N,H) frozen
            soft_read = digit_probs @ digit_emb                               # (B,H) "the value as a token"
            inject = read + soft_read
            emb = torch.cat([inject.unsqueeze(1), emb], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
        out = self.net(inputs_embeds=emb, attention_mask=mask,
                       use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        h_last = out.hidden_states[-1][:, -1, :]
        thoughts = []
        cur_mask = mask
        for _ in range(n_latent):
            proj = self.thought_proj(h_last)
            t = (h_last + proj if self.cfg.residual_thought else proj)
            if inject is not None:
                t = t + inject                            # keep the read available every step
            t = t.unsqueeze(1)
            cur_mask = torch.cat([cur_mask, torch.ones_like(cur_mask[:, :1])], dim=1)
            out = self.net(inputs_embeds=t, attention_mask=cur_mask, past_key_values=past,
                           use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            h_last = out.hidden_states[-1][:, -1, :]
            thoughts.append(h_last)
        answer_logits = out.logits[:, -1, :]
        return answer_logits, thoughts, h_last, addr_logits


def load(cfg: Config) -> CoconutReasoner:
    return CoconutReasoner(cfg)
