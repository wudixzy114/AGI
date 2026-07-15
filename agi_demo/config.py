"""Configuration for the AGI demo experiment.

One dataclass holds every knob. Defaults are tuned for an Apple M3 / 16GB Mac
(small model, small batch); on a B100 just bump `model_id`, `batch_size`,
and `train_steps`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    # --- model (the frozen "不变" base) ---
    model_id: str = "Qwen/Qwen2.5-0.5B"
    # If a local copy of the weights exists here, use it instead of hitting the Hub.
    local_model_dir: str = "models/Qwen2.5-0.5B"
    device: str = ""            # "" -> auto-detect
    dtype: str = "float32"      # fp32 is the stable choice on MPS

    # --- task: mod-N multi-hop arithmetic chains ---
    task: str = "arith"         # "arith" (mod-N +/- chain), "perm" (lookup), or "session" (linked)
    modulus: int = 10           # every value is a single digit 0..9
    train_hops: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    eval_hops: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    # hops 5,6 are out-of-distribution -> test length generalization

    # --- Project 3: short-term latent memory (session task) ---
    use_memory: bool = False       # attach addressable latent memory slots
    n_slots: int = 8               # number of memory slots (>= session length)
    session_len: int = 6           # problems per session (share one memory)
    ref_prob: float = 0.5          # probability a problem references an earlier slot (@k)
    session_hops: int = 2          # ops per problem inside a session (small; depth isn't the point)
    mem_write_weight: float = 0.5  # M2: weight on the "slot should equal its answer" CE
    # Attribution experiment: unfreeze the last N transformer blocks of the base (0 = fully frozen).
    # Tests whether "the frozen base can't USE an injected latent value" is a plasticity limit.
    unfreeze_last_n: int = 0
    lr_unfrozen: float = 2e-5   # unfrozen base layers: small LR (1e-4 diverged in bf16, verified)

    # --- the trainable "留白" modules (变) ---
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    thought_hidden_mult: float = 2.0   # thought_proj bottleneck width = H*mult
    # Q1: residual thought loop. t = h_last + proj(h_last) instead of t = proj(h_last).
    # With proj near-zero-init this makes the loop start at Coconut's identity (t=h_last),
    # so gradients flow through deep latent chains from step 0 (highway/ResNet intuition).
    residual_thought: bool = False

    # --- arm selection: A (result-only, no latent), B (latent), C (latent+process) ---
    arm: str = "B"
    # auxiliary-hint scale: keep answer CE dominant (λ=0.5 swamps it, verified)
    process_loss_weight: float = 0.1   # arm C: weight on intermediate-value CE
    process_warmup_steps: int = 200    # ramp process loss in only after answer-CE builds the chain
    # Causal ablation (①): WHICH latent steps get process supervision.
    #   "all"  -> every step (standard arm C)
    #   "first_half" / "second_half" -> only the shallow / deep half of the chain
    #   "1,3,5" -> explicit 1-indexed step set
    # Lets us test whether making a SUBSET of steps decodable causally drives OOD generalization.
    process_steps: str = "all"

    # --- training ---
    batch_size: int = 16       # keep small on unified-memory Macs; raise on B100
    train_steps: int = 1000
    lr_lora: float = 2e-4
    lr_head: float = 3e-4       # thought_proj/step_head; >1e-3 destabilizes (verified)
    warmup_steps: int = 30
    grad_clip: float = 1.0
    log_every: int = 50
    seed: int = 0

    # --- curriculum (shallow -> deep, competence-gated) ---
    # Fix for the difficulty-mixing collapse: start at hops=1 and only unlock the next
    # depth once running accuracy at the current max depth clears `curriculum_threshold`.
    curriculum: bool = False
    curriculum_threshold: float = 0.85   # ema-acc at current max depth needed to advance
    curriculum_min_steps: int = 80       # min steps at a level before it may advance
    curriculum_patience_steps: int = 400 # force-advance if stuck this long (avoid deadlock)

    # --- evaluation / probe ---
    eval_examples_per_hop: int = 256
    # >> hidden_size (896) so the linear probe isn't in the p>>n overfitting regime
    probe_examples_per_hop: int = 2000
    probe_test_frac: float = 0.4

    # --- io ---
    out_dir: str = "agi_demo/outputs"

    def resolve(self) -> "Config":
        import glob
        if not self.device:
            self.device = get_device()
        # Prefer a local weights folder if it has weights (avoids the Hub).
        # Handles both single-file (model.safetensors) and sharded (index + shards) layouts.
        d = self.local_model_dir
        single = os.path.join(d, "model.safetensors")
        has_single = os.path.exists(single) and os.path.getsize(single) > 100_000_000
        has_sharded = (os.path.exists(os.path.join(d, "model.safetensors.index.json"))
                       and len(glob.glob(os.path.join(d, "*.safetensors"))) > 0)
        if has_single or has_sharded:
            self.model_id = os.path.abspath(d)
        os.makedirs(self.out_dir, exist_ok=True)
        return self

    @property
    def max_hops(self) -> int:
        return max(self.eval_hops + self.train_hops)
