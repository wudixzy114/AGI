"""30-second smoke test: model loads, task generates, one forward+backward runs finite.

Run:  python -m agi_demo.smoke_test
"""
from __future__ import annotations

import random

import torch

from .config import Config
from .task import make_batch, make_example, digit_token_ids
from .model import CoconutReasoner
from .train import compute_loss


def main():
    print("== task sanity ==")
    rng = random.Random(0)
    for h in (1, 3, 5):
        e = make_example(h, 10, rng)
        assert e.answer == e.intermediates[-1]
        print(f"  hops={h}: {e.prompt:<26} inter={e.intermediates} ans={e.answer}")

    import os
    def mkcfg():
        c = Config()
        if os.environ.get("AGI_MODEL_DIR"):
            c.local_model_dir = os.environ["AGI_MODEL_DIR"]
        if os.environ.get("AGI_DTYPE"):
            c.dtype = os.environ["AGI_DTYPE"]
        return c.resolve()

    cfg = mkcfg()
    print(f"\n== loading {cfg.model_id} on {cfg.device} (dtype={cfg.dtype}) ==")
    model = CoconutReasoner(cfg)
    print(f"  hidden_size={model.hidden_size}  trainable={model.num_trainable():,}")

    digit_ids = digit_token_ids(model.tokenizer, cfg.modulus)
    print(f"  digit token ids: {digit_ids}")

    for arm in ("A", "B", "C"):
        cfg2 = mkcfg()
        cfg2.arm = arm
        m = CoconutReasoner(cfg2)
        m.train()
        batch = make_batch(m.tokenizer, 4, 3, cfg2.modulus, rng,
                           device=cfg2.device, digit_ids=digit_ids)
        out = compute_loss(m, batch, cfg2)
        loss = out["total"]
        loss.backward()
        gnorm = sum(
            p.grad.norm().item() ** 2
            for g in m.trainable_param_groups() for p in g["params"] if p.grad is not None
        ) ** 0.5
        finite = torch.isfinite(loss).item()
        print(f"  arm {arm}: loss={loss.item():.4f} finite={finite} "
              f"ans={out['ans'].item():.3f} proc={out['proc'].item():.3f} "
              f"grad_norm={gnorm:.4f} acc={out['acc'].item():.2f}")
        assert finite and gnorm > 0, f"arm {arm} failed: non-finite loss or zero grad"
        del m

    print("\nSMOKE OK: forward+backward works for all arms, gradients flow.")


if __name__ == "__main__":
    main()
