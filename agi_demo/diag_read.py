"""Diagnostic v2: after training M2 briefly, separate the two failure hypotheses.
  (a) ADDRESSING: does the read attention point at the correct ref_slot? (address accuracy)
  (b) USABILITY: given a CORRECT read, can the model produce the answer?
We measure: address-accuracy (argmax attn == ref_slot) and read-probe (does read encode the value).
"""
import torch, random, numpy as np
import agi_demo.config as C, agi_demo.model as M, agi_demo.train as T
from agi_demo.task import make_session_batch, digit_token_ids
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

cf = C.Config()
cf.local_model_dir = "/media/cfs/9n-das-admin/llm_models/Qwen2.5-3B-Instruct"
cf.dtype = "bfloat16"; cf.arm = "M2"; cf.use_memory = True; cf.task = "session"
cf.batch_size = 128; cf.device = "cuda"; cf.session_len = 6; cf.n_slots = 8
cf.train_steps = 200; cf.log_every = 100
cf = cf.resolve()
model = T.train_session(cf, verbose=False)   # quick-train M2
model.eval()
did = digit_token_ids(model.tokenizer, cf.modulus); rng = random.Random(5)

addr_hit = addr_tot = 0
Xr, yr = [], []
with torch.no_grad():
    for _ in range(8):
        sb = make_session_batch(model.tokenizer, 128, cf, rng, device="cuda", digit_ids=did)
        model.memory.reset(128, "cuda", next(model.parameters()).dtype)
        for t in range(sb.T):
            pos = sb.positions[t]
            emb = model.embed(pos["input_ids"]); mask = pos["attention_mask"]
            q = (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            read, addr_logits, attn = model.memory.read(q, return_attn=True)
            isref = pos["is_ref"].bool(); refslot = pos["ref_slot"]
            pred_slot = attn.argmax(-1)
            for b in range(128):
                if isref[b] and t > 0:
                    k = int(refslot[b]); v = sb.sessions[b].slot_after[t - 1][k]
                    if v >= 0:
                        addr_hit += int(pred_slot[b].item() == k); addr_tot += 1
                        Xr.append(read[b].float().cpu().numpy()); yr.append(v)
            _, _, ah, _ = model.forward_problem(pos["input_ids"], pos["attention_mask"], cf.session_hops)
            model.memory.write(ah, pos["write_slot"])

Xr = np.array(Xr); yr = np.array(yr); n = len(yr); nt = n // 3
idx = np.arange(n); np.random.RandomState(0).shuffle(idx); tr, te = idx[nt:], idx[:nt]
sc = StandardScaler().fit(Xr[tr]); clf = LogisticRegression(max_iter=200).fit(sc.transform(Xr[tr]), yr[tr])
print(f"(a) ADDRESS accuracy (argmax attn == ref_slot): {addr_hit/max(1,addr_tot):.3f} (chance {1/cf.n_slots:.3f})")
print(f"(b) READ-vector encodes referenced value: {clf.score(sc.transform(Xr[te]), yr[te]):.3f} (chance 0.10, n={n})")
print("If (a) low -> addressing failed (fix: harder address sup / more steps).")
print("If (a) high but (b) low -> reads the right slot but slot vec lost the value.")
print("If both high but ref_acc low -> value is right & readable, base still can't use it.")

