# Project 4 — 非 Transformer 快权重内核 + 分层记忆

这是 AGI 研究系列的第 4 个项目,一个**独立的成果包**(不含 Projects 1–3 的内容;那些在各自的包里)。

## 先看这个

**`architecture_kernel.html`** — 一张自包含的架构全景图(浏览器直接打开,无需联网)。
它按顺序讲清楚:Project 3 撞的墙 → 内核内部结构(快权重 cell + delta 规则)→ 多时间尺度分层记忆 →
可证伪的任务与三个实验臂 → 全矩阵结果 → 监控基础设施。**建议任何后续读者从这张图开始。**

## 一句话结论

Project 3 证明:在**冻结的 Transformer** 上外挂记忆,值能被写入(0.62)、寻址(0.72)、读出(0.69),
却**无法被当作算子使用**(ref_acc = 0.10 = 瞎猜)。Project 4 换掉架构本身——一个从零训练、**记忆读
即原生算子**的递归内核——把同一个指标从 **0.10 推到 1.00**,并用断读对照(K0=chance)因果地证明是那
一次联想读造成了可用性。**"可观测 ≠ 可用";可用性是架构属性,不是训练问题。**

## 内核为什么不是 Transformer

每层携带一个**固定大小矩阵状态** `S ∈ ℝ^{d×d}`(快权重 / 线性注意力式联想记忆)。逐 token:

```
k,v,q = W_k·h, W_v·h, W_q·h ;  β = σ(W_β·h) ;  k,q ← ℓ2 归一化
S ← γ·S + β·(v − S·k)·kᵀ        # delta 规则:覆盖式联想写入
r = S·q                          # 读 = 算子,在 cell 内部算出
out = h + W_o·r                  # 读直接加回残差流
```

无注意力、无 softmax over sequence、无越堆越大的 KV cache。读 `r=S·q` 在 cell 内部产生并加回残差流
——这就是"记忆即算子"原生成立的原因。多层用铺开的衰减率 `γ=[0,0.7,0.95,0.99]`:快层=工作记忆,
慢层=会话级存储。

## 关键结果(全矩阵,8 配置,B200)

| 配置 | K0 (断读) | K1 (仅答案) | K2 (+读监督) | 说明 |
|------|----------|------------|-------------|------|
| hops0 纯召回 | 0.09 | **1.00** | **1.00** | 头条;Δ1–5 全 1.0 |
| 5-seed 纯召回 | 0.10±.01 | **1.00±.00** | **1.00±.00** | 零方差,非 seed 噪声 |
| 长会话 (len=10) | 0.11 | **1.00** | **1.00** | 每个延迟(到 Δ=9)都 1.0 |
| hops2 召回+2步链 | 0.09 | 0.10 | 0.10* | 算术 grokking 墙,非记忆失败(*K2 lit=1.0) |

- **多时间尺度层级被探针实测**:慢层 γ=0.99 全延迟 1.0,快层 γ=0 过 Δ=1 即掉 chance。层级是量出来的,不是断言的。
- **诚实边界**:从零学 mod-N 算术是 grokking-slow,所以 hops≥2 被算术能力(而非记忆)卡住 → hops=0 纯召回才是无混淆判据。
- 完整分析见 **`FINDINGS_kernel.md`**。

## 目录结构

```
architecture_kernel.html   ← 架构全景图,从这里开始
FINDINGS_kernel.md         ← 完整实验分析
kernel/                    ← 从零内核(无 transformers/peft/Qwen 依赖)
  cell.py                    FastWeightCell, MultiTimescaleKernel, KernelModel
  encode.py                  符号词表 + 会话/算术 token 流编码
  kconfig.py                 KernelConfig
  train.py                   三个臂 K0/K1/K2 的训练
  probe.py                   eval_session, probe_timescales(层级图), probe_arith
  run.py / smoke.py          编排 / Mac 结构冒烟测试
  run_kernel.sh              B200 全矩阵驱动
outputs/                   ← 8 个配置的 metrics.json + run.log + 图
infra/                     ← 可复用监控基础设施(服务后续所有项目)
  scheduler/                 资源感知调度器(按显存装箱并发)
  dashboard/                 数据驱动看板(按指标形状自动出图)
```

## 如何复现

```bash
python -m agi_demo.kernel.smoke                            # 1) Mac 结构冒烟测试(秒级)
python -m agi_demo.kernel.run --session --session-hops 0   # 2) 头条使用测试(K0/K1/K2)
python -m agi_demo.kernel.run --arith --arm K1 --curriculum # 3) 算术长度泛化
bash agi_demo/kernel/run_kernel.sh                         # 4) 全矩阵(B200)
# 或用调度器按显存自动并发跑整个矩阵:
python3 agi_demo/scheduler/schedule.py agi_demo/scheduler/jobs_kernel.json
```

## 下一步(Project 5,已规划)

一个巩固闭环:① 记忆覆写任务(让 delta-rule 真正赢过 Hebbian)→ ② 睡眠巩固(慢层离线蒸馏进持久
存储,测跨会话免重读召回)→ ③ 换 bAbI 式实体跟踪(更真实但保留精确 ground truth)→ ④ 适度放大。
