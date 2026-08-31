# AGI — Latent Reasoning, Self-Evolution, and Hierarchical Memory

> **一个可证伪的研究计划，主题是"训练后不冻结权重"能不能造出更强的推理系统。**
> 这不是"AGI 概念验证"，是一个有 ground truth、能被线性探针测量的实证项目。

## 核心命题

论文分三块（详见 `docs/thesis_memory-and-dimensionality_20260713.md`）：

1. **潜在空间深度推理** — 不通过解码 token 来"思考"，而是在连续向量空间里推（Coconut 风格）
2. **"留白" + 迭代自演化** — 训练完成后模型继续变化；稳定-可塑性分离（frozen base + plastic modules）
3. **分层潜变量做短时记忆** — 分层思考 ↔ 不同时间尺度的分层记忆

统一视角：**有巩固机制的层级化记忆** —— 工作记忆/计算 → 短时记忆 → 巩固（短→长）→ 冻结基座作为长期记忆。

## 实验设计

每个实验都用合成的 mod-N 多跳算术，**每个中间步骤都有精确 ground truth**，
所以可以用线性探针测量潜在推理链是否仍可被解码。这个可观测性是所有结论的支柱。

## 5 个子项目

| # | 问题 | 结论 | 报告 |
|---|---|---|---|
| **1** | 0.5B frozen + LoRA 能在潜在空间推理吗？ | 难度均匀时潜在路径坍缩，但在**能力门控课程**下 OOD 优于 baseline。 | [`agi_demo/FINDINGS.md`](agi_demo/FINDINGS.md) |
| **2** | 3B 规模还成立吗？ | 过程监督是潜在推理"划算"的前提；**OOD 泛化 ↔ 中间链可解码性**。 | [`agi_demo/FINDINGS_3B.md`](agi_demo/FINDINGS_3B.md) |
| **2b** | 因果消融 + 鲁棒性 | 只改"哪几步潜在步骤受过程监督"，OOD 单调变化（无 < 深 < 浅 < 全），探针镜像这个趋势——**监督 ↔ 因果** ↔ 可解码性。跨任务（perm）成立，量级有种子噪声。 | [`agi_demo/FINDINGS_causal.md`](agi_demo/FINDINGS_causal.md) |
| **3** | 给 frozen Transformer 加外挂记忆？ | **干净的负结果**：写 ✓、寻址 ✓、读 ✓，但**用 = 随机**。frozen Transformer 从未学过把注入的潜在当操作数；部分解冻会破坏稳定性。**可观测 ≠ 可用**。 | [`agi_demo/FINDINGS_memory.md`](agi_demo/FINDINGS_memory.md) |
| **4** | 从头做一个"记忆当操作数"是原生的内核？ | 自研的 **fast-weight kernel**（矩阵状态 S，delta-rule 写入，in-cell 读取）把 Project 3 的 `USE=0.10` 提升到 **`ref_acc=1.00`**，并测出**按衰减 λ 排序的时间尺度层级**。 | [`agi_demo/FINDINGS_kernel.md`](agi_demo/FINDINGS_kernel.md) |
| **5** | 更真实任务上的巩固循环 | **进行中**（B200 上：覆写任务 → 睡眠巩固 → bAbI 风格实体追踪 → 中等规模 scale-up）。 | — |

## 仓库结构

```
AGI/
├─ agi_demo/                  # 5 个子项目的实验代码 + 报告
│  ├─ FINDINGS.md             # Project 1
│  ├─ FINDINGS_3B.md          # Project 2
│  ├─ FINDINGS_causal.md      # Project 2b
│  ├─ FINDINGS_memory.md      # Project 3
│  └─ FINDINGS_kernel.md      # Project 4
├─ docs/
│  └─ thesis_memory-and-dimensionality_20260713.md   # 核心论文
├─ requirements.txt
└─ README.md (本文件)
```

## 技术栈

- **语言**：Python 3.11+
- **ML 框架**：PyTorch + transformers + peft（LoRA）
- **实验追踪**：wandb
- **硬件**：B200（H100/4090 也能跑小规模实验）
- **可观测性**：线性探针（sklearn）、t-SNE / PCA 可视化

## 关键发现摘要

1. **潜在推理"能用"是有条件的**——均匀难度下坍缩，能力门控课程下能 OOD
2. **过程监督是潜在推理的前提**，不是装饰
3. **frozen base + 外挂记忆 ≠ 记忆系统** —— Transformer 不会"读"注入的向量
4. **fast-weight kernel 是更优的记忆原语**，天然支持时间尺度分层
5. **可观测性是研究的支柱**——没探针就只能讲故事

## 状态

- **v0.x** 状态：5 个子项目已完成 4 个，Project 5（巩固循环）进行中
- 论文草稿在 `docs/`
- 不在计划内：上生产、做产品

## License

MIT（实验代码）；数据合成脚本遵循 Apache 2.0。
