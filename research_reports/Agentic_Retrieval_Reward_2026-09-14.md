#  Agentic RL 动态检索奖励优化专项调研 - 2026-09-14

## 🧠 核心研究范式 (Core Paradigms)

### 任务损失端到端优化的稀疏上下文/检索选择
将上下文单元选择器的连续分数注入注意力 logits，使固定预算下的硬 Top-K 选择可由语言建模损失端到端训练，避免仅蒸馏稠密注意力分布。

#### 📄 [SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking](https://huggingface.co/papers/2609.13141)
- **奖励设计**: 以语言建模损失作为优化目标；把选择器的连续分数以 log 形式置于注意力 softmax 内，并用归一化 softmax 门控校准历史上下文与当前块，保留连续分数以学习相对优先级。
- **💡 启发**: 动态检索策略应通过最终任务损失直接优化检索/上下文排序，而不是仅模仿 oracle 注意力；可以用连续门控或 soft Top-K 缓解硬选择的梯度阻断。

---

### 可验证奖励与过程证据激励
在可验证的下游任务上使用 RLVR，并加入过程级 CoB 奖励鼓励模型在推理轨迹中使用可审计证据。

#### 📄 [CARDEA: Auditable Reasoning Grounded in Spatial Evidence for End-to-End Coronary Angiography Interpretation](https://huggingface.co/papers/2609.06931)
- **奖励设计**: RLVR 阶段使用可验证奖励，并加入 CoB（Chain-of-Box）奖励，鼓励推理轨迹中出现 bounding-box 证据；摘要指出仅有 RLVR 提升了零样本报告生成能力。
- **💡 启发**: 在动态检索中引入过程奖励以鼓励模型在推理轨迹中显式调用或引用检索到的证据，而非仅给最终答案的稀疏奖励。

---

### 检索-推理迭代互补的动态证据收集
通过受控实验证明推理与检索互补：仅推理对稀有实体提升不显著，仅检索可提升稀有实体但可能损害总体，二者结合最优。

#### 📄 [Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking](https://huggingface.co/papers/2609.10745)
- **奖励设计**: 无训练、无显式奖励函数；动态框架让 VLM 迭代搜索并推理 Wikipedia。受控实验揭示了检索与推理的互补性，并提示仅检索无推理可能损害整体精度。
- **💡 启发**: 奖励函数不应只奖励是否检索，还需奖励检索证据是否被有效推理利用；否则可能鼓励过度检索或无效检索。

---

### 强化查询生成与并行证据聚合
将长文档读取与深度推理解耦：多个冻结子代理并行读取块，主代理通过多轮 scatter-gather 广播查询、聚合证据并生成后续查询，并由 RL 优化。

#### 📄 [PARSER: Read in Parallel, Reason in Depth for Long-Context LLM Agents](https://huggingface.co/papers/2609.06702)
- **奖励设计**: 摘要仅指出 lead agent 通过强化学习优化，未披露具体奖励函数；可观察的动作空间包括每轮查询广播
- **💡 启发**: N/A

---

## ⚠️ 现有研究的局限与空白 (Gaps & Contradictions)

暂无

## 🚀 Top 3 可立即尝试的行动思路 (Actionable Ideas)

