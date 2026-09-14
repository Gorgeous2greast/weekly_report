#  Agentic RL 动态检索奖励优化专项调研 - 2026-09-14

## 🧠 核心研究范式 (Core Paradigms)

### 端到端下游损失驱动的上下文排序优化
将检索/上下文选择器的连续分数注入下游模型计算，使语言建模损失可直接反传并优化排序，而不是用硬 Top-K 或稠密注意力蒸馏间接监督。核心是在固定注意力/上下文预算下，让排序直接对齐预测效用。

#### 📄 [SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking](https://huggingface.co/papers/2609.13141)
- **奖励设计**: 没有独立的外部奖励；用语言建模损失端到端优化上下文排序。训练时把 selector 的连续分数注入 attention logits，并采用 log 形式门控、归一化 softmax 门控、保留连续 selector 分数，使损失能更新 selector。
- **💡 启发**: 把最终任务损失当作检索动作的隐式过程奖励，避免硬 Top-K 阻断梯度；在固定检索预算下直接奖励能提升预测的上下文单元。

---

### 多样性感知的检索集合重排
将检索/技能路由从独立点式相关性排序转为集合选择，显式平衡相关性与非冗余性，避免把预算浪费在功能冗余的候选上。

#### 📄 [Beyond Top-k Skill Retrieval: Diversity-Aware Skill Routing for LLM Agents](https://huggingface.co/papers/2609.05824)
- **奖励设计**: 用 Determinantal Point Process 做多样性感知重排，并用 query-residual diversity kernel 惩罚冗余技能重叠，同时降低仅由共享查询相关性带来的惩罚；目标为相关性+非冗余性联合。
- **💡 启发**: 在检索奖励中加入集合级互补性/去冗余内在奖励，奖励能覆盖复杂任务所需互补证据的检索集合，而非独立 Top-k 相关性。

---

### 迭代检索与推理协同的动态证据收集
让模型在推理过程中迭代搜索外部知识并动态收集证据，强调检索与推理互补；单独检索可能提升稀有实体但伤害整体准确率，二者结合最佳。

#### 📄 [Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking](https://huggingface.co/papers/2609.10745)
- **奖励设计**: 摘要中为 training-free 框架，无显式 RL 奖励；通过推理-capable VLM 在 Wikipedia 上迭代搜索和推理。受控实验发现 reasoning 与 retrieval 互补，检索无推理可提升稀有实体准确率但可能伤害整体准确率。
- **💡 启发**: 检索奖励应与推理正确性/整体任务收益耦合，而不是只奖励检索覆盖或稀有实体命中；需惩罚会伤害整体准确率的无效检索。

---

### 可验证结果奖励与证据使用过程奖励
对封闭任务使用可验证奖励，并对推理轨迹中是否使用可审计空间证据施加过程/格式奖励，使模型学会给出有证据支撑的结论。

#### 📄 [CARDEA: Auditable Reasoning Grounded in Spatial Evidence for End-to-End Coronary Angiography Interpretation](https://huggingface.co/papers/2609.06931)
- **奖励设计**: 采用 reinforcement learning with verifiable rewards (RLVR)，并设计 CoB reward 鼓励在 reasoning trace 中使用 bounding-box；训练阶段包括视觉特征对齐、自蒸馏 Chain-of-Box 冷启动和 RLVR。
- **💡 启发**: 在动态检索奖励中组合可验证结果奖励与证据使用过程奖励（如是否引用检索到的证据/边界框），提升可审计性和证据落地。

---

### 世界模型奖励的偏差与噪声修正
用世界模型替代昂贵环境执行来扩展 agent RL，但世界模型奖励会带偏差和噪声；通过在线去偏与逆方差去噪修正奖励，提升收敛和训练速度。

#### 📄 [Scaling Automatic Research Agents via World Models](https://huggingface.co/papers/2608.12564)
- **奖励设计**: World Model RL (WMRL) 用世界模型替代环境执行，并针对不完美世界模型奖励的 bias 与 noise，引入 Online Debiasing 与 Inverse-Variance Denoising；理论证明二者严格改善收敛保证。
- **💡 启发**: 若用模拟检索环境或 LLM 裁判生成动态检索奖励，应加入在线去偏和方差加权去噪，避免把有偏/高噪的模拟反馈直接当作真实检索收益。

---

### 负向自蒸馏与动态门控的缺陷规避
不模仿特权正确解，而是让模型生成问题特定的负面条件，并推动学生分布远离该负面教师；用动态门控隔离 reasoning-critical tokens，只惩罚行为缺陷。

#### 📄 [Negative Self-Distillation: Learning to Reason by Avoiding Flaws](https://huggingface.co/papers/2609.11699)
- **奖励设计**: Negative Self-Distillation (NSD) 使用模型自身生成 question-specific negative condition（如 careless reasoner），推动学生分布远离自生成负面教师；动态门控自动识别并隔离 reasoning-critical tokens，使梯度只针对行为缺陷，保留语言能力。
- **💡 启发**: 可把“错误检索/无证据推理/冗余搜索”构造为负向条件，设计负奖励或发散目标，并用门控只惩罚检索-推理关键动作，避免损害基础语言能力。

---

## ⚠️ 现有研究的局限与空白 (Gaps & Contradictions)

当前动态检索奖励设计仍高度分裂：SAS 用最终语言建模损失隐式优化上下文排序，DSR 用相关性+非冗余集合目标做重排，Think Before You Link 发现检索与推理互补但未给出可训练奖励，CARDEA 使用可验证结果奖励加证据使用过程奖励，WMRL 和 NSD 则分别处理模拟奖励偏差与负向缺陷规避。共同空白是缺少一个统一奖励，能同时覆盖何时检索、检索什么、检索多少、如何重写查询，并对每个检索动作做反事实信用分配；信息增益、不确定性、冗余度、推理可利用性、可验证证据使用这些信号尚未被整合为动作级过程奖励。另一个关键矛盾是“检索更多并不总是更好”：检索可提升稀有实体准确率，却可能伤害整体准确率，说明检索奖励必须与推理和最终任务收益耦合，否则会鼓励无效或有害检索。

## 🚀 Top 3 可立即尝试的行动思路 (Actionable Ideas)

1. **用最终任务损失或可验证结果作为检索动作的隐式过程奖励：参考 SAS，训练时把检索/上下文 selector 的连续分数注入下游模型，让语言建模损失或 RLVR 结果直接反传，在固定预算下奖励能提升预测的上下文/文档。**

2. **在检索奖励中加入集合级非冗余/互补性项：参考 DSR 的 DPP 与 query-residual diversity kernel，对冗余技能/文档重叠施加惩罚，奖励能覆盖复杂任务所需互补证据的集合，而非独立 Top-k 相关性。**

3. **设计“推理耦合+负向门控”的检索奖励：参考 Think Before You Link 的检索-推理互补结论和 NSD 的负面条件/动态门控，只奖励推理能利用并提升整体答案的检索动作，对无证据使用、冗余检索、伤害整体准确率的检索施加负向惩罚；可结合 CARDEA 的 CoB/可验证奖励做证据使用校验。**

