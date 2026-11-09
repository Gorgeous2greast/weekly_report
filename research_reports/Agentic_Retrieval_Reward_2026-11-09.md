#  Agentic RL 动态检索奖励优化专项调研 - 2026-11-09

## 🧠 核心研究范式 (Core Paradigms)

### 完整轨迹/完整输出级奖励模型引导的策略优化
该范式不再只对单步输出打分，而是由奖励模型评估完整生成结果，并用强化学习引导策略面向结果级与长期目标优化。OneTrans-GSR将完整曝光slate作为评估对象，由奖励模型评价完整slate并引导RL；解码时显式执行业务约束。虽然场景是推荐系统而非动态检索，但其完整轨迹奖励与解码约束可直接迁移到检索-回答轨迹优化。

#### 📄 [OneTrans-GSR: A Generative Slate Ranking Paradigm Unifying Reranking and Mix-Ranking with One Autoregressive Transformer](https://huggingface.co/papers/2611.04720)
- **奖励设计**: 摘要说明训练时由奖励模型评估完整slates，并引导强化学习朝向slate-level与长期目标；摘要未给出具体公式。解码时显式执行业务约束。
- **💡 启发**: 可把查询改写、何时检索、检索文档集合、证据引用与最终答案视为一个完整动作序列，由轨迹级奖励模型整体打分，而不是只对单步检索动作给稀疏奖励；同时把合法性约束放在解码或动作过滤阶段。

---

### Rubric/Checklist式细粒度奖励分解
该范式将复杂生成或编辑质量拆解为可核查的rubric与checklist奖励，分别优化不同能力，再通过策略整合保持专门化能力。WeLM-Image用rubric- and checklist-based rewards分别优化生成与编辑，再用noise-aware on-policy distillation整合策略。对动态检索而言，这提示可用checklist将证据相关性、覆盖度、忠实性等过程质量显式化。

#### 📄 [WeLM-Image: A Unified Image Generation Model towards Visual Intelligence](https://huggingface.co/papers/2611.07392)
- **奖励设计**: 摘要说明分别使用基于rubric和checklist的奖励优化生成与编辑，随后通过noise-aware on-policy distillation整合专门化策略；摘要未披露具体条目或公式。
- **💡 启发**: 可将检索过程奖励拆成可核查checklist，例如证据是否支持答案、是否覆盖关键子问题、引用是否忠实、是否存在冗余检索；这比单一终局正确性奖励更适合过程性检索策略优化。

---

### 可验证性验证与可行性硬门控的方向感知评分
该范式强调对最终产物进行可验证、可复现的机器检查，并用硬门控排除不可行答案，同时采用方向感知单侧评分避免机械匹配参考。TB-MathModel对最终数值核心按多个Outstanding论文复现评分，direction-aware one-sided scoring不惩罚超过anchor，且用程序化不变量检查与Lean 4义务进行两层可行性验证，硬门控将不可行答案置零。对检索奖励的启发是引入证据可验证性与动作可行性门控。

#### 📄 [Terminal-style Mathematical Modeling Bench: The First Agentic and Verifiable Benchmark for Mathematical Modeling](https://huggingface.co/papers/2611.08237)
- **奖励设计**: 最终数值核心对照Outstanding论文复现进行评分；采用direction-aware、one-sided scoring，不惩罚超过anchor；使用两层可行性验证，包括六类可判定不变量形式的程序化门控与624个Lean 4义务，并用硬门控将不可行答案置零；同时对照多个人类Outstanding论文面板评分。
- **💡 启发**: 检索奖励可加入可验证硬门控，例如无证据引用不得进入最终答案、越界或重复无效检索动作被零化；同时采用单侧评分鼓励超过参考证据集，而非强迫匹配某个固定检索路径。

---

## ⚠️ 现有研究的局限与空白 (Gaps & Contradictions)

当前与任务最接近的显式奖励设计主要来自完整输出评估、rubric/checklist分解和可验证硬门控，但几乎没有论文在摘要中直接给出动态检索的细粒度奖励函数。主要矛盾在于：终局稀疏奖励可靠但信用分配困难，过程细粒度奖励可塑性强但容易与最终正确性偏离；可行性硬门控能抑制无效检索和幻觉证据，但可能压缩探索空间；rubric/checklist可解释但摘要未说明权重、冲突处理与防奖励黑客机制。另一个明显空白是成本、延迟与可靠性的统一：AgentBaton显示模型handoff可显著降本并保持准确率，Beneath the Surface显示可靠性提升常伴随部署成本增加且演化可能遗忘已覆盖任务，但这些工作没有给出可直接用于动态检索的奖励公式；EvoMemState虽提到RL-based state evolution，但摘要未披露记忆使用策略的具体奖励信号。

## 🚀 Top 3 可立即尝试的行动思路 (Actionable Ideas)

1. **构建轨迹级检索奖励模型：输入完整检索轨迹（查询改写序列、检索文档集合、证据引用、最终答案），输出整体奖励并与最终任务奖励联合训练；直接借鉴OneTrans-GSR用奖励模型评估完整slate并引导RL面向长期目标的做法。**

2. **为动态检索设计rubric/checklist过程奖励：把证据相关性、关键信息覆盖度、引用忠实性、答案可验证性、冗余检索惩罚拆成可核查条目，借鉴WeLM-Image的rubric- and checklist-based rewards，并与终局正确性奖励做加权或分阶段整合。**

3. **引入方向感知单侧评分与可行性硬门控：对检索动作设置可判定的合法性/证据约束（如无证据引用不得进入答案、重复检索与越界查询零化），只惩罚低于参考的质量而不惩罚超过参考；借鉴TB-MathModel的direction-aware one-sided scoring与hard gate，同时可把API成本/延迟作为软惩罚以贴近AgentBaton的成本-准确率Pareto目标。**

