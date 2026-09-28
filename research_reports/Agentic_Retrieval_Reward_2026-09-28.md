#  Agentic RL 动态检索奖励优化专项调研 - 2026-09-28

## 🧠 核心研究范式 (Core Paradigms)

### 分阶段奖励可靠性与 Search Agent 后训练
通过串行后训练阶段，让模型先从硬、可验证奖励中学习基础能力，再逐步过渡到更软的基于 judge 的信号；奖励可靠性被用作安排阶段顺序的实践原则，并对 RL prompts 做难度过滤以保持有效学习范围。

#### 📄 [Rufus-Air: An Open LLM Post-Training Recipe](https://huggingface.co/papers/2609.29421)
- **奖励设计**: 八阶段串行后训练流水线包含 Search Agent 阶段；阶段从硬、可验证奖励逐步过渡到更软的基于 judge 的信号；reward reliability 被用作排序阶段的实践原则，并做难度过滤使 RL prompts 处于有效学习范围。
- **💡 启发**: 在动态检索 RL 中，可先建立可验证的检索/答案结果奖励，再按奖励可靠性逐步引入 judge 软奖励，并用难度过滤控制检索 prompt 的学习价值。

---

### 角色解耦与过程 rubric 的信用分配
将深度搜索中的规划、证据使用与合成解耦为不同角色，使用持续摘要状态降低上下文噪声；训练时结合终端结果奖励与轮级 rubric 评估，并计算角色特定优势，从而更精确地分配每一步检索与合成决策的信用。

#### 📄 [IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis](https://huggingface.co/papers/2609.29444)
- **奖励设计**: Role-Decoupled Policy Optimization (RDPO) 将终端结果奖励与轮级 rubric 评估结合，并计算角色特定优势以进行更精确的信用分配；训练角色解耦的 Planner 与 Synthesizer，并用 summary 作为搜索的持续状态。
- **💡 启发**: 用角色解耦降低一个策略同时规划、检索、合成的耦合，并用角色特定优势对检索决策和证据整合分别给信用，适合动态检索中的细粒度 reward shaping。

---

### 动作评判与状态编辑的动态反馈
不模拟高熵工具响应，而是建模推理与动作如何改变未来任务进展；通过 Action Judge 区分关键、探索性与噪声决策，并用 State Revision 编辑噪声推理-动作延续，最终与真实执行整合以改变后续决策状态。

#### 📄 [Agent-Editing World Model: Rethinking World Modeling for LLM Agents](https://huggingface.co/papers/2609.28416)
- **奖励设计**: Action Judge 区分 Critical、Exploratory、Noisy 决策；State Revision 编辑来自同一观察历史的 noisy reasoning-action continuations；EditAct 与真实执行整合，直接改变后续决策状态；AEWM-RFT 对 verified EditAct trajectories 做 rejection sampling fine-tuning。
- **💡 启发**: 可把检索历史中的噪声动作显式标注与编辑，而不是让错误假设累积；并用 verified 轨迹做离线 RFT，提升检索策略对动态反馈的利用效率。

---

### 基于内在质量与新颖性度量的过程奖励
定义与下游效用相关的内在有趣性指标，并用 proof difficulty 作为计算这些度量的 primitive；优化该指标可产生更 out-of-distribution 的结果，并可用于排序候选与指导搜索。

#### 📄 [Learning to Discover Interesting Mathematics](https://huggingface.co/papers/2609.28603)
- **奖励设计**: 将 theorem 的 intrinsic interestingness 定义为 proof length 与 statement length 之比；该指标与定理下游效用相关；使用 conditioned on premises 的 proof difficulty 作为计算这些度量的 primitive；优化该指标可产生更 interesting theorems，并用于 ranking conjectures 和 guiding proof search。
- **💡 启发**: 跨域启发是：检索奖励可考虑加入证据集合的内在质量与新颖性度量，例如奖励紧凑、非冗余、有信息量的证据，而非仅依赖最终正确性。

---

## ⚠️ 现有研究的局限与空白 (Gaps & Contradictions)

现有工作存在三类未统一：Rufus-Air 强调从硬可验证奖励过渡到软 judge 信号，并按 reward reliability 排序阶段，但未说明动态检索动作本身的奖励如何随步校准；IterSynth 用终端结果奖励、轮级 rubric 与角色特定优势解决信用分配，但 rubric 的可靠性与校准未在摘要中说明；AEWM 用 Action Judge 区分 Critical、Exploratory、Noisy 并编辑 noisy 历史，但判断标准与真实任务收益之间的校准仍开放。Calibration 论文指出标准校准需要 confidence score 与 correctness judgment，而开放式生成中二者定义仍是开放挑战，这直接对应动态检索中何时停止检索、是否继续检索以及置信度能否作为奖励的空白。另一个空白是：多数方法仍以结果或规则/评分器信号为主，缺少对检索动作信息增益、冗余检索惩罚、证据新颖性与紧凑性的统一定义；Learning to Discover Interesting Mathematics 的 proof length/statement length 比与 proof difficulty 提供跨域启发，但尚未被迁移到检索步级奖励。

## 🚀 Top 3 可立即尝试的行动思路 (Actionable Ideas)

1. **先复现 Rufus-Air 的 Search Agent 阶段奖励顺序：在动态检索 RL 中先用可验证最终答案或引用正确性作为硬奖励，再按 reward reliability 逐步加入 judge-based 软奖励，并对检索 prompt 做难度过滤。**

2. **实现 IterSynth/RDPO 式检索信用分配：将策略拆为 Planner 与 Synthesizer 或等价角色，用终端结果奖励加轮级 rubric 评估，并计算角色特定优势，奖励有效检索与证据整合，惩罚无效检索与上下文噪声。**

3. **引入 AEWM 式 Action Judge 与 State Revision：把每步检索、停止、重写动作标为 Critical、Exploratory、Noisy，对 Noisy 历史做状态修正，并用 verified EditAct 轨迹做拒绝采样微调；同时将校准，即 confidence 与最终 correctness 的对齐，作为停止或继续检索的辅助奖励。**

