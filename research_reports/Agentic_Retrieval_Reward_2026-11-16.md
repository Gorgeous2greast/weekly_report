#  Agentic RL 动态检索奖励优化专项调研 - 2026-11-16

## 🧠 核心研究范式 (Core Paradigms)

### 记忆/证据层阈值门控的按需检索
该范式不默认执行检索，而是用多层记忆、匹配阈值和生成器对证据充分性的判断来决定是否查询schema或外部证据。其核心是在准确率、输入token和延迟之间做按需权衡。

#### 📄 [MEM-SQL: Memory-Empowered Text-to-SQL](https://huggingface.co/papers/2611.18338)
- **奖励设计**: 摘要未给出显式RL奖励函数；MEM-SQL存储整查询模板、可复用SQL块和领域概念到schema元素的映射，仅当某层所有需求匹配高于该层阈值时使用该层；On-demand Schema Linking在无层合格或生成器判断检索证据不足时才查询schema。评估指标为执行准确率、输入token和延迟。
- **💡 启发**: 可把层匹配阈值与证据充分性判断转化为是否触发检索的门控奖励，从而优化何时查schema/记忆的决策，并同时约束token与延迟。

---

### 多证据形态的自适应组合与成本约束
该范式在检索到记忆后，不暴露单一压缩表示，而是保留摘要、caption、 grounded visual views、原始图像和交互历史等多种证据形态，并用轻量可训练selector为每个检索记忆组合证据子集。其目标是在提升任务性能的同时显著降低token消耗。

#### 📄 [HiVe-Mem: Interleaving Horizontal Memory Expansion and Vertical Evidence Composition for Multimodal Agents](https://huggingface.co/papers/2611.12872)
- **奖励设计**: 摘要未给出显式奖励公式；横向记忆扩展用文本和视觉锚点将语义检索种子扩展为关系图邻居，纵向为每个检索记忆保留多种证据形式，轻量可训练selector独立组合证据子集；以三个多模态记忆基准的任务性能和约90% token降低作为评估信号。
- **💡 启发**: 将证据子集组合视为检索后的细粒度动作，用任务性能与token成本的多目标信号训练selector，可为过程奖励提供可操作粒度。

---

### 工具增强检索的收益评估
该范式通过给模型额外终端和网络访问，观察其是否检索缺失知识以及性能变化，用以评估检索工具对模型能力的增益。摘要显示所有模型均有提升，差距缩小但仍有约4分，且模型排名大体不变。

#### 📄 [JFinBench: A Continuously Refreshable Japanese Financial Benchmark from the Latest Certification Exams](https://huggingface.co/papers/2611.18130)
- **奖励设计**: 摘要未给出奖励函数设计；JFinBench在模型可额外使用终端和网络访问时，评估其检索缺失知识后的性能变化，并报告所有模型提升、差距缩小但约4分仍存、模型排名大体不变。
- **💡 启发**: 可把工具访问前后的性能差作为检索收益的离线评估信号，用于校准动态检索奖励，而不是仅依赖最终答案正确性。

---

## ⚠️ 现有研究的局限与空白 (Gaps & Contradictions)

严格基于摘要，这些论文没有给出稀疏/稠密奖励、过程奖励、信息增益奖励或不确定性惩罚等显式RL奖励公式；MEM-SQL的层阈值与证据不足判断、HiVe-Mem的可训练selector证据组合属于系统门控或选择机制，并非明确的奖励函数；JFinBench主要是 benchmark 评估工具增强后的性能变化。矛盾在于：按需检索强调token和延迟成本，但MEM-SQL摘要指出On-demand Schema Linking的准确率收益取决于记忆设计，HiVe-Mem报告约90% token降低但摘要未给出成本-准确率权衡细节，JFinBench显示工具检索普遍提升但仍有约4分差距且排名大体不变。空白是：缺少将证据充分性、检索收益、无效检索和成本转化为可优化奖励信号的明确方法，动态检索的奖励函数设计在这些工作中尚未显式化。

## 🚀 Top 3 可立即尝试的行动思路 (Actionable Ideas)

1. **把MEM-SQL的‘层匹配阈值+生成器证据不足判断’显式化为检索触发奖励：证据充分且不检索时给成本节省奖励，证据不足时触发检索并以执行准确率提升作为收益，检索后仍不足则给无效检索惩罚。**

2. **把HiVe-Mem的轻量selector证据子集组合作为动作空间，用任务性能与token消耗（摘要报告约90%降低）构造多目标奖励，训练selector在证据覆盖和成本间权衡。**

3. **利用JFinBench的工具增强前后性能差作为检索收益的离线校准信号：先比较终端/网络访问前后的答案正确率变化，再将该收益差引入在线RL，避免仅使用最终答案的稀疏奖励。**

