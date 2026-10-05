#  Agentic RL 动态检索奖励优化专项调研 - 2026-10-05

## 🧠 核心研究范式 (Core Paradigms)

### 面向可复用检索子技能的分层过程奖励与自蒸馏
该范式把多步 Agent 的扁平动作流按可复用性合并为经验树节点，每个节点表示高频且承载成功结果的技能如何由子技能组成。奖励不再均匀分配给所有 token，而是通过离线 RL 训练实例、在线 RLVR 自适应技能 bonus、on-policy self-distillation 特权上下文来强化可复用的过程结构。

#### 📄 [X-Tree: Tokenizing Reusable Experience for Efficient Agent Generalization](https://huggingface.co/papers/2609.32993)
- **奖励设计**: 摘要原文：offline RL 中每个 X-Tree 节点作为训练实例；online RLVR 中引入 adaptive skill bonus；on-policy self-distillation 中 X-Tree 作为 self-teacher 的 privileged context。动作 span 按可复用性评分，并将 canonicalized actions 合并为 reusable eXperience tree。
- **💡 启发**: 可把动态检索拆成 query 重写、检索源选择、证据验证等可复用技能节点，对高频成功检索子过程给自适应技能 bonus，并用检索经验树指导新任务，而不是对整条轨迹 token 均匀奖励。

---

### 未来信息需求驱动的前瞻选择与稀疏-稠密过程监督
该范式反对仅按当前内容相关性或局部视觉合理性做选择/生成，而是预测未来需要什么信息，并用未来需求表示、目标帧或未来隐状态对齐来指导中间过程。监督信号上结合显式稀疏目标与隐式稠密逐步引导，缓解长程信用分配问题。

#### 📄 [FrameMorrow: Future-guided Frame Selection with Prospective Tokens for Long-Horizon Video Generation](https://huggingface.co/papers/2609.38839)
- **奖励设计**: 摘要未提出显式 RL 奖励；其选择依据是预测一组 prospective tokens 来表示未来信息需求，并据此从历史中选显式历史帧，而不是仅根据当前内容判断历史相关性。
- **💡 启发**: 动态检索奖励可评估所选文档是否满足未来推理/答案生成的信息需求，而非只奖励当前 query 相似度；可引入未来需求表示作为检索动作的过程信号。

#### 📄 [ProAR: Learning Prospective Reasoning with Autoregressive Video Models](https://huggingface.co/papers/2610.03664)
- **奖励设计**: 摘要原文：goal-frame prediction 通过非对称 attention mask 将生成锚定到长期结果；future representation self-alignment 鼓励当前 hidden states 预测 upcoming temporal dynamics；二者结合 explicit sparse target supervision 与 implicit dense step-wise guidance。
- **💡 启发**: 检索 RL 可采用稀疏最终成功奖励加稠密未来表示对齐，奖励中间检索动作是否让模型更接近目标答案所需状态，从而改善只有最终奖励时的 credit assignment。

---

### 证据接地与防 reward hacking 的奖励门控
该范式强调直接优化表面指标或显式 slop-aware prompting 会触发 reward hacking，因此贡献被限制在可验证证据支持的范围内。修订或优化只在实验记录支持时发生，避免模型通过改写表面文本而非真正改善推理来获取奖励。

#### 📄 [Science or Slop?: Benchmarking and Mitigating Scientific Slop in AI-Generated Papers](https://huggingface.co/papers/2610.00531)
- **奖励设计**: 摘要原文：direct slop-aware prompting triggers reward hacking；SciSlopHarness guides a fixed LLM to revise slop only where the experiment records support the change；在不需要 human reference targets 的情况下，将剩余 AI-human gap 较最强 revision baseline 降低 63%。
- **💡 启发**: 动态检索奖励应把可验证证据/引用支持作为门控，只有检索到的证据真正支持当前推理或答案时才给奖励，防止策略优化检索数量、引用形式或表面相关性。

---

### 检索决策模型的奖励归一化偏差与校准
该范式将检索训练的 sentence encoder 转为 schema-constrained typed decision 模型，一次前向输出可阈值化概率。其关键发现是 RL 式 recipe 的奖励归一化会放大噪声 score-function 项，导致性能落后；无偏 leave-one-out 估计器可恢复大部分差距，温度缩放后 cross-entropy 的校准也不差。

#### 📄 [From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders](https://huggingface.co/papers/2610.02486)
- **奖励设计**: 摘要原文：released RLCD recipe 的 reward normalisation 使 noisy score-function term 膨胀 3.6-15 倍；unbiased leave-one-out estimator 恢复大部分性能差距；温度缩放后没有目标明显优于 cross-entropy 的校准。
- **💡 启发**: 若用 RL 优化检索动作，必须检查 reward normalization 与方差，采用无偏 LOO/基线；检索/不检索等 typed decisions 需要校准阈值，而不是盲目优化原始奖励分数。

---

### Agentic search 的训练配方缺口与作者判断监督
该范式揭示在作者标注的文献检索任务中，Agentic search 即使调用同一 retriever，也不优于 embedding retrieval，说明仅靠工具调用或多步搜索不会自动带来专家检索直觉。作者提供的判断与 rationale 可作为训练/评估信号来源，但摘要未给出显式奖励函数。

#### 📄 [ScholarCatalyst: A Benchmark for Retrieving Papers That Inspire New Research](https://huggingface.co/papers/2610.02202)
- **奖励设计**: 摘要未提出显式奖励函数；提供 184 位 lead authors 对 207 篇近期计算机科学论文候选是否推进项目的判断和详细 rationale，并报告 Agentic search 与 embedding retrieval 的 Recall@20 对比：0.42 vs. 0.48，Claude Fable 5.1 agent 仅 0.51。
- **💡 启发**: 不要默认 Agentic search 优于 embedding retrieval；可用作者 rationale 构造检索过程偏好或过程奖励，训练模型从广语料中找真正启发新研究的文献。

---

## ⚠️ 现有研究的局限与空白 (Gaps & Contradictions)

当前奖励设计存在明显分裂：一类方法偏好可复用技能 bonus 或未来需求对齐等过程信号，但检索特有的“何时检索、检索多少文档、如何重写查询、何时停止”仍缺乏统一奖励定义；另一类方法强调证据接地和防 reward hacking，却可能限制探索和召回。ScholarCatalyst 进一步显示，Agentic search 即使调用同一 retriever 也不优于 embedding retrieval，说明现有多步搜索训练/奖励并未捕捉专家检索直觉。SBERT2S1 则提示 RL 奖励归一化本身可能引入巨大偏差，而很多 agentic RL 工作未系统处理 reward normalization、校准与信用分配的联合问题。

最大空白在于：如何把可验证证据门控与未来需求导向的稠密过程奖励结合，同时保持奖励无偏、校准且不易被 hack；如何把动态检索轨迹拆成稳定、可复用的技能节点，并在稀疏最终成功与稠密中间信号之间做可靠 credit assignment。这些矛盾是创新的直接突破口。

## 🚀 Top 3 可立即尝试的行动思路 (Actionable Ideas)

1. **构建动态检索的 X-Tree：将检索轨迹按可复用性合并为 query 重写、源选择、证据验证、停止检索等技能节点；在线 RLVR 中对成功且高频的检索子过程给 adaptive skill bonus，离线 RL 用节点作为训练实例，on-policy self-distillation 用检索经验树作为 privileged context。**

2. **引入未来需求表示作为检索过程奖励：借鉴 FrameMorrow 的 prospective tokens 和 ProAR 的 future representation self-alignment，奖励所选文档对未来推理/目标答案所需信息的贡献，并用稀疏最终成功奖励加稠密未来对齐奖励缓解长程信用分配。**

3. **用证据门控加无偏奖励归一化防 reward hacking：借鉴 SciSlopHarness 的“只在实验记录/可验证证据支持处更新”思想门控检索奖励，同时借鉴 SBERT2S1 的 leave-one-out 无偏估计与温度校准，修正 RL 检索中 reward normalization 导致的 noisy score-function 膨胀。**

