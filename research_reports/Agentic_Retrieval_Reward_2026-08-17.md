#  Agentic RL 动态检索奖励优化专项调研 - 2026-08-17

## 🧠 核心研究范式 (Core Paradigms)

### 基于验证器反馈与检索结果校准的记忆增强检索
该范式在参数冻结且不依赖外部专家工具的条件下，通过检索可复用经验/教训来指导模型推理；经验被赋予可校准的可靠性分数，该分数根据后续检索结果动态更新，形成基于检索结果反馈的自进化闭环。

#### 📄 [Spatial Memory Agent: Experience-Grounded Procedure Memory for Spatial Intelligence](https://huggingface.co/papers/2608.12743)
- **奖励设计**: 在可验证空间环境中获得预测答案和奖励；使用验证器引导的反思将已验证的空间经验蒸馏为紧凑可迁移教训；每条教训分配一个 Transfer Reliability Score (TRS)，初始统一，并根据后续检索结果作为未来可迁移可靠性的访问证据进行校准。
- **💡 启发**: 可将检索到的文档、查询模板或工具输出视为可复用经验，维护其可靠性分数，并把该分数作为检索排序的一部分和内在奖励信号，在无需更新模型权重的情况下持续优化动态检索。

---

### 基于验证器适应度与种群选择的智能体 Harness 进化
该范式将智能体能力视为 prompts、tools、skills 和 control flow 等 harness 组件，通过搜索/进化 harness 而非更新模型权重来提升能力；一些方法采用种群选择、档案重组和 preserve-and-extend 契约，以缓解单线搜索的路径依赖和局部回归；另一些方法用元优化器基于 rollout 反馈递归改进 harness。

#### 📄 [DarwinX: Evolving Agent Harnesses Through Natural Selection](https://huggingface.co/papers/2608.07545)
- **奖励设计**: 每个基准任务使用其自带验证器计算适应度，没有黄金解或人工挑选赢家；preserve-and-extend 契约只接受扩展覆盖率且不回退的变体；失败、教师和自我衍生的证据共享统一编辑接口。
- **💡 启发**: 将动态检索策略（何时检索、查询重写模板、检索文档数量、检索工具选择）作为 harness 的可进化组件，利用任务验证器反馈与种群选择来避免局部最优，并保持跨任务泛化。

#### 📄 [AutoDesign: Meta-Harness Optimization for Long-Horizon Agentic Design](https://huggingface.co/papers/2608.13560)
- **奖励设计**: 元 harness 优化器基于 rollout 反馈引导代码代理递归改进 harness；在论文-海报生成任务上以 PosterBench Score 作为评估/反馈信号，学习到的 DesignHarness 在多种代码代理模型配置上持续提升平均得分。
- **💡 启发**: 可以用元优化器基于长程任务的 rollout 反馈自动调整检索相关 harness（例如检索提示、工具调用逻辑），将终端得分作为稀疏奖励，通过递归改进稳定提升检索驱动任务质量。

---

### 参数空间探索与奖励估计分组优化
该范式将探索从动作空间（如温度缩放）扩展到参数空间：从后验分布采样不同策略参数生成 rollout，以互补方式控制探索；同时通过不同 rollout 分组进行奖励估计，改善 RLVR 的稳定性和样本效率。

#### 📄 [Parameter Exploration for RLVR via Variational Learning](https://huggingface.co/papers)
- **奖励设计**: N/A
- **💡 启发**: N/A

---

## ⚠️ 现有研究的局限与空白 (Gaps & Contradictions)

暂无

## 🚀 Top 3 可立即尝试的行动思路 (Actionable Ideas)

