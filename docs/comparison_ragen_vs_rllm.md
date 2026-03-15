# RAGEN vs rLLM 对比分析

本文档基于对 RAGEN 和 [rLLM](https://github.com/agentica-project/rllm) 两个代码库的深入阅读，梳理二者在架构、功能、优势上的差异，以及 RAGEN 值得保留的核心竞争力。

---

## 一、架构层面差异

| 维度 | RAGEN | rLLM |
|------|-------|------|
| **核心范式** | StarPO (State-Thinking-Actions-Reward Policy Optimization) | Agent/Env + Workflow 双范式 |
| **训练后端** | veRL（单后端，深度集成） | Verl / Tinker / Fireworks / OpenAI（多后端可插拔） |
| **推理引擎** | vLLM 直接集成 | vLLM / SGLang / OpenAI API / LiteLLM |
| **配置系统** | Hydra + OmegaConf（resolver 支持） | Hydra + OmegaConf（类似） |
| **执行模式** | 同步 + 线程池并行 | Async/Await + ThreadPool，支持 128-256 并行 workflow |
| **数据抽象** | veRL 的 DataProto + TensorDict | 自定义 Step/Trajectory/Episode + DataProto 转换 |

### 关键架构差异说明

- **RAGEN** 的三层架构：Agent（LLMAgentProxy）、Environment（EnvStateManager）、Context（ContextManager），训练逻辑集中在 `RayAgentTrainer` 中。
- **rLLM** 的分层更多：Agent → Workflow → ExecutionEngine → RolloutEngine → Trainer，通过 Workflow 抽象将交互逻辑与训练逻辑解耦。

---

## 二、RAGEN 的独特优势（值得保留）

### 1. SNR-Adaptive Rollout Filtering（信噪比自适应过滤）

**文件**: `ragen/trainer/rollout_filter.py`（811 行）

rLLM 没有这个机制。RAGEN 在 PPO 更新前根据组内 reward 方差过滤低信号 prompt，有效减少梯度噪声。

- 策略丰富：`top_p`, `top_k`, `top_k_abs`, `min_p`
- 支持 `linear`（score-sum）和 `softmax`（prob-mass）两种概率模式
- 可配置 metric：`reward_variance`（默认）、`reward`、`entropy`
- rLLM 的 Rejection Sampling 解决类似问题，但思路不同（过量生成后筛选成功轨迹 vs 按方差过滤低信号组）

### 2. Collapse Detection（坍缩检测）

**文件**: `ragen/trainer/collapse_metrics.py`（854 行）

这是 RAGEN V2 的核心学术贡献，rLLM 完全没有类似机制。区分两种坍缩类型：

- **Entropy Collapse**: H(Z|X) 低 — 同一输入下推理多样性不足
- **Template Collapse**: I(X;Z) 低 — 推理与输入无关（模板化输出）

通过 cross-scoring 在经验输入分布下估计互信息。相关指标：
- Retrieval Accuracy（检索准确率）
- MI Estimate（互信息估计）
- Conditional Entropy（条件熵）

### 3. 三种上下文窗口模式

**文件**: `ragen/llm_agent/ctx_manager.py`（1625 行）

每种模式对应不同的 RL problem formulation，配合 episode-level deduplication：

| 模式 | 描述 | 采样方式 |
|------|------|---------|
| `full` | 完整历史在上下文中 | 每个环境每步一个样本 |
| `limited_multi_turn` | 最近 k 轮滑动窗口 | 每轮一个独立样本，episode-ID 去重 |
| `single_turn` | 无历史，每轮独立 | 每轮独立样本，turn 数倍增 batch |

rLLM 的 `CumulativeWorkflow` 大致对应 `full` 模式，但没有 `limited_multi_turn` 的精细控制和 episode 去重机制。

### 4. Bi-level GAE（双层广义优势估计）

**文件**: `ragen/trainer/core_algos.py`

- 轨迹级 + 回合级的层次化优势计算
- `high_level_gamma`（默认 0.95）单独控制跨回合折扣
- rLLM 只有标准的 trajectory-level / step-level（broadcast / per-step）优势

### 5. 分组环境管理

**文件**: `ragen/llm_agent/es_manager.py`（499 行）

```
train_batch_size = env_groups × group_size
例: 8 groups × 16 samples = 128 total
```

- 同组 = 同配置不同种子（prompt），组间可混合不同环境类型
- 支持单次训练中混合多种环境，有利于 curriculum learning
- 线程池并行，按环境类型分组执行
- rLLM 的环境管理更扁平，没有显式的分组抽象

### 6. Reasoning Token 独立追踪

- `<think>...</think>` 中的推理 token 单独提取、记录 `reasoning_ids`
- 推理 ID 参与互信息计算（collapse detection 的基础）
- 支持 token 级别的 reasoning 分析
- rLLM 虽支持 DeepSeek 的 thinking tag，但不做独立追踪和分析

### 7. Response Masking 精细控制

- 只在 assistant token 上计算 loss，防止 value function 学到 P(state) 成分
- 支持 Qwen 和 Llama-3 的模型特定 token 检测
- 经验上提升 rollout 稳定性和优势估计质量

### 8. Modular Memory System

**文件**: `ragen/llm_agent/ctx_manager.py`

- 环境特定的历史格式化（工厂模式）
- `SimpleMemory`（默认）、`AlfWorldMemory`（ALFWorld 专用）
- 可扩展：新环境无需修改核心逻辑

---

## 三、rLLM 的优势（RAGEN 可借鉴）

### 1. Workflow 抽象

- 将 agent/env 交互封装为 Workflow，支持任意控制流（分支、循环、多 agent）
- `SingleTurnWorkflow`, `MultiTurnWorkflow`, `CumulativeWorkflow`, `DistillationWorkflow`, `EvalProtocolWorkflow`
- RAGEN 的交互逻辑直接写在 `agent_proxy.py`，耦合度更高

### 2. 多后端支持

- Verl / Tinker / Fireworks / OpenAI 可插拔切换
- RAGEN 深度绑定 veRL，灵活性较低

### 3. SDK 自动轨迹收集

```python
from rllm.sdk import session, trajectory

with session(experiment="v1") as sess:
    @trajectory(name="solver")
    async def solve(problem):
        response = await llm.chat.completions.create(...)
        return response.content
```

- 装饰器自动收集轨迹
- OpenTelemetry 分布式追踪
- SQLite / 内存存储后端
- RAGEN 没有类似的轻量级数据收集工具

### 4. 更丰富的生产级 Agent

- MathAgent, CodeAgent, SWEAgent, WebArenaAgent, ToolAgent, MiniWoBAgent 等
- 已发布模型：DeepScaleR-1.5B (43.1% AIME), DeepCoder-14B (60.6% LiveCodeBench), DeepSWE-32B (59% SWEBench)
- RAGEN 的环境更偏学术（Sokoban, FrozenLake, Sudoku, Countdown 等）

### 5. Tool 集成与 MCP 支持

- 完整的 Tool Registry + JSON Schema + MCP 协议
- Web tools, Code execution tools
- RAGEN 没有独立的 tool 系统

### 6. Rejection Sampling

- 过量生成（multiplier 倍）后筛选成功轨迹
- 降低 batch 方差
- RAGEN 通过 SNR filtering 从不同角度解决类似问题

### 7. 大规模异步并行

- 128-256 个并行 workflow，async/await 原生支持
- ThreadPoolExecutor（64 workers）处理 env 操作
- Retry 机制（默认 3 次）
- RAGEN 的线程池并行规模相对较小

### 8. Token-Level Trajectory Validation

- 跨多步轨迹检测 token 不匹配
- 验证累积 chat history 的一致性
- 优雅处理 retokenization 问题

---

## 四、共同点

- 都基于 **veRL + Ray + vLLM** 技术栈
- 都支持 **GRPO, REINFORCE, RLOO** 等优势估计器
- 都用 **Hydra** 做配置管理
- 都支持**多轮 agent-env 交互**
- 都做 **token-level 的 reward/mask** 处理
- 都使用 **FSDP** 做分布式训练

---

## 五、总结

### RAGEN 最值得保留的核心竞争力

| 优先级 | 特性 | 原因 |
|--------|------|------|
| P0 | Collapse Detection (MI-based) | 学术独创性最强，rLLM 完全没有 |
| P0 | SNR-Adaptive Filtering | 训练稳定性的关键机制，rLLM 无对应 |
| P1 | 三种 context window mode + episode dedup | multi-turn RL 的精细建模 |
| P1 | Bi-level GAE | 层次化优势估计，rLLM 只有单层 |
| P2 | Reasoning token 独立追踪 | 支撑 collapse 分析的基础设施 |
| P2 | 分组环境管理 | 混合环境训练、curriculum learning |

### 可从 rLLM 借鉴的方向

| 优先级 | 方向 | 收益 |
|--------|------|------|
| P1 | Workflow 抽象 | 解耦交互逻辑与训练逻辑，提升可扩展性 |
| P1 | 多后端支持 | 不绑死 veRL，适配更多部署场景 |
| P2 | Tool/MCP 集成 | 扩展到 tool-use 场景 |
| P2 | SDK 轨迹收集 | 降低数据采集门槛 |
| P3 | 大规模异步并行 | 提升 rollout 吞吐量 |
