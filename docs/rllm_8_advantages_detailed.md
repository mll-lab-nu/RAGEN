# rLLM 八项可借鉴方向：详细实现分析

本文档深入分析 rLLM 在八个方面的具体实现，作为 RAGEN 未来改进的参考。

---

## 目录

1. [Workflow 抽象](#1-workflow-抽象)
2. [多后端支持](#2-多后端支持)
3. [SDK 自动轨迹收集](#3-sdk-自动轨迹收集)
4. [更丰富的生产级 Agent](#4-更丰富的生产级-agent)
5. [Tool 集成与 MCP 支持](#5-tool-集成与-mcp-支持)
6. [Rejection Sampling](#6-rejection-sampling)
7. [大规模异步并行](#7-大规模异步并行)
8. [Token-Level Trajectory Validation](#8-token-level-trajectory-validation)

---

## 1. Workflow 抽象

### 1.1 是什么

Workflow 是 rLLM 中将 **agent-env 交互逻辑** 从训练器中解耦出来的核心抽象。每个 Workflow 定义一次完整 episode 的执行流程（如何 reset、如何交互、如何收集轨迹），而训练器只负责调度和梯度更新。

**对比 RAGEN**：RAGEN 的交互逻辑直接写在 `agent_proxy.py`（403 行）和 `agent_trainer.py`（1548 行）中，交互流程和训练逻辑耦合。

### 1.2 基类设计

**文件**：`rllm/workflows/workflow.py`

```python
class Workflow(ABC):
    def __init__(self, rollout_engine, executor, timeout=1e6, gamma=0.0, reward_bonus_coeff=0.0):
        self.rollout_engine = rollout_engine     # 推理引擎
        self.executor = executor                 # 线程池（环境操作用）
        self.timeout = int(timeout)              # 超时时间
        self._completed_trajectories = []        # 已完成轨迹

    @abstractmethod
    async def run(self, task, uid, **kwargs) -> Episode | None:
        """核心执行方法，子类必须实现"""

    def commit(self, name=None, agent=None, trajectory=None, reset=False):
        """手动提交一条完整轨迹（深拷贝后存储）"""

    def collect_trajectories(self) -> Episode:
        """自动收集所有轨迹：显式 commit 的 + agent 属性上的"""

    def reset(self, task=None, uid=None):
        """重置所有 agent 和 env 属性（通过反射发现）"""

    def postprocess_episode(self, episode, termination_reason=None, error=None):
        """7 步后处理流水线：清理 → 计算 reward → 调整 reward → 判断正确性 → 收集指标"""

    def run_with_termination_handling(self, task, uid, **kwargs) -> Episode:
        """包装 run()，统一处理超时、异常、终止事件"""

    async def run_in_executor(self, fn, *args, **kwargs):
        """在线程池中运行阻塞操作（如 env.step）"""

    def is_multithread_safe(self) -> bool:
        """检查所有 env 属性是否线程安全"""
```

**关键设计**：
- **反射式发现**：`reset()` 和 `collect_trajectories()` 通过 `dir()` + `getattr()` 自动发现类上所有 `BaseAgent` 和 `BaseEnv` 实例
- **深拷贝安全**：所有轨迹在收集时深拷贝，防止后续修改污染
- **UID 去重**：通过 trajectory UID 避免重复收集

### 1.3 终止处理系统

```python
class TerminationReason(Enum):
    MAX_PROMPT_LENGTH_EXCEEDED = auto()
    MAX_RESPONSE_LENGTH_EXCEEDED = auto()
    ENV_DONE = auto()
    MAX_TURNS_EXCEEDED = auto()
    TIMEOUT = auto()
    UNKNOWN = auto()
    ERROR = auto()

class TerminationEvent(Exception):
    """Workflow 内部抛出，由 run_with_termination_handling 统一捕获"""
    def __init__(self, reason: TerminationReason): ...
```

`run_with_termination_handling()` 保证 **无论如何都返回 Episode**（即使出错也能保留部分轨迹）。

### 1.4 六种 Workflow 实现

#### (a) SingleTurnWorkflow
单次交互：env.reset → agent.update → model.generate → env.step → 结束。

#### (b) MultiTurnWorkflow
多轮交互：循环最多 `max_steps` 次，每轮 model.generate → env.step，直到 `done=True` 或达到上限。

#### (c) CumulativeWorkflow
累积式多轮：与 MultiTurnWorkflow 类似，但**动态管理 prompt 长度**。每轮计算当前累积 prompt token 数，动态调整 `max_tokens = max_response_length - (current_prompt_length - initial_prompt_length)`。当 `max_tokens <= 0` 时提前终止。

#### (d) DistillationWorkflow
知识蒸馏：student 模型生成 → teacher 模型计算 per-token advantage → 用于训练。支持 advantage clipping（`clip_min`, `clip_max`），1% 采样可视化。

#### (e) SimpleWorkflow
最简 Workflow：单次 LLM 调用 + reward function，无环境交互。

#### (f) EvalProtocolWorkflow
评估专用：集成 eval-protocol 框架，支持 MCP 工具，将评估结果转换为 Episode。

### 1.5 Reward 计算流水线

```
postprocess_episode() 的 7 步流程：
1. 设置 episode.id 和 episode.task
2. 清理不完整的最后一步（空 chat_completions）
3. compute_trajectory_reward()  → 默认 sum(step.reward)
4. adjust_step_rewards()        → reward shaping + Monte Carlo return
5. assign_episode_correctness() → 默认 sum(traj.reward) > 0
6. collect_metrics()            → 按 trajectory name 分组计算 mean reward
7. 存储 error 信息和 termination_reason
```

**Reward Shaping**（`reward_bonus_coeff > 0`）：
```python
s[i].reward += bonus_coeff * (raw_reward[i] - raw_reward[i-1])  # 鼓励进步信号
```

**Monte Carlo Return**（`gamma > 0`）：
```python
G_t = R_{t+1} + γ * G_{t+1}  # 反向折扣累积
```

### 1.6 TimingTrackingMixin

通过 Mixin 为 Workflow 添加计时功能，记录每步的 LLM 时间、Env 时间、Reward 时间、总时间，以及 ISO 8601 时间戳。非侵入式设计。

### 1.7 AgentWorkflowEngine

**文件**：`rllm/engine/agent_workflow_engine.py`

负责调度多个 Workflow 实例并行执行：

```python
class AgentWorkflowEngine:
    def __init__(self, workflow_cls, workflow_args, rollout_engine,
                 n_parallel_tasks=128, retry_limit=3):
        ...

    async def initialize_pool(self):
        """创建 asyncio.Queue，预填充 n_parallel_tasks 个 workflow 实例"""

    async def process_task_with_retry(self, task, task_id, rollout_idx):
        """从 Queue 获取 workflow → 执行 → 归还。ERROR 时重试"""

    async def execute_tasks(self, tasks, task_ids=None):
        """asyncio.as_completed 并发执行所有 task"""

    def transform_results_for_verl(self, episodes, task_ids) -> DataProto:
        """Episode → verl 训练格式：prompt/response padding、mask、multimodal 处理"""
```

---

## 2. 多后端支持

### 2.1 是什么

rLLM 支持三种训练后端：**Verl**（默认，分布式 PPO）、**Tinker**（阿里内部框架）、**Fireworks**（云端 pipeline）。用户通过 `backend="verl"` 一行参数切换。

**对比 RAGEN**：RAGEN 深度绑定 veRL，训练器直接继承 `VerlRayPPOTrainer`，无法切换后端。

### 2.2 路由层：AgentTrainer

**文件**：`rllm/trainer/agent_trainer.py`

```python
class AgentTrainer:
    def __init__(self, ..., backend: Literal["verl", "fireworks", "tinker"] = "verl"):
        assert backend in ["verl", "fireworks", "tinker"]
        # fireworks 只支持 workflow_class，不支持 agent/env classes
        # tinker 不支持 workflow_class

    def train(self):
        if self.backend == "verl":
            if self.workflow_class:
                trainer = AgentWorkflowTrainer(...)  # workflow 模式
            else:
                trainer = AgentPPOTrainer(...)        # agent/env 模式
        elif self.backend == "fireworks":
            trainer = AgentWorkflowTrainerFireworks(...)
        elif self.backend == "tinker":
            trainer = TinkerAgentTrainer(...)
        trainer.train()
```

### 2.3 Verl 后端

**两种训练模式**：

**(a) Agent/Env 模式 (`AgentPPOTrainer`)**

继承 verl 的 `RayPPOTrainer`，重写 `_generate_and_fit_batch()`：

```python
class AgentPPOTrainer(RayPPOTrainer):
    def _generate_and_fit_batch(self, batch):
        # Stage 1: 生成 rollout（通过 AgentExecutionEngine）
        trajectories = asyncio.run(self.execution_engine.trajectory_generator(...))

        # Stage 2: 转换为 verl 格式
        batch = self.execution_engine.transform_results(trajectories)

        # Stage 3: 计算 advantage
        batch = compute_advantages(batch, self.config)

        # Stage 4: PPO 更新（调用 verl 原生 actor/critic update）
        self.actor_rollout_ref.update(batch)
```

**(b) Workflow 模式 (`AgentWorkflowTrainer`)**

```python
class AgentWorkflowTrainer:
    def _train_batch_async(self, batch):
        # Stage 1: 执行 workflows（通过 AgentWorkflowEngine）
        episodes = await self.workflow_engine.execute_tasks_verl(batch)

        # Stage 2: Rejection Sampling（可选）
        if rejection_sampling_enabled:
            groups, episodes = apply_rejection_sampling(episodes, groups, config, state)

        # Stage 3: 计算 advantage（GRPO/REINFORCE/RLOO）
        groups = compute_advantages(groups, self.config)

        # Stage 4: 转换为 verl DataProto
        batch = transform_to_verl(groups)

        # Stage 5: 策略更新
        self.actor_rollout_ref.update(batch)
```

### 2.4 Fireworks 后端

**文件**：`rllm/trainer/verl/agent_workflow_trainer_fireworks.py`

Pipeline 模式，适合云端训练：
- 使用 OpenAI API 兼容接口做推理
- Workflow 产生 episodes → 转换为训练数据 → 上传到 Fireworks 进行训练
- 不需要本地 GPU 做推理

### 2.5 Tinker 后端

**文件**：`rllm/trainer/tinker/`（标记为 deprecated）

阿里内部框架，支持 LoRA 微调，提供替代的分布式训练基础设施。

### 2.6 Rollout Engine 抽象

**文件**：`rllm/engine/rollout/rollout_engine.py`

```python
class RolloutEngine(ABC):
    """推理引擎的统一接口"""
    async def get_model_response(self, messages, ...) -> ModelOutput:
        """发送 messages，返回 ModelOutput（content, reasoning, tool_calls, token_ids, logprobs）"""

    @property
    def chat_parser(self) -> ChatTemplateParser: ...

    @property
    def tokenizer(self) -> PreTrainedTokenizer: ...
```

**四种实现**：

| Engine | 推理方式 | 特点 |
|--------|---------|------|
| `VerlEngine` | verl 内置 vLLM rollout | 直接使用 verl 管理的 vLLM 实例 |
| `OpenAIEngine` | OpenAI API 兼容 | 支持任何 OAI 兼容服务（vLLM、SGLang 等） |
| `TinkerEngine` | Tinker rollout | 阿里内部推理引擎 |
| `FireworksEngine` | Fireworks API | 云端推理 |

---

## 3. SDK 自动轨迹收集

### 3.1 是什么

rLLM SDK 提供 **装饰器 + 上下文管理器** 的方式自动收集 LLM 调用轨迹，无需手动构建 Step/Trajectory。用户代码几乎不用改动，只需加 `@trajectory` 装饰器和 `session()` 上下文。

**对比 RAGEN**：RAGEN 必须通过 `ContextManager` 手动构建 prompt、解析 response、追踪 token，没有自动收集机制。

### 3.2 核心用法

```python
from rllm.sdk import session, trajectory, get_chat_client

llm = get_chat_client(api_key="sk-...")

with session(experiment="math_v1") as sess:
    @trajectory(name="solver")
    async def solve(problem: str):
        response = await llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": problem}]
        )
        return response.choices[0].message.content

    traj = await solve("What is 2+2?")
    # traj 是 TrajectoryView，包含自动收集的 steps
    traj.reward = 1.0  # 手动设置 reward
```

### 3.3 Session 上下文管理

**两种后端实现**：

**(a) ContextVar 后端（默认，单进程）**

**文件**：`rllm/sdk/session/contextvar.py`

- 使用 Python `contextvars.ContextVar` 实现线程安全、async 安全的上下文传播
- Session UID：`ctx_{uuid}`
- **层次化**：维护 `_session_uid_chain`（从 root 到 current 的 UID 列表）
- **Metadata 继承**：子 session 自动继承并合并父 session 的 metadata
- **存储**：`SessionBuffer`（线程安全内存字典）

```python
class ContextVarSession:
    def __enter__(self):
        # 读取父 session 的 UID chain
        # 生成当前 UID，追加到 chain
        # 合并父 metadata + 当前 metadata
        # 设置 contextvars

    def __exit__(self):
        # 通过 context token 恢复上一个 session 的 contextvars

    @property
    def llm_calls(self) -> list[Trace]:
        """查询 SessionBuffer，返回当前 session 下的所有 LLM 调用"""
```

**(b) OpenTelemetry 后端（分布式）**

**文件**：`rllm/sdk/session/opentelemetry.py`

- 使用 **W3C Baggage 作为唯一真实来源**（不传递 session 对象）
- Session UID 从 span ID 派生
- 自动跨进程/服务传播（通过 HTTP baggage header）
- SQLite 后端持久化

```python
class OpenTelemetrySession:
    def __enter__(self):
        # 读取父 baggage
        # 创建新 OTel span（span ID 作为 session UID）
        # 构建 UID chain = 父 chain + [当前 UID]
        # 合并 metadata
        # 写入 baggage（JSON 编码）

    @property
    def llm_calls(self) -> list[Trace]:
        """从 SQLite 按 session UID 查询"""
```

### 3.4 @trajectory 装饰器

**文件**：`rllm/sdk/decorators.py`

```python
def trajectory(name: str, **metadata):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 1. 捕获函数输入：inspect.signature().bind()
            func_input = capture_args(func, args, kwargs)

            # 2. 创建内部 session
            with session(trajectory_name=name, **metadata) as sess:
                # 3. 执行用户函数
                result = await func(*args, **kwargs)

                # 4. 从 session 收集所有 LLM 调用
                traces = sess.llm_calls  # 自动收集！

                # 5. 转换为 StepView
                steps = [trace_to_step_view(t) for t in traces]

                # 6. 返回 TrajectoryView（不是原函数返回值！）
                return TrajectoryView(
                    name=name,
                    steps=steps,
                    reward=0.0,        # 用户后续手动设置
                    input=func_input,
                    output=result,     # 原始返回值存在这里
                    metadata=metadata,
                )
        return wrapper
    return decorator
```

### 3.5 LLM 调用拦截

**文件**：`rllm/sdk/chat/openai.py`

**核心策略**：继承 `OpenAI` 客户端，重写 `request()` 方法。

```python
class TrackedChatClient(OpenAI):
    def request(self, cast_to, options, *args, **kwargs):
        # 1. 提取 model, messages
        # 2. 如果启用代理：将 session metadata 编码到 URL slug
        # 3. 调用原始 request()
        # 4. 如果流式：包装为 TrackedStream（累积 chunks → 重建完整响应）
        # 5. 如果非流式：立即 _log_trace()
        # 6. 返回原始 response（用户无感知）
```

**TrackedStream 处理流式响应**：
- 累积所有 chunks
- 在 `StopIteration` 时重建完整 response dict
- 重建 tool_calls（从流式 delta 中按 index 组装）

**_log_trace() 记录**：
```python
def _log_trace(tracer, *, model, messages, response, metadata, latency_ms):
    ctx_metadata = {**get_current_metadata(), **metadata}  # 合并 session metadata
    token_ids = extract_completion_tokens(response)
    tracer.log_llm_call(
        name="chat.completions.create",
        model=model,
        input={"messages": messages},
        output=response,
        metadata=ctx_metadata,
        latency_ms=latency_ms,
        tokens=extract_usage_tokens(response),
        session_name=get_current_session_name(),  # 自动从 context 获取
        session_uids=get_active_session_uids(),   # 完整 UID chain
    )
```

### 3.6 存储后端

**(a) SessionBuffer（内存）**：线程安全 dict，trace 存储到 UID chain 中每个 UID 下（层次可见性）。

**(b) SqliteTraceStore（持久化）**：
```sql
traces(id, context_type, namespace, data JSON, metadata JSON, created_at, updated_at)
trace_sessions(trace_id, session_uid, created_at)  -- 多对多关联表
```
- 复合索引：`(session_uid, created_at)` 快速按 session 查询
- 后台 worker 线程异步写入

### 3.7 LiteLLM Proxy 集成

**架构**：`Client → MetadataRoutingMiddleware → LiteLLM → TracingCallback → Backend`

- **Metadata Slug**：将 session context 编码到 URL 路径中（`/meta/rllm1:BASE64/v1`），不修改请求体
- **Middleware**：拦截请求，从 URL 提取 metadata，注入到 JSON body 的 `metadata.rllm_metadata` 字段
- **TracingCallback**：在 LiteLLM 的 `async_post_call_success_hook` 中记录 trace
- **VerlProxyManager**：自动发现 verl 管理的 vLLM 实例地址，生成 LiteLLM 负载均衡配置

### 3.8 端到端数据流

```
用户代码 @trajectory
    ↓
创建 session context（ContextVar 或 OTel）
    ↓
LLM 调用（client.chat.completions.create）
    ↓
TrackedChatClient.request() 拦截
    ↓
Tracer.log_llm_call()（附带 session UID chain）
    ↓
存储到 SessionBuffer 或 SQLite
    ↓
session.llm_calls 查询
    ↓
trace_to_step_view() 转换
    ↓
TrajectoryView（steps + reward）
    ↓
训练数据提取（token sequences + logprobs）
    ↓
RL 训练循环
```

---

## 4. 更丰富的生产级 Agent

### 4.1 是什么

rLLM 提供 8 种面向不同任务类型的 Agent 实现，覆盖数学推理、代码生成、软件工程、Web 自动化等场景。

**对比 RAGEN**：RAGEN 的 agent 交互通过 `LLMAgentProxy` 统一处理，环境特定逻辑通过 Memory 系统（`SimpleMemory`、`AlfWorldMemory`）适配，没有独立的 Agent 类。

### 4.2 BaseAgent 接口

**文件**：`rllm/agents/agent.py`

```python
class BaseAgent(ABC):
    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def update_from_env(self, observation, reward, done, info): ...

    @abstractmethod
    def update_from_model(self, response) -> Action: ...

    @property
    def chat_completions(self) -> list[dict]:
        """OpenAI 格式的消息历史"""

    @property
    def trajectory(self) -> Trajectory:
        """当前轨迹"""
```

核心数据结构：
- **Step**：单次交互原子单元（prompt_ids, response_ids, logprobs, observation, thought, action, reward, done, mc_return, advantage）
- **Trajectory**：Step 序列 + 轨迹级 reward
- **Episode**：一个或多个 Trajectory + 终止原因 + 正确性标记

### 4.3 八种 Agent 实现

#### (a) ToolAgent — 通用工具调用
- 集成 `MultiTool`（工具注册表或直接 tool_map）
- 从 JSON schema 生成 tool prompt
- 支持 Qwen / R1 两种 tool call 解析格式
- 回退机制：解析失败时默认调用 "finish" tool
- MCPToolAgent 变体：专门处理 MCP 协议工具

#### (b) MathAgent — 数学推理
- `<think>` 标签支持 chain-of-thought
- `accumulate_thinking` 标志控制是否保留历史中的思考过程
- 灵活的 observation 处理：None/空 dict = 更新 reward；问题文本 = 创建新 step

#### (c) CompetitionCodingAgent — 代码生成
- 测试结果格式化为自然语言反馈
- Public/Private 测试分离（避免数据泄露）
- 迭代式问题解决：生成代码 → 执行测试 → 反馈 → 改进

#### (d) SWEAgent — 软件工程
- 双格式解析：OAI function calling + XML 格式（`<function=name>args</function>`）
- 支持 `r2egym` 和 `sweagent` 两种 scaffolding 框架
- bash 命令执行、文件编辑等操作

#### (e) WebArenaAgent — Web 自动化
- 复杂状态管理：accessibility tree 解析/剪枝、HTML DOM 扁平化、截图 base64 编码
- 交互历史追踪：observation 描述、action 原因、action 本身
- 元素 ID 系统：维护 `id2node` 映射
- 动作集：click, type, go_back, stop

#### (f) MiniWobAgent — Mini Web 任务
- BrowserGym HighLevelActionSet 集成
- 多模态观测：axtree + HTML + 截图 + tab 信息
- Step 验证：检查 thought != action（解析成功）、上一步无错误

#### (g) FrozenLakeAgent — 网格导航
- 5-shot in-context learning prompt
- 方向解析：从 code block 提取 Up/Down/Left/Right
- 无效动作检测：检测玩家是否移动

#### (h) AppWorldReactAgent — ReAct 模式
- 生成可执行 Python 代码（非简单 action）
- Jinja2 模板化 prompt（注入用户信息）
- 执行结果格式化：success/failure + output + stdout + stderr

---

## 5. Tool 集成与 MCP 支持

### 5.1 是什么

rLLM 提供完整的 Tool 系统：Tool 基类 → 注册表 → 多工具路由 → MCP 协议支持 → 代码沙箱执行。

**对比 RAGEN**：RAGEN 的环境通过 `BaseEnv` 接口接收 string action，没有独立的 Tool 抽象层。action 解析在 `ContextManager` 中通过分隔符（`||`）完成。

### 5.2 Tool 基类

**文件**：`rllm/tools/tool_base.py`

```python
class Tool(ABC):
    def __init__(self, name, description, function=None):
        self.name = name
        self.description = description

    @property
    def json(self) -> dict:
        """返回 OpenAI function calling 格式的 JSON Schema"""

    def forward(self, *args, **kwargs) -> ToolOutput:
        """同步执行"""

    async def async_forward(self, *args, **kwargs) -> ToolOutput:
        """异步执行"""

    def __call__(self, *args, use_async=False, **kwargs):
        """自动路由到 sync/async"""
```

**ToolOutput 数据结构**：
```python
@dataclass
class ToolOutput:
    name: str
    output: str | list | dict | None = None
    error: str | None = None
    metadata: dict | None = None
```

### 5.3 Tool Registry（单例模式）

**文件**：`rllm/tools/registry.py`

```python
class ToolRegistry:
    _instance = None  # 单例

    def register(self, name, tool_cls): ...
    def instantiate(self, name, *args, **kwargs) -> Tool: ...
    def get(self, name) -> type[Tool]: ...
    def list_tools(self) -> list[str]: ...
    # 支持 registry[name]、name in registry 等字典式操作
```

### 5.4 MultiTool（多工具路由）

**文件**：`rllm/tools/multi_tool.py`

```python
class MultiTool(Tool):
    def __init__(self, tools=None, tool_map=None):
        # tools: list[str] → 从 Registry 查找
        # tool_map: dict[str, type[Tool]] → 直接实例化

    @property
    def json(self) -> list[dict]:
        """返回所有工具的 JSON Schema 列表"""

    def forward(self, tool_name, **kwargs) -> ToolOutput:
        """根据 tool_name 路由到对应 tool"""
```

### 5.5 MCP Tool（Model Context Protocol）

**文件**：`rllm/tools/mcp_tool.py`

```python
class MCPTool(Tool):
    def __init__(self, session, tool_name, tool_description, tool_schema):
        self.session = session  # MCP session 对象

    async def async_forward(self, **kwargs) -> ToolOutput:
        result = await self.session.call_tool(self.name, kwargs)
        content_str = extract_text(result.content)
        return ToolOutput(name=self.name, output=content_str)
```

### 5.6 Tool Parser 系统

**文件**：`rllm/parser/tool_parser.py`

```python
class ToolParser(ABC):
    @abstractmethod
    def parse(self, model_response: str) -> list[ToolCall]: ...

    @abstractmethod
    def get_tool_prompt(self, tools_schema: str) -> str: ...

    @classmethod
    def get_parser(cls, tokenizer) -> ToolParser:
        """根据 tokenizer 名称自动选择 parser"""
        if "deepseek" in name: return R1ToolParser()
        elif "qwen" in name: return QwenToolParser()
```

**R1ToolParser**：使用特殊 token（`<｜tool▁calls▁begin｜>` 等）
**QwenToolParser**：使用 XML 格式（`<tool_call>{"name": ..., "arguments": ...}</tool_call>`）

### 5.7 代码执行工具

**E2B Python Interpreter**（`rllm/tools/code_tools/e2b_tool.py`）：
- 远程隔离沙箱执行 Python 代码
- 多沙箱轮转（round-robin）
- 失败自动重启
- 超时支持（默认 20s）

### 5.8 Tool Environment

**文件**：`rllm/environments/tools/tool_env.py`

将 Tool 系统包装为 Gym 兼容的 Environment：

```python
class ToolEnvironment(BaseEnv):
    def step(self, action):
        if is_tool_calls(action):
            outputs = self._execute_tool_calls(action)  # 多线程并行执行
            return {"tool_outputs": outputs}, reward, done, info
        elif is_finish(action):
            return {}, reward, True, info
```

---

## 6. Rejection Sampling

### 6.1 是什么

在 RL 训练中，过量生成轨迹后筛选掉全部失败或全部成功的 group，只保留**部分成功**的 task，以提供更有信息量的梯度信号。

**对比 RAGEN**：RAGEN 通过 SNR-Adaptive Filtering 按组内 reward 方差过滤，角度不同。rLLM 按"是否有部分 episode 成功"过滤。

### 6.2 配置

**文件**：`rllm/experimental/common/rejection_sampling.py`

```python
@dataclass
class RejectionSamplingConfig:
    mode: Literal["none", "episode", "group"] = "none"
    min_trajs_per_group: int = 2          # group 模式：每组最少轨迹数
    min_partial_solve_tasks: int = 1      # episode 模式：最少部分成功 task 数
```

YAML 配置：
```yaml
rejection_sample:
  enable: False
  multiplier: 1    # 过量生成倍数（batch_size × multiplier）
  min_partial_solve_tasks: 1
  min_trajs_per_group: 2
```

### 6.3 核心逻辑

```python
def apply_rejection_sampling_and_filtering(episodes, groups, config, state):
    # Step 1: 按最少轨迹数过滤 group
    filtered_groups, dropped_groups = filter_groups(groups, config, metrics)
    filtered_episodes = filter_episodes(episodes, dropped_groups)

    # Step 2: 统计 episode 级指标
    update_episode_metrics(filtered_episodes, metrics)
    # solve_none: 所有 episode 都错
    # solve_all: 所有 episode 都对
    # solve_partial: 部分 episode 对（最有训练价值！）

    # Step 3: episode 模式 — 累积直到有足够的 partial solve
    if config.mode == "episode":
        state.accumulated_groups.extend(filtered_groups)
        state.accumulated_episodes.extend(filtered_episodes)

        if metrics.solve_partial >= config.min_partial_solve_tasks:
            return state.accumulated_groups.copy(), ...  # 发送训练
        else:
            return [], ...  # 跳过这个 batch，继续累积
```

### 6.4 与 RAGEN SNR Filtering 的对比

| 维度 | RAGEN SNR Filtering | rLLM Rejection Sampling |
|------|---------------------|------------------------|
| 过滤粒度 | 按 prompt group 的 reward 方差 | 按 task 的 episode 成功率 |
| 过滤标准 | 低方差 = 低信号，丢弃 | 全对/全错 = 低信号，保留 partial |
| 过量生成 | 不需要 | 需要 multiplier 倍过量生成 |
| 累积机制 | 无 | 跨 batch 累积直到条件满足 |
| 适用场景 | 通用 RL | 稀疏 reward 场景（成功率低时尤其有用） |

---

## 7. 大规模异步并行

### 7.1 是什么

rLLM 使用 `asyncio` + `ThreadPoolExecutor` 实现大规模并行的 agent-env 交互，支持 128-256 个 agent/workflow 同时运行。

**对比 RAGEN**：RAGEN 使用线程池并行处理环境操作，但规模较小，且不使用 async/await。

### 7.2 并行架构

```
asyncio event loop（主循环）
    │
    ├── asyncio.Semaphore（128-256 并发限制）
    │     │
    │     ├── Workflow/Agent 实例
    │     │     │
    │     │     └── ThreadPoolExecutor（64 workers）
    │     │           ├── env.reset()    [阻塞操作在线程中]
    │     │           ├── env.step()     [阻塞操作在线程中]
    │     │           └── reward_fn()    [阻塞操作在线程中]
    │     │
    │     └── RolloutEngine.get_model_response()  [async I/O]
    │
    ├── asyncio.Queue（Workflow 对象池，复用实例）
    │
    └── Retry 逻辑（每 task 最多 3 次）
```

### 7.3 AgentExecutionEngine

**文件**：`rllm/engine/agent_execution_engine.py`

```python
class AgentExecutionEngine:
    def __init__(self, n_parallel_agents=128, max_workers=64, retry_limit=3):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def trajectory_generator(self, ...):
        semaphore = asyncio.Semaphore(self.n_parallel_agents)

        async def launch_one(env_idx):
            async with semaphore:  # 等待信号量
                return await self.run_agent_trajectory_with_retry(idx=env_idx, ...)

        tasks = [launch_one(i) for i in range(len(self.envs))]
        for coro in asyncio.as_completed(tasks):
            yield await coro  # 谁先完成谁先返回

    async def run_agent_trajectory_with_retry(self, idx, ...):
        for _ in range(self.retry_limit):
            try:
                return await asyncio.wait_for(
                    self.run_agent_trajectory_async(idx, ...),
                    timeout=7200  # 单轨迹 2 小时超时
                )
            except Exception:
                continue
        raise Exception(f"Trajectory {idx} failed after {self.retry_limit} attempts")

    async def run_agent_trajectory_async(self, idx, ...):
        loop = asyncio.get_event_loop()
        # 环境操作在线程池中执行
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        for step in range(self.max_steps):
            model_output = await self.get_model_response(...)  # async I/O
            next_obs, reward, done, info = await asyncio.wait_for(
                loop.run_in_executor(self.executor, env.step, action),
                timeout=(self.trajectory_timeout - elapsed)  # 动态超时
            )
```

### 7.4 AgentWorkflowEngine

**文件**：`rllm/engine/agent_workflow_engine.py`

```python
class AgentWorkflowEngine:
    def __init__(self, workflow_cls, workflow_args, n_parallel_tasks=128, retry_limit=3):
        self.executor = ThreadPoolExecutor(max_workers=n_parallel_tasks)

    async def initialize_pool(self):
        self.workflow_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
        for _ in range(self.n_parallel_tasks):
            workflow = self.workflow_cls(rollout_engine=..., executor=self.executor, ...)
            self.workflow_queue.put_nowait(workflow)

    async def process_task_with_retry(self, task, task_id, rollout_idx):
        workflow = await self.workflow_queue.get()  # 从池中获取
        try:
            for attempt in range(self.retry_limit):
                episode = await workflow.run_with_termination_handling(task=task, uid=uid)
                if episode.termination_reason != TerminationReason.ERROR:
                    return task_id, rollout_idx, episode
            return task_id, rollout_idx, episode  # 最后一次的结果
        finally:
            await self.workflow_queue.put(workflow)  # 归还到池

    async def execute_tasks(self, tasks, task_ids=None):
        futures = [self.process_task_with_retry(t, tid, idx) for ...]
        with tqdm(total=len(tasks)) as pbar:
            for future in asyncio.as_completed(futures):
                task_id, rollout_idx, episode = await future
                results.append(episode)
                pbar.update(1)
```

### 7.5 关键设计点

| 设计 | 说明 |
|------|------|
| **Semaphore** | 限制同时执行的 agent 数（避免 OOM） |
| **asyncio.Queue** | Workflow 对象池，复用实例避免重复创建 |
| **as_completed** | 先完成的先处理，最大化吞吐 |
| **run_in_executor** | 阻塞的 env 操作放到线程池，不阻塞 event loop |
| **动态超时** | 每步超时 = trajectory_timeout - 已用时间 |
| **Retry** | 只对 ERROR 终止原因重试，其他直接返回 |

---

## 8. Token-Level Trajectory Validation

### 8.1 是什么

多轮交互中，每轮的 prompt 是上一轮 prompt + response 的累积。由于 tokenizer 的特性（如 BPE 合并规则），重新 tokenize 整个累积历史可能与逐步拼接的 token 序列不一致（retokenization 问题）。rLLM 在训练前检测并处理这种不一致。

**对比 RAGEN**：RAGEN 在 `ContextManager` 中处理多轮 token 拼接，通过 `all_turns_prompt_ids` 和 `all_turns_reasoning_ids` 追踪，但没有显式的 token 一致性验证机制。

### 8.2 累积历史一致性检查

**文件**：`rllm/agents/agent.py`

```python
class Trajectory:
    def is_cumulative(self) -> bool:
        """检查每步的 chat_completions 是否是前一步的严格超集（前缀关系）"""
        prev = None
        for step in self.steps:
            if prev is not None:
                prev_cc = prev.chat_completions
                curr_cc = step.chat_completions
                if not (len(curr_cc) >= len(prev_cc) and curr_cc[:len(prev_cc)] == prev_cc):
                    return False
            prev = step
        return True
```

### 8.3 Token 不匹配检测

**文件**：`rllm/engine/agent_execution_engine.py` — `assemble_steps()` 方法

```python
def assemble_steps(self, steps):
    initial_prompt_ids = steps[0]["prompt_ids"]
    accumulated_sequence = initial_prompt_ids.copy()
    response_tokens, response_masks = [], []
    is_valid_trajectory = True

    for i, step in enumerate(steps):
        current_prompt_ids = step["prompt_ids"]
        current_completion_ids = step["completion_ids"]

        if i == 0:
            response_tokens.extend(current_completion_ids)
            response_masks.extend([1] * len(current_completion_ids))
            accumulated_sequence.extend(current_completion_ids)
        else:
            # 关键验证：当前 prompt 的前缀必须与累积序列完全一致
            if current_prompt_ids[:len(accumulated_sequence)] != accumulated_sequence:
                # TOKEN 不匹配！
                diff_pos = find_first_diff(accumulated_sequence, current_prompt_ids)
                logger.warning(f"Token mismatch at position {diff_pos}. "
                               f"Setting response_masks to all 0s.")
                is_valid_trajectory = False
                break

            # 有效 → 追加新 token
            new_context = current_prompt_ids[len(accumulated_sequence):]  # env response tokens
            response_tokens.extend(new_context + current_completion_ids)
            response_masks.extend(
                [0] * len(new_context) +           # env response 不参与 loss
                [1] * len(current_completion_ids)   # model response 参与 loss
            )
            accumulated_sequence = current_prompt_ids + current_completion_ids

    # 如果检测到不匹配，mask 整条轨迹
    if config.filter_token_mismatch:
        response_masks = response_masks * int(is_valid_trajectory)

    return prompt_tokens, response_tokens, response_masks, is_valid_trajectory
```

### 8.4 累积式 Tokenization

**文件**：`rllm/parser/chat_template_parser.py`

```python
def tokenize_and_mask_cumulative(self, messages):
    """对多步轨迹做累积 tokenization，自动生成 response mask"""
    # 找到第一个 assistant message
    first_assistant_idx = find_first_assistant(messages)

    # 初始 prompt = 所有 assistant 之前的 messages
    prompt_ids = tokenize(messages[:first_assistant_idx])

    response_ids, response_mask = [], []
    for i in range(first_assistant_idx, len(messages)):
        ids = tokenize([messages[i]])
        response_ids.extend(ids)
        if messages[i]["role"] == "assistant":
            response_mask.extend([1] * len(ids))  # assistant → 参与 loss
        else:
            response_mask.extend([0] * len(ids))  # user/tool → 不参与 loss

    return prompt_ids, response_ids, response_mask
```

### 8.5 验证流程总结

```
多步轨迹:
  Step 0: chat = [sys, user, assistant_0]
  Step 1: chat = [sys, user, assistant_0, tool_1, assistant_1]
  Step 2: chat = [sys, user, assistant_0, tool_1, assistant_1, tool_2, assistant_2]

验证:
  1. Step 0 的 chat 是 Step 1 的前缀？ ✓
  2. Step 1 的 chat 是 Step 2 的前缀？ ✓
  3. Tokenize Step 0 的 prompt + response
  4. Tokenize Step 1 → 验证 Step 0 部分的 token 完全匹配
  5. Tokenize Step 2 → 验证 Steps 0-1 部分的 token 完全匹配

如果任何步骤 token 不匹配:
  → is_valid_trajectory = False
  → response_masks = [0, 0, 0, ...]（整条轨迹不参与训练）
  → token_mismatch metric = 1.0（用于监控）
```

### 8.6 配置

```yaml
filter_token_mismatch: True  # 默认开启：mask 掉有 token 不匹配的轨迹
```

---

## 总结对照表

| # | 特性 | rLLM 实现方式 | RAGEN 现状 | 借鉴价值 |
|---|------|-------------|-----------|---------|
| 1 | Workflow 抽象 | 6 种 Workflow + Engine 调度 | 交互逻辑耦合在 agent_proxy + trainer | 高：解耦交互与训练 |
| 2 | 多后端支持 | Router + 3 Backend + 4 RolloutEngine | 深度绑定 veRL | 中：按需扩展 |
| 3 | SDK 轨迹收集 | 装饰器 + Session + Proxy + SQLite | 无 | 中：降低数据采集门槛 |
| 4 | 生产级 Agent | 8 种 Agent 覆盖 math/code/swe/web | 统一 AgentProxy + Memory 适配 | 中：扩展任务类型时参考 |
| 5 | Tool/MCP | Registry + MultiTool + MCP + CodeSandbox | 无独立 Tool 层 | 中：扩展 tool-use 场景 |
| 6 | Rejection Sampling | episode 级累积过滤 | SNR Filtering（不同角度） | 低：已有替代方案 |
| 7 | 异步并行 | asyncio + Semaphore + Queue 池化 | 线程池并行 | 高：提升 rollout 吞吐 |
| 8 | Token 验证 | 累积前缀匹配 + mask 失效轨迹 | 无显式验证 | 高：防止训练数据污染 |
