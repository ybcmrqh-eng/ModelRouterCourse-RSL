# 作业二：在 ModelRouter 上实现有界递归自我改进

## 1. 原始作业要求与实现结论

原始作业要求二选一：

1. 在现有 ModelRouter 基础上添加 RSL（Recursive Self-Learning）；
2. 自己实现一个简单的 RSL 系统。

本仓库选择第 1 种方案，并在 ModelRouter 中新增了独立的 RSL 模块。原有调用链、Provider、路由和管理接口保持兼容。

实现按参考文件优先对齐：

- `recursive-self-improvement-handbook.pdf`
- `E:/aicourse/JEV/test_jev.py`

当旧实现与这两个参考文件冲突时，以手册和 JEV 为准。参考文件没有覆盖的工程细节，再使用原始作业要求补足。

最终系统属于手册定义的 **有界自我改进（bounded self-improvement）**：任务、评价器和预算固定，系统在 3–5 轮内搜索更好的候选答案，不修改模型权重，也不改写评价标准。

## 2. 手册概念如何映射到代码

| 手册概念 | 本项目实现 |
| --- | --- |
| Agent | ModelRouter 中的 RSL 引擎和本地语言模型 |
| Harness | 提示词、候选池、输出规范化、预算、停止条件、SQLite 轨迹 |
| Evaluator | 硬约束 + 外部检查 + 留出集 + 本地 LLM 评审 + 可选 TypeSafe 评审 |
| Self-improvement | 只修改候选答案和版本化记忆，不修改模型权重 |
| 基线 | 第 0 轮单候选生成，记录 `baseline_score` |
| 生成—评价—修订 | 每轮最少保留候选 0 的修订链，记录 `generate_evaluate_revise` 控制组 |
| 候选搜索 | 每轮生成 `candidates_per_round` 个候选，按分数保留 `top_k` |
| 停止条件 | 目标分数、连续无提升、模型调用预算、token 预算、时间预算 |
| 记录器 | `rsl_runs`、`rsl_rounds`、`rsl_candidates`、`rsl_memories` |
| 版本化记忆 | `version`、`status`、`evidence_count`、`previous_version_id`、回滚原因 |

## 3. 最小闭环和主流程

每次运行依次执行：

1. 读取 active 记忆和经验；
2. 生成第 0 轮基线候选；
3. 使用硬约束、外部检查、留出集和 LLM 评审评分；
4. 每轮生成 k 个候选；
5. 对每个候选独立评分；
6. 保留 top-k 候选组成下一轮候选池；
7. 计算多样性、重复率、分数方差和校准相关性；
8. 达到目标、连续无提升或预算耗尽时停止；
9. 只保留全程最佳候选；
10. 只有通过全部独立验证的经验才允许写入记忆库。

候选搜索不是盲目多生成。系统会记录：

- `average_score`
- `score_std`
- `diversity`
- `duplicate_rate`
- `best_candidate_index`
- 每个候选的 `evaluator_score`、`format_score`、`external_score`
- 候选原始答案 `raw_answer` 和是否经过后处理 `postprocessed`

## 4. 三类控制组

手册要求区分三种方案，代码会自动计算：

| 控制组 | 含义 | 返回字段 |
| --- | --- | --- |
| 单次生成 | 只看第 0 轮基线 | `controls.single_generation` |
| 生成—评价—修订 | 只看每轮候选 0 的修订链 | `controls.generate_evaluate_revise` |
| 候选搜索 top-k | 全部候选中的最佳结果 | `controls.candidate_search_topk` |

只有外部验证或留出集也改善时，才计算 `verified_improvement`。如果只是模型自评分变高，结果不会标记为“已验证提升”。

## 5. 评价器设计

评价器使用多信号组合，而不是把模型自评当作唯一证据。

默认权重：

```text
总分 = 0.6 × 外部检查 + 0.2 × 格式约束 + 0.2 × 过程/LLM 评审
```

缺少某一类信号时，权重会重新分配：

- 没有格式约束：外部检查 0.8 + LLM 评审 0.2；
- 没有外部检查：格式约束 0.6 + LLM 评审 0.4；
- 两者都没有：只使用 LLM 评审，并标记为 `internal_only`，不能写入长期记忆。

支持的评价信号：

1. 形式化/规则检查：JSON、Python AST 语法、正则、包含与排除规则；
2. 执行前检查：默认只做静态 AST 解析，不执行模型生成的任意代码；
3. 学习型评价器：本地 LLM-as-a-judge；
4. 独立结构化评价器：JEV 风格的 TypeSafe `system_one`；
5. 留出集：`holdout_checks` 不进入生成提示，只在最终候选上验证。

### 5.1 JEV / TypeSafe 集成

JEV 示例使用：

```python
from typesafe_sdk import TypeSafeClient, Choice, Noul

response = client.system_one(
    model="jev-latest",
    state={...},
    questions={
        "department": Choice(...),
        "urgent": Noul(...),
    },
)
```

本项目新增 `app/core/typesafe_verifier.py`，使用相同的 `system_one`、`Choice` 和 `Noul` 形式提出结构化问题：

- `passed`：候选是否完整满足任务；
- `task_fit`：pass / partial / fail；
- `primary_issue`：缺失要求、格式错误、事实风险、不安全内容或无问题；
- `hallucination_risk`：是否存在无依据或矛盾陈述。

配置环境变量后启用：

```env
TYPESAFE_BASE_URL=https://api.ergouzi.life
TYPESAFE_API_KEY=your_api_key_here
```

没有 API Key 或接口失败时，系统不会伪造结构化结果，而是记录日志并回退到本地评价器。

## 6. 输出规范化和可追踪性

真实实验中发现的失败案例：

- 本地模型生成了正确 JSON、SQL 或 Python 内容；
- 但外层包了 Markdown 代码围栏；
- JSON 解析和 Python 语法检查因此失败；
- 旧实现没有正确识别这是 harness 问题；
- 修复前系统没有把结果错误标记为“已验证提升”。

修复方式：

- 在候选进入评价器前执行可追踪的输出规范化；
- 仅在任务要求原始 JSON、Python 或禁止 Markdown 时移除代码围栏；
- 保留 `raw_answer`；
- 记录 `postprocessed=true/false`；
- 外部检查仍然独立执行，规范化不会修改检查标准。

这符合手册把工具和脚手架视为系统组成部分的原则，同时保留完整轨迹和失败案例。

## 7. 安全边界

- 默认不执行模型生成的 Python 代码，只使用 `ast.parse` 做语法检查；
- 发现破坏性命令、危险 shell 调用、明文密钥或私钥时，外部分数清零；
- 安全失败候选不能成为最佳通过候选，也不能写入记忆；
- 记忆默认只使用 active 状态；
- 旧版未验证记忆迁移为 `legacy`，不参与检索；
- 支持手动回滚并恢复上一版本。

## 8. 失败模式与处理

| 手册失败模式 | 本项目处理 |
| --- | --- |
| 自我确认循环 | 外部规则、留出集、LLM 评审和 TypeSafe 多信号组合 |
| Reward hacking / Goodhart | `verified_improvement` 只认独立检查通过；记忆写入门槛严格 |
| 模型坍缩 | 不训练模型权重，只在固定任务内改进外部答案 |
| 多样性坍缩 | 记录 diversity 和 duplicate rate；低于阈值时注入多样化提示 |
| 评价器漂移 | `verifier_version=rsl-verifier-v2`，提示词版本单独记录 |
| 持久化污染 | 连续两次通过独立验证才从 candidate 升级为 active；支持回滚 |
| 方向选择失败 | 保持任务和预算固定，不做无边界开放式 RSI |

## 9. API

已有接口：

- `POST /v1/rsl/run`
- `GET /v1/rsl/runs`
- `GET /v1/rsl/runs/{run_id}`
- `GET /v1/rsl/memory`
- `GET /v1/rsl/health`

新增接口：

- `POST /v1/rsl/memory/{memory_id}/rollback`

`/v1/rsl/health` 会返回：

- `verifier_version`
- `typesafe_available`

## 10. 真实实验结果

### 10.1 三个客观验证任务

实验脚本：`scripts/run_rsl_experiment.py`

任务：

1. 生成可解析 JSON，`skills` 恰好三个字符串；
2. 生成参数化 SQLite 查询，不得使用字面数字 18；
3. 生成带类型注解、docstring 和零除保护的 Python 函数。

最终结果：

| 任务 | baseline | best | verified improvement | 停止原因 |
| --- | ---: | ---: | ---: | --- |
| JSON | 0.9700 | 0.9700 | 0.0000 | target_reached_at_baseline |
| SQL | 0.9700 | 0.9700 | 0.0000 | target_reached_at_baseline |
| Python | 0.9700 | 0.9700 | 0.0000 | target_reached_at_baseline |

解释：

- 输出规范化修复后，基线已经通过全部外部检查和格式检查；
- 因基线达到目标，继续候选搜索没有额外收益；
- 系统没有虚构“候选搜索提升”，而是如实返回 0.0000；
- 三种控制组数值相同，说明对本机 1.5B 模型和这些任务，增加搜索轮数没有必要。

### 10.2 修复前失败案例

修复前三个任务分别失败于：

- JSON：Markdown 围栏导致 JSON 解析失败；
- SQL：Markdown 围栏违反格式约束；
- Python：Markdown 围栏导致 Python 语法解析失败。

系统当时的 `verified_improvement` 为 `None`，没有把内部高分误报为已验证提升。这个案例说明外部检查确实独立于模型自评。

### 10.3 记忆晋级

真实任务：只输出字符串 `OK`。

- 第一次通过独立检查：写入 candidate 记忆，`evidence_count=1`；
- 第二次再次通过：升级为 active 记忆，`evidence_count=2`；
- 第三次同类任务：`memory_used=1`，成功检索 active 记忆；
- 回滚接口验证：记忆状态变为 `rolled_back`，上一版本恢复为 active。

证据文件：

- `docs/evidence/rsl-evidence-summary.json`
- `docs/evidence/rsl-experiment-20261015T114936Z.json`
- `docs/evidence/rsl-demo-v2.json`
- `docs/evidence/rsl-memory-promotion-v2.json`
- `docs/evidence/rsl-memory-retrieval-rollback-v2.json`

## 11. 单元测试

运行：

```powershell
E:\aicourse\ModelRouterCourse\.venv\Scripts\python.exe -m unittest tests.test_rsl_engine -v
```

覆盖：

- 基线、候选搜索、控制组和完整轨迹；
- 模型调用预算停止；
- 留出集失败阻止记忆写入；
- 两次独立通过后记忆晋级；
- 回滚恢复上一版本。

## 12. 运行方法

启动本地模型和 ModelRouter：

```powershell
powershell -ExecutionPolicy Bypass -File E:\aicourse\scripts\start_rsl_router.ps1 -Restart
```

运行三项对照实验：

```powershell
E:\aicourse\ModelRouterCourse\.venv\Scripts\python.exe E:\aicourse\ModelRouterCourse\scripts\run_rsl_experiment.py
```

RSL 请求示例：

```json
{
  "task": "输出原始 JSON，包含 name、age、skills；skills 恰好三个字符串。",
  "model": "qwen2.5-1.5b-instruct",
  "max_rounds": 2,
  "candidates_per_round": 3,
  "top_k": 2,
  "target_score": 0.9,
  "constraints": [
    {"name": "no fence", "kind": "not_contains", "value": "```"}
  ],
  "external_checks": [
    {"name": "json", "kind": "json_valid"},
    {"name": "keys", "kind": "contains_all", "values": ["name", "age", "skills"]}
  ],
  "holdout_checks": [
    {"name": "skills", "kind": "regex", "value": "\"skills\"\\s*:\\s*\\[[^\\]]+\\]"}
  ],
  "budget": {
    "max_model_calls": 24,
    "max_total_tokens": 16000,
    "max_duration_seconds": 300
  }
}
```

## 13. 边界与未实现项

- 不修改模型权重；训练时自迭代被主动排除；
- 不执行模型生成代码，避免越权工具调用；
- 没有可靠评价器时不进行大规模自训练；
- TypeSafe 结构化评价需要 API Key；
- 本地 1.5B 模型能力有限，复杂任务可能无法通过多轮修订改善；
- 候选搜索只有在基线未达到目标时才有实际意义。
