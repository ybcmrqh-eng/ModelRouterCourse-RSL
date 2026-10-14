# 作业 2：在 ModelRouter 上增加 RSL

## 1. RSL 是什么

RSL（Recursive Self-Learning）在本实现中不修改模型权重，也不重新训练模型。它做的是：

1. 模型先生成答案。
2. 评估器给答案评分并指出缺点。
3. 硬约束检查精确格式要求。
4. 把缺点写回下一轮提示词，让模型重新回答。
5. 只保留分数最高的答案。
6. 把有效经验保存到记忆表，供以后相似任务使用。

因此它是“外部策略和记忆驱动的递归自我改进”，不是让模型自己偷偷改权重。

## 2. 在当前 ModelRouter 基础上的新增内容

新增文件：

- `app\core\rsl.py`：RSL 核心引擎。
- `app\api\rsl.py`：RSL REST API。
- `app\schemas\rsl.py`：请求、评分、轮次和记忆的数据模型。
- `tests\test_rsl_engine.py`：递归改进与记忆持久化单元测试。
- `config\providers.yaml`：本地 llama.cpp Provider 配置。
- `requirements-rsl.txt`：运行依赖。

修改文件：

- `app\main.py`：初始化 RSL 引擎并挂载新路由。
- `app\core\router.py`：增加 `local` Provider 类型，映射到 OpenAI 兼容协议。

## 3. API

RSL 功能开关：

`ROUTER_ENABLE_RSL_API=true`

端点：

- `GET /v1/rsl/health`：RSL 状态。
- `POST /v1/rsl/run`：执行一次递归学习任务。
- `GET /v1/rsl/runs`：查看运行历史。
- `GET /v1/rsl/runs/{run_id}`：查看某次完整运行和每一轮答案。
- `GET /v1/rsl/memory`：查看可复用经验。

鉴权沿用 ModelRouter 的 Bearer Token。

## 4. 数据保存

RSL 使用 ModelRouter 的 SQLite 数据库，位置为：

`E:\aicourse\runtime\router.db`

相关表：

- `rsl_runs`：一次完整运行。
- `rsl_rounds`：每一轮答案、评分、反馈和 token 用量。
- `rsl_memories`：跨运行复用的改进经验。

所有运行数据都在 `E:\aicourse` 内，不使用云端数据库。

## 5. 硬约束验证

仅依赖 1.5B 小模型自评并不可靠，它可能给出“有缺点但通过”的矛盾结果。因此增加了可选硬约束：

- `contains`：必须包含某文本。
- `not_contains`：不得包含某文本。
- `line_count`：必须恰好若干行。
- `min_lines` / `max_lines`：行数范围。
- `line_exact_chars`：指定行必须恰好若干汉字。
- `line_starts_with`：指定行必须以某文本开头。
- `line_contains`：指定行必须包含某文本。

最终分数由模型评分和硬约束通过率共同计算。只要硬约束失败，该轮就不会被判定为通过。

## 6. 真实运行效果

演示任务要求写四行中文诗，每行七个汉字，第二行以“风”开头，第四行包含“月”。

实际运行结果：

- 第 1 轮：0.8536 分，第四行缺少“月”。
- 第 2 轮：0.7821 分，出现退化。
- 第 3 轮：0.9250 分，全部硬约束通过。
- 最终系统保留第 3 轮答案。
- 相比第一轮提高 0.0714 分。

最终答案：

```text
秋水共长天一色，
风起波涛带秋声。
明月高悬照影清，
月照湖影水波轻。
```

完整证据：

- `E:\aicourse\evidence\hw2\rsl-demo-final.json`
- `E:\aicourse\evidence\hw2\rsl-demo-summary.json`
- `E:\aicourse\evidence\hw2\rsl-demo-attempts.json`

## 7. 运行方式

启动：

`powershell -ExecutionPolicy Bypass -File E:\aicourse\scripts\start_rsl_router.ps1 -Restart`

演示：

`powershell -ExecutionPolicy Bypass -File E:\aicourse\scripts\run_rsl_demo.ps1`

测试：

`E:\aicourse\ModelRouterCourse\.venv\Scripts\python.exe -m unittest tests.test_rsl_engine -v`

## 8. 通俗效果说明

以前的 ModelRouter 是“用户提问 -> 选一个模型 -> 返回答案”。

加入 RSL 后变成“用户提问 -> 先回答 -> 自己打分 -> 找出哪里不好 -> 再回答 -> 比较两轮分数 -> 输出最好的一版 -> 记住有效经验”。

例如第一次诗里没有按要求写“月”，系统看到这个缺点后会在下一轮重点修正；即使中间一轮变得更差，它也只会保留历史最高分答案，不会因为最后一次输出而覆盖最好结果。

## 9. 边界和后续增强

- 当前学习不修改权重，属于提示词、策略和记忆层面的递归学习。
- 当前记忆匹配使用轻量词元相似度，没有额外下载向量模型。
- 评估器与生成器默认共用同一个本地小模型，因此仍需硬约束兜底。
- 如果使用更大模型，可以把评估器拆成独立模型，并增加人工反馈或奖励模型。
