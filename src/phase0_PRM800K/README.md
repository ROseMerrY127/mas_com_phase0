# Phase0_PRM800K / MATH500 实验说明

## 现在做到了什么

这个版本把 Phase0_PRM800K 改成了一个面向后续 RL 数据集构建的 replay-ready 多智能体实验框架，并把默认实验数据切换到本地 `MATH500/test.jsonl`。

核心能力：

- 默认使用 `MATH500/test.jsonl`，共 500 条有效样本。
- 默认随机切分为 400 条 train、100 条 test。
- 三层通信拓扑：`Planner`、两个 `Solver`、`Judger`。
- 跨层全连接：`Planner <-> SolverA/SolverB/Judger`，`SolverA/SolverB <-> Planner/Judger`。
- 不允许两个 solver 直接通信：没有 `SolverA -> SolverB` 或 `SolverB -> SolverA`。
- 外部只保留两条边：`Input -> Planner` 和 `Judger -> Output`。
- `Judger -> Output` 一旦发生，当前题目结束，不再向系统内部广播。
- 同步轮次调度：一轮是 `Planner stage -> Solver stage -> Judger stage`，不是单条边传递。
- Solver 每轮只输出一个 PRM800K 风格 step。
- `max_rounds` 限制完整协作轮数；到达上限时可强制 Judger 输出最终答案。
- 系统状态消息化：agent 的输出只依赖显式收到的 inbox messages 和 Scheduler control messages。
- Solver 的自我历史通过 `SolverA -> SolverA` / `SolverB -> SolverB` self-message 显式进入 inbox。
- Judger 的历史反馈通过 `Judger -> Judger` self-message 显式进入 inbox。
- 每条通信边先生成 edge candidate，再由 edge policy 决定是否投递。
- 当前内置默认策略是 `identity`，即所有边原样投递；真实剪枝/压缩策略后续接入同一接口。
- 支持边级别 re-execution replay：checkpoint 前从旧日志恢复，checkpoint 后重新调用 LLM。

每次运行会生成这些主要文件：

- `predictions.jsonl`：每道题的最终结果和兼容旧流程的汇总字段。
- `traces.jsonl`：已投递消息形成的兼容 trace。
- `messages.jsonl`：所有真正进入 recipient inbox 的消息。
- `activations.jsonl`：每次 agent 调用的输入消息、control 消息、完整 prompt、prompt hash、模型配置和输出。
- `edge_candidates.jsonl`：每条待决策通信边。
- `edge_decisions.jsonl`：edge policy 对每条 candidate 的动作结果。
- `rl_edge_samples.jsonl`：后续训练通信边剪枝/压缩策略所需的边动作样本。
- `summary.json`：本次运行的配置、统计量和输出路径。

## 如何运行普通实验

先进入项目根目录：

```powershell
cd C:\Users\ASUS\Desktop\mas_com
```

配置环境变量，或写入项目根目录的 `.env`：

```text
OPENAI_API_KEY=你的 key
OPENAI_BASE_URL=可选，OpenAI-compatible 网关地址
OPENAI_MODEL=可选，用来覆盖 config 里的 model
```

运行一个最小 smoke test：

```powershell
python run_phase0_prm800k.py --sample-size 1
```

运行时会读取默认配置：

```text
config/phase0_PRM800K.yaml
```

当前默认配置为：

```yaml
data_path: MATH500/test.jsonl
train_size: 400
test_size: 100
split: test
sample_size: null
```

这表示：先从 500 条 MATH500 样本中随机切分 400 条训练集和 100 条测试集，然后默认运行完整 test split，也就是 100 条。

常用参数示例：

```powershell
python run_phase0_prm800k.py --split test --sample-size 10 --max-rounds 4 --stall-rounds 2
```

如果只想切分数据并生成 split 文件，不调用模型：

```powershell
python run_phase0_prm800k.py --prepare-only
```

输出会写到：

```text
runs/phase0_PRM800K_<timestamp>/
```

## 如何查看一次运行

假设运行目录是：

```text
runs/phase0_PRM800K_YYYYMMDDTHHMMSSZ
```

查看总览：

```powershell
Get-Content runs\phase0_PRM800K_YYYYMMDDTHHMMSSZ\summary.json
```

查看每道题预测：

```powershell
Get-Content runs\phase0_PRM800K_YYYYMMDDTHHMMSSZ\predictions.jsonl -TotalCount 1
```

查看通信边候选：

```powershell
Get-Content runs\phase0_PRM800K_YYYYMMDDTHHMMSSZ\edge_candidates.jsonl -TotalCount 5
```

查看每条边的策略动作：

```powershell
Get-Content runs\phase0_PRM800K_YYYYMMDDTHHMMSSZ\edge_decisions.jsonl -TotalCount 5
```

查看 agent 调用 prompt：

```powershell
Get-Content runs\phase0_PRM800K_YYYYMMDDTHHMMSSZ\activations.jsonl -TotalCount 1
```

## 如何运行 re-execution replay

先跑一次 baseline：

```powershell
python run_phase0_prm800k.py --sample-size 1
```

从 baseline 的 `edge_candidates.jsonl` 中选择一个 `candidate_id`，例如：

```text
c000002
```

然后从该边动作前恢复，并从这个 checkpoint 后重新调用 LLM：

```powershell
python run_phase0_prm800k.py --replay-from-run runs\phase0_PRM800K_YYYYMMDDTHHMMSSZ --checkpoint-candidate-id c000002
```

当前 replay 仍使用默认 `identity` policy，因此行为上主要用于验证恢复和重新执行流程。后续加入剪枝或压缩策略后，可以通过同样的 checkpoint 入口，在同一个状态起点上比较不同通信动作带来的结果变化。

replay run 会生成新的运行目录，并在 `summary.json` 中记录：

- `parent_run_id`
- `replay_from_run`
- `checkpoint_candidate_id`
- `replay_policy_name`

## 建议的本地检查

修改代码后可以运行：

```powershell
python C:\Users\ASUS\Desktop\mas_com\tests\test_phase0_prm800k.py
```

再做一次语法检查：

```powershell
python -m compileall C:\Users\ASUS\Desktop\mas_com\src C:\Users\ASUS\Desktop\mas_com\tests
```