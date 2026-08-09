# Phase0_MATH500 实验说明

## 现在做到了什么

这是一个面向后续 RL 数据集构建的 replay-ready MATH500 多智能体实验框架，默认读取本地 `MATH500/test.jsonl`。

核心能力：

- 默认使用 `MATH500/test.jsonl`，共 500 条有效样本。
- 默认随机切分为 400 条 train、100 条 test。
- RL 数据生成默认运行完整 train split，即 400 条；100 条 test 留作策略评估。
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
- 每个 stage 先收齐全部 edge candidate，再由 edge policy 通过固定顺序的联合 action mask 决定投递。
- `identity`、LOMO 和单次随机 stage 子集剪枝分别对应全零、one-hot 和 multi-hot mask。
- random_drop 强制从 identity parent replay，只在 checkpoint 所属完整 stage 执行一次，后续恢复 identity。
- 支持 re-execution replay：新日志和已迁移日志按 stage 联合决策边界分叉。

每次运行会生成这些主要文件：

- `predictions.jsonl`：每道题的最终结果和兼容旧流程的汇总字段。
- `traces.jsonl`：已投递消息形成的兼容 trace。
- `messages.jsonl`：所有真正进入 recipient inbox 的消息。
- `activations.jsonl`：每次 agent 调用的输入消息、control 消息、完整 prompt、prompt hash、模型配置和输出。
- `edge_candidates.jsonl`：每条待决策通信边。
- `edge_decisions.jsonl`：edge policy 对每条 candidate 的动作结果。
- `stage_actions.jsonl`：每个 stage 的状态哈希、固定边顺序、候选边和联合 action mask。
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
python run_phase0_math500.py --sample-size 1
```

运行时会读取默认配置：

```text
config/phase0_MATH500.yaml
```

当前默认配置为：

```yaml
data_path: MATH500/test.jsonl
train_size: 400
test_size: 100
seed: 0
split: train
sample_size: null
```

这表示：使用 seed 0 从 500 条 MATH500 样本中固定划分 400 条训练集和 100 条测试集，然后默认运行
完整 train split，也就是 400 条。test 100 条不参与 RL 数据生成，留作策略评估。

常用参数示例：

```powershell
python run_phase0_math500.py --split train --sample-size 10 --max-rounds 4 --stall-rounds 2
```

如果只想切分数据并生成 split 文件，不调用模型：

```powershell
python run_phase0_math500.py --prepare-only
```

输出会写到：

```text
runs/phase0_MATH500_<timestamp>/
```

## 如何查看一次运行

假设运行目录是：

```text
runs/phase0_MATH500_YYYYMMDDTHHMMSSZ
```

查看总览：

```powershell
Get-Content runs\phase0_MATH500_YYYYMMDDTHHMMSSZ\summary.json
```

查看每道题预测：

```powershell
Get-Content runs\phase0_MATH500_YYYYMMDDTHHMMSSZ\predictions.jsonl -TotalCount 1
```

查看通信边候选：

```powershell
Get-Content runs\phase0_MATH500_YYYYMMDDTHHMMSSZ\edge_candidates.jsonl -TotalCount 5
```

查看每条边的策略动作：

```powershell
Get-Content runs\phase0_MATH500_YYYYMMDDTHHMMSSZ\edge_decisions.jsonl -TotalCount 5
```

查看 agent 调用 prompt：

```powershell
Get-Content runs\phase0_MATH500_YYYYMMDDTHHMMSSZ\activations.jsonl -TotalCount 1
```

## 如何运行 re-execution replay

先跑一次 baseline：

```powershell
python run_phase0_math500.py --sample-size 1
```

从 baseline 的 `edge_candidates.jsonl` 中选择一个 `candidate_id`，例如：

```text
c000002
```

然后从该边动作前恢复，并从这个 checkpoint 后重新调用 LLM：

```powershell
python run_phase0_math500.py --replay-from-run runs\phase0_MATH500_YYYYMMDDTHHMMSSZ --checkpoint-candidate-id c000002
```

上面的普通 replay 使用默认 `identity` policy，主要用于验证恢复和重新执行流程。若要在同一个
identity 状态执行一次随机 multi-edge 动作，使用独立剪枝入口：

```powershell
python -B run_phase0_pruning.py `
  --replay-from-run runs\phase0_MATH500_YYYYMMDDTHHMMSSZ `
  --checkpoint-candidate-id c000002 `
  --replay-policy-name random_drop `
  --random-policy-seed 7
```

checkpoint 可以是 stage 内任意 candidate；程序会自动回退到完整 stage 开头。若 parent 包含多道
题，程序自动建立临时单题 scoped parent。random_drop 只在目标 stage 执行一次 multi-hot mask，
后续 stage 恢复 identity，因此 child 与 baseline 的奖励差对应这一次联合动作及其下游影响。

replay run 会生成新的运行目录，并在 `summary.json` 中记录：

- `parent_run_id`
- `replay_from_run`
- `checkpoint_candidate_id`
- `replay_policy_name`

## 建议的本地检查

修改代码后可以运行：

```powershell
python C:\Users\ASUS\Desktop\mas_com\tests\test_phase0_math500.py
```

测试命令建议使用 `python -B`，避免生成 `__pycache__` 和 `.pyc`。
