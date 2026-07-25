# Phase0 独立删边运行说明

删边相关代码与原 Phase0 文件分离：

- `src/phase0_PRM800K/edge_pruning.py`：LOMO、稳定哈希随机删边及保护规则。
- `src/phase0_PRM800K/run_pruning.py`：独立删边 runner。
- `run_phase0_pruning.py`：命令行入口。
- `config/phase0_PRM800K_pruning.yaml`：独立配置。


## edge_candidates.jsonl 在哪里

普通 Phase0 baseline 实际运行后，文件位于本次输出目录：

```text
runs/phase0_PRM800K_<UTC时间>/edge_candidates.jsonl
```

例如：

```bash
python run_phase0_prm800k.py --sample-size 1
sed -n '1,10p' runs/phase0_PRM800K_20260724T010203Z/edge_candidates.jsonl
```

终端会打印真实输出目录。`--prepare-only` 不调用 Agent，因此不会生成该文件。

## LOMO

先跑 identity baseline，从 `edge_candidates.jsonl` 选择一条 `plan`、`solver_step` 或
`judge_feedback` 跨 Agent 消息。然后从该 candidate 前恢复，只删除这一条消息：

```bash
python run_phase0_pruning.py \
  --replay-from-run runs/phase0_PRM800K_20260724T010203Z \
  --checkpoint-candidate-id c000002 \
  --replay-policy-name lomo
```

checkpoint id 默认就是 LOMO 删除目标，也可以额外指定：

```text
--lomo-candidate-id c000002
```

LOMO 保留其他所有 candidate。默认禁止删除 `Input -> Planner`、`Judger -> Output` 和
Agent 自环。批量 runner 会先从 parent shard 中提取 candidate 所属题目的 messages、activations、
candidates 和 decisions，保留原始事件 ID，并计算 Router counter 偏移。child run 因此只包含
一道目标题；同一 shard 的其他题不会进入 child 的预测或通信日志。

## 稳定随机删边

```bash
python run_phase0_pruning.py \
  --edge-policy random_drop \
  --random-drop-probability 0.25 \
  --random-policy-seed 7 \
  --sample-size 10
```

稳定哈希会把“实验 seed + 数据位置 + 轮次 + 源消息 + sender + recipient + 消息类型”转换为
`[0, 1)` 内的固定数。当该数小于删除概率时删边。因此相同 seed 和相同逻辑消息总会得到相同
决定，不依赖程序运行时随机数的调用顺序，也不包含每次都会变化的 run 目录名。

如需把 Solver/Judger 的自环历史也纳入删边范围，增加：

```text
--pruning-include-self-edges
```

删除结果查看：

```text
edge_candidates.jsonl  # 所有候选边
edge_decisions.jsonl   # dropped/action 和实际投递结果
messages.jsonl         # 真正进入 Agent inbox 的消息
```

## 批量 LOMO：每条可删边各删除一次

下面的命令读取一个 identity baseline 的全部 candidate。每个 candidate 都从同一个 baseline
独立 replay，只删除自身一次，因此不同边的干预不会相互污染：

```bash
python run_lomo_batch.py \
  --replay-from-run runs/phase0_PRM800K_20260724T010203Z
```

建议先限制为一道题或少量边确认成本：

```bash
python run_lomo_batch.py \
  --replay-from-run runs/phase0_PRM800K_20260724T010203Z \
  --lomo-question-id 0 \
  --lomo-limit 5
```

批次输出结构：

```text
runs/lomo_batch_<UTC时间>/
  manifest.json                 # 总数、成功数、失败数
  lomo_results.jsonl            # 每条边一行汇总结果
  <candidate_id>/               # 每次独立 replay 的完整通信记录
    predictions.jsonl
    edge_candidates.jsonl
    edge_decisions.jsonl
    messages.jsonl
    activations.jsonl
    traces.jsonl
    rl_edge_samples.jsonl
    summary.json
```

`lomo_results.jsonl` 包含 candidate 信息、baseline/LOMO 最终输出、token 变化、轮数变化和
child run 路径。完整的 messages、activations、decisions 和 traces 保存在相应 candidate 目录。
单题 scoped parent 只是在运行期间从原通信记录生成的临时过滤视图，不调用模型，并在批次结束后
自动清理，因此结果目录不再保存 `scoped_parents/`。

每个 child 的 `predictions.jsonl` 必须恰好为一行，否则批次会将该干预标记为失败。正式全量
运行前仍建议先使用 `--lomo-question-id` 或 `--lomo-limit` 做成本估算。

## 为并行 shard 生成 1000 条分层 LOMO 计划

对于包含多个 shard 的 baseline，可以先生成计划而不调用模型：

```bash
python build_lomo_plan.py \
  --master-dir '通信记录/通信记录/phase0_PRM800K_parallel_20260709T151013Z' \
  --output-dir experiment_plans/math500_lomo1000_seed0 \
  --lomo-sample-size 1000 \
  --seed 0
```

输出文件：

```text
plan_summary.json          # 总量和各层分布
lomo_candidates.jsonl      # 分层抽取的单边 LOMO candidate
```

LOMO 采样会排除 baseline 失败题、输入/输出边和自环，按轮次设置配额，并在每个轮次内
尽量均衡不同有向边。执行已生成的 1000 条计划：

```bash
python run_lomo_plan.py \
  --lomo-plan experiment_plans/math500_lomo1000_seed0/lomo_candidates.jsonl \
  --plan-run-dir runs/lomo_stratified_1000 \
  --plan-continue-on-error
```

如果运行中断，可以复用同一个输出目录并指定从哪个 shard 开始。例如从 shard 7 开始：

```bash
python run_lomo_plan.py \
  --lomo-plan experiment_plans/math500_lomo1000_seed0/lomo_candidates.jsonl \
  --plan-run-dir runs/lomo_stratified_1000 \
  --plan-start-shard 7 \
  --plan-continue-on-error
```

已有输出目录的 shard 会被整体跳过，不会再次调用模型；根目录的 `lomo_results.jsonl` 和
`manifest.json` 会根据已有 shard 结果重新汇总。若某个 shard 中断后只完成了一部分，指定下一
个 shard 即表示保留该部分结果并放弃该 shard 剩余 candidate。

计划执行器只运行清单中的 1000 个 `(parent_run, candidate_id)`，不会自动补跑其余可删边。
输出按 shard 分组，批次根目录的 `lomo_results.jsonl` 汇总全部结果。
