# Phase0 stage 联合剪枝运行说明

删边相关代码与原 Phase0 文件分离：

- `src/phase0_MATH500/edge_pruning.py`：LOMO、稳定哈希随机删边及保护规则。
- `src/phase0_MATH500/run_pruning.py`：独立删边 runner。
- `src/phase0_MATH500/random_stage_batch.py`：400 题随机 stage 计划、并发执行与一致性校验。
- `src/phase0_MATH500/lomo_parallel.py`：LOMO 计划的并发执行、续跑和分叉一致性校验。
- `run_phase0_pruning.py`：命令行入口。
- `run_random_stage_replays.py`：随机 stage 批量入口。
- `run_lomo_parallel.py`：高并发 LOMO 计划入口。
- `config/phase0_MATH500_pruning.yaml`：独立配置。

## 当前执行总结（2026-08-09）

数据使用 MATH500 的 seed 0 固定划分：train 400 条用于 RL 数据生成，test 100 条保留评估。当前
identity、random drop 和 LOMO 状态如下：

| 数据 | 规模 | 结果 | 配置与校验 |
| --- | ---: | --- | --- |
| identity | 400 | 400 完成 | 旧记录迁移 398 条，失败的 q216、q330 已补跑 |
| random drop | 400 | 400 完成，0 失败 | 40 并发、temperature 0、每题一个稳定 stage multi-hot 动作 |
| LOMO | 1000 | 1000 完成，0 失败 | 40 并发、temperature 0、每个 candidate 一个 one-hot 动作 |

random drop 和 LOMO 的每个 completed child 都已验证：分叉前的 identity stage 状态一致、目标 stage
候选内容一致、mask 符合策略约束、目标 stage 外没有额外剪边，并且 child 只包含一道题。分布为：

- random drop：Planner 191、Solver 185、Judger 24，单次删除 2 至 4 条边；
- LOMO：Planner 329、Solver 428、Judger 243，覆盖 316 道题。

可复现实验的代码、配置、random drop 计划和 LOMO 计划纳入 Git。完整通信轨迹保存在本地 `runs/`
并由 `.gitignore` 排除，避免将约 343 MB 的生成结果上传 GitHub。项目内旧版 MAS、失效 LOMO 计划、
Python bytecode 和缓存均已删除。


## edge_candidates.jsonl 在哪里

普通 Phase0 baseline 实际运行后，文件位于本次输出目录：

```text
runs/phase0_MATH500_<UTC时间>/edge_candidates.jsonl
```

例如：

```bash
python run_phase0_math500.py --sample-size 1
sed -n '1,10p' runs/phase0_MATH500_20260724T010203Z/edge_candidates.jsonl
```

终端会打印真实输出目录。`--prepare-only` 不调用 Agent，因此不会生成该文件。

## 统一的 stage action mask

Router 会先收齐同一题、同一轮、同一 stage 的全部候选边，再执行一次联合动作。三种策略使用
同一个动作表示：

- `identity`：全零 mask，不删除边。
- `lomo`：one-hot mask，只删除一条指定边。
- `random_drop`：multi-hot mask，从同一 stage 的边子集中一次删除至少两条边。

默认不包含自环时，各 stage 的 bit 顺序固定为：

```text
Planner: [Planner->SolverA, Planner->SolverB, Planner->Judger]
Solver:  [SolverA->Planner, SolverA->Judger, SolverB->Planner, SolverB->Judger]
Judger:  [Judger->Planner, Judger->SolverA, Judger->SolverB]
```

例如 Solver mask `1001` 同时删除 `SolverA -> Planner` 和 `SolverB -> Judger`。指定
`--pruning-include-self-edges` 后，Solver 和 Judger 的自环也按固定位置加入 mask。

## LOMO

先跑 identity baseline，从 `edge_candidates.jsonl` 选择一条 `plan`、`solver_step` 或
`judge_feedback` 跨 Agent 消息。然后从该 candidate 前恢复，只删除这一条消息：

```bash
python run_phase0_pruning.py \
  --replay-from-run runs/phase0_MATH500_20260724T010203Z \
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
一道目标题；同一 shard 的其他题不会进入 child 的预测或通信日志。LOMO 在 stage action 日志中
表现为 one-hot mask，因此可以和 identity、联合随机剪枝直接比较。新格式 baseline 的 checkpoint
会自动对齐到目标 candidate 所属 stage 的联合决策边界；旧格式日志仍按 candidate 边界 replay。

## 稳定随机 stage 子集剪枝

`random_drop` 不再支持从头独立运行。它必须从新格式或已迁移的 identity parent 中选择一个
checkpoint，精确 replay identity 前缀，在 checkpoint 所属完整 stage 上执行一次 multi-hot
动作，然后使用 identity 投递规则继续生成后续轨迹：

```bash
python run_phase0_pruning.py \
  --replay-from-run \
    runs/phase0_MATH500_parallel_identity_migrated_20260709T151013Z/shard_0000_start_0_n_10/phase0_PRM800K_20260709T151016Z \
  --checkpoint-candidate-id c000003 \
  --replay-policy-name random_drop \
  --random-policy-seed 7
```

checkpoint 可以是目标 stage 中任意一条 candidate，不必是第一条。ReplayController 根据
`stage_action_id` 自动回退到该 stage 的第一条边，保证同一个联合动作不会混用 identity 和
random_drop 决策。若 parent shard 含多道题，runner 会自动创建临时单题 scoped parent，保留原事件
ID 和 counter offset；child 输出只包含 checkpoint 所属题目，临时 parent 在运行结束后清理。

目标 stage 先按固定顺序构造动作边集合，再从所有“至少包含两条边”的删除子集中均匀采样一个
multi-hot mask。相同 identity state 和 seed 总会得到相同 mask，不依赖程序运行时随机数的调用
顺序，也不包含 run 目录名。该 mask 只应用一次；目标 stage 后的所有 stage 都恢复全零 identity
mask，因此相对 identity 的最终奖励差只包含这一次联合剪枝及其下游影响。

目标 stage 的候选消息也是 identity prefix 的一部分：runner 先 replay 产生该 stage 候选边的 Agent
输出，再在投递前应用 mask。批量验证要求目标 stage 之前的 `state_hash`、拓扑、candidate id 和
全零 action 与 identity 一致，并要求目标 stage 每条 candidate 的正文、局部 state hash 和
`stage_state_hash` 与 identity 一致。只有联合 action 执行后才允许轨迹分叉。child 的 `run_id`、
`replayed` 标记和计时属于运行元数据，不要求与 parent 相同。

最低删除边数默认为 2，可以调高：

```text
--random-drop-min-edges 3
```

如果 checkpoint 所属 stage 的可剪枝边少于最低数量，命令直接失败，要求改选 checkpoint 或降低
最低数量；不会静默生成一个没有干预的全零 child。不同 stage 的边绝不会进入同一个动作。

### 400 条 train 的高并发 random_drop

下面的命令对 train 400 题各稳定选择一个可剪 stage，并在该 stage 均匀采样一个至少含两条边的
删除子集。模型温度显式固定为 0；identity prefix 不调用模型，温度只影响分叉后的新调用：

```bash
/home/yyh/miniconda3/envs/phase0-lomo/bin/python -B run_random_stage_replays.py \
  --master-dir runs/phase0_MATH500_parallel_identity_migrated_20260709T151013Z \
  --plan-dir experiment_plans/math500_random_stage400_seed0 \
  --output-dir runs/phase0_MATH500_random_stage400_seed0 \
  --seed 0 \
  --workers 40 \
  --temperature 0
```

计划包含 400 个唯一 `(split, question_id, source_index)`，覆盖 `question_id=0..399`。每个问题的
stage 选择和 mask 都由 seed 及 identity 状态稳定决定。每个 branch 独立保存 `result.json`；中断后
用原命令增加 `--resume`，已完成且验证通过的 branch 会跳过，失败和未完成项才会重新执行。

批量根目录保存 `manifest.json` 和 `random_results.jsonl`。只有同时满足以下条件才记为 completed：

- child 只有一道对应题目的 prediction；
- 分叉前的 stage 状态与 identity 严格一致；
- 目标 stage 至少删除两条边，其他 stage 全部是 identity mask；
- summary 声明 `checkpoint_stage_once_then_identity`。

seed 0 的本次运行已完成 400/400，失败 0 条，400 条均通过 identity prefix 校验。抽到的目标
stage 分布为 Planner 191、Solver 185、Judger 24；每个 mask 实际删除 2 至 4 条边。计划和结果位于：

```text
experiment_plans/math500_random_stage400_seed0/
runs/phase0_MATH500_random_stage400_seed0/
```

如需把 Solver/Judger 的自环历史也纳入删边范围，增加：

```text
--pruning-include-self-edges
```

删除结果查看：

```text
edge_candidates.jsonl  # 所有候选边
edge_decisions.jsonl   # dropped/action 和实际投递结果
stage_actions.jsonl    # stage state、固定边顺序和联合 action mask
messages.jsonl         # 真正进入 Agent inbox 的消息
```

## 批量 LOMO：每条可删边各删除一次

下面的命令读取一个 identity baseline 的全部 candidate。每个 candidate 都从同一个 baseline
独立 replay，只删除自身一次，因此不同边的干预不会相互污染：

```bash
python run_lomo_batch.py \
  --replay-from-run runs/phase0_MATH500_20260724T010203Z
```

建议先限制为一道题或少量边确认成本：

```bash
python run_lomo_batch.py \
  --replay-from-run runs/phase0_MATH500_20260724T010203Z \
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
    stage_actions.jsonl
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

## 为并行 shard 生成 1000 条分层 LOMO

对于包含多个 shard 的 baseline，可以先生成计划而不调用模型：

```bash
python build_lomo_plan.py \
  --master-dir runs/phase0_MATH500_parallel_identity_migrated_20260709T151013Z \
  --output-dir lomo_plans \
  --lomo-sample-size 1000 \
  --seed 0
```

早期 MATH500 baseline 曾错误使用 `phase0_PRM800K_*` 目录前缀；计划生成器仍兼容这些历史目录，
但新运行统一生成 `phase0_MATH500_*`。

输出文件：

```text
plan_summary.json          # 总量和各层分布
lomo_candidates.jsonl      # 分层抽取的单边 LOMO candidate
```

LOMO 采样会排除 baseline 失败题、输入/输出边和自环，按轮次设置配额，并在每个轮次内
尽量均衡不同有向边。使用 40 并发、温度 0 执行已生成的 1000 条计划：

```bash
/home/yyh/miniconda3/envs/phase0-lomo/bin/python -B run_lomo_parallel.py \
  --lomo-plan lomo_plans/lomo_candidates.jsonl \
  --output-dir runs/phase0_MATH500_lomo1000_seed0_temp0 \
  --workers 40 \
  --temperature 0
```

如果运行中断，复用同一个计划和输出目录并增加 `--resume`：

```bash
/home/yyh/miniconda3/envs/phase0-lomo/bin/python -B run_lomo_parallel.py \
  --lomo-plan lomo_plans/lomo_candidates.jsonl \
  --output-dir runs/phase0_MATH500_lomo1000_seed0_temp0 \
  --workers 40 \
  --temperature 0 \
  --resume
```

每个 candidate 单独保存 `result.json`；续跑会跳过 completed 分支，只重试失败或未完成分支。
根目录的 `lomo_results.jsonl` 和 `manifest.json` 在整批结束时汇总。每个 completed 分支必须满足：

- 分叉前的 stage 状态和目标 stage 候选内容与 identity 一致；
- 目标 mask 恰好有一个 `1`，且对应计划中的 candidate；
- 目标 stage 之外所有 mask 都是全零；
- child 只包含一道对应题目的 prediction。

计划执行器只运行清单中的 1000 个 `(parent_run, candidate_id)`，不会自动补跑其余可删边。

旧的 398-success 通信路径计划已经删除。当前 `lomo_plans/lomo_candidates.jsonl` 从完整 400 条
identity 重新生成：候选池 4360 条，抽取 1000 个唯一 candidate，全部 parent 都指向迁移后目录并
带有完整 stage 元数据。LOMO child 从这些 identity 的完整 stage 边界生成，不需要重跑 identity。

temperature 0 的本次 LOMO 已完成 1000/1000，失败 0 条；1000 条均通过 identity prefix 和 one-hot
检查。抽样覆盖 316 道题，目标分布为 Planner 329、Solver 428、Judger 243。结果位于：

```text
runs/phase0_MATH500_lomo1000_seed0_temp0/
```

## 本次设计结论

### 数据集与命名

- `PRM800K/phase1_train.jsonl` 是 949 条嵌套 PRM 标注记录；`MATH500/test.jsonl` 是 500 条
  扁平 MATH500 记录。两者格式不同，题目精确重合数为 0，不是同一个数据集。
- 旧实验虽然使用 `phase0_PRM800K_*` 名称，但其 `summary.json` 中的实际数据路径是
  `MATH500/test.jsonl`。错误发生在实验、入口和输出目录的命名，不在真实 PRM800K 数据目录。
- 当前实验代码、配置、入口和新输出统一使用 `phase0_MATH500` / `phase0_MATH500_*`；真实
  `PRM800K/` 数据保留。计划生成、消息合并和评分代码继续兼容历史误名目录。
- RL 数据生成配置默认使用 seed 0 划分出的 train 400 条；test 100 条不参与训练，保留用于策略
  评估。旧并行通信任务也选择了这 400 条 train，但其中 398 条成功，2 条因网络连接中断失败。

### 稳定随机剪枝与旧 identity 记录
上传到
稳定随机 stage 子集剪枝现在强制读取 identity replay parent。不提供 `--replay-from-run` 和
`--checkpoint-candidate-id` 时命令直接失败，不会从头重新生成一个未配对的 random trajectory。
采样哈希包含 seed、题目位置、round、stage、`stage_state_hash` 和固定顺序的候选边，不包含
`run_id`，因此同一个 identity 状态可以用不同 seed 生成可对齐的反事实分支。

旧通信记录可以作为 identity baseline，因为已确认其中 `edge_policy` 为 `identity`，且全部
`edge_decisions` 都是 `keep`。语义上它等价于每个 stage 的全零 mask，但旧格式缺少：

- `stage_action_id`、`stage_state_hash` 和 `stage_edge_index`；
- `stage_actions.jsonl`；
- 显式的 stage 联合全零动作。

ReplayController 对新格式或已迁移记录会自动回退到 checkpoint 所属 stage 的开头。未经迁移的
旧格式只有 candidate 边界，random_drop 会拒绝使用，避免同一个 stage 出现前几条边沿用 identity、
后几条边执行新 mask 的混合动作。

旧 identity 记录应先迁移而不是重新调用模型：按题目、round、stage 聚合候选边，补齐 stage 字段
并生成全零 mask 的 `stage_actions.jsonl`。迁移后，同一个原始状态可以安全分叉为 identity、多个
random seed 和多条 one-hot LOMO 策略；未经迁移的旧记录不能作为 random_drop parent。

旧 identity 记录已经使用以下命令迁移，源目录保持不变：

```bash
python -B migrate_legacy_identity_records.py \
  --source-root '/home/yyh/MARL/mas_com_phase0-prm/通信记录/通信记录/phase0_PRM800K_parallel_20260709T151013Z' \
  --output-root runs/phase0_MATH500_parallel_identity_migrated_20260709T151013Z
```

迁移结果包含 40 个 shard、6,427 条候选边和 2,067 个 stage action。迁移只补充 stage 字段、
全零联合动作和本地有效的 summary 路径；原 messages、activations、predictions 和 split 文件保持
字节级不变。旧根目录元数据另存为 `legacy_parallel_manifest.json` 和
`legacy_parallel_summary.json`，新根目录使用本地可访问的 `parallel_manifest.json`、
`parallel_summary.json` 和 `migration_manifest.json`。

旧任务最初得到 398 条完整 prediction，以下两条因远端连接中断只保留了失败前缀：

- `question_id=216`、`source_index=483`；
- `question_id=330`、`source_index=25`。

旧失败前缀和对应 `errors.jsonl` 保持不变；两题现已分别补跑到原 shard 下的新目录：

```text
shard_0021_start_210_n_10/phase0_MATH500_20260809T073718Z
shard_0033_start_330_n_10/phase0_MATH500_20260809T073723Z
```

选择 identity parent 时，新 `phase0_MATH500_*` 记录优先于同一题的旧失败前缀，因此当前 train
已有 400 条完整 identity 回报。

### 当前保存格式

输入 MATH500 使用 JSONL，每行一条 JSON 题目，包含 `problem`、`solution`、`answer`、`subject`、
`level` 和 `unique_id`。一次新运行默认保存为：

```text
runs/phase0_MATH500_<timestamp>/
  splits/train.jsonl       # 训练题目划分
  splits/test.jsonl        # 测试题目划分
  predictions.jsonl        # 每道题的最终输出、步骤、轮数和 token
  messages.jsonl           # 实际成功投递的消息
  activations.jsonl        # Agent 输入、prompt、输出和耗时
  edge_candidates.jsonl    # 所有候选有向边，包括被删除的边
  edge_decisions.jsonl     # 每条边最终的 keep/drop 结果
  stage_actions.jsonl      # stage 状态、有序动作空间和联合 mask
  rl_edge_samples.jsonl    # 联合动作展开后的 edge-level 样本
  traces.jsonl             # 实际投递边的事件轨迹
  summary.json             # run 级汇总
```

除 `summary.json` 外均为 JSONL，即每行一个独立 JSON 对象。`messages.jsonl` 只包含实际投递的
消息；分析被删除的消息必须联合读取 `edge_candidates.jsonl` 和 `edge_decisions.jsonl`。

### stage 状态、动作与 RL transition

当前策略在 Agent 已经生成候选消息之后、消息投递之前选择 mask。因此 `stage_state_hash` 表示
“候选消息已经产生，但联合投递动作尚未执行”的状态指纹。它由以下内容计算：

- split、source index、question id、round 和 stage；
- 固定顺序的候选边；
- 每条边的 sender、recipient、kind 和候选消息正文；
- 每条边生成时对应的局部消息状态哈希。

`stage_state_hash` 不是完整状态本身；完整信息仍保存在 messages、activations 和 candidates 中。
它主要用于状态校验、稳定采样、去重和 transition 关联。当前 mask 约定为 `0 = keep`、`1 = drop`，
必须和同一条 `stage_actions.jsonl` 中的 `edge_order` 一起解释。

当前尚未保存 `next_state_hash`。后续建议将它定义为：当前 mask 执行并影响后续 Agent 后，到达的
下一个可剪枝 stage 的 `stage_state_hash`。终止状态使用 `next_state_hash: null` 和 `done: true`。
完整的 stage-level RL 样本应包含：

```json
{
  "run_id": "phase0_MATH500_...",
  "stage_action_id": "test:12:3:r1:solver",
  "stage_state_hash": "state-before-action",
  "edge_order": ["SolverA->Planner", "SolverA->Judger", "SolverB->Planner", "SolverB->Judger"],
  "action_mask": "1001",
  "reward": 0.5,
  "next_state_hash": "state-after-transition",
  "done": false
}
```

当前 `stage_actions.jsonl` 已保存 state 和联合 action，但 `reward` 仍为 `null`，并且没有显式的
`next_state_hash` 和 `done`，所以它是 replay-ready 日志，还不是完整的 RL transition。

### run_id 的用途

`run_id` 应继续保存在原始日志和训练数据索引中，但不应作为策略网络输入，也不应加入
`stage_state_hash` 或随机 mask 的采样哈希。原因如下：

- message、activation、candidate 和 decision 编号会在每个 run 中重新开始，并非全局唯一；
- `run_id` 用于区分 identity、LOMO、不同 random seed 和不同 shard；
- `run_id` 用于把 stage 串成同一条轨迹，并追溯配置、模型与 parent baseline；
- 排除 `run_id` 后，同一 baseline 状态在不同策略分叉中仍可得到相同状态哈希。

推荐使用 `(run_id, stage_action_id)`、`(run_id, candidate_id)` 和 `(run_id, message_id)` 作为日志
联合键。当前局部 `state_hash` 仍包含 message id；如果以后需要跨独立 run 按纯内容去重，可以
额外增加一个只依赖消息角色、类型和正文的 `semantic_state_hash`，不要替换用于严格 replay 的原哈希。
