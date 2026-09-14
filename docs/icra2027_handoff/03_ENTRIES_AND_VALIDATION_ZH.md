# 已搭建的入口、运行方式与验证边界

新增代码统一在 [scripts/so101_pipeline/paper_experiments](../../scripts/so101_pipeline/paper_experiments/)。这些入口复用已经跑通的 `ovcr_adapter/v2_experiments` 训练器与队列，不另造一套优化器，不修改当前模型服务。

## 1. 已准备的cylinder机制对照（是否已训练另查运行记录）

```bash
cd /home/fenghao/workspace/AHA-WAM
export PYTHONPATH=src:.
PY=/home/fenghao/miniconda3/envs/ahawam/bin/python

# 已执行，拒绝覆盖存在的run目录；无需重复执行。
"$PY" scripts/so101_pipeline/paper_experiments/prepare.py controls
```

输出在 `runs/icra2027_cylinder_controls/`：

- [full_context_truncated.yaml](../../runs/icra2027_cylinder_controls/configs/full_context_truncated.yaml)：训练HUMAN上下文仅段内。模型仍history6；验证仍使用与C-H相同的跨界上下文，便于同条件评价。
- [full_video_off.yaml](../../runs/icra2027_cylinder_controls/configs/full_video_off.yaml)：真正将 `model.loss.lambda_video` 置0。
- [plan.json](../../runs/icra2027_cylinder_controls/plan.json)：预期初始化、步数、科学变量与**计划**最终权重；`trained=false` 是生成时状态，不是动态监控。

上述两个配置均从基本C-H原配方复制，初始化C-SFT，3285步，全局batch12，lr1e-5。生成器检查科学变量变化集合；验证配置保持固定。上下文截断同时影响其有效历史/时间位置，这是数据上下文模式的对照，不是仅遮掉某个KV值的实验。

优先训练 `full_context_truncated`，其余是备用，不建议截止前无条件全跑。

```bash
# CPU数据检查，不加载大模型、不训练、不占用三张训练卡：
bash scripts/so101_pipeline/paper_experiments/run.sh data \
  runs/icra2027_cylinder_controls full_context_truncated

# 下面会真正占用三张GPU；当前任务未替用户执行。
# 在已有队列释放GPU后，先两步smoke，成功后再正式训练。
bash scripts/so101_pipeline/paper_experiments/run.sh smoke \
  runs/icra2027_cylinder_controls full_context_truncated

# 建议在tmux里执行，以便断开终端后保留任务。
WANDB_MODE=online GPU_IDS=0,1,2 \
bash scripts/so101_pipeline/paper_experiments/run.sh train \
  runs/icra2027_cylinder_controls full_context_truncated
```

wrapper复用旧queue的输出拒绝覆盖、失败即停和GPU显存检查；使用独立端口29593，可通过 `MAIN_PORT` 改变。不会终止其他队列或服务。smoke写独立后缀输出，不能放入论文正式结果。

数据检查自动把HF datasets缓存放在仓库 `runs/cache/huggingface/datasets_v2`。本次直接调用runner最初因默认用户目录只读而失败，切换到上述已有可写缓存后通过；不是数据/schema失败。

## 2. power v3：先固定验证划分，再复用入口

```bash
PYTHONPATH=src:. /home/fenghao/miniconda3/envs/ahawam/bin/python \
  scripts/so101_pipeline/paper_experiments/prepare.py power \
  --hil-root data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v3 \
  --camera-note '本轮固定全局/腕部相机标识、重新校正日期与记录' \
  --run-root runs/icra2027_power_new_camera
```

**当前不能原样启动：** v3目录已存在，但四个session均没有`_val`后缀，上述入口会拒绝生成。需要先固定独立验证session，或扩展显式整轨迹验证白名单后再执行；不移动正在采集的目录，不随机拆相邻帧。命令保留为划分准备后的入口示例。约束和行为：

- 拒绝已作废的旧power HIL-v1/v2目录及其子目录。
- 需要独立的 `_val` 结尾session，并且train/val均有有效HUMAN。
- 沿用P-SFT `step_005328`、原power demo白名单与stats，只把纠正集替换成明确的新集合。
- 沿用power full预算2664步、lr1e-5、1:1、history6、跨界执行帧48、video loss1、无未来延伸。
- 输出 `full_new_camera_context`，并冻结parquet清单、camera-note和训练/验证切分。
- camera-note是人工记录，不代表工具自动验证了相机标定；生成后应检查图像、动作同步和P-SFT在同setup下的表现。

固定划分后使用同一wrapper依次 `data` / `smoke` / `train`。默认是无未来延伸的基本full；若选vidext为跨任务主配方，还需明确生成video extension配置并检查成功标记。此次只更新文档与读取元数据，没有生成或启动power训练。

## 3. 真机测试表与结果统计

```bash
PY=/home/fenghao/miniconda3/envs/ahawam/bin/python
"$PY" scripts/so101_pipeline/paper_experiments/trials.py plan \
  --task cylinder --protocol full_task \
  --models C-SFT C-H C-HV --conditions 20 \
  --output /tmp/cylinder_trials.csv

# 完成真实试验后填写CSV，再统计：
"$PY" scripts/so101_pipeline/paper_experiments/trials.py summarize /tmp/cylinder_trials.csv
```

已经生成一份 [cylinder待填表](cylinder_trials_template.csv)。每个物理初始条件下随机化模型顺序；实验前另外冻结每个condition ID对应的位姿/重置方法。该表是测试计划，不含已运行结果。正式实验请复制到自己的结果目录，不覆盖计划模板。

字段：`status=pending/completed`、`physical_success=0/1`、实际`interventions`、`duration_s`、精确checkpoint、camera_id。发生接管后成功只计物理完成，不计自主成功。pending不进入分母；duplicate condition、混合checkpoint、混合camera和非法数值会拒绝统计。统计含Wilson95%区间、配对条件中各模型独有成功数、未配对数量；不自动给出“显著提升”的结论。

完整任务用 `full_task`，人工孔口初始化用 `local_recovery`，开发筛选用 `development`。三个协议分组计算，不能混在一个成功率中。现有工具要求每模型每condition ID唯一；重复试验可使用`offset_left_near_rep01`等预定义ID，并在两个模型中保持对应。另保存基础条件、重复编号与session，避免工具将重复ID视为独立初始化。当前工具不做簇相关推断；用户已有不等次数汇总不能自动变成配对记录。

## 4. 给作者刷新证据清单

```bash
PYTHONPATH=src:. /home/fenghao/miniconda3/envs/ahawam/bin/python \
  scripts/so101_pipeline/paper_experiments/collect_evidence.py \
  --output docs/icra2027_handoff/evidence_after_training.json
```

拒绝覆盖既有快照；新文件名表达新的核查时刻。会读取运行时配置、源文件hash、日志、验证值、权重大小/存在性和队列状态，不加载15GB权重，不把loss转成成功率。`suite_record=completed`和最终checkpoint存在应一起看；运行中日志的最后step仅是瞬时快照。

论文表格应绑定一个冻结快照、checkpoint与真实试验记录。不能用旧清单的配置名推断文件已训练完成。

## 5. 原入口的历史验证记录与本轮边界

- 新增测试覆盖：上下文控制不改变验证/骨干模式、重复/混合权重拒绝、人工协助成功排除、pending分母、Wilson区间、旧power目录拒绝。
- 之前准备的控制（包括本文关注的`full_context_truncated`、`full_video_off`）使用真实数据跑CPU data-only，预期并核对长度 `[19708, 3460, 661, 169]`，四类数据样本字段可加载。
- 生成器所用C-H配置与其运行时 `experiment_config.yaml` 相等；新增科学变量之外只改变输出、实验标识与来源记录。
- 新增shell入口通过语法检查；测试表模板只有pending数据。
- **尚未执行**新增配置的GPU两步训练、正式训练或真机比较。复用既有trainer降低实施成本，不等于新配置已经验证性能。
- 新power v3已确认存在，但未完成有效HUMAN审计与划分；该分支仍需完成真实数据data-only与smoke。

本轮文档更新未运行训练或修改部署。新实机汇总单独登记于[user_report_results.json](user_report_results.json)，原始工程快照不被覆盖。
