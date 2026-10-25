# 成功帧与区间标注

在仓库根目录运行：

```powershell
.\scripts\ahawam\frame_annotation\run_annotation.ps1
```

使用现有 `so101-ahawam-collect` 环境的 Python、Tkinter、PyAV、Pillow、NumPy、PyArrow，不升级 LeRobot。默认载入上次统计的六批 cylinder HIL 数据，共 103 条保存的 episode，包括成功和失败。丢弃的 rollout 没有保留视频，无法事后逐帧标注。

## 窗口怎么用

左侧选择 episode，条目显示采集时间、数据 episode_index 和原 outcome。右侧同时显示 fixed/wrist 原图，没有应用采集端以外的裁剪。标注以 Parquet 的 `frame_index` 为准，从 0 开始；画面显示原有数据 `timestamp`。视频先实际解码检查帧数与表行数一致，定位时顺序解码到指定帧，不依赖压缩视频的不精确随机 seek。大范围拖动可能需要等待解码；只缓存每个相机最多 24 帧。

1. 用 Space 播放/暂停，用左右方向键逐帧查看。拖动滑条跳转，也可点击 ±15 跳约一秒。
2. Label 默认 `success`。找到首次**已经达到任务成功判据**的画面，按 I 设置区间起点，再到成功状态结束的帧按 O 加入闭区间。起止帧都算正例。
3. 若成功状态从当前帧一直保持到录像末尾，按 E 一次标到末尾。只有确认最后都成功才这样标；插进后又拔出、摆放失稳或人为复位，就用 I/O 只圈真实成功段。允许多段。
4. 如果只需要标首次成功的那个瞬间，可以按 K 标一个单帧。注意：若将来训练视觉成功分类器，单帧之后仍处于成功状态的帧也应该圈进成功区间，否则完成审阅会把它们误标为负例。
5. S 保存草稿。确认**整条视频针对当前标签已经审阅**后，按 R 完成审阅并保存，弹窗会确认将该标签下所有未圈中的帧视作负例。失败 episode 如果确实全程没有成功状态，直接 R，得到完整负例；原 outcome=success 也必须实际查看视频，不能自动把最后一帧判正。
6. N 下一条；Previous 上一条。离开或关闭时若有未保存修改，会提示保存/放弃/取消。已保存的区间会在下次打开时恢复，左下 Reviewed labels 显示哪些标签已审阅。
7. 选中区间后点 Delete selected interval 删除错误区间，再重新标。Notes 可以写遮挡、成功判据或不确定处。修改已完成审阅的区间后仍需保存；该标签继续视作全片已审阅。

按钮、列表聚焦时也应支持上述快捷键；Label 和 Notes 输入框聚焦时保留正常文字输入，快捷键不生效，点一下视频后再用快捷键。

对于 cylinder，建议统一使用“圆柱已插入匹配孔且达到预先约定的深度/稳定状态”作为成功判据；是否必须松爪由你的实验定义决定，先固定定义再标全部数据。可以用额外 `release` 标签单独记录松爪。不可见的深度、接触力不能仅从画面臆测；有遮挡时先存草稿并写备注，不要把不确定的整片标为已审阅。

## 标签放在哪里

每个采集目录新增：

```text
recording_context/
  frame_annotations/episode_000000.json   # 人工编辑保存的事实来源
  frame_labels/chunk-000/episode_000000.parquet  # 按需导出的逐帧训练辅助表
```

复用了原来已有的 `recording_context` 扩展区。原有 `episodes_manifest.jsonl` 的 `outcome`/`outcome_class` 是整段结果，含人工干预后的成功；它们没有精确成功时间。原 `data/*.parquet` 没有现成 reward/success/done 列。逐帧人工标签不适合直接塞进 episode outcome，更不适合自动修改原数据表和 info/stats。因此以小 JSON 保存人工区间，需要训练时展开，视频不复制、不重编码。

JSON 记录原 episode_index、logical_episode_id、帧数、原 Parquet SHA256、区间、已审阅标签、备注及更新时间。保存使用临时文件+原子替换；原数据被重新编号/改写后，身份检查会阻止默默复用错位标签。将数据上传服务器时，应带上整个 `recording_context`。支持单人单窗口标注，同一 episode 不要开两个窗口同时保存。

标签支持任意小写英文名（字母开头、可含数字和下划线）。在脚本 `$Labels` 增加预设，或者在窗口 Label 输入新名字，如 `grasp`、`fine_align`。区间按标签分别审阅；完成 `success` 审阅不会把 `release` 自动判负。当前简单实现支持二值帧/区间标签，不涉及框、关键点或分割。

## 导出、检查与更换数据

把 `run_annotation.ps1` 开头 `$Mode` 从 `annotate` 改为 `export`，运行同一个命令即可处理六个目录中所有已保存标注；改为 `audit` 则只检查表结构和标注身份。窗口的 Export saved labels 按钮也能导出。重复导出会更新辅助表，不改原始数据；未创建 JSON 的 episode 不导出。

更换数据：修改 `$DatasetBase` 和 `$Runs`；支持多批次。未来不同任务使用另一份脚本默认值即可，不必更换环境。

导出的辅助 Parquet 按原索引连接原表：

| 字段 | 含义 |
| --- | --- |
| episode_index / frame_index / timestamp / index | 从源表原样保留；索引只在各采集目录内唯一 |
| label.success | 当前观察帧是否处于成功状态 |
| label.success.valid | 是否有有效人工标签；false 不能当负例 |
| success_event | 进入成功区间的首帧事件，可多次出现 |
| success_event.valid | 当前及前一帧标签是否足够确定成功边界 |
| label.其他标签、label.其他标签.valid | 使用相同区间展开逻辑 |

草稿只对圈中的正例帧有效；未圈中是未知。R 审阅完成后，该标签所有帧有效，未圈中为负例。`success_event` 是状态进入事件，不是直接可用的 transition reward：只标一个单帧时，也不能据此声称已经给整段轨迹提供了可靠的稀疏奖励。

## 给 RL 训练端的说明

先连接每个采集目录自己的辅助表与原表，以 `frame_index`/`episode_index` 精确对齐。不要跨目录只凭 episode_index 合并，也不要跨 episode 或原始控制时间断点构建 transition。原 `control_source`、`action_supervision_mask`、原有时序字段仍应保留；视觉标签可以包含 SYSTEM_HOLD 的画面，但这些画面不因此变成有效的人类 action 监督。RL 是否纳入 hold、用哪个实际 action/执行时长，需要你的具体算法处理；模仿学习的 mask 不自动等于 RL transition mask。

训练成功分类器：只用 `label.success.valid=true` 的观察帧，`label.success` 作为二值目标；建议按 episode/采集批次划分训练验证，避免临近几帧随机分到两侧造成泄漏。成功图像和失败图像都需要，人工干预后完成的图像也可以是成功状态正例。

构建离线 RL：当前标注描述 `s_t`（发动作前的观察），而常见 transition `(s_t,a_t,r_t,s_{t+1})` 的 `r_t` 描述动作后的结果。因此从第 k 帧成功事件生成奖励时，通常应对齐到实际导致它的上一条有效 transition；不能盲目把它放到源表第 k 行 action。若终止于首次成功，则同时截断该 transition 之后的数据；超时/人为停止与任务终止要分别处理，不能把每段录像末尾自动设成成功或 terminal。本工具不生成 `reward`、`next.reward`、`done`、`discount`，因为这些需要先确定你服务器训练端的 transition 约定。

这是一套可迁移的人工视觉事实标注，并非把 HIL 数据直接转成任意 RL 算法都可训练的数据。奖励约定确定后，训练端可以将这些辅助标签映射到对应库字段，不需要重新人工标注。

## 方案依据

- [LeRobot HIL-SERL 官方流程](https://huggingface.co/docs/lerobot/hilserl)：支持人工成功反馈和视觉奖励分类器；建议采集成功后的多帧观察训练分类器。因此这里同时支持单帧事件和持续成功区间。
- [HIL-SERL 奖励分类器源码](https://github.com/rail-berkeley/hil-serl/blob/main/examples/train_reward_classifier.py)：分开加载正负样本并训练二值标签；整条成功轨迹不是整段正例。
- [RLDS 数据语义](https://github.com/google-research/rlds)：观察、动作、奖励、terminal/truncation 需有明确的时间和终止含义。因此本工具保存观察帧事实，将 RL 奖励映射留给训练适配层。
- [CVAT 官方格式说明](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/)：提供帧标签、属性等通用标注。当前只有本地双视频和时间区间需求，用已有 Tkinter/PyAV 做轻量窗口更便于直接对齐本仓库的 Parquet，无需部署完整标注服务。

本次未自动写入任何真实成功帧标签。需要你根据任务完成判据在窗口中审阅。

本地验证已检查六批共 103 条 Parquet 的长度、帧索引和 episode 索引，抽查第一条真实 AV1 双视频的帧数与前后跳转，完成窗口创建、切帧、换 episode 和关闭测试；合成数据测试覆盖区间正负/未知、成功事件边界、保存导出、源文件变化检测及视频精确帧定位。

此前聊天中逐批数量正确、合计计算有误；正确合计为 90 条成功、13 条失败、34 次丢弃、137 次总尝试。按无干预成功才算自主成功的口径，自主成功率是 7/137≈5.11%。本工具未改写这些原始结果标签。
