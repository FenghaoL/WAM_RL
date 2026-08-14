# SO101 AHA-WAM Dashboard

这个目录只保留当前 SO101 在线部署客户端。它连接 Windows 上的 SO101 follower 和
两路相机，通过 SSH 隧道调用 GPU 服务器上的 AHA-WAM，再以 15 Hz 执行服务器返回
的 8 步动作块。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run_remote_client.ps1` | 唯一启动入口；常用设置都在文件顶部 |
| `open_policy_tunnel.ps1` | 建立动作或视频服务的本地 SSH 转发 |
| `ahawam_dashboard_client.py` | Dashboard、机械臂控制、相机处理和 TCP 客户端 |
| `README.md` | 当前使用说明 |
| `SERVER_INTERACTION.md` | 提供给本地 Codex 的服务器接口契约 |

## 设计

- 一个 worker 线程独占机械臂、相机和 TCP socket；Tk 线程只负责界面，避免多个
  线程同时操作串口。
- 当前 checkpoint 直接使用 LeRobot `use_degrees=True` 的 calibrated-degree frame。
  客户端不再应用旧 G0.5 的 signs/offsets，服务器返回的动作也直接是机械臂角度。
- fixed 相机先从 `640x480` 裁去右侧 160 像素。模型输入为 `384x320` RGB：fixed
  占上方 `320x256`，wrist 占左下 `160x128`，右下 `160x128` 补黑。这与当前训练
  数据和服务器预处理一致。
- 动作模式每次请求 `[8,6]` 动作块，在推理期间保持当前姿态，然后以 15 Hz 执行
  完整动作块。每个 tick 最多变化 3 度。标定范围只在 Dashboard 显示，不会裁剪、
  拒绝或中止动作；这是按操作者此前要求保留的行为。
- Home 是可选工具，不是 Run 的前置条件。机械臂可以从任意机械上有效的姿态直接
  Run；但超出训练数据分布时，模型效果没有保证。
- PowerShell 默认 `$EnableMotion = $false`。首次部署先 dry-run，确认状态、画面和
  动作后再手动改为 `$true`。

## 本地环境

`run_remote_client.ps1` 默认使用：

```text
%USERPROFILE%\.conda\envs\so101-ahawam-collect\python.exe
```

该环境需要已有的本地 LeRobot SO101/Feetech 支持，以及 `numpy`、`opencv-python`、
`Pillow` 和 Tk。机械臂标定文件由 LeRobot 根据 `$FollowerId` 读取。

运行前编辑 `run_remote_client.ps1` 顶部设置：

```powershell
$Task = "Pick up the white block."
$Mode = "action"
$PolicyPort = 12000
$EnableMotion = $false
$FollowerPort = "COM4"
$FixedCameraIndex = 2
$WristCameraIndex = 0
$HomeOnStartup = $false
```

服务器当前只缓存两个精确任务，大小写和句号都不能变：

```text
Pick up the white block.
Put the white block on the paper roll.
```

当前默认 `2/0` 与 2026-08-13 的实际训练采集设置一致；抽查该训练集时，索引 `0`
保存的是夹爪附近的腕部视角。Windows 相机索引会在重新插拔、禁用设备或更换 USB
控制器后变化，因此不要根据更早的 `1/2` 记录或数字本身猜测设备。每次部署都必须
查看 Dashboard：fixed 应是完整桌面固定视角，wrist 应是夹爪附近视角。若语义不对，
先修改脚本顶部索引并重新检查，再允许真机运动。

## 启动服务器

在 GPU 服务器的 AHA-WAM 仓库中启动动作服务：

```bash
bash scripts/so101_pipeline/online_deploy/run_action_server.sh
```

它监听服务器回环地址 `127.0.0.1:12000`。当前部署配置使用四轮训练的
`step_003440.pt`、新 SO101 stats 和新文本 embedding cache。

未来视频诊断使用另一个进程和端口：

```bash
bash scripts/so101_pipeline/online_deploy/run_video_server.sh
```

该服务监听 `127.0.0.1:12001`。单卡时不要同时启动两个完整模型服务。

## 建立 SSH 通道

在 Windows PowerShell 新开一个窗口，确认 `open_policy_tunnel.ps1` 顶部的
`$ServiceMode = "action"`，然后保持脚本一直运行：

```powershell
cd D:\workspace\so101
.\scripts\ahawam\dashboard_client\open_policy_tunnel.ps1
```

这样本地 Dashboard 访问 `127.0.0.1:12000` 时，数据会被转发到服务器动作服务。

若使用未来视频服务，先关闭动作服务和原通道，把 `$ServiceMode` 改成 `"video"` 后
重新运行脚本。它等价于：

```powershell
ssh -N -L 12001:127.0.0.1:12001 fenghao@chickadee.engin.umich.edu
```

同时把 PowerShell 设置改为 `$PolicyPort = 12001`，并将 `$Mode` 改为
`"video_preview"` 或 `"infer_with_video"`。

## 运行 Dashboard

可从仓库根目录直接执行：

```powershell
cd D:\workspace\so101
.\scripts\ahawam\dashboard_client\run_remote_client.ps1
```

首次测试建议：

1. 保持 `$EnableMotion = $false`，启动动作服务器和 12000 SSH 通道。
2. Dashboard 连接后点击 `Health`，确认 6D、chunk 8、horizon 40、图像
   `[384,320,3]` 和 cached text。
3. 检查 model input：上方 fixed，左下 wrist，右下纯黑。
4. 点击 `Run`，观察至少 10 个动作块；state sent to server 应与 robot state 相同。
5. 点击 `Stop + reset`，确认服务器返回 reset acknowledgement。
6. 确认目标角度连续且在标定范围内后，关闭客户端，把 `$EnableMotion` 改为 `$true`。
7. 真机首次运行保持手靠近急停，只运行少量 chunk。Home 按钮可用，但完全可选。

## 三种模式

- `action`：请求动作并在非 dry-run 时控制机械臂，使用端口 12000。
- `video_preview`：只生成未来视频，不返回或执行动作，使用端口 12001。
- `infer_with_video`：一次请求同时返回动作块和未来视频，但 Dashboard 不执行该动作，
  使用端口 12001。

未来视频保存到本目录的 `outputs/ahawam_previews`，请求时序记录保存到
`outputs/ahawam_client_timing.jsonl`。

动作请求默认超时 120 秒；视频生成默认超时 1800 秒。两者分别由脚本顶部的
`$ActionTimeoutS` 和 `$VideoTimeoutS` 控制，避免动作服务卡住时无提示等待半小时。

## 当前与旧部署的区别

本目录是唯一有效的本地部署实现，按 `SERVER_INTERACTION.md` 中的
`so101_ahawam_square_v1 / step_003440.pt` 契约工作：原生 calibrated-degree 坐标，
以及左下腕部、右下补黑的模型图像。仓库 `docs/DEPLOYMENT_GUIDE_ZH.md` 描述的是旧
`so101_g05_square_v1 / step_006524.pt` 服务，其 signs/offsets 和整行腕部图像不能
用于当前服务。切换服务器 checkpoint 时必须同时切换客户端契约，不能混合配置。

## 常见问题

- `Connection refused`：服务器未启动、SSH 窗口已关闭，或本地/远端端口不一致。
- `No cached text embedding`：任务文字与上面两个 prompt 不完全一致。
- `health mismatch`：连接到了旧服务或错误端口，不能继续真机运行。
- 相机 shape 错误：fixed 必须能裁成 `480x480`，wrist 必须是 `640x480`。
- Run 不要求 Home；`Torque off` 后必须先 `Torque on` 才能执行真实动作。
