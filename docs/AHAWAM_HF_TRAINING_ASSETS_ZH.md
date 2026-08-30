# AHA-WAM SO101 训练资产：Hugging Face 目录与自动下载

> 更新时间：2026-08-30<br>
> 默认仓库：<https://huggingface.co/datasets/Jill111/AHA-WAM-SO101-HIL-training-assets><br>
> 仓库类型：Hugging Face Dataset；下载脚本按 **Public** 仓库实现

## 1. 最重要的目录规则

远端使用两个 ZIP 表示大量数据文件；Hugging Face 本身不负责解压，训练脚本下载后在本机安全展开。远端只要求以下 7 个运行资产：

| Hugging Face 仓库内路径 | 本服务器 `/home/fenghao/workspace/AHA-WAM` 下的源路径 | bytes | 用途 |
|---|---|---:|---|
| `checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt` | 同路径 | 14,699,235,086 | 官方 AHA-WAM 初始化权重 |
| `runs/so101_ahawam_plug_all/assets/dataset_stats.json` | 同路径 | 60,639 | 固定 02+04 normalization stats |
| `data/so101_ahawam_square_v1_text_embeds_cache/*` | 同路径 | 3,155,607 | 3 个 T5 embedding cache 文件 |
| `so101_ahawam_plug.zip` | 项目根目录的同名 ZIP | 965,095,995 | 包含 01/02/04；下载器只展开 02+04 |
| `ahawam_hil_raw.zip` | 项目根目录的同名 ZIP | 797,366,154 | 全部原始 HIL/HG-DAgger 数据 |

两个 ZIP 的固定校验值：

```text
83f48092b0db647435af7d9e0a1295d12a2d8c04e183a34c18318dfe485cbed0  so101_ahawam_plug.zip
c024b22b41a9346fe2c729e91e85f3155f83e847f3729d7f1029b73a2096bfcd  ahawam_hil_raw.zip
```

解压后的本地训练数据审计基准仍是 **696 个文件、16,331,861,500 bytes（约 15.21 GiB）**。

最终远端树的关键部分应为：

```text
Jill111/AHA-WAM-SO101-HIL-training-assets
├── README.md
├── checkpoints
│   └── AHA-WAM-Pretrained
│       └── AHA-WAM-pretrained.pt
├── runs
│   └── so101_ahawam_plug_all
│       └── assets
│           └── dataset_stats.json
├── so101_ahawam_plug.zip
├── ahawam_hil_raw.zip
└── data
    └── so101_ahawam_square_v1_text_embeds_cache
        └── 3 个 .pt 文件
```

## 2. 从另一台本地电脑上传

先创建一个 **Dataset** 仓库，仓库 ID 建议保持为：

```text
Jill111/AHA-WAM-SO101-HIL-training-assets
```

将其设为 Public。上传端需要 write token，但公开下载端不需要 token。

如果需要手工补传，可在 AHA-WAM 根目录运行：

```bash
hf auth login

hf upload Jill111/AHA-WAM-SO101-HIL-training-assets \
  scripts/so101_pipeline/hil_human_only_training/HF_ASSETS_REPO_CARD.md \
  README.md \
  --repo-type dataset

hf upload Jill111/AHA-WAM-SO101-HIL-training-assets \
  checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt \
  checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt \
  --repo-type dataset

hf upload Jill111/AHA-WAM-SO101-HIL-training-assets \
  runs/so101_ahawam_plug_all/assets/dataset_stats.json \
  runs/so101_ahawam_plug_all/assets/dataset_stats.json \
  --repo-type dataset

hf upload Jill111/AHA-WAM-SO101-HIL-training-assets \
  data/so101_ahawam_square_v1_text_embeds_cache \
  data/so101_ahawam_square_v1_text_embeds_cache \
  --repo-type dataset --exclude "**/.DS_Store" --exclude "**/Thumbs.db"

hf upload Jill111/AHA-WAM-SO101-HIL-training-assets \
  so101_ahawam_plug.zip so101_ahawam_plug.zip \
  --repo-type dataset

hf upload Jill111/AHA-WAM-SO101-HIL-training-assets \
  ahawam_hil_raw.zip ahawam_hil_raw.zip \
  --repo-type dataset
```

README 不计入 7 个运行资产。如果上传中断，重新运行对应命令即可；请勿使用 `--delete`。

远端曾上传过展开版 02 和部分 04。新下载器忽略这些旧路径，只读取两个根目录 ZIP；不必为了训练删除旧文件。

## 3. 下载器与训练入口

公开仓库不要求 `hf auth login`。代码环境只需要安装 `huggingface_hub`：

```bash
python -m pip install -U huggingface_hub
```

两个无参数训练入口都已经在 GPU preflight 和真正训练之前调用下载器：

```bash
scripts/so101_pipeline/hil_human_only_training/run_natural_5ep.sh
scripts/so101_pipeline/hil_human_only_training/run_balanced_2plus2_5ep.sh
```

也可以独立准备资产：

```bash
python scripts/so101_pipeline/hil_human_only_training/prepare_training_assets.py
```

下载器会：

1. 检查本地是否已经具备完整审计指纹；
2. 本地不完整时，匿名读取 public Dataset 的递归文件树；
3. 校验远端运行资产的路径和大小（默认 7 个；已手动放置的 ZIP 会从必需清单里移除，见 3.1）；
4. 取得两个 ZIP（优先用仓库根目录的本地副本，否则从 Hub 下载）并核对固定 SHA-256；
5. 拒绝路径穿越、符号链接、重复 member 和不符合审计数量的 ZIP；
6. 从 SO101 ZIP 只展开 02+04，明确忽略其中的 01；
7. 通过临时文件加 `os.replace` 逐文件落盘，避免留下半写文件；
8. 最终校验本地 696 个文件和总字节数；
9. 写入 `.cache/huggingface/ahawam_training_assets_state.json`，然后才继续 preflight。

强制联网核对远端清单：

```bash
python scripts/so101_pipeline/hil_human_only_training/prepare_training_assets.py \
  --force-online-check
```

在你完成全部上传之前，这条强制检查会有意报“远端 bundle 不完整”；这可以防止别人拿半套数据启动训练。

### 3.1 手动放置两个 ZIP（不经过 Hugging Face）

两个数据 ZIP 也可以线下传给运行方（微信、网盘、`scp` 等），不必先上传到 Hub。
运行方只需要把它们放到 **clone 出来的仓库根目录**，也就是和 `scripts/`、`docs/` 同一层：

```
AHA-WAM/                          # git clone 出来的目录
├── so101_ahawam_plug.zip         # 965,095,995 bytes
├── ahawam_hil_raw.zip            # 797,366,154 bytes
├── scripts/
├── docs/
└── ...
```

文件名必须完全一致，不要改名、不要解压、不要放进 `data/` 或子目录。
`*.zip` 已经在 `.gitignore` 里，不会污染 `git status`。

放好之后直接运行既有入口即可：

```bash
scripts/so101_pipeline/hil_human_only_training/run_natural_5ep.sh
```

下载器发现根目录已有合法 ZIP 时会：

- 打印 `[assets] using local archive <path>`，直接用本地文件；
- **不再要求 Hub 上存在这两个 ZIP**，只校验 checkpoint、`dataset_stats.json` 和 3 个文本缓存共 5 个远端文件；
- 仍然逐个核对 SHA-256，再按同样的安全规则解压。

建议运行方在启动前自校验一次（应与下面的值一致）：

```bash
sha256sum so101_ahawam_plug.zip ahawam_hil_raw.zip
```

```
83f48092b0db647435af7d9e0a1295d12a2d8c04e183a34c18318dfe485cbed0  so101_ahawam_plug.zip
c024b22b41a9346fe2c729e91e85f3155f83e847f3729d7f1029b73a2096bfcd  ahawam_hil_raw.zip
```

只放一个 ZIP 也可以：放好的那个走本地，另一个仍然从 Hub 下载。
如果传输过程损坏（大小对但哈希不对），下载器会明确报错并提示重新传，不会静默回退去下载。

## 4. 修改仓库地址或使用私有镜像

默认仓库 ID 定义在 `training_assets.py`，也可以不改源码而覆盖：

```bash
export AHAWAM_TRAINING_ASSETS_REPO="organization/repository-name"
```

默认匿名下载。若以后把镜像改回 Private，可在训练调度器的 secret 中提供 read-only token：

```bash
export HF_TOKEN="hf_..."
```

不要把 token 写入脚本、YAML、日志或 Git。

## 5. 数据语义与公开风险

`data/ahawam_hil_raw` 保留 POLICY、SYSTEM_HOLD 和 HUMAN 原始行，用于审计和重建逻辑 correction。训练 loader 仍按已经实现的 HUMAN-only 规则取监督目标；远端公开或下载方式不会改变这一训练语义。

Public 仓库意味着任何人都能下载相机录像和权重。正式切换公开前，仍应检查：

- 全局/腕部相机画面、文件名和元数据是否含隐私信息；
- AHA-WAM 权重及所含第三方资产是否允许公开再分发；
- 数据许可、免责声明和引用信息是否清楚。

这里按你的决定把训练端实现为 public/匿名下载，但公开内容审计仍由上传者负责。
