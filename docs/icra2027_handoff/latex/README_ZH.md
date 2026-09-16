# LaTeX 工程入口

- `main.tex` / `main.pdf`：英文论文。
- `main_中文全文.tex` / `main_中文全文.pdf`：中文阅读版，同一工程，使用 XeLaTeX 或 Tectonic。
- `figures/method_overview.tex`：可编辑 TikZ 方法图；编译导出PDF供论文引用以固定中英文字体。
- `figures/rollout_comparison.pdf`：正文关键帧拼图。
- `figures/rollout_comparison_all_frames.pdf`：完整33帧拼图，适合放大查看。
- `old/`：原始论文草稿；`archive/`：改版前方法图，均不参与当前编译。

从本目录运行 `./build_paper.sh`、`./build_paper_zh.sh`。英文默认检查8页上限；需要保存超长工作稿时，显式设置 `ALLOW_OVERLENGTH=1`。已有图件PDF可直接用于编译，无需重新生成。

当前验证：英文8页（含参考文献）、中文7页；无缺失字符、未定义引用或越界排版警告。

当前服务器命令：

```bash
TECTONIC_BIN=/tmp/icra_tex_tool/tectonic XDG_CACHE_HOME=/tmp/icra_tex_cache ALLOW_OVERLENGTH=1 ./build_paper.sh
TECTONIC_BIN=/tmp/icra_tex_tool/tectonic XDG_CACHE_HOME=/tmp/icra_tex_cache ./build_paper_zh.sh
```

重新生成图件：运行 `./build_figures.sh`。原始照片仍在 `../figures/rollouts/`，构图脚本只引用图片，不改写原始像素；清单记录SHA256、帧选择与裁剪视口。单独分享此LaTeX目录可以编译论文；重新生成照片拼图需要保留上层原始图片目录。
