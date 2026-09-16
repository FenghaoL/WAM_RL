# 中文LaTeX阅读版

主文件：[main_中文全文.tex](latex/main_中文全文.tex)。编译结果：[main_中文全文.pdf](latex/main_中文全文.pdf)。

保留英文稿同目录的`ieeeconf.cls`、`references.bib`和`figures/`，直接在同一项目中把主文档切换为`main_中文全文.tex`，**编译器选择XeLaTeX**。无需独立建项目，也无需安装Windows宋体：中文使用TeX Live提供的Fandol字体、ctex中文支持包。Tectonic也可编译。不要使用英文项目可能默认选择的pdfLaTeX。

本地编译：

```bash
cd docs/icra2027_handoff
./build_paper_zh.sh
```

本服务器本次使用已有临时Tectonic编译器：

```bash
TECTONIC_BIN=/tmp/icra_tex_tool/tectonic XDG_CACHE_HOME=/tmp/icra_tex_cache ./build_paper_zh.sh
```

脚本的输出是`main_中文全文.pdf`，不会覆盖英文`main.pdf`。中文源稿可以直接编辑，不依赖运行Markdown转换程序；后续Markdown或英文修改不会自动同步，需要同时维护。

已实际编译检查：6页双栏；四个编号公式、四张表和两个图位；无缺字、越界或未解析引用。中文比英文紧凑，因此页数不同，不要求强行排满八页。图1复用现有英文标注的矢量图，图题及论文文字为中文；图2沿用待真实实验素材的占位。
