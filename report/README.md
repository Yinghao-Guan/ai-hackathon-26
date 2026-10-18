# RoleKV report

使用 MacTeX 中的 XeLaTeX 编译，`latexmk` 自动处理需要重复编译的引用。

```text
report/
  main.tex         # 正文和研究框架
  references.bib   # 已核验的参考文献；正文通过 \cite 引用
  figures/        # 可编辑的 TikZ 缓存交接示意图
  methods_manifest.json # 方法核验时的数据、源码哈希与待确认设置
  .latexmkrc       # XeLaTeX 与输出路径配置
  Makefile         # 编译入口
  AuxiliaryFiles/  # 编译中间文件及工作 PDF（自动生成，不提交）
  output/          # 最终 main.pdf（自动生成，不提交）
```

在项目根目录执行：

```sh
make -C report
```

也可以进入 `report` 后执行 `make` 或 `latexmk main.tex`。

- `make watch`：监听源文件变化并自动重新编译；按 Ctrl+C 停止。
- `make clean`：清理辅助文件，保留 PDF。
- 最终 PDF 位于 `output/main.pdf`。

默认采用 Letter 纸张、10pt 双栏排版，页边距为 0.75 英寸，栏间距为 0.25 英寸。
标题、摘要与缓存交接示意图跨双栏，训练设置表使用单栏宽度。
`output/main-single-column.pdf` 保留本次改版前的单栏 PDF 快照，方便对比；
该快照不会随 `make` 自动更新。

`main.tex` 已启用中英文支持，字体使用 TeX Live 自带字体，不依赖额外系统字体。
正文已包含 Introduction、Background and Related Work 初稿与实验设计草案，
其余章节保留研究框架，没有预设实验结果。
Methods 区分已核验的数据和运行时、团队训练约定及尚未实现的比较基线。
当前默认工具反馈会执行全部 MBPP asserts，不能解释为隐藏测试评测。
参考文献由 BibTeX 管理，`latexmk` 会自动处理引用和编号。编辑器应选择 XeLaTeX/latexmk，
并在 `report` 目录运行，以读取 `.latexmkrc`。
