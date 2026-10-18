# RoleKV report

使用 MacTeX 中的 XeLaTeX 编译，`latexmk` 自动处理需要重复编译的引用。

```text
report/
  main.tex         # 完整正文初稿
  references.bib   # 已核验的参考文献；正文通过 \cite 引用
  figures/        # 可编辑的 TikZ 示意图和 PGFPlots 实验曲线
  tables/         # 从原始日志生成的结果表
  analyze_results.py # 无第三方依赖的离线结果核验与图表生成
  results_summary.json # 经核验的评测、固定轨迹与训练记录摘要
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
本 worktree 不包含先前主目录中的单栏 PDF 快照，生成文件不随 Git 同步。

`main.tex` 已启用中英文支持，字体使用 TeX Live 自带字体，不依赖额外系统字体。
正文已包含摘要、引言、相关工作、实际实验方法、结果分析、讨论与结论初稿。
结果基于六种方法各 100 个任务的 4090 日志，以及固定四轮轨迹的 60 次成本测量。
Methods 已按五个 adapter 的实际训练记录更新，未运行的比较列为局限与后续工作。
当前默认工具反馈会执行全部 MBPP asserts，不能解释为隐藏测试评测。
最终通过率是可经过两次修订的 session 成功率，不是初次 Coder 输出的准确率。
日志中的 `gpu_s` 是各轮模型耗时之和，不包含工具执行、最终评分或加载。
H1 与 A2 的 61/100、59/100 不构成质量等效证明，第二轮错误批准单独报告。
引言列明运行时、评测和实证发现三项贡献，明确 aLoRA 是已有工作。
缓存机制图放在引言后；结果中加入 MBPP task 135 的逐轮案例，
用于解释错误批准，非代表性样本。案例片段由脚本直接从原始日志提取。

重新核验原始日志并生成结果表、矢量图和数据摘要：

```sh
python3 report/analyze_results.py
make -C report
```

该脚本只聚合已提交日志，不调用模型、不执行生成的代码，不产生新的 GPU 结果。
会检查任务配对、context 一致性、逐轮计数/耗时汇总及训练输入哈希。
图表使用 PGFPlots 绘制，机制示意图仍为 TikZ；两者均为可编辑矢量源文件。

参考文献由 BibTeX 管理，`latexmk` 会自动处理引用和编号。编辑器应选择 XeLaTeX/latexmk，
并在 `report` 目录运行，以读取 `.latexmkrc`。
