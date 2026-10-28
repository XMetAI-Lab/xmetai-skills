# 台风误差报告工作流

生成可追溯、可复现的台风预报误差报告，覆盖确定性模型与集合平均预报。默认用中文输出 Markdown 报告、相关图表和机器可读统计结果；用户指定语言或格式时遵从用户要求。

这是 `xmetai-evaluation` skill 的台风报告分支，由原 `typhoon-error-report` 整合而来，
只消费已有 evaluation 产物，不重跑中心诊断。通用计算与渲染在随附运行库
`scripts/reporting/reports/typhoon_report.py` 与 `typhoon_plots.py`；skill 脚本只做入口，
`render_report.py` 已接通标准台风产物的自动派发。
不要假定其他项目具有相同的目录、字段、模型或依赖。

## 可执行入口

```bash
python skills/evaluation-reporting/scripts/render_report.py \
  evaluation_results/weather_typhoon_ens_fgvp \
  --model FGVP \
  --compare "FuXi=evaluation_results/weather_typhoon_ens_fuxi" \
  --result reports/typhoon_ens_comparison \
  --leads 24 48 72 96 120 \
  --fixed-leads 24 48 72 96 120
```

- `--model NAME=目录` 可重复，支持单模型；输入可为标准产物根或场次曲线目录。
- 预报类型从每个场次 `_meta.json` 的 `forecast_type` 读取，不能混合确定性/集合。
- 旧确定性元数据没有类型时可显式给 `--forecast-type det`；与已有声明冲突则报错。
- 旧集合汇总缺少有效时间、观测列或存在陈旧聚合标签时，用
  `--verify-legacy-members` 从对应 `_members.csv` 只读核验。路径检验平均位置距离，
  AT/CT 核对成员投影平均与平均位置投影，强度核对成员有符号误差平均；
  容差为 0.01 对应单位，仅用于归档数值精度，不能掩盖真实口径差异。
  核验不一致立即报错，不修正或覆盖输入；匹配方法与历史差异写入汇总和报告。
- 不给 `--leads` 就使用真实数据中的时效；不固定模型名、日期、台风或成员数。
- `--bootstrap` 默认 2000，`--seed` 默认 20250910；`--bootstrap 0` 禁用区间。
- `--fixed-leads` 可选，仅对路径计算跨指定时效均完整的固定样本敏感性。
- 报告分析方向显式配置：`--primary-model` 指定主模型（默认第一个 `--model`），
  `--previous-model` 指定旧版本，不从模型名推断新旧关系。
- `--primary-leads` 指定主要讨论时效，`--exploratory-leads` 指定探索性时效，
  其他报告时效只作个例参考；两组必须互不重叠且包含在 `--leads` 中。
  未指定主要时效则按路径样本阈值选择（全部少样本时仅作方向性观察），
  正式报告应预先明确窗口，不能按结果优劣挑选。
- 少样本标记默认 `N_case < 10` 或 `N_storm < 3`，可用 `--small-cases` /
  `--small-storms` 改；它只是披露门槛，不是显著性判断。
- 输出目录必须不存在或为空，不能在输入目录内；不覆盖已有报告、不改写 CSV。
- 依赖已有 `viz` 依赖组中的 matplotlib；缺依赖时显式报错。

输出：`REPORT.md`（六章）、`summary.json`（原始精度统计、逐文件元数据/字段映射/
SHA256、配对键、排除计数、配置与依赖版本）、`statistics.csv`、`differences.csv`、
`track.svg`、每个分量/强度指标各一张的 `mae_<指标>.svg` 与 `me_<指标>.svg`，
以及 `track_plots/tc<编号>_<名称>.svg` 逐台风路径图。没有有效共同分量/强度数据时
不生成占位图片，报告注明不比较的指标；没有有效共同路径数据则报错。
`mean` 列在路径指标上是平均距离，其他指标上是 MAE，`statistic` 列明确区分；
`me` 是有符号均值。区间缺失时 JSON 为 `null`、CSV 为空、报告为“未计算”。

台风报告统一走 `render_report.py`（要求标准产物目录带 `manifest.json`）：

```bash
python skills/evaluation-reporting/scripts/render_report.py \
  evaluation_results/weather_typhoon_ens_fgvp --model FGVP \
  --compare "FuXi=evaluation_results/weather_typhoon_ens_fuxi" \
  --result reports/typhoon_ens_comparison
```

这里 `--compare` 是产物目录（TS 分支仍是宽表 CSV）；高级统计选项
（`--leads` / `--fixed-leads` / `--bootstrap` 等）也由 `render_report.py` 直接提供。
当前自动报告输出为 `REPORT.md`，报告章节与模板对齐，
真实数字和图表由同一统计对象生成，不把 Markdown 骨架当模板引擎。

正文以用户提供的完成报告为质量参考，采用“模型 × 时效”横表，
逐指标在图表附近列主模型配对差值与小结；ME 表保留区间。
完整模型对/时效/原始精度统计与诊断元数据留在 CSV/JSON，不在正文堆长表或 JSON。
差值展示统一为“比较模型 − 主模型”，负值表示比较模型误差更低，
而机器可读差值仍保持 `first - second` 并带模型名。
结论区分均值下降、配对区间不跨 0 的信号、单模型系统偏差和固定样本观察，
不能以均值排名替代显著性检验。图的模型顺序决定颜色，各图保持一致，
浅灰/灰色区分探索时效与个例参考时效。

例如新旧版本比较：

```bash
python skills/evaluation-reporting/scripts/render_report.py \
  /data/new_typhoon --model fgvp_new \
  --compare "FGVP ENS=/data/old_typhoon" \
  --primary-model fgvp_new --previous-model "FGVP ENS" \
  --leads 24 48 72 96 120 144 168 192 216 \
  --primary-leads 24 48 72 96 120 --exploratory-leads 144 \
  --fixed-leads 24 48 72 96 120 --result reports/typhoon_version_comparison
```

## 配套资源

- 计算前阅读 [统计口径与数据检查](./typhoon-verification.md)。
- 确定性与集合平均共用 [weather_typhoon_single.md](../assets/templates/weather_typhoon_single.md)：
  六章结构相同，集合版只多 §2 的成员规模表、§3.1 的集合口径句和 §5 的限制句，不另设模板。
- 删除不适用的部分并填入真实结果；没有分量/强度样本或最大风速字段时按模板的条件块收口。

当前标准产物包括 `manifest.json`、`scores.csv`、`typhoon/tc<编号>_<起报>.csv`
和对应 `_meta.json`。逐时效报告必须读场次曲线，不能从 `scores.csv`
的跨时效均值还原；场次表与拼接总表不能重复计入样本。
中心诊断参数、种子、观测时区和成员完成信息按实际元数据披露；
缺少成员明细时不计算离散度，不强制输出雷达图；逐台风路径图只依赖场次 CSV 的位置列，按模板输出。
下列严格配对与统计规则只针对台风分支，不改变现有 TS/RMSE 报告口径。

## 1. 确定输入与范围

先从用户请求和工作区提取：

- 数据根目录、模型及显示名、确定性/集合预报类型。
- 台风、起报时间范围、检验时效及观测来源。
- 输出目录、报告名称、是否允许覆盖已有产物。
- 是否需要轨迹个例、偏差分析、置信区间或固定样本敏感性分析。

缺少关键输入或存在多种合理的统计定义时，逐项询问用户。可先只读检查数据，但不能凭模型名猜预报类型，不能猜时区、风速单位、观测基准或集合平均算法。

默认采用起报个例等权、逐指标逐时效严格配对的模型比较。已有项目若采用不同口径，显式记录并确认是否保留；不要悄悄改变历史口径。探索性全可用样本统计可另外提供，但不能和配对结果混用。

## 2. 勘察数据与既有工具

只读检查代表性 CSV、元数据、现有报告和脚本，确认：

1. 数据布局、字段映射、时间和单位；起报个例标识能否唯一构造。
2. 集合数据是成员明细、集合平均位置结果，还是成员平均误差。
3. 现有脚本的真实 CLI、输入输出路径、硬编码模型/时效、缺失值处理、样本筛选、依赖及写入行为。
4. 原始数据和现有统计结果是否一致，元数据是否有陈旧模型标签。

用户提供的 `typhoon_report/` 目录保留旧脚本与完成报告作为参考，不直接执行：
旧脚本绑定模型、路径和时效，部分重算脚本会改写输入。
通用实现沿用其逐指标配对、台风分组 bootstrap 和图形布局，
不复制特定批次的数字或结论。参考工具包括：

- `plot_track_error.py`：确定性路径误差及其他指标 MAE。
- `plot_ens_track_error.py`、`plot_ens_error_metrics.py`：集合路径误差、MAE/ME。
- `summarize_ens_errors.py`：集合配对统计和报告；有些实现仅在特定模型选项下支持报告输出。
- `recompute_ens_track_error.py`：可能原地改写集合 CSV，不能当成只读报告步骤。
- `plot_typhoon_tracks.py`、`plot_ens_typhoon_tracks.py`：可选轨迹个例图，可能依赖 Basemap。

读取脚本或确认安全的帮助入口后才能给出实际运行命令。不要把无参数、绑定脚本目录的程序描述成支持任意数据目录；也不要盲目执行脚本来探测其行为。

当前入口要求场次元数据完整（`tcid`、`init_utc`、`forecast_type`、`tz_shift_h`），
逐行 `valid_bjt` 必须与起报加时效/时区一致。当前集合数据需明确标注方案 B 字段；
历史 `*_mean_*` 字段需方案 B 声明，或用成员核验确认实际数值。
核验模式允许有效时间/观测从成员明细取得，但不靠模型名猜测类型或算法。
重复键、负路径距离、非数值内容、观测冲突与成员数冲突明确报错；
空值/非有限值计数并排除，位置缺测或越界排除路径类指标，不把缺测填零。

## 3. 校验并构造比较样本

按照参考文档建立规范化记录：

`model × storm_id × init_time_utc × lead_h × metric`

- 同时统计模型可用起报个例数、台风数和成员数分布。
- 对每个指标、每个时效，取所有待比较模型都有有限数值的共同键。
- 每个指标独立筛选；不同指标不能共享未经校验的样本数。
- 单列空值、非有限值、无效坐标、重复键、时间错位及缺失观测的数量与处理方法。
- 无有效样本时输出缺测说明，不能填零、插值或生成成功形态的占位统计。
- 输入冲突、单位不明、算法不明时停止相关计算并显式报告问题。

默认不改写原始数据。确需重算或转换时，在单独输出位置生成派生数据；原地修改必须先征得用户同意，并保留备份和变更记录。

## 4. 计算统计结果

先确定统计定义，再运行已核实的工具：

- 路径误差：逐时效平均大圆距离，单位 km。
- AT/CT、最低气压、最大风速：提供适用的 MAE 和 ME/Bias，注明单位与符号定义。
- 集合主结果：集合平均位置相对观测的误差；成员平均误差只能作为独立指标。
- 集合强度：必须确认是集合平均强度的有符号误差，而不是成员 MAE 的平均。
- 每个指标/时效同时输出配对起报个例数 `N_case` 和独立台风数 `N_storm`。
- 模型差值使用完全相同的配对键，声明减法方向；不要用不同样本的均值相减冒充配对比较。

需要不确定性时，采用台风分组 bootstrap，保留组内起报相关性；建议 2,000 次、固定随机种子、95% 百分位区间，并记录配置。比较模型时对配对差值计算区间，不通过两条均值曲线的区间是否重叠判断显著性。

机器可读汇总应包含：数据来源、模型/字段映射、时间范围与时效、单位、误差定义、配对和权重口径、排除计数，以及各指标/时效的模型值、样本数和适用的置信区间。无值用 JSON `null`，不要输出 NaN 或 Infinity。

## 5. 作图与撰写报告

最小交付：

1. 路径误差随时效变化图及数值表。
2. 适用的强度与 AT/CT MAE 图表；有符号误差另列 ME/Bias 图表。
3. 样本覆盖表、方法说明、结果解释、限制和复现命令。
4. 同口径机器可读汇总；数据量小也不能只交付图片。

图表使用一致的模型颜色和名称，显示单位、时效、样本口径；置信区间注明方法。为少样本时效作醒目标注。确定性报告按模板自动输出逐台风路径图（一个台风一张，`track_plots/`，依赖 basemap）；不自动添加概率检验或其他用户未要求的大规模输出。

按照模板撰写，所有数字取自实际统计结果。结论区分：

- 确定性性能或集合平均预报性能。
- 误差大小与有符号偏差。
- 逐时效结论与跨时效汇总的权重定义。
- 样本充分的结论与少样本个例观察。

没有一致有效的风速字段时，明确写“不比较最大风速”及原因，而非补造数值。只有集合平均结果时，不能声称已检验离散度、可靠性、CRPS 或概率技巧。

## 6. 验收与交付

- 抽查至少一个起报个例，核对有效时间、单位、误差符号和关键计算；有成员数据时抽查集合定义。
- 核对配对键及每指标每时效样本数，确保所有模型比较确实使用相同样本。
- 验证 bootstrap 保留同一台风全部相关记录，差值使用配对重采样。
- 核对报告、表格、图片和机器可读结果的数字、单位、模型名与样本口径一致。
- 确认产物存在且非空、图片可正常读取、Markdown 相对链接可解析、JSON 为合法有限数值或 `null`。
- 核对输入未被意外修改，未未经许可覆盖现有报告。
- 记录实际运行命令、输入和输出位置、依赖版本、随机种子及任何适配。
- 在最终回复链接报告和主要产物，简述统计口径与限制。失败或不完整必须明确说明，不能宣称已成功生成。

## 开发验证

在具备项目依赖的评测环境运行：

```bash
python -m pytest tests/unit/test_typhoon_report.py \
  tests/unit/test_typhoon_metrics.py tests/unit/test_typhoon_io_writer.py
```

新增测试覆盖确定性/集合完整链路、逐指标配对、MAE/ME、固定样本、分组 bootstrap
重复抽样与模型间共享抽样、历史字段、缺测、输入冲突、输出保护和统一入口。
真实数据需另行运行上面的报告命令，检查图片与统计，不能用合成测试代替真实批次验收。

旧归档验证示例：

```bash
python skills/evaluation-reporting/scripts/render_report.py \
  /home/lenovo/workspace_1/tc/ens/fuxi --model "FuXi ENS" \
  --compare "FGVP ENS=/home/lenovo/workspace_1/tc/ens/fgvp" \
  --verify-legacy-members --leads 24 48 72 96 120 \
  --result reports/legacy_typhoon_ens
```

> 旧归档目录也必须带 `manifest.json`；统一入口不接受裸场次目录。
