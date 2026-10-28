# RMSE 批次家族：口径与产出

覆盖 `weather_rmse_single_<模型>` / `weather_rmse_ens_<模型>` 两条配置支线。
吃同一批产物的 `_wave` 谱检验补充报告现已并入前两份的球谐带小节
（单成员 §4.5 / 集合 §4.8），`render_wave_report.py` 只作为 legacy 入口保留。
**模型名在最后**，别写成 `weather_rmse_<模型>_single`——那是旧文档的写法。

**它们的产物形状就是标准评估产物目录**（`scores.csv` + `manifest.json` +
`diagnostics/`），与其它流程没有区别，`resolved_config.pipeline` 是
`weather_field_scores`。**形状不是它们单开入口的原因，口径才是**：

| | 吃几份产物 | 出什么 |
|---|---|---|
| `render_report.py` → `field_report.py` | **一份** | 单模型的连续场报告（RMSE / ACC / 活跃度 / 谱） |
| `scripts/render_det_report.py` → `det_report.py` | **N 份**（≥2） | N 个模型**横向对比**的报告（相对 RMSE、配对 Wilcoxon、名次） |

`render_report.py` 认得出这些产物（有 `manifest.json`、有
`resolved_config.pipeline`），它只是**不做多模型对比**。反过来，把 TS 的产物目录
丢给 `render_det_report.py` 会当场报错——它要 `scores.csv` 里
`metric="rmse"` 且 `product_kind="deterministic"` 的行。

**冲突时以代码为准。** 本文件是代码的散文镜像：

| 事实 | 权威来源 |
|---|---|
| 产物目录的形状 | 评测流水线的写盘代码（`scores.csv` / `manifest.json` / `diagnostics/`）；本 skill 只读，落盘格式定义见 `scripts/reporting/core/writer.py` |
| 老批次归档 ↔ 新长表的等价关系 | `scripts/reporting/reports/det_report.py` 的**模块 docstring**（有一张三列对照表） |
| 派生指标的定义 | 同上（那几个指标**没有生成脚本，是反推出来的**，docstring 里写明了） |
| 报告长什么样 | `assets/templates/weather_rmse_single.md` / `weather_rmse_ens.md` 两份骨架 |

改口径先改代码，再回来改这里。

---

## 0. 两条支线

| 支线 | config | 吃什么 | 骨架 | 渲染器 |
|---|---|---|---|---|
| 确定性单成员 | `weather_rmse_single_<模型>` | 评估产物目录（`metric="rmse"` + `product_kind="deterministic"`） | `weather_rmse_single.md`（§4.5 球谐带） | ✅ `scripts/reporting/reports/det_report.py` + `scripts/render_det_report.py` |
| 集合 | `weather_rmse_ens_<模型>` | 评估产物目录（多 `crps` / `spread` / `spread_error_ratio` 指标） | `weather_rmse_ens.md`（§4.8 球谐带） | ✅ `scripts/reporting/reports/ens_report.py` + `scripts/render_ens_report.py` |
| 谱检验补充（legacy） | 吃上面任一份产物 | 评估产物目录 + `spherical_bands` 指标 | —（已并入单成员 §4.5 / 集合 §4.8） | `scripts/reporting/reports/wave_report.py` + `scripts/render_wave_report.py` |

两条**各有各的入口，按支线选**：`_single` 用 `render_det_report.py`，
`_ens` 用 `render_ens_report.py`。`render_wave_report.py` 仍可跑，
但球谐带与谱分析已经写进单成员报告，它只作为 legacy 入口保留、没有独立骨架。
骨架是**格式契约**，两份章节口径互不相同——**拿错入口比报错危险**：
集合与单成员的产物目录形状**一模一样**，`render_det_report.py` 拿集合产物
不会报错（它挑 `product_kind="deterministic"` 的行，集合的 rmse 行恰好也有
确定性那一份），出来的却是 `_single` 的章节口径。

单模型的入口用法：

```bash
python skills/evaluation-reporting/scripts/render_det_report.py \
  --model FGVP=/path/to/weather_rmse_single_fgvp \
  --model FuXi=/path/to/weather_rmse_single_fuxi \
  --result report/det_2models
```

`--model NAME=目录` 可重复、**至少两个**（单模型对比没有意义），
`NAME` 是**报告里显示的模型名**，由调用方给、**不从目录名或
`manifest.run_id` 猜**——这个仓库的规矩是身份不靠目录名。
其余选项：`--result`（必给，报告输出目录）、`--title`、`--change`、`--archive`、
`--declared NAME=v1,v2,...`（对应骨架第 6.2 节的「声明变量数」，不给则该列记 `—`）、
`--spatial-csv FILE` + `--spatial-model NAME`（可选，给就出 §4.6 空间分布图，
不给则该节写「本批未出」）。`render_ens_report.py` 有同样两个开关；
集合版吃的 CSV 还必须带 spread 列（由 `extract_spatial_distribution_ens.py` 产出）。

---

## 1. 产物形状

### 1.1 评估产物目录

一个模型一个目录，由评测框架（`pipeline/runner.py` + `output/store.py`）直接落盘：

```
<model_root>/
  scores.csv                  统一长表，25 列，见 §1.2——报告唯一的取数来源
  manifest.json               运行记录：statuses / valid_times / regions /
                              n_results / metrics / resolved_config / artifacts
  diagnostics/
    spectrum_<变量>.csv       全体样本加权平均谱：wavenumber, pred_mean, obs_mean
    spectrum_by_init.csv      逐起报谱：variable, init_time, wavenumber, pred, obs
  .states/                    逐段状态缓存（pkl），resume 用，**报告不读**
```

**没有 `summary.csv` / `batch_meta.json` / `<YYYYMMDD>/` 了**，
那是中间层还在时的形状。逐日明细的等价物是 `scores.csv` 的 `init_time` 列。

**集合成员数是运行参数，产物里不记。** 长表与 `manifest.json` 都只记「算出来的数」，
`n_members` 这类运行参数一律读不到——`ens_report._unique_summary_value(archive, "n_members")`
找不到列，只能把「成员数」渲染成「—」。**这个数只能人工维护**，现行取值见 §4；
出报告时用 `render_ens_report.py --members N` 对所有模型一律填同一个数
（不给就还是「—」），不要指望从产物里读出来。

> ⚠️ **报告的「成员数」列写「—」不是缺数据**，是产物确实没记这个数。
> 要让它自动进报告，得在 reader 侧把成员数写进产物；在此之前，
> 报告里的「成员数」列与「各模型成员数是否一致」那句判断都是照 §4 的表读的。

> ⚠️ **`manifest.json` 里的 `artifacts` 存的是写盘时的绝对路径**
> （`/workspace/.../scores.csv`）。**产物目录被拷到别处后这些路径就失效了**，
> 别拿 `artifacts` 定位文件——一律用产物目录内的相对路径。

### 1.2 长表 `scores.csv`

一行一个 `(变量, 指标, 区域, 起报, 时效)` 组合，**现在 25 列**：

```
run_id, model_id, dataset_id, variable, level, metric, product_kind, aggregation,
weights_id, region, group, threshold, window_h, lead_h, init_time, valid_time,
sample_unit, storm, value, unit, status, n_valid, n_requested, metric_version,
protocol_id
```

> `group` 是**分组指标的组标签**（球谐带的 `1_4` 等、分类检验的阈值名），
> 非分组指标为空；`storm` 是台风编号，其余链路为空。
> **早于 `group` 修复跑出来的归档只有 23 列**——数一下表头列数就能认
> （`manifest.json` 的 `score_columns` 也记着）。
> **球谐带必须靠 `group` 区分频带**：没有它时整节走 `SPHERICAL_ABSENT`、
> 标「本批未出」，只能重跑评测补齐（见 §5 第 6 条）。

报告取数只看其中几列：

| 报告里的东西 | 长表里怎么认 |
|---|---|
| 分变量 RMSE | `metric="rmse"` **且** `product_kind="deterministic"`，按 `init_time` × `lead_h` 透视 |
| ACC | `metric="acc"` |
| FA ratio | `metric="activity_ratio"` |
| 纬向谱偏差 | `metric="spectrum_power_ratio"`（标量汇总，恒全球）；逐波数曲线在 `diagnostics/`。**指标名叫 `zonal_spectrum`，落盘的 `metric` 值却是 `spectrum_power_ratio`**，两者不是同一个字符串 |
| 起报日 | `init_time` 前 10 位（`2025-01-02T00:00:00.000000` → `20250102`） |
| 全球行 / 纬度带行 | `region` 为空 = 全球，非空 = 该带（见 §1.3） |

> ⚠️ **`metric="rmse"` 有两套**：确定性 RMSE（`product_kind="deterministic"`）
> 与集合 `spread_error` 展开出来的那个 rmse（`product_kind="ensemble"`）。
> **名字一样、含义不同**，取数必须先按 `product_kind` 过滤。
> 渲染器里这一步是 `det_report._deterministic()`，别绕过它。

**ACC 与 FA 需要气候态**（`reference_reader`），配置里没给就不会有这两类行——
报告里对应的表会整块消失，**这不是缺数据**。

**哪些变量配哪些指标，由配置的 `VAR_METRICS` 逐条写死，各变量可以不一样。**
所以「某个变量没有某个指标」**先去看配置，不要当成跑丢了**——长表里没有的行，
就是配置没让它算。现行的 `weather_rmse_ens_fuxi.py` 是这么排的（**有意如此，不是漏配**）：

| 变量 | rmse | spread_error | acc | activity | 谱 |
|---|---|---|---|---|---|
| `z500` | ✓ | ✓ | ✓ | ✓ | ✓（含 crps） |
| `msl` | ✓ | ✓ | ✓ | — | — |
| `u200` / `v200` / `ws200` | ✓ | ✓ | — | ✓ | ✓ |

- **ACC 只给 `z500` 和 `msl`**，`u200` / `v200` / `ws200` 不算 ACC。
- **`msl` 只算 `rmse` / `spread_error` / `acc`**，不做 FA、不做谱。

> 这两条**是口径选择，不是漏配**——2026-09-22 已跟维护者确认「当前就是想要的」。
> **不要再当成缺口提**。真要改，动的是 `VAR_METRICS`，不是渲染器。
- 附带一个容易误判的点：`smoke_tp.log` 里**气候态确实加载了 `u200` / `v200`**
  （`ws200` 不单独加载，按 u/v 合成）。所以上表的取舍**不是数据不够**——
  真要补是「改配置 + 重跑」，不涉及新输入。

`manifest.json` 的 `metrics` 是**所有变量指标的并集**（如
`["acc","activity","crps","rmse","spherical_bands","spread_error","zonal_spectrum"]`），
**看不出哪个变量配了哪个**——想知道逐变量的矩阵，直接数长表的 `variable` × `metric`。

### 1.3 全球行与纬度带行

配了 `options["regions"]` 时，**格点标量指标**的每一个 `(起报, 时效)` 会写一行
全球行（`region` 为空）加每个带一行。实拍的三带是：

```
tropics[-20, 20]      nh_extratropics[20, 90]      sh_extratropics[-90, -20]
```

`n_valid`（格点数）三种行各不相同：全球 **1038240**、两个中高纬带各 **404640**、
热带 **231840** —— **这个比例可以用来核对区划是否真的切开了**。

> 这三个数就是「纬圈数 × 经圈数」，边界**闭区间、两端带重叠**，所以三条带加起来
> 比全球行大（±20° 那两条纬圈被热带和中高纬各算了一次）。ERA5 0.25°（721×1440）：
> 全球 721 圈、中高纬各 281 圈（20°→90°）、热带 161 圈（−20°→20°）。
> **换网格或换边界，这三个数就变**——别拿它们当常数去比不同网格的归档。

> **哪些指标分带、哪些不分**：`rmse` / `acc` / `activity_*` / `crps` /
> `spread` / `spread_error_ratio` **分带**；**谱类恒全球**——
> `spectrum_power_ratio` 与 `spherical_band_power_*` 的行 `region` 全为空。
> 判据是「这个量能不能在子区域上重新定义」，谱类不能。
> 所以「附 L」那一块拿到的是这些标量指标的分带值，**谱没有分带版本**，
> 骨架里分纬度带那张谱表会标「本批未出」，这不是数据没跑到。

> ⚠️ **分带靠的是指标自己认 `valid_mask`，执行层一个格点都不裁**：
> `restrict_to_latitude_band` 只把带掩码叠进 `batch.valid_mask`，
> 「分带」这件事**完全落在指标实现里**。`metrics/ensemble.py` 的
> `crps` / `spread_error` 在 2026-09-22 之前整段没读这个掩码，于是
> **全球标量被原样复制进每一个纬度带行**——四行的 `crps` / `spread` /
> `spread_error_ratio` 与 `n_valid` 完全相同，只有 `rmse` 等确定性指标
> 真的分带。**该日期之前生成的集合归档都带这个指纹，不要拿它们的分带行
> 做带间比较**；判别方法：看同一批数据的四个 `region` 行里
> `crps` 是不是同一个数（或 `n_valid` 是不是都等于全球行）。

> ⚠️ **两类行混在一起 `pivot_table(aggfunc="mean")` 会静默把带间差异平均掉**，
> 得到一堆看着正常、其实无意义的数。报告正文一律先过 `_global()` 只留全球行；
> 「附 L 分纬度带」那一块反过来只留 `region` 非空的行。**两者永不合并。**

带名与边界**照搬配置**，报告不翻译、也不写死——`det_report.region_names()`
是从 `scores.csv` 的 `region` 列现读的，所以换个切法重跑，报告那边一个字都不用改
（`manifest["regions"]` 存的是带名 + 边界的显示形式，两者对不上时以长表为准）。

### 1.4 谱诊断表（`diagnostics/`）

逐波数的谱**不在长表里**（长表一行一个变量的总量），走 `spectrum` writer 落到
`diagnostics/`：

| 文件 | 形状 | 谁在用 |
|---|---|---|
| `spectrum_<变量>.csv` | `wavenumber, pred_mean, obs_mean`，全体样本加权平均 | `field_report` 的分波段那节 |
| `spectrum_by_init.csv` | `variable, init_time, wavenumber, pred, obs`，逐起报一条 | `det_report.load_spectrum()`，报告里按日期画的谱曲线都取自它 |

**没有 `spectrum_by_init.csv` = 配置里没挂 `spectrum` writer**，报告保留节位、
写明「本批未出」，不静默省略。

**谱曲线没有逐时效的版本**（长表里 `zonal_spectrum` 是标量，没有波数轴；
`spectrum_by_init` 也没有 lead 列），所以曾经那节「谱比随时效」永远只能是
「本批未出」。**2026-09-22 已把这一节从 RMSE 骨架和渲染器里删掉**
（留着纯是噪声，图号顺延），**不要再按这一节去找产物**——真要做逐 lead 的谱比，
得先让 `spectrum` writer 多写一列 lead，那是改评测侧的事。

---

## 2. 统一口径与派生指标

### 2.1 共同日期（与 `--own-dates`）

默认口径下所有统计（表与图）都用 N 个模型的**共同日期**，逐日结果按 lead 平均后
比较。因此各表的样本量一致、可直接横向比较。**跨批次比较必须用同一批起报日**——
"涨到几倍"这个说法只在当前这套输入下成立。

各模型覆盖时段差很多时（实测五模型 27 / 228 / 348 / 349 / 29 天），为了对齐会把长的
那几份砍到最短那份，这时用 `render_ens_report.py --own-dates`：每个模型按**自己的
全部起报日**统计，代价是**同表各列来自不同时段、位次里混着时段差异**，不是公平对比。
两档的差别只在下面这几处（渲染器会逐处点名）：

- §2 表多一列「**本次使用日期**」= 该模型真正进了统计的天数。默认口径下它等于
  「有效共同日期」，`--own-dates` 下等于「日期目录」——看这两列相不相等就知道
  这一份有没有被砍；
- 跨模型基线（相对 RMSE 的几何均值、分带基线）在**各列自己的样本**上重算；
- **只有第 6 节的配对检验永远退回共同日期**：日期不对齐的样本做不了 Wilcoxon，
  所以那一节是全报告唯一一份公平对比，也是唯一能读"显著"的地方；
- 共同日期仍必须 ≥2 天，否则直接报错（报错里点名各模型的日期跨度与两两交集）。

`Bundle.dates_by_model` 是这条口径的落点：**凡是要给单个模型算平均的地方读它，
不要读 `Bundle.dates`**（后者永远是共同日期，配对检验与 §2 那一行用）。

**哪些入口有这个开关**：`render_ens_report.py` 与 `render_wave_report.py`
有 `--own-dates`；**`render_det_report.py`（`_single`）还没有**——底层
`det_report.analyze` 已经支持，只是那个 CLI 没把参数接出来，要用得先补一行
（`build_det_report` 的签名也要跟着透传）。legacy 的 `_wave` 入口谱是数据主体，各模型取
不同时段时曲线高低里混着季节与过程差异，渲染器会在 §1、§2、§4.6 与附录各点
一次名；它的配对检验同样永远退回共同日期。

### 2.2 派生指标

| 指标 | 定义 | 方向 |
|---|---|---|
| **相对 RMSE** | 逐变量 `100 × v_m / geomean_模型(v)`，再对**共享变量**等权平均 | 越低越好，**100 = 几何均值基线** |
| **ACC (%)** | 共同日期 × 全 lead 的 ACC 变量平均 | 越高越好 |
| **FA 偏差 (pp)** | `100 × mean(\|ratio − 1\|)`，对全部 FA 变量、全 lead 平均 | 越低越好 |
| **频谱对数 RMS×100** | `100 × mean_v sqrt(mean_k (log10(pred_k / obs_k))²)`，纬向谱，log10 | 越低越好 |
| **球谐谱 RMS×100** | `100 × RMS[ln(pred/obs)]`，对球谐总阶数频带池化，**ln 不是 log10** | 越低越好 |
| **平均排名分** | 上述各指标上名次的平均 | 越低越好 |

**纬度加权**：格点评估的方差/协方差一律 cos(lat) 加权（`transforms/regrid.compute_latitude_weights`），
不是简单格点平均；站点协议按 `options.weights` 决定。

**确定性 vs 集合**：长表里 `metric="rmse"` 有两套——`product_kind=deterministic`
是确定性 RMSE，`product_kind=ensemble` 是 `spread_error` 展开出来的 rmse。
名字一样，取数必须先按 `product_kind` 过滤。

**全球行**：同一 (起报, 时效, 指标) 会同时写全球行与 `config_options.regions`
的区域行，**只有 `region` 为空的那行是全球**，报告正文一律取全球行。

> **相对 RMSE 的 100 是"这几个模型的几何均值"**，不是某个绝对标准。
> 换一批模型参赛，同一个模型的分就变了——**它只在本次对比内部有意义**。
> 基线由 `det_report.py` 的 `BASELINE = 100.0` 钉住。

### 2.3 时效分段

```
LEAD_BANDS = ((0,120], (120,240], (240,360)]   单位 h，左开右闭
```

**空段不输出**，段名按段内实际的 lead 现写——所以模型时效范围不同时，
表的行数可能不一样。

### 2.4 显著性

逐日期配对的 **Wilcoxon 符号秩检验** + **Holm 多重校正**（手写在 `_holm`，
五行，不为它引入 statsmodels）。**统计显著 ≠ 业务重要**：样本量够大时
0.001 的差距也会显著，报告里必须同时给出差值本身。

### 2.5 FA 这个缩写的歧义

这里的 **FA = Forecast Activity**（预报活跃度，距平加权标准差），
**不是**降水二分类里的 False Alarm。两份报告都用这个缩写，所以
`weather_rmse_*.md` 的骨架里都有一段专门声明，**不许省**。
且 FA 统一用**全程**（首–末时效）统计、不做短中长分段——分段只给 RMSE。

---

## 3. 判读要点

### 3.1 相对 RMSE 看"倍数"而不是"差值"

`100` 是几何均值基线：`89.3` 表示比几何均值好约 11%，`117.6` 表示差约 18%。
**跨变量比倍数是可以的**（比值无量纲），比原始 RMSE 的差值是安全的——
原始 RMSE 的单位随数据源走（`z500` 可能是 `m²/s²` 也可能是 `m`），
**不同单位的 RMSE 不能画在同一根纵轴上比大小**。

### 3.2 ACC 饱和要单独提示

`det_report.py` 里有 `SATURATED_ACC = 99.0`：首时效 ACC 高于 99% 就算"饱和"，
写进 6.3 的风险提示。ACC 到 99% 以上时，**模型之间的差异已经淹没在噪声里**，
这时候比 ACC 的名次没有意义，要看 RMSE 和谱。

### 3.3 FA 偏差是"偏离 1 的幅度"

`ratio = pred 活跃度 / obs 活跃度`，理想值 1。报告用的是
`mean(|ratio − 1|)`——**取了绝对值，方向就丢了**：两个模型一个偏平滑（0.8）
一个偏噪（1.2），FA 偏差都是 0.2，但含义完全相反。
**要方向必须回去看长表里 `metric="activity_ratio"` 的行**
（`activity_bias` / `activity_forecast` / `activity_observation` 三行也在，
分别是分子分母的原始量）。

### 3.4 谱指标

`频谱对数 RMS×100` 是对**全波数**算的，高波数的近零观测功率会主导结果。
要定位"从第几个波数开始失真"必须看谱曲线
（`diagnostics/spectrum_by_init.csv` 取单日起报，或 `spectrum_<变量>.csv` 看平均谱），
一个汇总数说不出来。

> ⚠️ **底数在不同报告之间不一致，这是现行代码的真实状态、不是文档过时**：
> `det_report.py` 用 **log10**（`:810`），`ens_report.py`（`:282`）与
> `wave_report.py`（`:253`、`:389`）用 **ln**，两者差 ×2.3026。
> 两个渲染器都**在正文里自己声明了底数并互相点名**，所以单看一份报告不会错；
> **跨份报告比这个数之前必须先统一底数**。要不要把各报告对齐是一个
> 待定决策（改了会让已有归档的数整体变样，所以没有擅自动）。

---

## 4. 口径参数（跨批次比较必须一致）

现在这些参数都写在**评测配置**（`examples/evaluation/configs/weather_rmse_*.py`）里，
换一个值就是换一个口径，**两批结果的数不可横向比**：

| 配置项 | 作用 | 坑 |
|---|---|---|
| `options["sample_by"]` | 采样口径：`"init_lead"`（每起报报满整段时效）或 `"valid_time"`（每有效时刻一个样本、最新起报获胜） | **这是当年 `pred_shift` 那个坑的现代表达**。逐日起报配 15 天时效**必须**用 `init_lead`，否则每个起报只剩头 4 个时效 |
| `forecast_reader["step_hours"]` | 时效轴步长（小时） | 等于当年的 `lead_step`。写错整条时效轴就整体缩放 |
| `reference_reader`（`type: daily_climatology` / `climatology`） | 气候态 | **ACC 与 FA 必需**；没配就没有这两类行（见 §1.2） |
| `reference_reader["smooth_days"]` | 日序气候态的环形平滑天数（实测 15 = ±7.5 天） | 等于当年的 `climo_window`。**不同平滑窗的气候态不可比** |
| `metric_options["acc"]["centered"]` | `True` = 经典皮尔逊（距平去均值）；`False` = **uncentered**（FDP / WeatherBench2） | 实测配置用的是 `False`。**两种口径的 ACC 不可比** |
| `metric_options["zonal_spectrum"]["max_wavenumber"]` | 谱取到第几个波数（720 = 0.25° Nyquist） | 写小了只截曲线前段。等于当年的 `spec_lead_range` 那一类"谱口径" |
| `metric_options`（每个指标各自的 `variables` 列表） | 算哪些变量 × 哪些指标 | 只有 `rmse` 时，ACC / 谱**整块没有**；漏点名的变量会拿到整批数据、`single_variable` 直接报错（**框架故意的，防静默出假数**） |
| `options["regions"]` | 分纬度带区划 | 改了区划，`region` 列的值就变了——**带间不可跨批次比**（见 §1.3） |
| **集合成员数** | 每个起报的成员个数 | **产物里读不到**（见 §1.1），只能照下表人工维护。**CRPS 与离散度对成员数敏感**，成员数不同的集合之间比这两个指标要格外小心 |

现行集合的成员数（改动 reader 配置后**必须回来改这张表**，产物不会自己更新）：

| 集合 | `model_id` | reader | 每例成员数 |
|---|---|---|---|
| FuXi 集合 | `fuxi_ens` | `fuxi_ens_phys` | **51**（1 控制 + 50 扰动） |

> **`pred_shift` / `spec_lead_range` 这两个老参数在新框架里没有对应物**，
> 对齐由框架按 `sample_by` 自己算。旧文档里"pred 不含 0 时次取 1"那条经验
> 是针对老数据的，**不要往新配置里搬**。

---

## 5. 坑

1. **全球行和纬度带行不能混着平均**——`pivot_table(aggfunc="mean")` 会把带间
   差异**静默**抹平，出一堆看着正常的假数（见 §1.3）。正文只留全球行，
   「附 L」只留带行。
2. **`--model` 少于两个要报错**，不要拿单模型的产物去套多模型骨架
   （骨架里到处是「{模型数}模型」「四条曲线」，一个模型填不出来）。
3. **`metric="rmse"` 有两套口径**，取数必须先按 `product_kind` 过滤
   （见 §1.2）。忘记过滤 = 把集合的 `spread_error` 展开项混进确定性 RMSE。
4. **模型名由调用方给**（`--model NAME=目录`），**不从目录名或
   `manifest.run_id` 猜**——这个仓库的规矩是身份不靠目录名。
   `manifest["run_id"]` 是**配置名**（如 `weather_rmse_single_fuxi`），
   与目录名、与报告里的模型名**三者可以全都不一样**；实测把 FuXi 的产物
   拷进一个叫 `..._fgvp` 的目录，`run_id` 照样是 `..._fuxi`。
5. **谱变量默认优先 `z500`**（`PREFERRED_VARIABLE`），缺了才退到字典序第一个。
   换变量会让频谱那条线整体变样，**别拿不同谱变量的数比大小**。
6. ⚠️ **球谐带功率行在旧产物里认不出频带**。长表契约 25 列里有 `group`
   （球谐带的 `1_4` / `5_20` …，见 `output/table.py` 的 `SCORE_COLUMNS`）；
   但早期归档的 `scores.csv` 只有 **23 列、没有 `group`**。此时 5 个频带的行
   混在一起无法区分，确定性报告 §4.5 与集合报告 §4.8（以及 legacy
   `wave_report` 的对应小节）会走 `SPHERICAL_ABSENT`、标「本批未出」。
   **修法是重跑评测**（配置里带 `spherical_bands` 指标即可），不是改报告端。

---

## 6. 按需子集对比（口语请求 → 长表取数）

出报告时用户会临时点名一个切片，典型说法：

> 「热带 春季 rmse 对比」
> 「北半球夏季的 ACC 谁高」
> 「JJA 240h 以后的 CRPS」

**这类请求不需要重跑评测**——长表里每个维度都已经在列上了，取数是纯筛选加
按报告既有口径重算。**渲染器不认这类请求**（两份骨架的「附 S」现在固定输出
「本批未出」），要**智能体自己取数、自己把段落补进报告**。本节就是那套工序。

### 6.1 口语词 → 长表列

| 用户说的 | 落到哪一列 | 取值怎么写 |
|---|---|---|
| 全球 / 热带 / 南北半球中高纬 | `region` | 全球 = **空串 `""`，不是 `"global"`**；带名照配置 `options["regions"]`，本仓库默认 `tropics` / `nh_extratropics` / `sh_extratropics`。报告里显示的 `tropics [-20, 20]` 是渲染时拼的界——**长表里只有裸名**，边界要去 `manifest["regions"]`（顶层列表，形如 `["tropics[-20, 20]", …]`）或配置 `options["regions"]` 取，别去 `resolved_config` 里翻 |
| 春季 / MAM、夏季 / JJA… | **没有这一列**，从 `init_time` 推 | 见 §6.2 |
| rmse / acc / crps / spread… | `metric` | ⚠️ `rmse` **有两套口径**，必须同时筛 `product_kind`（§1.2、§5.3）：场 RMSE 是 `deterministic`，集合离散度那个展开项是 `ensemble` |
| z500 / msl / ws200… | `variable` | 大小写敏感，写错就是空集 |
| 240h 以后 / 短时效段 | `lead_h` | **浮点字符串**（`'6.0'`、`'240.0'`），别按整型比 |
| 哪个模型 | 归档目录 | 每个模型一份 `scores.csv`，**一次读一个目录**；模型名由调用方给，不从目录名或 `run_id` 猜（§5.4） |

### 6.2 季节：按**起报时刻**切（已定口径，别再改）

季节列长表里没有，从 `init_time` 的日期部分取月份推：

| 季节 | 起报日月份 |
|---|---|
| MAM | 3、4、5 |
| JJA | 6、7、8 |
| SON | 9、10、11 |
| DJF | **12、次年 1、次年 2** |

**两条 2026-09-22 跟维护者确认过的口径：**

1. **按起报时刻切，不按 `valid_time`。** 这套报告的契约是「同一天 = 同一个
   `init_date`」（`_common_dates_or_raise`、逐日配对检验、逐日文件都建在这上面），
   一组 `init_date` 必须干净地属于一个季节。按 `valid_time` 切会让同一个
   `init_date` 的 60 个 lead 散进不同季节，逐日配对的样本就不再属于同一个季节，
   季节边界上的样本量也偏少。
   → 副作用要知道：一次 15 天预报的**有效时刻会跨出起报季节**。所以季节口径是
   「**这批起报的**预报」，不是「春季的天气」，结论里要写清是哪一个。
2. **DJF 按气象冬季归组**：`2025-01`、`2025-02` 属于 DJF(2024/25)，`2025-12`
   属于 DJF(2025/26)。**于是不满整年的批次，两个冬季各只有一截**——实测归档
   `20250102–20251215`：DJF(2024/25) 只有 1 月 2 日–2 月 28 日（缺 2024-12），
   DJF(2025/26) 只有 12 月 1–15 日（缺 2026-01/02）。**注意不是「没有 12 月」**，
   12 月的数在表里，只是它不属于前一个冬季。
   所以报 DJF 时必须说清报的是**哪一个冬季**，或者干脆只出 MAM / JJA / SON；
   **不要**把同一自然年的 12 月硬拼进去充数（那不是真冬季，中间断了 9 个月）。

### 6.3 三条不能破的口径

1. **`region` 行与全球行永不混**（§1.3）。要热带就只取 `region == "tropics"`，
   不要把全球行掺进来平均——`pivot_table(aggfunc="mean")` 会**静默**把带间差异抹平。
2. **相对基线在子集内重算。** 报告里的 `100` 是**本次参赛模型的几何均值**，不是常数。
   切片之后模型的集合、样本的日期都变了，沿用全球的 `100` 就是在比两个基线。
   要么在同一子集内重算几何均值，要么直接报**绝对 RMSE + 比值**并把基线写在表下。
3. **先按 lead 平均、再跨日平均**，与正文同序。没有缺格时两种顺序结果相同；
   一旦有缺格（某天缺某个 lead），顺序反了会让「日期少而 lead 全」的日子权重过高。

另外：配对检验仍然**按 `init_date` 分组**（`paired_tests`），样本是子集内的共同日期
——**注意 `--own-dates` 下正文各表已不是共同日期，切片时更要盯住这一点**：
季节子集要么在共同日期上切（与第 6 节同源），要么各模型各切各的、并在结论里
写明各季各模型的天数不同，别印一个"共 N 天"。切片会把日期数从 348 砍到几十天，
**必须把天数与 `n_valid` 一起写进结论**；天数掉到小样本阈值以下就照
`small_sample` 的写法点名「方向性高于显著性」。

### 6.4 取数片段（照抄后改筛选条件）

**切之前先验归档能不能切。** 分带不生效的归档（2026-09-22 修复之前跑的那一批，
见 §1.3）切出来**不报错、也不为空**——它把全球的数原样交给你，挂上热带的表头。
识别方法只有一条，看 `n_valid`：

```python
# 同一天数、同一变量、不同纬度带的 n_valid 若完全相同 → 分带没生效，别切了
for reg in ["tropics", "nh_extratropics", "sh_extratropics"]:
    f = metric_rows(root, "rmse", product_kind="deterministic", region=reg)
    f = f[f["variable"] == "z500"]
    print(reg, int(f["n_valid"].astype(int).sum()), f["value"].astype(float).mean())
```

实测（`weather_rmse_ens_fuxi`）：

| `product_kind` | tropics | nh_extratropics | sh_extratropics | 结论 |
|---|---|---|---|---|
| `ensemble` | 95,518,080 | 95,518,080 | 95,518,080 | **三带全同 → 分带没生效** |
| `deterministic` | 21,329,280 | 37,226,880 | 37,226,880 | 正确 |

⚠️ **集合口径和确定性口径要分别验**，不能验了一个就当整份归档都行——上面这份
归档正是「确定性对、集合错」。撞上不生效的那一档，只有两条路：**重跑归档**，
或者**改用生效的那一档并把这个换口径的事实在表下写明**（报告里必须写，不能默认
读者知道）。别默不作声地切。

> `metric_rows` 的 `region=""` **取不到全球行**（它内部读 CSV 时把空串吞成 `NaN`），
> 返回 0 行。要全球口径就**把 `region` 整个省掉**。§6.4 上面那段用 pandas 自己
> 读表，所以 `keep_default_na=False` + `region == ""` 是有效的——**两条路的写法
> 不一样，别混**。

```python
import pandas as pd

root = r"D:\xmetai-evaluation-result\weather_rmse_ens_fuxi"
# keep_default_na=False 是**必须的**：否则空 region 会被读成 NaN，
# 于是 df["region"] == "" 一行都选不出来（全球口径静默变空集）
df = pd.read_csv(root + r"\scores.csv", dtype=str, keep_default_na=False)

df["init_date"] = df["init_time"].str[:10]
SEASON = {3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
          9: "SON", 10: "SON", 11: "SON", 12: "DJF", 1: "DJF", 2: "DJF"}
df["season"] = pd.to_datetime(df["init_date"]).dt.month.map(SEASON)

sub = df[
    (df["region"] == "tropics")                  # 全球写 ""，不是 "global"
    & (df["season"] == "MAM")
    & (df["metric"] == "rmse")
    & (df["product_kind"] == "deterministic")    # rmse 两套口径，必须筛
    & (df["variable"] == "z500")
    & (df["status"] == "success")
].copy()

# dtype=str 读进来后**所有列都是字符串**，数值列要自己转：
# lead_h 不转则 unstack 按字典序排（'100.0' 排到 '6.0' 前面），
# value 不转则 groupby().mean() 直接抛 TypeError。
sub["lead_h"] = sub["lead_h"].astype(float)
sub["value"] = sub["value"].astype(float)

# 先逐 (日期, lead) 取值，再按 lead 平均、最后跨日平均——与正文同序
by_lead = (sub.groupby(["init_date", "lead_h"])["value"].mean()
              .unstack("lead_h").sort_index(axis=1))
per_date = by_lead.mean(axis=1)
print("天数", len(per_date), "样本", int(sub["n_valid"].astype(int).sum()),
      "综合RMSE", per_date.mean())
```

每个模型跑一遍上面这段（换 `root`），把结果并起来才能横向比。

**跑完先自检**（实测归档 `20250102–20251215` 的已知值，对得上才算筛对了）：

| 检查 | 实测 | 说明 |
|---|---|---|
| 热带样本量 | 92 天 × 60 lead × **231840** = 1279756800 | 231840 是热带格点数（§1.3） |
| MAM / JJA / SON 天数 | 92 / 92 / 91 | 自然月数，对得上就说明季节映射没错 |
| **DJF 天数** | **73** | ⚠️ 见下 |
| lead 列数 | 60（6.0 → 360.0） | 若不是数值序，就是 `lead_h` 忘了转 float |
| 热带 MAM z500 综合 RMSE | 74.02（全球同口径 445.45） | 热带 z500 方差小，比全球低一个量级是正常的 |

⚠️ **DJF 的 73 天是个陷阱**：它 = 58（DJF(2024/25) 的 1/2–2/28）+ 15
（DJF(2025/26) 的 12/1–12/15），**两截不相连**。天数和 MAM/JJA/SON 看起来
是一个量级（73 vs 92/92/91），**光看天数发现不了**，很容易当成一个正常季节
写进结论。报 DJF 前必须先看它由哪几个冬季组成。

数量级不对就是筛选条件写错了（多半是 `region` 的 `NaN` 或 `product_kind` 漏筛）。

### 6.5 补进报告的形态

- **表**：与「附 L」同形——一行一个变量（或一个季节、一个带），各模型各占一列，
  末列 `Best`。基线若在子集内重算过，表下必须写明。
- **判读句**：一条，带具体数值 + **天数**；有缺季/缺带就点名。
- **节位**：
  - 只按季节切 → 落在 **`## 附 S 分季节结果（可选）`**，把那段「本批未出」的
    静态文案整段换成真结果；
  - 只按纬度带切 → 落在 **`## 附 L`**（那里本来就有数）；
  - **带 × 季节都要**（如「热带 春季」）→ 落 **附 S**，标题下第一句写明
    「本块是 `{带名} × {季节}` 的切片，不是全球口径」。
- **图**：默认**只补表与判读句**，不出图。骨架给 `season_summary.png` /
  `season_rmse_vs_lead.png` 留了 `图 S1`/`图 S2` 两个号，**没出图就别留号**——
  不指不存在的文件。
- **别做的比较**：这个切片的数和正文全球口径的数**不能并列比大小**，
  基线、样本日期、样本量三者都不一样。
- **口径若被迫换过，必须在表下写明**（见 §6.4）——归档分带不生效时改用另一档
  `product_kind`，是**这份报告**在替归档背锅，不写清楚下一手会拿它当集合口径用。
- ⚠️ **手改的 REPORT.md 是临时的。** 渲染器每次跑都整份重写 `REPORT.md`，
  手工补进「附 S」的这段**会在下次 `render_*_report.py` 时被冲掉**，退回那句
  「本批本批未出」的静态文案。所以要么每次重跑后重补，要么把切片做成渲染器的
  可选输入。**交给用户的报告里如果含手补内容，要当面说清这一点**——否则用户
  以为它跟正文一样是自动产出的。
- **显著优胜列在单归档上必然退化**：fuxi1 / 伏羲 喂同一份 `scores.csv` 时两侧
  逐位相同，配对检验只会给「无显著差异」。这不是筛选写错了，**要在表下点明**，
  否则读的人会以为真做了模型对比。

---

## 7. 不要做的事

1. 不要给相对 RMSE 的 `100` 编绝对含义（它是**本次参赛模型的几何均值**）；
2. 不要把不同单位的 RMSE 画在同一根纵轴上；
3. 不要用 FA 偏差的绝对值断言"偏平滑"或"偏噪"——它丢了方向；
4. 不要把全球行与纬度带行混着平均（见 §1.3）；
5. 不要用"结果里没有"反推"数据源里没有"（ACC/FA 缺席多半是没配气候态；
   谱曲线缺席多半是没挂 `spectrum` writer；球谐带认不出多半是旧产物没有 `group` 列）；
6. 不要把集合的产物喂给 `render_det_report.py`——**形状一样、读得进去**，
   出的却是 `_single` 的章节口径，**拿错模板比报错危险**；
7. 不要跨口径比较（不同起报集合、不同 `acc` 口径、不同气候态平滑窗、
   不同 `max_wavenumber`、不同 `regions` 区划的结果不可直接横向比）；
8. 不要跨份报告比「频谱对数 RMS」——各报告的底数不一致（见 §3.4）。

---

**维护者**：沈哲文 (szw)
**最后更新**：2026-09-22（§6.4 补「切之前先验归档能不能切」——拿 `n_valid` 跨带
比，实测 `weather_rmse_ens_fuxi` 集合口径三带全同、确定性口径正确，两个口径要
分别验；记下 `metric_rows(region="")` 取不到全球行；§6.5 补「手改 REPORT.md 会被
重跑冲掉」。同日早些时候新增 §6 按需子集对比：季节按**起报时刻**切、DJF 按
气象冬季归，两条口径当日与维护者确认；原 §6 顺延为 §7。此前 2026-09-21 按
`pipelines/evaluation/` 的实际产物形状重写 §1 与 §4；作废 `vfc/` 系列路径与
`--summarize-*` 开关）
