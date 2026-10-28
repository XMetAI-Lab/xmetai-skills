# 评估指标词典

各指标的**定义、范围、方向、恒等式**。本文件只回答"这个数是什么"，
**不回答"这个数算好还是坏"**——判读门槛、分级、建议触发条件在别处：

| 想知道 | 去哪 |
|---|---|
| 降水分类检验（TS 系列）怎么判读 | `diagnostic-rules.md` |
| 连续场（RMSE / ACC / 活跃度）怎么判读 | `field-evaluation.md` |
| TS 系列的口径、产出、踩过的坑 | `precipitation-evaluation.md` |
| RMSE 批次家族（single / ens；wave 为 legacy）的口径与判读 | `rmse-batch-evaluation.md` |

**冲突时以代码为准。** 本文件是代码的散文镜像：分类指标的**计算**在评测流水线里，
本 skill 不重算，只在 `scripts/reporting/reports/ts_report.py` 做汇总与判读；
连续场指标逐条对应 `scripts/reporting/reports/field_report.py`（详见 `field-evaluation.md`）；
谱族（纬向谱 / 球谐带）在 `scripts/reporting/spectra/specialized.py`。
**改公式先改代码，再回来改这里。**

---

## 1. 分类指标（站点降水）

降水检验本质是**二分类**：预报是否 ≥ 阈值、观测是否 ≥ 阈值。

### 1.1 混淆矩阵

|  | 观测 ≥ 阈值 | 观测 < 阈值 |
|---|---|---|
| **预报 ≥ 阈值** | Hits (a) | False Alarms (b) |
| **预报 < 阈值** | Misses (c) | Correct Negatives (d) |

四个量在产物里就是 `hits` / `false_alarms` / `misses` / `n_pairs`（= a+b+c+d）。

### 1.2 TS（Threat Score，又名 CSI）

```
TS = a / (a + b + c)          范围 [0,1]，越高越好
```

去掉了"正确否定"（没下雨也没报）的影响——降水是**稀有事件**，d 极大，
算进去会让所有模型都看着很好。**代价是不能区分漏报和空报谁更严重**，
也不能反映频率偏差（BIAS=1 时 TS 仍可能很低）。

### 1.3 POD（命中率）

```
POD = a / (a + c)             范围 [0,1]，越高越好
```

实际发生的降水中被报出来的比例。别名：Hit Rate / Recall / Sensitivity。
只看漏报、不看空报——**全报可以让 POD=1**。

### 1.4 FAR（空报率）

```
FAR = b / (a + b)             范围 [0,1]，越低越好
```

所有"报了"的当中报错的比例。注意它**不是**"空报数 / 总样本数"，分母是预报事件数。
只看空报、不看漏报——**全不报可以让 FAR=0**。

### 1.5 漏报率

```
漏报率 = 1 − POD = c / (a + c)      范围 [0,1]，越低越好
```

产物里是独立一列（`漏报率`），缺失时由 `1 − POD` 推算并注明。

### 1.6 BIAS（频率偏差）

```
BIAS = (a + b) / (a + c)      范围 [0, +∞)，理想值 1
```

预报事件数 ÷ 观测事件数。**它不是平均误差**（不是预报减观测的平均），
只回答"报多了还是报少了"：>1 偏多、<1 偏少、=1 频率匹配。
**BIAS=1 不代表报得准**——每次都在错的地方报同样多的次数，BIAS 照样是 1。
分级门槛见 `diagnostic-rules.md` §3。

### 1.7 两个恒等式（判读时最常用）

```
TS = 1 / (1/POD + 1/(1−FAR) − 1)        TS 是 POD 与 (1−FAR) 的调和平均
POD = BIAS × (1 − FAR)                  三者的约束关系
```

推论：

- 想提高 TS 必须**同时**提高 POD、降低 FAR；只提高 POD（全报）会让 FAR 上升，TS 不一定动；
- POD 与 FAR 已知时 BIAS 就被定死了，**三个量不是独立的**——
  报告里同时列出它们时，出现 `POD = BIAS×(1−FAR)` 对不上，就是口径混用了。

### 1.8 阈值与窗口

所有分类指标都依赖**降水阈值**，阈值表又随**累积窗口**变：

| 窗口 | 常用阈值 (mm) | 说明 |
|---|---|---|
| 24h | 0.1 / 10 / 25 / 50 / 100（部分配置到 250） | 日降水，最常用 |
| 6h | 0.1 / 4 / 13 / 25 | 短时强降水；**与 24h 的阈值不是一回事** |

**不同窗口、不同阈值的结果不可直接横向比较。** 各模型的等级列表还可能不一样
（有的没有 ≥250），对比时按**共有等级**对齐。

---

## 2. 概率指标（集合，站点）

集合预报的价值在**分布**而不只是均值。这一节的口径由
`metrics/probabilistic.py` 计算，与第 1 节的站点配对口径**不是一回事**。

### 2.1 事件概率怎么来

对每个成员、每个站，**逐成员插值**到站点后按**超越式阈值**（`x ≥ 阈值`）判事件，
再对成员求比例，得到 `p ∈ [0,1]`。对照观测 `o ∈ {0,1}`。

> 与第 1 节的区别：第 1 节是**先把集合平均场插值到站**再判事件；
> 这里是**逐成员插值**再算概率。同一个集合，两条路的数不可互换。

### 2.2 BS / BS_ref / BSS

```
BS     = mean( (p − o)² )            范围 [0,1]，越低越好
BS_ref = mean( (p_气候 − o)² )        气候基准的 BS
BSS    = 1 − BS / BS_ref             >0 才算有正技巧，=0 与气候持平
```

`p_气候` 来自**外部的逐 6h 气候概率参考**（`--ref`，逐站 × 阈值，缺站跳过）。
两个必须注意的地方：

- **`BS_ref` 只有给了 `--ref` 的 6h 窗才有值**，其余窗口整列是 NaN，BSS 也就无从算起；
- 有 ref 时 **`BS` 会改用 ref 覆盖的那批站重算**，与 `BS_ref` 同批样本——
  所以"有 ref"和"没 ref"两次数出来的 BS 不可比。

### 2.3 AROC

ROC 曲线下的面积。

```
AROC 范围 [0.5, 1]，0.5 = 无技巧（等于瞎猜），1 = 完美区分
```

它只关心**排序**：报的概率高的地方，是不是真的更容易出现事件。
因此 AROC **对频率偏差不敏感**——一个系统性高报的模型，AROC 依然可以很高。
**AROC 高但 BSS 低**是常见组合，意思是"能分出轻重，但绝对概率不准"。

### 2.4 base_rate

该档事件的**出现频率**（气候概率量级）。它决定了这条评分线有多难：
`base_rate = 0.005`（≥25mm）时，一个永远报 0 的模型 BS 就已经很小了，
所以**高阈值档的 BSS 比 BS 更有信息量**。

### 2.5 汇总行（`grade = AVG`）

每个时效末尾有一行 `AVG`：对**可算的阈值档**做等权平均（该档无事件的跳过），
`n_used` 记参与平均的档数。**不是按样本数加权**，所以它只表示"跨量级的总体水平"，
不能替代逐档看。

---

## 3. 集合连续场指标

对应 `weather_rmse_ens_<模型>` 系列。这些量都以**评估产物目录**的形式落盘，
数值在长表 `scores.csv` 里：`crps` / `spread` / `spread_error_ratio` 三行
（`product_kind="ensemble"`），集合平均场的 `rmse` 则与确定性同名、
靠 `product_kind` 区分。产物形状与判读见 `rmse-batch-evaluation.md`。

### 3.1 CRPS（连续排序概率评分）

```
CRPS = ∫ (F(x) − H(x − o))² dx       越低越好，单位同被评变量
```

`F` 是集合的累积分布，`H` 是观测的阶跃函数。**它是 MAE 的概率版**：
集合只有一个成员时 CRPS 退化成 MAE。既惩罚偏差、也惩罚离散度不对，
所以它比 RMSE 更难"蒙"。

### 3.2 Spread（离散度）

集合成员的标准差。它**单独看没有意义**——离散度大不等于好，
必须和误差一起看（见 3.3）。

### 3.3 Spread / RMSE

```
比值 = 集合成员标准差 / 集合平均场的 RMSE     理想值 1
```

- **< 1：欠离散**（over-confident）——集合太窄，真值经常落在成员之外，
  概率预报会给出过分自信的 0/1；
- **> 1：过离散**（under-confident）——集合太宽，概率被稀释，失去了区分度。

这是判断"集合到底有没有用"的核心指标：**RMSE 好但比值远离 1，
说明这份集合只能当确定性预报用**。

### 3.4 集合平均场口径

`RMSE` / `ACC` / `FA` 在集合产品上评的都是**集合平均场**（ensemble mean），
不是单个成员、也不是逐成员评分后再平均。**这个口径下的数不能和"集合的
概率技巧"混为一谈**——平均场平滑掉了成员的离散信息（见 3.3）。

---

## 4. 确定性连续场指标

| `metric` | 方向 | 理想值 |
|---|---|---|
| `rmse` | 越小越好 | 无（所以不画参考线） |
| `acc` | 越大越好 | 1.0（本流程是 **uncentered** 口径，不是经典皮尔逊） |
| `activity_ratio` | 越接近 1 越好（**两侧都坏**） | 1.0 |
| `activity_bias` | 越接近 0 越好 | 0.0 |
| `activity_forecast` / `activity_observation` | 原始量，无方向 | 无 |
| `spectrum_power_ratio` | 越接近 1 越好 | 1.0（**全波数求和后**的总功率比） |

**单位随数据源走**（`z500` 可能是 `m²/s²` 也可能是 `m`），跨变量不可比大小。
逐条语义、`n_valid` 的坑、判读规则见 `field-evaluation.md`。

---

## 5. 谱指标

### 5.1 纬向 FFT 功率谱

对每个纬圈沿经度做一维 rFFT，得到波数 `k = 1..N`（`k=0` 的纬向平均已置零）
的功率谱，预报与实况各一条。**只保留功率、不含相位**——所以它能说明
"能量在各尺度上分布得对不对"，**不能**说明"位置对不对"。

实现：`scripts/reporting/spectra/specialized.py` 的 `zonal_spectrum()` 与
`ZonalSpectrum`（`max_wavenumber` 控制取到第几个波数）。
逐波数曲线不在长表里，走 `spectrum` writer 落到
`diagnostics/spectrum_<变量>.csv`（全体平均）与 `spectrum_by_init.csv`（逐起报）。

### 5.2 log-RMS 口径

```
谱 log-RMS×100 = 100 × mean_v sqrt( mean_k ( log10(pred_k / obs_k) )² )
```

取对数是为了让**各个波数等权**——不取的话高波数的大功率会淹没低波数。
×100 是把它变成"百分数量级"便于读。

> ⚠️ **底数在不同报告之间不一致，这是现行代码的真实状态**：
> `det_report.spectrum_rms` 用 **log10**，`ens_report` 与 `wave_report.log_rms_ln`
> 用 **ln**，两者差一个 ×2.3026 的因子。
> 两个渲染器都**在正文里自己声明了底数并互相点名**，所以单看一份报告不会错；
> **跨份报告比这个数之前必须先统一底数**。要不要把各报告对齐是待定决策
> （改了会让已有归档的数整体变样，所以没有擅自动）。

### 5.3 球谐带功率

把场展开成球谐系数，按**总阶数**分带汇总功率。与纬向 FFT 的区别：
球谐是**二维**展开，能同时刻画经向与纬向尺度；纬向 FFT 只沿经度做。

**实现：`metrics/specialized.py` 的 `spherical_band_power()` 与
`SphericalBands`，频带定义在 `DEFAULT_SPHERICAL_BANDS`。**
结果作为 `spherical_bands` 指标展开成三个量落在长表里：
`spherical_band_power_forecast` / `_observation` / `_ratio`，
靠 **`group` 列**区分频带（`1_4` / `5_20` …）。

**球谐带恒全球，这是刻意的**——在掩掉一部分纬度的子区域上做二维球谐展开，
得到的不是同一个物理量。**纬向谱同样恒全球**（实测 `spectrum_power_ratio`
的行 `region` 全为空，与球谐带一样）。

分带的只有**格点标量指标**：`rmse` / `acc` / `activity_*` / `crps` /
`spread` / `spread_error_ratio`——每一个 (起报, 时效) 写一行全球加每个带一行。
判据是「这个量能不能在子区域上重新定义」，谱类不能、标量能。

> ⚠️ **现存归档认不出频带**：长表契约是 **25 列**（含 `group`，见
> `output/table.py` 的 `SCORE_COLUMNS`），而修复前跑出来的产物只有 **23 列、没有 `group`**——
> 5 个频带的行混在一起无法区分，确定性报告 §4.5（集合 §4.8）会标「本批未出」。
> **重跑评测即可**，不是报告端的问题。

---

## 6. 统计检验

多模型比较用**逐日期配对**的 Wilcoxon 符号秩检验，再对同一指标族内的
多个模型对做 **Holm 多重校正**（手写在 `scripts/reporting/reports/det_report.py` 的 `_holm`，
不为它引入 statsmodels）。

**统计显著 ≠ 业务重要**：样本量足够大时，0.001 的差距也会显著。
报告里说"显著优于"必须同时给出**差值本身**。

---

## 参考文献

1. Wilks, D. S. (2011). *Statistical methods in the atmospheric sciences* (Vol. 100). Academic press.
2. Jolliffe, I. T., & Stephenson, D. B. (Eds.). (2003). *Forecast verification: a practitioner's guide in atmospheric science*. John Wiley & Sons.
3. Hersbach, H. (2000). Decomposition of the continuous ranked probability score for ensemble prediction systems. *Weather and Forecasting*, 15(5), 559–570.

---

**维护者**：沈哲文 (szw)
**最后更新**：2026-09-15
