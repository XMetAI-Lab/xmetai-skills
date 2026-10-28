# 出图规范：配色、图宽与字号

> **以代码为准**。色值、字号表和宽度档位都在
> `scripts/reporting/core/style.py`；本文是它的散文镜像，两者不一致时改代码。

## 1. 配色

四类色池各管一种编码任务，不要混用：

| 任务 | 入口 | 规则 |
|---|---|---|
| 模型身份 | `model_colors(names)` | 已知模型固定色；对照/旧版统一中性灰；未知模型按名字排序取空槽 |
| 降水等级（有序） | `grade_colors(grades)` | 从浅到深一条色带，不循环分类色 |
| 连续量 | `CMAP_SEQ` / `CMAP_DIV` / `CMAP_SKILL` | 单向用 `CMAP_SEQ`；正负差值用 `CMAP_DIV`（0 落在米白）；相对表现热力图用 `CMAP_SKILL`（蓝差、中灰、红好） |
| 中性色 | `NEUTRAL` | 观测、基准、参考线、网格、边框、文字都从这里取 |

- 模型色池取 **NPG（Nature Publishing Group）配色**：饱和度和明度比旧版莫兰迪
  色高一档，屏幕上更醒目；相邻色相仍拉得开，灰度打印可辨。
- 不再使用 `Set1`、`viridis`、`YlGnBu` 这类外部色板；`palette=` 只作为临时覆盖参数保留。
- 网格色本身已经很浅，**不要再乘 `alpha`**，否则在浅底上直接看不见。
- 新增模型时在 `MODEL_COLOR_MAP` 里加一行固定绑定，不要靠调用顺序碰运气。
- **坐标轴一律纯黑**：轴标题、刻度数字、刻度短线、spine 全部 `#000000`；
  spine 线宽 1.0 pt、刻度线宽 1.0 pt。不要再把刻度数字弱化成灰色。
- 局部放大框走 `ZOOM_HIGHLIGHT`：**灰色虚线**描边 1.8 pt + 极浅中性灰底
  （`face` 10% 不透明度），底色压在曲线下方，只做高亮。不要用琥珀/橙色：
  太抢眼，且会和 S2B 的橙线混淆。
- 图例边框走 `LEGEND_FRAME`：**只描边、不填底**（`facecolor="none"`），
  `#B8B8B8` 细边。填浅灰底会变成一块灰色方块/阴影，画面发闷。
  **所有图例统一用它**，不要不带框，也不要各图手写颜色。

## 2. 图宽档位

Nature 单栏 89 mm、双栏 183 mm。全仓只用下面五档（英寸）：

| 档位 | 宽度 | 用途 |
|---|---:|---|
| `single` | 3.5 | Nature 单栏 89 mm |
| `onehalf` | 4.7 | 1.5 栏 / 小面板 |
| `double` | 7.2 | Nature 双栏 183 mm；论文版式 |
| `wide` | 12.0 | 屏幕报告常规图 |
| `full` | 16.0 | 类别多、面板多的长图 |

所有 `figsize` 的宽度都要走 `snap_width(requested)`：

- 取**不小于** `requested` 的最小档位，宽度只升不降，现有版式不会被挤窄；
- 超过 16 in 的动态宽度（类别轴太长）保持原值，字号继续按实际宽度放大；
- 高度不设档位，由面板行数/内容决定。

```python
from reporting.core.style import snap_width

width = snap_width(12)
fig, ax = plt.subplots(figsize=(width, 6))
```

## 3. 字号表

基准是 **183 mm 双栏（7.2 in）**。任何文字不小于 **5 pt**（Nature 下限）。

| 角色 | 键 | 7.2 in 字号 | 管什么 |
|---|---|---:|---|
| 刻度 | `tick` | 10 | 坐标轴刻度、色标刻度 |
| 轴标签 | `label` | 12 | 横/纵轴标签、色标标题 |
| 子图标题 | `title` | 12（加粗） | 子图标题、变量名、模型名 |
| 图例 | `legend` | 10 | 图例条目与图例标题 |
| 注释 | `annot` | 8 | 图内注释、格内标数、等值线标签、脚注、inset 刻度 |
| 面板字母 | `panel` | 12（粗体） | `(a)` / `(b)` |
| 总标题 | `suptitle` | 12 | 整张图的总标题；只有一根轴的图，主标题也归这一档 |

宽度超过 7.2 in 时按 `scale = 宽 / 7.2` **等比放大**，四舍五入到整数 pt：

| 图宽 | tick | label / title | legend | annot | panel / suptitle |
|---:|---:|---:|---:|---:|---:|
| 3.5–7.2 in | 10 | 12 | 10 | 8 | 12 |
| 12 in | 17 | 20 | 17 | 13 | 20 |
| 16 in | 22 | 27 | 22 | 18 | 27 |

这样屏幕报告图缩到 183 mm 印刷时，视觉字号仍然符合 Nature；密图不用重排。

## 4. 绘图代码怎么写

```python
from reporting.core.style import font_sizes, snap_width

width = snap_width(requested_width)
sizes = font_sizes(width)
fig, ax = plt.subplots(figsize=(width, height))

ax.set_title("...", fontsize=sizes["title"])
ax.set_xlabel("...", fontsize=sizes["label"])
ax.set_ylabel("...", fontsize=sizes["label"])
ax.tick_params(labelsize=sizes["tick"])
ax.legend(fontsize=sizes["legend"])
ax.annotate("...", fontsize=sizes["annot"])
```

硬规则：

1. 绘图模块里**不写字面量字号**。`grep -n "fontsize="` 只应看到 `sizes[...]`，
   或由 `font_sizes` 解析出的变量（如 `plot_model_panels` 的 `tick_labelsize`）。
2. 色标：刻度用 `annot`，标题用 `label`。
3. 子图标题一律**加粗**。`plot_model_panels(panel_letters=True)` 把 `(a)` 与
   变量名排在同一行，整行取 `panel`/`title` 同档（12 pt）并加粗，不会一大一小。
4. 字体家族保持 **CJK 优先**（Noto Sans CJK / 思源黑体 / SimHei …）。不要把
   Arial 放到第一位：matplotlib 3.5 没有逐字回退，中文会掉进回退字体或豆腐块。
5. 字号只跟**图宽**走，不要按“这张图看起来挤”单独改某一个标签的大小。
6. 局部放大图（inset）的纵轴用 `annot` 档（8 pt），**隔一个刻度标一个数字**，
   刻度线不省略；放大框用 `ZOOM_HIGHLIGHT`（灰色虚线描边 1.8 pt + 10% 浅灰底），
   框到 inset 的引导线走网格色虚线、1.1 pt。
7. 图例用 `legend` 档（10 pt），**统一套 `LEGEND_FRAME`**：不填底色
   （`facecolor="none"`）、`#B8B8B8` 细边、frame 线宽 0.8 pt。只描边不填底，
   图例不会变成一块灰色方块；不要再在各图里手写图例框参数。子图有空格时图例
   就放进空格（5 面板两列排 → `(d)` 正下方、`(e)` 右侧），否则横向贴整图底部。
8. 例外：集合谱图（`ens_report` 图 7+k）画布 16 in，但字号固定取 183 mm 基准档
   （`font_sizes(BASE_WIDTH)`），与图 2–6 的绝对字号一致，不按图宽放大——否则
   在报告里它的字会明显比邻居大。其余图仍按图宽走。

## 5. 当前约定与例外

- **集合报告论文图**（`ens_report` 图 2–6）：两列、单面板 3.6×2.7 in，总宽 7.2 in，
  因此自动取 刻度 10 / 轴标签与标题 12 / 图例 10 / inset 刻度 8 pt；
  `dpi=2400` 暂保持不变。5 面板图的图例放在第 3 行右侧空格里，不占底部。
  图 7+k（谱图；k = 空间变量数 × 月份数）画布 16 in 但字号同样固定在这组基准值，图内文字全英文、面板标
  「(a) u200」加粗。
- **总体指标热力图**（`ens_report` 图 1）：`CMAP_SKILL` 发散色（深蓝 0 =
  最差 → 浅蓝 0.25 → 中灰 0.50 → 浅红 0.75 → 深红 1 = 最优），白色细线分格、
  不画外框和刻度短线、色标不写标题；该色是**列内**归一化的相对表现，各列量纲
  不同，跨列不可比——图注必须写明。归一化复用 `radar_scores`，与雷达图同一套
  规则。格内字色按实际填充色的亮度选黑/白，不能用固定阈值。
- **球谐带柱图**（`wave_report._grouped_bars`）：图例横排整图底部、套
  `LEGEND_FRAME`（只描边不填底），不压在柱子上方。
- **动态宽度**（43 个暴雨过程、多个降水等级）：宽度由类别数算出后走
  `snap_width`，超过 16 in 时保持原值；字号继续按实际宽度放大。
- **线宽**目前仍是各图自己的 `linewidth`，尚未纳入这套等比表；改线宽时按
  Nature 的 0.25–1 pt（缩放后）自行换算，不要顺手改动字号。
