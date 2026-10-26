# xmetai-skills

面向 XMetAI 类气象模型工作的 agent skills 仓库：既包含建模侧工作流，也包含评测结果报告。

## Skills

| Skill | 入口 | 用途 |
|---|---|---|
| `xmetai-weather-modeling` | [`skills/xmetai-weather-modeling/SKILL.md`](skills/xmetai-weather-modeling/SKILL.md) | 工作区检查、LazyConfig 审查、Zarr/NetCDF/静态数据检查、模型 contract、tensor shape 调试、训练与评测**执行**、ONNX 导出、部署审查。 |
| `xmetai-datakit-skills` | [`skills/xmetai-datakit-skills/SKILL.md`](skills/xmetai-datakit-skills/SKILL.md) | 技能族入口：把族级请求或不确定归属的请求分诊到成员 skill。 |
| `reporting` | [`skills/xmetai-datakit-skills/reporting/SKILL.md`](skills/xmetai-datakit-skills/reporting/SKILL.md) | 消费已完成的评测产物（`manifest.json` + 结果表），产出图表与 Markdown 报告。**不重算指标**。 |

`SKILL.md` 是宿主唯一会加载的文件，其中的 `name` + `description` 是唯一的检索依据；
`agents/*.yaml` 是可选元数据，不影响选中。

## 安装

### 方式一：注册为 agent skill

**Codex** —— 把每个 skill 目录链接到 Codex 的 skills 目录：

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
git clone git@github.com:XMetAI-Lab/xmetai-skills.git
mkdir -p "$CODEX_HOME/skills"

ln -s "$(pwd)/xmetai-skills/skills/xmetai-weather-modeling" "$CODEX_HOME/skills/xmetai-weather-modeling"
ln -s "$(pwd)/xmetai-skills/skills/xmetai-datakit-skills" "$CODEX_HOME/skills/xmetai-datakit-skills"
ln -s "$(pwd)/xmetai-skills/skills/xmetai-datakit-skills/reporting" "$CODEX_HOME/skills/reporting"
```

**Claude Code** —— 把每个 skill 复制（或链接）到 `.claude/skills/<name>/`：

```bash
mkdir -p .claude/skills
cp -r skills/xmetai-weather-modeling .claude/skills/xmetai-weather-modeling
cp -r skills/xmetai-datakit-skills/reporting .claude/skills/reporting
```

宿主**不会递归发现**嵌套的 `SKILL.md`：只注册族容器不等于注册了 `reporting`。
希望哪个 skill 能被选中，就单独链接哪一层。

### 方式二：放在 `xmetai-core` 或 `xmetai-datakit` 旁边

项目内协作时，把本仓库作为 sibling checkout 或 submodule 放在目标仓库旁边：

```bash
git submodule add git@github.com:XMetAI-Lab/xmetai-skills.git xmetai-skills
git submodule update --init --recursive
```

然后让 agent 直接加载 `xmetai-skills/skills/<skill>/SKILL.md`。不要把 `xmetai-core`
或 `xmetai-datakit` vendor 到本仓库里；skill 检查的是用户当前正在工作的 active
workspace。

## 依赖

只有 reporting 有 Python 依赖；建模 skill 只用标准库。

```bash
python -m pip install -r requirements.txt
# 需要空间取数（zarr / netCDF4）与台风地图（basemap）时：
python -m pip install "./skills/xmetai-datakit-skills/reporting[all]"
```

`requirements.txt` 只指向 skill 的 `pyproject.toml`，版本与 extras 都维护在那里。
安装是可选的：reporting 的入口脚本会自己把 `scripts/` 插进 `sys.path`，复制目录
即可运行。

## 使用

### 建模侧工作流

让 agent 使用 `xmetai-weather-modeling` 处理：

- 创建或审查 LazyConfig。
- 检查 Zarr、NetCDF、静态数据与数据集 contract。
- 审查模型 contract，调试 tensor shape。
- 训练、评测/推理执行、ONNX 导出与部署检查。

常用只读辅助命令：

```bash
python skills/xmetai-weather-modeling/scripts/inspect_workspace.py /path/to/xmetai-core
python skills/xmetai-weather-modeling/scripts/check_config_contract.py /path/to/config.py
python skills/xmetai-weather-modeling/scripts/inspect_static_nc.py /path/to/static.nc
```

建模 skill 只覆盖**执行**评测任务与如实汇报跑过什么，不计算指标、不产出分析报告。

### 评测结果报告

评测产物已生成时，让 agent 使用 `reporting`：

```bash
python skills/xmetai-datakit-skills/reporting/scripts/render_report.py \
  outputs/results/weather_ts_single_fgvp --result report/fgvp --model FGVP
```

多模型对比（TS 系列）：

```bash
python skills/xmetai-datakit-skills/reporting/scripts/render_report.py \
  outputs/results/weather_ts_single_fgvp --result report/fgvp_vs_fuxi --model FGVP \
  --compare "FuXi=outputs/results/weather_ts_single_fuxi/diagnostics/categorical_wide.csv"
```

其他入口：

| 场景 | 命令 |
|---|---|
| 多模型 RMSE（单成员 / 集合） | `render_det_report.py` / `render_ens_report.py --model NAME=目录` |
| 暴雨过程个例 | `render_rainstorm_report.py --model NAME=目录 --process 202503,202521` |
| 挑暴雨过程 | `render_rainstorm_summary.py` → `summary_long.csv` |
| 空间分布取数（服务器） | `extract_spatial_distribution.py` / `extract_spatial_distribution_ens.py` |
| 空间分布出图 | `render_spatial_distribution.py --csv ... --model ...` |
| 台风统计 | `render_report.py` + `--leads` / `--fixed-leads` / `--bootstrap` |

输入契约（各能力必需文件、必需列、缺值处理）见
[`reporting/SKILL.md`](skills/xmetai-datakit-skills/reporting/SKILL.md)。

## 验证

在仓库根执行：

```bash
python -m pytest tests -q
```

- `tests/test_skill_structure.py`：检查建模 skill 的入口、references、agent 元数据，
  以及它的脚本能否编译。
- `tests/test_reporting.py`：把 `reporting` 复制到临时目录，在不安装的情况下跑通
  全部入口；缺少科学计算依赖时自动跳过。

CI 目前编译建模 skill 的脚本并运行测试；reporting 的测试只在装有科学计算栈的
环境中实际执行。

## 仓库结构

```text
skills/
  xmetai-weather-modeling/             建模侧工作流 skill。
    SKILL.md
    agents/openai.yaml
    assets/templates/
    references/
    scripts/
  xmetai-datakit-skills/               datakit 技能族。
    SKILL.md                           族入口（router）。
    reporting/                         评测产物 → 图表 + REPORT.md。
      SKILL.md  README.md  pyproject.toml
      agents/
      assets/templates/
      references/
      scripts/
        render_*.py  extract_*.py      CLI 入口
        reporting/                     运行库
          dispatch.py                  认能力 + 派发
          core/                        样式、表格、writer、语义
          sources/                     契约、ERA5、布局、台风几何
          spectra/                     指标基类、球谐带
          reports/                     TS / 连续场 / RMSE / 暴雨 / 台风 / 空间
docs/                                  网站文档页面。
tests/                                 skill 结构与独立运行测试。
requirements.txt                       环境安装（指向 reporting/pyproject.toml）。
```

## 文档

`docs/` 用于网站文档页面。安装与本地设置说明统一放在本 README，避免出现多个
setup 入口。

- `docs/usage.md`：面向网站的使用概览。
- `docs/tool-compatibility.md`：面向网站的工具兼容说明。
- `docs/release.md`：发布检查清单。

## 安全边界

- 读取 Zarr 是正常检查工作。写入、追加、覆盖、删除、原地合并、rechunk 或任何
  store 变更，必须在当前对话中获得用户明确授权。
- reporting 只读输入；报告输出必须是新目录或空目录，绝不覆盖已有报告。

## 新增 Skill

1. 建 `<name>/`，包含 `SKILL.md`、`agents/`、`assets/`、`references/`、`scripts/`。
2. 触发词写在 `SKILL.md` 的 `description` 里；`name` 与目录名保持一致。
3. `SKILL.md` 保持 router 形态，长文规则放进 `references/`。
4. 如果属于某个技能族，把成员加进族入口的路由表；同时更新本 README。
5. 在你使用的每个宿主里注册它。
