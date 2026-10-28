# xmetai-skills

面向 XMetAI 类气象模型工作的 agent skills 仓库，包含原始资料下载、建模侧工作流与评测结果报告。

## Skills

| Skill | 入口 | 用途 |
|---|---|---|
| `skills` | [统一路由](skills/SKILL.md) | 将不确定归属或跨阶段请求分派到具体 skill。 |
| `dataset-downloader` | [下载工作流](skills/dataset-downloader/SKILL.md) | 依赖 Datakit 仓库，规划、预览、执行或恢复 ERA5 / ERA5-Land / ORAS5 下载，并核对下载报告。 |
| `evaluation-reporting` | [评测报告工作流](skills/evaluation-reporting/SKILL.md) | 消费已完成的评测产物，生成图表与 Markdown 报告；不执行评测流水线、不重新计算其指标。 |
| `xmetai-weather-modeling` | [建模工作流](skills/xmetai-weather-modeling/SKILL.md) | 工作区检查、LazyConfig 审查、Zarr/NetCDF/静态数据检查、模型契约、张量形状调试、训练与评测/推理执行、ONNX 导出、部署审查。 |

三个具体 skill 在 `skills/` 下并列，`skills/SKILL.md` 为统一路由入口。
宿主通过 `SKILL.md` 的 `name` 与 `description` 选择技能；
`agents/*.yaml` 是可选元数据，不决定技能选择。

## 安装

### 注册为 Agent Skill

Codex：克隆本仓库后，分别链接需要使用的技能目录：

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
git clone git@github.com:XMetAI-Lab/xmetai-skills.git
mkdir -p "$CODEX_HOME/skills"

ln -s "$(pwd)/xmetai-skills/skills/dataset-downloader" "$CODEX_HOME/skills/dataset-downloader"
ln -s "$(pwd)/xmetai-skills/skills/evaluation-reporting" "$CODEX_HOME/skills/evaluation-reporting"
ln -s "$(pwd)/xmetai-skills/skills/xmetai-weather-modeling" "$CODEX_HOME/skills/xmetai-weather-modeling"
# 可选的统一路由入口：
ln -s "$(pwd)/xmetai-skills/skills" "$CODEX_HOME/skills/skills"
```

Claude Code：在本仓库根目录执行：

```bash
mkdir -p .claude/skills
cp -r skills/dataset-downloader .claude/skills/dataset-downloader
cp -r skills/evaluation-reporting .claude/skills/evaluation-reporting
cp -r skills/xmetai-weather-modeling .claude/skills/xmetai-weather-modeling
```

宿主不会递归注册嵌套的 `SKILL.md`。注册路由不等于分别注册三个具体技能。
可选路由的复制注册示例见 [英文 README](README.md#register-agent-skills)；
其相对链接必须能够访问三个具体技能。已有同名注册时先检查，不覆盖。

VS Code / Copilot 可以将具体技能链接到工作区 `.github/skills/` 或用户级
`~/.copilot/skills/`，名称与 frontmatter 保持一致；可选路由注册名称为 `skills`。

### 使用相邻工作区或子模块

将本仓库放在 `xmetai-core` 或 `xmetai-datakit` 旁边，也可以作为子模块引入：

```bash
git submodule add git@github.com:XMetAI-Lab/xmetai-skills.git xmetai-skills
git submodule update --init --recursive
```

直接加载 `xmetai-skills/skills/<skill>/SKILL.md`，或通过
`xmetai-skills/skills/SKILL.md` 选择具体技能。不要把模型或 Datakit 源码复制进本仓库；
技能检查的是当前目标工作区。

## 依赖

建模辅助脚本和下载分派入口使用 Python 标准库。下载报告渲染器需要 PyYAML；
实际下载及预览的依赖由 Datakit 下载器维护，详见
[下载技能 README](skills/dataset-downloader/README.md)。

评测报告需要科学计算依赖，版本及 extras 统一由其 `pyproject.toml` 维护：

```bash
python -m pip install -r requirements.txt
# 可选的空间取数及台风地图依赖：
python -m pip install "./skills/evaluation-reporting[all]"
```

`requirements.txt` 只引用该项目，不重复维护依赖清单。技能包安装是可选的：
入口脚本会自行将 `scripts/` 加入 `sys.path`，但运行时依赖仍须可用。

## 使用

### 原始资料下载

使用 `dataset-downloader`，明确数据需求并确认保存目录后，生成配置并预览计划：

```bash
python skills/dataset-downloader/scripts/run_download.py \
  --repo-root /absolute/path/to/xmetai-datakit \
  --source cds /absolute/path/to/batch.yaml
```

默认仅离线预览；实际下载需要确认具体计划、完成相应认证，并添加 `--execute`。
完整契约见 [下载技能](skills/dataset-downloader/SKILL.md)。

### 建模侧工作流

使用 `xmetai-weather-modeling` 创建或审查配置，检查数据与模型契约，调试张量形状，
执行训练、评测/推理、ONNX 导出及部署检查：

```bash
python skills/xmetai-weather-modeling/scripts/inspect_workspace.py --root /path/to/xmetai-core
python skills/xmetai-weather-modeling/scripts/check_config_contract.py --config /path/to/config.py
python skills/xmetai-weather-modeling/scripts/inspect_static_nc.py --input /path/to/static.nc
```

该技能支持评测任务执行与如实汇报运行情况，不提供指标计算实现或分析报告生成。

### 评测结果报告

评测产物已生成时，使用 `evaluation-reporting`：

```bash
python skills/evaluation-reporting/scripts/render_report.py \
  outputs/results/weather_ts_single_fgvp --result report/fgvp --model FGVP
```

TS 系列多模型对比：

```bash
python skills/evaluation-reporting/scripts/render_report.py \
  outputs/results/weather_ts_single_fgvp --result report/fgvp_vs_fuxi --model FGVP \
  --compare "FuXi=outputs/results/weather_ts_single_fuxi/diagnostics/categorical_wide.csv"
```

| 场景 | 入口 |
|---|---|
| 多模型 RMSE（确定性 / 集合） | `render_det_report.py` / `render_ens_report.py --model NAME=目录` |
| 暴雨过程个例 | `render_rainstorm_report.py --model NAME=目录 --process 202503,202521` |
| 暴雨过程选择 | `render_rainstorm_summary.py`，生成 `summary_long.csv` |
| 服务器端空间取数 | `extract_spatial_distribution.py` / `extract_spatial_distribution_ens.py` |
| 空间分布绘图 | `render_spatial_distribution.py --csv ... --model ...` |
| 台风统计 | `render_report.py` 配合 `--leads` / `--fixed-leads` / `--bootstrap` |

必需文件、列名、缺值处理与输出约束见 [报告技能契约](skills/evaluation-reporting/SKILL.md)。

## 验证

在仓库根目录运行：

```bash
python -m pytest tests -q
```

结构测试覆盖统一路由、并列目录、名称、英文文档、相对链接、元数据与建模脚本编译。
下载测试覆盖分派、认证向导、模板和报告渲染。真实 Datakit 兼容性及 CDS 在线测试须显式启用，
普通测试不会提交下载。

报告测试将技能复制到临时宿主目录，不安装技能包即可验证入口和运行库；
缺少科学计算依赖时跳过。CI 运行测试并编译建模脚本，报告运行测试需要对应科学计算环境。

## 仓库结构

```text
skills/
  SKILL.md                            统一路由入口
  dataset-downloader/
    SKILL.md  README.md
    agents/  assets/templates/  references/  scripts/
  evaluation-reporting/
    SKILL.md  README.md  pyproject.toml
    agents/  assets/templates/  references/
    scripts/
      render_*.py  extract_*.py        CLI 入口
      reporting/           Python 运行库
        dispatch.py
        core/  sources/  spectra/  reports/
  xmetai-weather-modeling/
    SKILL.md
    agents/  assets/templates/  references/  scripts/
docs/                                 网站文档
tests/                                结构与运行测试
requirements.txt                       评测报告依赖安装入口
```

## 文档与安全边界

安装及本地设置统一维护在 README。网站页面包括 [使用说明](docs/usage.md)、
[工具兼容性](docs/tool-compatibility.md) 与 [发布清单](docs/release.md)。

- 任何 Zarr 写入、追加、覆盖、删除、原地合并或 rechunk，均须在当前对话中获得明确授权。
- 报告技能只读输入，输出必须使用新目录或空目录，不覆盖已有报告。
- 下载须确认具体计划与输出目录；凭据保持私密，认证配置和预览不自动授权下载。

## 新增 Skill

1. 创建 `skills/<name>/`，按需包含 `SKILL.md`、`agents/`、`assets/`、`references/` 与 `scripts/`。
2. frontmatter 的 `name` 与目录名一致，`description` 包含具体触发场景。
3. 具体技能维护操作契约，长篇说明放入其 references。
4. 更新统一路由、本 README 与直接相关测试。
5. 在需要使用的宿主中分别注册。
