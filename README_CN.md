# xmetai-skills

面向 XMetAI 类气象模型工作的 agent skills 仓库，覆盖工作区检查、气象数据需求提取与下载前规划、LazyConfig 审查、Zarr/NetCDF/静态数据检查、模型 shape 调试、训练评估、ONNX 导出和部署审查。

可安装的 skill 目录是：

```text
skills/xmetai-weather-modeling/
```

主入口文件是：

```text
skills/xmetai-weather-modeling/SKILL.md
```

## 安装

### 方式一：作为 Codex skill 安装

克隆本仓库，并把可安装 skill 目录暴露到 Codex skills 目录：

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
git clone git@github.com:XMetAI-Lab/xmetai-skills.git
mkdir -p "$CODEX_HOME/skills"
ln -s "$(pwd)/xmetai-skills/skills/xmetai-weather-modeling" "$CODEX_HOME/skills/xmetai-weather-modeling"
```

如果你的 Codex home 目录不是 `~/.codex`，先设置 `CODEX_HOME` 再执行命令。最终路径应该等价于：

```text
$CODEX_HOME/skills/xmetai-weather-modeling/SKILL.md
```

### 方式二：放在 `xmetai-core` 旁边

如果是项目内协作，可以把本仓库作为 sibling checkout 或 submodule 放在目标工程旁边，然后让 agent 加载：

```text
xmetai-skills/skills/xmetai-weather-modeling/SKILL.md
```

示例 sibling submodule 安装：

```bash
git submodule add git@github.com:XMetAI-Lab/xmetai-skills.git xmetai-skills
git submodule update --init --recursive
```

不要把 `xmetai-core` vendor 到本仓库里。这个 skill 会检查用户当前正在工作的 active workspace。

## 验证

在本仓库根目录执行：

```bash
python -m pytest -q
```

测试会检查 skill 入口、references、agent metadata 和辅助脚本是否存在且有效。

## 使用

打开目标气象模型仓库后，让 agent 使用 `xmetai-weather-modeling` 处理以下任务：

- 创建或审查 LazyConfig。
- 提取气象数据需求并制定下载前计划，不执行下载或数据写入。
- 检查 Zarr、NetCDF、静态数据和数据集 contract。
- 审查模型 contract，调试 tensor shape。
- 检查训练、评估、实验报告、ONNX 导出和部署流程。

常用只读辅助命令示例：

```bash
python skills/xmetai-weather-modeling/scripts/inspect_workspace.py /path/to/xmetai-core
python skills/xmetai-weather-modeling/scripts/check_config_contract.py /path/to/config.py
python skills/xmetai-weather-modeling/scripts/inspect_static_nc.py /path/to/static.nc
```

优先从 `skills/xmetai-weather-modeling/SKILL.md` 开始；它会把 agent 路由到对应任务的 reference 文档和脚本。

## 仓库结构

```text
skills/xmetai-weather-modeling/
  SKILL.md                 skill 主入口。
  agents/openai.yaml       agent metadata。
  references/              分任务详细说明。
  scripts/                 小型只读检查和验证 CLI。
  assets/templates/        审查、报告和计划模板。
docs/                      网站文档页面。
tests/                     结构和脚本编译测试。
```

## 文档

`docs/` 用于网站文档页面。安装和本地设置说明统一放在本 README，避免出现多个 setup 入口。

有用页面：

- `docs/usage.md`：面向网站的使用概览。
- `docs/tool-compatibility.md`：面向网站的工具兼容说明。
- `docs/release.md`：发布检查清单。

## 安全边界

读取 Zarr 是正常检查工作。写入、追加、覆盖、删除、原地合并或修改 Zarr store 必须在当前对话中获得用户明确授权。
