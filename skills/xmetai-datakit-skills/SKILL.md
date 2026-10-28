---
name: xmetai-datakit-skills
description: >
  XMetAI datakit 技能族的入口与分诊。当用户提到 datakit、数据侧技能族、
  "这些 skill 分别能做什么""该用哪个 skill"、需要跨成员的流程（先取数或先评测、
  再出报告），或请求明显属于本族但不确定该落到哪个成员时使用。
  本 skill 自身不实现任何能力：每项能力的输入契约、命令与产物都在成员自己的
  SKILL.md 里，选中后按成员文档执行。
metadata:
  short-description: XMetAI datakit skill family router
---

# XMetAI Datakit Skills

数据侧技能族的入口。**这里只做分诊，不做实现**——每个成员有自己的输入契约与
工作流，位于同一个族目录下。是否依赖 Datakit 仓库，以成员文档为准。

## Start Here

1. 读「成员」表，判断请求属于哪个成员。
2. 打开该成员的 `SKILL.md`（如 `reporting/SKILL.md`），按它自己的输入契约与
   工作流执行。**不要再回到本文件找命令、列名或图表规则。**
3. 判断不了时，只问一个能决定归属的问题（例如"评测产物已经生成了吗？"），不要猜。
4. 成员目录缺失、为空或未安装时如实说明，不要凭记忆执行它的流程。

## 成员

| 成员 | 何时用 | 入口 | 状态 |
|---|---|---|---|
| `downloader/` | 下载 ERA5 / ERA5-Land / ORAS5，选择变量与数据源，生成配置、预览任务、执行或恢复下载 | `downloader/SKILL.md` | ✅ |
| `reporting/` | 已有评测产物（`manifest.json` + 结果表），要出图、写报告、多模型对比、台风检验、暴雨个例、空间分布 | `reporting/SKILL.md` | ✅ |

新增成员时：在族目录下建 `<name>/{agents,assets,references,scripts}` 与 `SKILL.md`，
再在本表加一行。**不要把成员内容抄进本文件。**

## Non-Negotiable Rules

- **不复述成员契约**。列名、路径、图号、硬性规则只在成员文档里维护；本文件抄一份就会漂移。
- **不在这里产出数字、图或报告**。本文件没有脚本，也不读数据。
- **不猜成员**。两个成员都像能接时，用一句话问清归属。
- **成员不可用就停下**，说明缺什么，不要用相似能力的命令顶替。

## Routing

| 请求 | 去向 |
|---|---|
| 下载原始资料、选择下载变量 / 区域 / 日期、生成下载配置、预览或恢复下载 | `downloader/SKILL.md` |
| 评测产物已生成，要出图 / 写报告 / 多模型对比 / 台风 / 暴雨个例 / 空间分布 | `reporting/SKILL.md` |
| 要**计算**评测指标、跑评测任务 | 不在本族：评测流水线仓库（本族只消费其产物） |
| 模型配置、Zarr/NetCDF 检查、训练、ONNX 导出、张量排错 | 不在本族：`xmetai-weather-modeling` skill |
| "这个族里都有什么""该用哪个" | 本文件 |

## 宿主注册

宿主只认**目录根部的 `SKILL.md`**，不会递归发现成员。按需要注册：

```bash
# 在 Datakit 仓库根执行；目标宿主的 skills 目录须已存在
# 注册成员：成员自身的触发词才会生效
ln -s "$(pwd)/examples/dataset_downloader/xmetai-datakit-skills/downloader" "$CODEX_HOME/skills/downloader"
ln -s "$(pwd)/examples/dataset_downloader/xmetai-datakit-skills/reporting" "$CODEX_HOME/skills/reporting"
# 注册本 router：让"该用哪个 / 族里有什么"这类请求也有入口
ln -s "$(pwd)/examples/dataset_downloader/xmetai-datakit-skills" "$CODEX_HOME/skills/xmetai-datakit-skills"
```

以上为 Codex 注册示例；VS Code / Copilot 可将成员链接到工作区 `.github/skills/`
或用户级 `~/.copilot/skills/`，名称与成员 `SKILL.md` 的 name 一致。

> 只注册容器、不注册成员时，成员不会被单独检索到——本文件的成员表与
> 路由仍可用，但成员自己的触发词不生效。
> 两者都注册时，**具体请求应由成员命中**，本 router 只接模糊请求。

## Validation Posture

本文件不附带脚本或测试。成员的验证方式（独立性测试、产物检查、报告回归）以成员
`SKILL.md` 与成员目录内的 `tests/` 为准。
