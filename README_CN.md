# xmetai-skills

面向 XMetAI 类气象模型研发、训练、评估、推理和部署的 agent skill 仓库。

- 可安装 skill：`skills/xmetai-weather-modeling/`
- 入口：`SKILL.md`
- 按需参考：`references/`
- 小型 CLI：`scripts/`
- 模板：`assets/templates/`

安全边界：读取 Zarr 可以直接检查；写入、覆盖、追加、删除、原地合并或修改 Zarr store 必须在当前对话中获得用户明确授权。
