# Tool Compatibility

- Core is Markdown plus standalone Python CLIs.
- `agents/openai.yaml` is optional metadata.
- If commands can run, prefer scripts for deterministic checks.
- If commands cannot run, follow checklists manually and say so.
- Use runtime approval/sandbox mechanisms for dataset writes, installs, long training, export, and remote operations.
- Report facts, inferences, assumptions, and pending approvals separately.
