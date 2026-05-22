# Install

Use `skills/xmetai-weather-modeling/` as the installable skill directory.

As a sibling submodule:

```bash
git submodule add <xmetai-skills-remote-url> xmetai-skills
git submodule update --init --recursive
```

Do not vendor `xmetai-core` here; the skill inspects the active workspace.
