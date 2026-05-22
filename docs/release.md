# Release

Before tagging:

1. `python -m pytest tests`
2. `python -m py_compile skills/xmetai-weather-modeling/scripts/*.py`
3. Check `SKILL.md` references existing files.
4. Check no credentials or private dataset paths are embedded.
