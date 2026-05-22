# xmetai-skills

Agent skills for XMetAI-style weather model work, including workspace inspection, LazyConfig review, Zarr/NetCDF/static-data checks, model-shape debugging, training/evaluation operations, ONNX export, and deployment review.

The installable skill is:

```text
skills/xmetai-weather-modeling/
```

The main entry file is:

```text
skills/xmetai-weather-modeling/SKILL.md
```

## Install

### Option 1: Install as a Codex skill

Clone this repository and expose the installable skill directory to your Codex skills folder:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
git clone git@github.com:XMetAI-Lab/xmetai-skills.git
mkdir -p "$CODEX_HOME/skills"
ln -s "$(pwd)/xmetai-skills/skills/xmetai-weather-modeling" "$CODEX_HOME/skills/xmetai-weather-modeling"
```

If your Codex home directory is not `~/.codex`, set `CODEX_HOME` before running the commands. The final path should be equivalent to:

```text
$CODEX_HOME/skills/xmetai-weather-modeling/SKILL.md
```

### Option 2: Keep this repo beside `xmetai-core`

For project-local development, keep this repository as a sibling checkout or submodule and ask the agent to load:

```text
xmetai-skills/skills/xmetai-weather-modeling/SKILL.md
```

Example sibling submodule setup:

```bash
git submodule add git@github.com:XMetAI-Lab/xmetai-skills.git xmetai-skills
git submodule update --init --recursive
```

Do not vendor `xmetai-core` into this repository. The skill is designed to inspect the active workspace where the user is working.

## Verify

From the root of this repository:

```bash
python -m pytest -q
```

The tests check that the skill entrypoint, references, agent metadata, and helper scripts are present and valid.

## Use

Open the target weather-model repository, then ask the agent to use `xmetai-weather-modeling` for one of these task types:

- LazyConfig creation or review.
- Zarr, NetCDF, static-data, and dataset-contract inspection.
- Model contract review and tensor-shape debugging.
- Training, evaluation, experiment reporting, ONNX export, and deployment checks.

Typical read-only helper commands:

```bash
python skills/xmetai-weather-modeling/scripts/inspect_workspace.py /path/to/xmetai-core
python skills/xmetai-weather-modeling/scripts/check_config_contract.py /path/to/config.py
python skills/xmetai-weather-modeling/scripts/inspect_static_nc.py /path/to/static.nc
```

Start with `skills/xmetai-weather-modeling/SKILL.md`; it routes the agent to the right reference document and script for the task.

## Repository Layout

```text
skills/xmetai-weather-modeling/
  SKILL.md                 Main skill entrypoint.
  agents/openai.yaml       Agent metadata.
  references/              Detailed task guidance.
  scripts/                 Small read-only inspection and validation CLIs.
  assets/templates/        Review, report, and planning templates.
docs/                      Website documentation pages.
tests/                     Structure and script-compilation tests.
```

## Documentation

`docs/` is for website documentation pages. Installation and local setup instructions belong in this README so users have one canonical setup path.

Useful pages:

- `docs/usage.md`: website-facing usage overview.
- `docs/tool-compatibility.md`: website-facing compatibility notes.
- `docs/release.md`: release checklist.

## Safety Boundary

Zarr reads are normal inspection work. Zarr writes, appends, overwrites, deletes, in-place merges, or store mutation require explicit user approval in the active conversation.
