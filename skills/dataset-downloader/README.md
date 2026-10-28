# XMetAI Dataset Downloader

Translate meteorological and oceanographic data requirements into configurations,
plans, and local source files according to Datakit conventions.
[SKILL.md](./SKILL.md) is the authoritative capability contract and execution policy.

## Scope and Layout

This skill requires the Datakit repository and invokes its existing downloaders.
It does not duplicate their implementation, package metadata, or dependency lists.

```text
dataset-downloader/
  SKILL.md
  README.md
  agents/
    openai.yaml
    claude.yaml
  assets/templates/
    cds_batch.yaml
    cds_hourly_single_profile.yaml
    cds_hourly_pressure_profile.yaml
    cds_hourly_land_profile.yaml
    cds_daily_single_profile.yaml
    cds_daily_pressure_profile.yaml
    cds_daily_land_profile.yaml
    marine_oras5_profile.yaml
    download_report.md
    marine_batch.yaml
  references/
    cds.md
    marine.md
    profile-templates.md
    variables.md
  scripts/
    setup_cds.py
    run_download.py
    render_download_report.py
```

## Usage

A request can specify ERA5 2 m temperature for a region and year, four daily times,
and a plan preview before execution. The skill clarifies requirements, verifies
variable conventions, generates task-specific Profiles and a Batch from its
templates, and invokes Datakit. Datakit example Profiles are references, not defaults.

Templates contain applicable variable, pressure-level, time, region, daily-statistic,
depth, and format fields. Comments identify candidate values, repository sources,
and snapshot dates. These are not exhaustive official catalogs or automatically
enabled selections; see [Profile templates](./references/profile-templates.md).

Confirm `storage.root_path` before generating a Batch. If absent, propose
`~/xmetai-downloads/<task-name>/`, not the skill repository, `output/`, or Datakit
source. Copy selected templates into the task directory, replace placeholders,
and point the Batch to the new Profile.

From the repository root:

```bash
python skills/dataset-downloader/scripts/run_download.py \
  --repo-root /absolute/path/to/xmetai-datakit \
  --source cds /absolute/path/to/batch.yaml
```

The default is an offline preview; `--execute` submits downloads. Use
`--source marine` for Marine and `--help` for all options. A copied or registered
skill still requires access to Datakit; specify `--repo-root` when necessary.

The entrypoint uses the Python standard library. Backend dependencies remain in
Datakit's `examples/dataset_downloader/cds_download/requirements.txt` and
`examples/dataset_downloader/copernicusmarine_download/requirements.txt`.
Dependencies, login, and downloads are not started automatically.

## CDS Queueing and Execution Terminals

Before submission, the skill explains potential CDS queueing and offers:

- **Agent terminal, default:** after plan confirmation and authentication
  preparation, execute and report observed request IDs, statuses, and elapsed time.
- **Another terminal:** provide the actual previewed Batch command and environment
  for the same machine and account. Do not submit concurrently. After the user
  reports completion, verify records and files and generate a report.

`accepted` means only that the request was accepted. Small files may still wait,
and unchanged logs do not establish a stalled process.
The [CDS Live Dashboard](https://cds.climate.copernicus.eu/live) shows global load,
not an individual queue position. Unavailable queue counts and estimates remain
unavailable; no year, month, or level category is guaranteed to be faster.
Do not stop or change terminals automatically after two minutes.

Before switching an active task, establish the old request's status. Stopping a
local process does not guarantee server cancellation; rerunning a Batch can create
duplicates. Do not run the same Batch in parallel terminals. See
[execution policy](./SKILL.md#cds-queue-notice-and-execution-terminal) and
[CDS queue reference](./references/cds.md#queueing-and-execution-terminals).
The agent-terminal default does not apply to credential entry or Marine's private
interactive authentication.

## CDS API Credential Configuration

Three options are available without reading downloader source:

- **Local interactive wizard:** the user runs
  `python skills/dataset-downloader/scripts/setup_cds.py` in a private terminal
  on the download machine and selects option 1. Hidden token input is saved to
  `~/.cdsapirc` on Linux/macOS, with mode 0600, or
  `%USERPROFILE%\.cdsapirc` on Windows, with a current-user-only ACL set and
  verified by Windows PowerShell. Existing files require confirmation of complete
  replacement. Never enter secrets through chat, agent tools, or captured tasks.
- **Manual guidance:** add `--manual`, or select option 2, to display instructions.
- **Fully manual, zero code execution:** the skill explains Windows/Linux paths,
  `url` and `key` fields, permissions, and dataset terms in the conversation.
  During configuration it runs no code, wizard, `--manual`, or credential-file
  inspection. After the user confirms completion, resume without requesting tokens,
  screenshots, or file contents. Confirmation neither verifies authentication nor
  authorizes a download.

See the [fully manual procedure](./references/cds.md#fully-manual-configuration).
The script retains its two menu options; fully manual configuration uses neither.
On Windows, protect an empty temporary file before writing the token; missing
PowerShell or permission failures stop saving. The script stores a user-obtained
token; it does not generate keys, connect, validate credentials, or install dependencies.
Under WSL, SSH, or containers, the file belongs to that environment's user home,
not the Windows host. Client environment overrides stop the wizard without revealing
their values. Users remain responsible for account access, token retrieval, and
accepting dataset terms. See [CDS authentication](./references/cds.md#source-and-authentication).

## Marine Authentication and Download Reports

Marine execution calls `copernicusmarine.login()`. Users may enter credentials then
without a separate prior login. Provide execution commands for a private interactive
terminal rather than capturing output or entering passwords through agent tools.
After execution, the skill verifies evidence and delivers results.

After either CDS or Marine execution, generate a Markdown report using the
[report template](./assets/templates/download_report.md) in the actual plan's
record directory. The default is
`<storage.root_path>/_records/<Batch-stem>/download_report_<UTC-timestamp>.md`.
Microsecond timestamps preserve history without overwrites.

Generate a draft explicitly:

```bash
python skills/dataset-downloader/scripts/render_download_report.py \
  /absolute/path/to/batch.yaml \
  --run-log /absolute/path/to/run.log --exit-code 0 \
  --request-id <request-id> --datakit-root /absolute/path/to/xmetai-datakit \
  --source cds --strict
```

The renderer reads Batch/Profile files, plans, `status.jsonl`, optional logs, and
file attributes. It does not connect, read credentials, or validate file contents.
It requires PyYAML, which backend preview already requires. Review the draft and
supplement unperformed checks. Missing exit codes or summaries remain unknown,
not zero or success.

Reports include requested contents, parameters, absolute configuration/data paths,
completion/skip/failure counts, actual targets and byte sizes, anomalies, and
recovery guidance. Describe Profiles separately and attach full inventories when
needed. Report failures and interruptions as well as successes. Do not infer this
run's results from historical records or invent unavailable versions or metadata.
Disclose when scientific-integrity checks were not performed.

The agent explicitly generates and verifies the report; `run_download.py` does
not generate Markdown automatically. Reporting does not read credentials or
authorize postprocessing or further downloads. See
[verification and delivery](./SKILL.md#6-verify-and-deliver).

## Tests

Run from the repository root:

```bash
python -m pytest -q tests/test_downloader_skill.py
python -m pytest -q tests/test_downloader_report.py
python -m pytest -q tests/test_downloader_cds_setup.py
python -m pytest -q tests/test_downloader_templates.py
```

- [Dispatcher tests](../../tests/test_downloader_skill.py) use stubs for argument
  gates, repository discovery, forwarding, and exit codes. They require pytest,
  not Datakit, authentication, or network access, and do not test templates or
  real downloader implementations.
- [Report tests](../../tests/test_downloader_report.py) use temporary Batches,
  plans, statuses, and files without network or credential access.
- [Wizard tests](../../tests/test_downloader_cds_setup.py) use temporary homes and
  synthetic tokens. They cover interactive/manual modes, hidden input, overwrite
  confirmation, permissions, cancellation, overrides, and write failures.
- [Template tests](../../tests/test_downloader_templates.py) validate fields and
  types for seven Profile classes after placeholder substitution. They require
  PyYAML and skip if it is unavailable.

Optional offline compatibility checks use Datakit's actual task builders and
`--dry-run` entrypoints:

```bash
XMETAI_DATAKIT_ROOT=/absolute/path/to/xmetai-datakit \
  python -m pytest -q tests/test_downloader_skill.py tests/test_downloader_templates.py
```

Without `XMETAI_DATAKIT_ROOT`, only real-Datakit compatibility cases skip.
These checks do not log in or connect remotely and do not validate official
variable availability or file contents. ERA5-Land daily statistics checks cover
local interfaces only; verify variables against the official dataset.

### Optional CDS Live Smoke Test

Configuration resides in [test fixtures](../../tests/fixtures/dataset-downloader/)
and [live tests](../../tests/test_downloader_cds_live.py). By default, only
configuration checks run; live cases skip even when credentials already exist.

Configure CDS credentials, accept ERA5 terms, and install Datakit CDS requirements
plus `xarray`, `numpy`, `h5netcdf`, `h5py`, and `pytest` in the test environment.
Explicitly authorize the live request from the repository root:

```bash
XMETAI_RUN_CDS_LIVE=1 \
XMETAI_DATAKIT_ROOT=/absolute/path/to/xmetai-datakit \
CDSAPI_RC=/absolute/path/to/.cdsapirc \
  conda run -n xmetai-skills python -m pytest -q tests/test_downloader_cds_live.py
```

`CDSAPI_RC` is optional; the default is the user's `.cdsapirc`. Provide only its
path, never keys in commands, test configuration, or version control. The test
does not inspect or display credential contents. Once enabled, missing configuration
or dependencies, authentication errors, unaccepted terms, and server failures fail
the test rather than causing a skip.

Each live case submits one real request for February 2021, 00 UTC daily, 2 m
temperature, latitude 30-30.25 degrees north and longitude 120-120.25 degrees east,
without retries, with a 600-second subprocess timeout. Timeout stops local waiting,
not necessarily the server task. Independent temporary Batches and outputs do not
overwrite existing smoke data. Checks cover completion records, plans, 28 times,
a 2-by-2 grid, kelvin units, and 112 finite values.
The offline fixture can also be previewed directly. Its
`output/downloader_smoke` destination is for local smoke tests only, not the skill's
default download directory.

## Registration

Repository presence does not guarantee host discovery. From the repository root:

```bash
mkdir -p .github/skills
ln -s ../../skills/dataset-downloader .github/skills/dataset-downloader
```

Inspect existing registrations before making changes; do not overwrite them.
For other hosts, register this directory according to their discovery rules.
Registration does not install runtime dependencies.
