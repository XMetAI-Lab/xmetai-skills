---
name: dataset-downloader
description: >
  Use when downloading meteorological or oceanographic source data through the
  XMetAI Datakit repository, including ERA5, ERA5-Land, and ORAS5; selecting
  variables, pressure levels, depths, regions, times, or daily statistics;
  configuring CDS API credentials or CDS / Copernicus Marine downloads;
  previewing or resuming downloads; locating sources and outputs; or reviewing
  CDS queue status and execution-terminal options. Generate YAML configurations
  according to repository conventions and invoke existing downloaders.
  Dataset processing, training, and evaluation reporting are outside this skill.
metadata:
  short-description: Plan and run repository-backed meteorological downloads
---

# XMetAI Dataset Downloader

Convert data requirements into verifiable configurations, task plans, and local
source files. This skill requires the Datakit repository. It includes configuration
templates and documented candidate values, not a separate downloader implementation
or an authoritative variable registry.

## Inputs and Outputs

Accept natural-language requirements or an existing Batch YAML. Before execution,
establish:

- The source or dataset, variables, and their meanings.
- Dates, region, times or statistical aggregation, and pressure levels or depths
  where applicable.
- The local output directory and whether the request is for preview or execution.
  Confirm the directory with the user. If unspecified, propose
  `~/xmetai-downloads/<task-name>/`; do not default to the skill repository,
  `output/`, or a source-code directory.

Ask one question at a time when missing information determines data semantics.
Do not assume global coverage, substitute variables, or extend the date range.
CDS currently requests complete months; disclose this limitation for partial-month
requests rather than silently downloading an entire month.

| Stage | Deliverables |
|---|---|
| Planning | Profile and Batch YAML, the actual task plan, and a requirements summary for confirmation |
| Execution | Source files, status records, and a download report covering parameters, actual paths, completed/failed/skipped tasks, and verification scope |

Download completion does not establish scientific integrity or completion of a
standardized training dataset.

## Workflow

### 1. Locate the Repository and Source

Use [scripts/run_download.py](./scripts/run_download.py). The entrypoint searches
its actual ancestor directories for Datakit. When the repositories are separate,
provide `--repo-root /absolute/path/to/xmetai-datakit`. Stop if the repository or
downloader is unavailable; do not substitute another implementation.

- CDS: ERA5 and ERA5-Land; see [CDS reference](./references/cds.md).
- Copernicus Marine: currently configured for ORAS5; see
  [Marine reference](./references/marine.md).
- Identify unsupported sources or variables explicitly rather than mapping them
  to existing datasets based only on their names.

### 2. Verify Variable Conventions

Consult [Profile templates and candidates](./references/profile-templates.md),
then verify the current downloader, dataset definitions, and schema against
[variable conventions and boundaries](./references/variables.md). Confirm external
variable names, levels, and statistical definitions. Datakit Profiles are examples,
not default tasks or exhaustive lists. Template candidates are not all enabled.
Standard product units in the schema may differ from downloaded source units;
do not claim a conversion has occurred.

### 3. Prepare Configuration

Generate task-specific Profiles from the supplied templates. Do not automatically
copy example variable combinations, regions, pressure levels, or times from Datakit.
Select hourly single-level, pressure-level, land, corresponding daily-statistics,
or Marine ORAS5 templates as appropriate.

Template comments document fields and a repository candidate snapshot, not an
exhaustive official catalog or a guarantee that every combination is supported.
Check official dataset and downloader rules for candidates outside the snapshot;
adding a variable does not itself require modifying the skill's candidate list.
Inspect user-supplied Profiles before reusing them; do not require regeneration
or overwrite shared configurations or templates.

Resolve relative Profile paths and storage roots against the **Batch directory**.
Before generating the Batch, obtain confirmation of `storage.root_path`. If absent,
propose `~/xmetai-downloads/<task-name>/`, resolve it to an absolute path, and write
it only after confirmation. Keep configurations, records, and data separate from
source code; do not default to the skill repository, `output/`, Datakit source,
or the current working directory.

Start from the [CDS Batch template](./assets/templates/cds_batch.yaml) or
[Marine Batch template](./assets/templates/marine_batch.yaml). Save the selected
Profile and Batch in the task directory and replace all placeholders. Clarify
unresolved critical values before proceeding. Dates belong only in the Batch.
Do not mix hourly `time` or format fields into daily-statistics Profiles, or CDS
fields into Marine Profiles. Validate YAML types, placeholders, and actual paths
before previewing. Use low concurrency and bounded retries rather than inheriting
large example retry limits.

### 4. Preview Offline and Confirm

In these commands, `<skill>` is this skill directory, `<datakit>` is the Datakit
repository root, and `<batch>` is the actual configuration:

```bash
python <skill>/scripts/run_download.py --repo-root <datakit> --source cds <batch>
python <skill>/scripts/run_download.py --repo-root <datakit> --source marine <batch> --plan-only
```

The default is `--dry-run`: no remote connection is made, but the underlying
downloader writes a local plan. `--plan-only` generates only the target list.
Neither mode verifies authentication or remote data availability. Preview still
requires the downloader's YAML dependencies; install missing dependencies from
that downloader's requirements.

After a successful offline preview, present a **Download Overview**, not a
"Preview passed" heading or opening. The overview supports confirmation of the
download contents, not notification of internal checks. It does not imply
authentication, remote availability, or request submission. If preview fails,
explain the error rather than presenting the configuration as ready.

Present the information in this order:

1. Briefly acknowledge the confirmed region, times, and directory without asking
   for those parameters again.
2. Provide a table or concise fields covering source, variables and levels,
   region, dates, times or aggregation, number of time steps when determinable,
   task count, file organization, and confirmed storage location. Use the actual
   preview; distinguish request counts, file counts, and time-step counts.
   Prefer a clickable directory link for storage. Show full target paths separately
   when needed; do not place long paths in selection controls.
3. State the current status, such as "Configuration prepared; no download request
   has been submitted." Use this statement only after successful offline checks
   and do not describe local checks as remote verification.
4. Present the next action. Do not invent file-size estimates. State that file
   sizes will be determined by the actual download without repetitive warnings.

Obtain approval of the concrete plan, including the output root, before the first
execution. Re-preview and obtain confirmation after configuration changes.

#### CDS Queue Notice and Execution Terminal

Before submitting a CDS request, briefly explain after the overview that processing
may require queueing and that sizes depend on the downloaded files. Do not promise
immediate handling for small regions, single variables or levels, or few time steps,
or provide unsupported completion estimates. Explain `accepted` only after that
status is observed; do not overload the confirmation question with status terms,
dashboard links, or detailed queue explanations.

After the overview and status, ask a short, separate execution question using the
host's question tool when available. Do not repeat the overview or long paths in
the question. Combine approval of the specific plan with these terminal choices:

- Start the download in the agent terminal (recommended).
- Provide the command for execution in another terminal.
- Retain the configuration without downloading.

Do not ask again if the user has selected a terminal. If execution is requested
without a terminal preference, use the agent terminal after obtaining confirmation
of the concrete plan. Terminal selection does not replace credential preparation
or acceptance of dataset terms.

- **Agent terminal, default:** execute in the confirmed environment. Report only
  observed request IDs, latest statuses, and elapsed waiting times. After observing
  `accepted`, explain that CDS has accepted the request and processing is pending.
  State that the process continues waiting only after verifying it is still running.
  Acceptance does not mean extraction or transfer has started. Logs may update only
  on state changes; silence does not establish a stalled process.
  Do not invent queue position, queue length, or completion time. If relevant,
  state that CDS has not supplied an estimate. During waiting, provide the
  [CDS Live Dashboard](https://cds.climate.copernicus.eu/live) as a view of overall
  service load, not this request's position. Do not promise faster processing for
  another year, month, or dataset. Attribute cited statistics with source, update
  time, and time zone, or identify unknown metadata.
  Once continued execution is confirmed, a response may end while awaiting the
  completion notification; do not repeatedly poll merely to keep responding.
  Do not stop, switch terminals, retry, or resubmit solely because waiting exceeds
  two minutes or the service is congested.
- **Another terminal:** provide a copyable execution command for the previewed
  Batch without starting a concurrent download. Use actual absolute paths and
  shell-appropriate quoting. Include `--repo-root`, `--source cds`, the Batch,
  and `--execute`; omit credentials and default forced redownloads.
  Include the actual conda activation or
  `conda run --no-capture-output -n <environment>` prefix where applicable.
  Require the same download machine and user account, and request retention of
  the final summary and exit code. After the user reports completion, verify
  evidence and generate the report under step 6. Mark unavailable exit codes unknown.

Before switching an active task from the agent terminal, establish local process
and server-task status. Stopping a local process does not guarantee cancellation
of the CDS request. The downloader cannot attach to an unfinished request by ID;
rerunning the Batch may submit duplicates. Do not execute the same Batch
concurrently in another terminal. First have the user inspect and cancel the old
request on the CDS task page, or retrieve its completed result, before deciding
whether to rerun. Local `started` or `retrying` records do not establish cancellation.

These choices apply to CDS downloads after authentication preparation. Credential
entry must remain in a private user terminal. Marine execution follows the private
interactive-terminal requirements below, not the agent-terminal default.

### 5. Prepare Authentication and Execute

Official clients consume authentication; consult the source-specific references.
Ask the user to select among these CDS options:

1. **Local interactive wizard:** the user runs
   `python <skill>/scripts/setup_cds.py` in a private terminal on the actual download
   machine and account, selects option 1, and enters the token with hidden input.
   The script detects the platform and uses `Path.home()` for the official default
   configuration: `~/.cdsapirc` on Linux/macOS or `%USERPROFILE%\.cdsapirc` on Windows.
   POSIX permissions are 0600. Windows PowerShell sets and verifies a current-user-only
   ACL on an empty temporary file before writing the token; permission failures stop
   the operation. Existing regular files require confirmation of whole-file replacement.
   Symbolic links and Windows reparse points are not followed. Saving is refused
   when hidden input is unavailable.
2. **Manual configuration guidance:** run
   `python <skill>/scripts/setup_cds.py --manual`, or select option 2, to display
   instructions without reading or writing credentials.
3. **Fully manual, zero code execution:** provide the
   [manual procedure](./references/cds.md#fully-manual-configuration) directly in the conversation.
   Do not run scripts, shell commands, dependency installations, authentication
   probes, or the wizard, and do not read or write credential files. If necessary,
   establish the operating system and whether execution is remote or under WSL.
   Explain the applicable path, file format, permissions, and dataset terms, then
   wait for the user to confirm completion. Do not request file contents, tokens,
   screenshots, or command output as evidence. Resume the original task after
   confirmation without requiring the wizard or online verification, and do not
   claim authentication has been verified. This zero-execution restriction applies
   to credential configuration; subsequent preview and execution retain their
   authorization and plan-confirmation requirements.

The agent may present wizard commands or instructions, but must not perform
secret entry through chat tools, output-capturing tasks, pipes, or automated input.
Never request keys in chat, read or display existing credentials, or place real
secrets in Profiles, Batches, source code, arguments, or logs. The local wizard's
official user-level authentication YAML is the sole credential-file write path;
it is not task YAML.

If `CDSAPI_RC`, `CDSAPI_URL`, or `CDSAPI_KEY` overrides are present, the wizard
reports only their names and stops. Ask the user to resolve them privately or use
manual configuration; do not inspect or print their values. Saving configuration
does not validate authentication. The wizard does not connect, submit requests,
or install dependencies. Users must accept dataset terms themselves.

The CDS wizard does not apply to Marine. The Marine executor calls
`copernicusmarine.login()` during execution; users may enter credentials or answer
replacement prompts at that point, without a separate prior login.
Provide Marine execution commands for an interactive private terminal in the
actual download environment. Do not receive or enter credentials through captured
tools, pipes, or automated input. Prior login does not guarantee a prompt-free run.
Do not inspect the terminal during credential entry or request passwords or screenshots.
After completion, check local plans, statuses, and files. If the current exit code
or summary is unavailable, mark it unknown rather than inferring success from history.
Offline previews may still run through agent tools.

```bash
python <skill>/scripts/run_download.py --repo-root <datakit> --source cds <batch> --execute
python <skill>/scripts/run_download.py --repo-root <datakit> --source marine <batch> --execute
```

Default recovery uses the underlying downloader's completion records. A task is
skipped only when both a matching completion record and its target file exist.
This is task-level recovery, not guaranteed byte-level resumption.
`--force-redownload` requires `--execute` and separate approval because it may
overwrite files. Use a new output directory when changing region, variable
semantics, or grid rather than overwriting another dataset.

### 6. Verify and Deliver

Check the process exit code, final summary, and actual status records. Never report
completion while the task is running. Disclose nonzero exits or failure counts,
retain failure evidence, and provide recovery commands.

After execution ends, including partial failure or confirmed interruption, verify
and save a Markdown report using the
[download report template](./assets/templates/download_report.md). A chat-only
completion statement is insufficient. Prefer
[scripts/render_download_report.py](./scripts/render_download_report.py) to generate
a draft from the Batch, plan, status records, optional run log, and file attributes.
It does not connect remotely, read credentials, or validate file contents.
Review the draft and supplement checks it cannot perform. The download entrypoint
does not generate a report automatically.

```bash
python <skill>/scripts/render_download_report.py <batch> \
  --plan <plan.txt> --status <status.jsonl> \
  --run-log <captured_output.log> --exit-code <code> \
  --request-id <id> --datakit-root <datakit> --source cds --strict
```

`--run-log` is captured entrypoint output, including its final summary. Without it,
the report infers evidence from status records. Do not replace missing exit codes,
summaries, or status evidence with zero or success. `--strict` returns nonzero for
evidence anomalies.

- Save reports in the record directory containing the **actual task plan**, using
  `download_report_<YYYYMMDDTHHMMSSffffffZ>.md` in UTC without overwriting history.
  The default is `storage.root_path/_records/<Batch-stem>/`; actual custom paths
  or runtime overrides take precedence. If the plan path cannot be established,
  explain the issue and confirm the report location. Report write failures explicitly;
  do not claim a report exists.
- Describe each Profile separately: source, dataset, variables, dates, region,
  pressure levels/depths, times or statistics, format, version selection, and
  execution parameters. Distinguish configuration/request values from inspected
  file values. Support native temporal resolution with verified evidence rather
  than file counts; do not invent versions, units, or resolution.
- Include absolute paths for Batches, Profiles, data, actual plans, and status
  records. Verify existence and actual size against this plan's targets. Separate
  current completions, historical completions skipped, failures, and unknowns.
  Do not count the entire data directory as the current download or double-count
  retries. Append-only logs may include earlier tasks: match current requests,
  targets, and execution windows. When current and historical completion cannot
  be distinguished, mark it unknown; path equality alone is insufficient.
- For large outputs, save a complete inventory beside the report and link it with
  counts and directory structure. Disclose missing, corrupt, or contradictory
  evidence. Treat missing files, zero-byte files, and residual `.part.nc` files
  as anomalies rather than declaring full completion.
- Do not read credential files or copy raw terminal transcripts or unsanitized
  errors into the report. If contents were not inspected, state that file-content
  and scientific-integrity verification was not performed. Label preview-only
  reports accordingly; do not classify preview targets as downloaded.

Deliver a report link, content summary, data location, and anomalies. While a task
is running, provide progress only, not a final completion report. Report generation
does not authorize retries, forced overwrites, or additional online requests.
Decompression, conversion, unit processing, standardized dataset construction,
and subsequent validation are separate workflows and must not start automatically.

## Mandatory Boundaries

- Do not move or duplicate downloader implementations; execution and dependencies
  remain in Datakit.
- Candidate snapshots are not authoritative registries. Current downloader rules
  govern fields; dataset definitions govern availability. Extend and verify Datakit
  before using unsupported sources or request structures.
- Do not bypass account permissions, service limits, or data terms.
- Obtain confirmation of `storage.root_path`; do not default to source directories.
- Preview success, plan generation, and file existence do not establish a successful
  download.

## Validation

Tests reside in the repository-level
[dispatcher tests](../../tests/test_downloader_skill.py) and
[report tests](../../tests/test_downloader_report.py):

```bash
python -m pytest -q tests/test_downloader_skill.py tests/test_downloader_report.py
```

Dispatcher tests use temporary stubs to verify argument gates, repository discovery,
forwarding, and exit-code propagation. They do not test documentation, templates,
or real downloaders and do not connect remotely. Report tests use temporary
configurations, plans, records, and files without network or credential access.
Report tests require PyYAML; neither suite requires a real Datakit checkout.
See [README validation instructions](./README.md#tests) for templates and optional
offline Datakit compatibility checks. Offline success does not prove remote support
for a variable combination.
