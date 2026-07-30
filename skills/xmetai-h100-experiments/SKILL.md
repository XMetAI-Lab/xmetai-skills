---
name: xmetai-h100-experiments
description: Operate XMetAI H100 training and evaluation Jobs on BingoCloud without a browser. Use when Codex needs to render and validate Job YAML, submit H100 Jobs, inspect Job/Pod/queue/event state, access the shared H100 workspace over SSH to read logs and results, or safely clean finished Pods and analyzed checkpoints.
---

# XMetAI H100 Job Operations

Use the bundled `scripts/bingo_job.py` for platform operations. Use SSH only for the shared H100 filesystem, logs, checkpoints, and result artifacts.

## Configure access

Require platform-specific values through the environment or matching CLI options:

```bash
export BINGO_BASE_URL="..."
export BINGO_DEX_URL="..."
export BINGO_CLUSTER_ID="..."
export BINGO_WORKSPACE_ID="..."
export BINGO_NAMESPACE="..."
export BINGO_USERNAME="..."
export BINGO_PROXY="..."  # optional
```

Never put a username, password, token, or cookie in this Skill, a Job YAML, a repository file, a log, or a command shown to the user.

- Prefer the interactive password prompt.
- Use `BINGO_PASSWORD` only as an ephemeral process environment value when the user explicitly authorizes it.
- Reuse the script's short-lived token cache; never print or copy the token.

Set reusable local variables without embedding credentials:

```bash
SKILL_DIR="/path/to/xmetai-skills/skills/xmetai-h100-experiments"
BINGO_JOB="$SKILL_DIR/scripts/bingo_job.py"
TEMPLATE="/path/to/train_h100.yaml"
```

## Prepare code and YAML

1. Read repository rules and inspect the worktree.
2. Confirm the code root visible inside the Pod, dataset mounts, output directory, base checkpoint, GPU count, batch per GPU, CPU, memory, learning rate, maximum iterations, and evaluation cadence.
3. Sync only the required code/config files to the H100 workspace. Avoid broad syncs when the worktree contains unrelated changes.
4. Render to `/tmp` before submitting:

```bash
python "$BINGO_JOB" render "$TEMPLATE" \
  --name "$JOB_NAME" \
  --gpus "$GPU_COUNT" \
  --cpu "$CPU_COUNT" \
  --memory "$MEMORY_LIMIT" \
  --shell-command "$TRAIN_COMMAND" \
  -o "/tmp/$JOB_NAME.yaml"
```

5. Parse the rendered YAML and verify:
   - `metadata.name` and Pod `job-name` label match.
   - The command has the intended `-g`, per-GPU `-b`, config, checkpoint, and output path.
   - GPU requests and limits match the command.
   - CPU/memory are proportional to the workload.
   - The output path is unique and located in the user-approved shared output root.
   - No username, password, token, network interface, tenant, or unrelated RDMA annotation leaked into the YAML.

Use a stable, explicit Job name. When submitting an already-rendered YAML, pass `--name` again because the submit step re-renders the document.

## Submit safely

Check for an existing name and queued Pods first:

```bash
python "$BINGO_JOB" jobs --keyword "$JOB_NAME" --page-size 100
python "$BINGO_JOB" queue --page-size 100
```

Submit only after the exact name is absent:

```bash
python "$BINGO_JOB" submit "/tmp/$JOB_NAME.yaml" \
  --name "$JOB_NAME" \
  --gpus "$GPU_COUNT" \
  --cpu "$CPU_COUNT" \
  --memory "$MEMORY_LIMIT"
```

The script refuses to submit while another Job Pod is queued. Add `--allow-queued` only when the user explicitly wants concurrent or queued submissions. Never retry blindly after a timeout: query the exact Job name first.

Do not cancel or replace an already-submitted Job unless the user explicitly asks.

## Query Jobs

Use the narrowest query:

```bash
python "$BINGO_JOB" jobs --keyword "$KEYWORD" --page-size 100
python "$BINGO_JOB" pods --job "$JOB_NAME" --page-size 100
python "$BINGO_JOB" status "$JOB_NAME"
python "$BINGO_JOB" events "$JOB_NAME"
python "$BINGO_JOB" queue --page-size 100
```

Use `status --watch` only for short interactive observation. For long monitoring, use the product's recurring-monitor mechanism rather than opening new tasks, terminals, or tmux windows.

Report platform state precisely: `Pending`, `Running`, `Complete`, or `Failed`; Pod node; current iteration; ETA; key losses; OOM/NaN; and checkpoint presence. Do not claim completion until the Job succeeded and the expected final artifact exists.

## Access logs and results over SSH

The platform helper does not stream container logs. Read the shared H100 workspace:

```bash
ssh "$H100_HOST" "find '$OUTPUT_DIR' -maxdepth 1 -type f -printf '%f\n' | sort"
ssh "$H100_HOST" "tail -n 80 '$OUTPUT_DIR'/log-*.txt"
ssh "$H100_HOST" "tail -n 1 '$OUTPUT_DIR'/metrics.json"
ssh "$H100_HOST" "grep -nE 'max_iter|total_batch_size|teacher_forcing|rollout|max|loss' '$OUTPUT_DIR'/config.yaml"
ssh "$H100_HOST" "du -sh '$OUTPUT_DIR'"
```

Prefer rank-0 logs. Treat missing output as a possible scheduling/startup delay until platform status and events are checked.

For result collection:

1. Confirm the checkpoint size is complete and stable.
2. Confirm evaluation uses the matching model/runtime options.
3. Read machine-readable CSV/JSON before summarizing logs.
4. Preserve configs, metrics, logs, and evaluation tables even when checkpoints are later removed.

## Clean finished resources

### Finished Pods

Always dry-run first:

```bash
python "$BINGO_JOB" cleanup-pods --keyword "$KEYWORD" --dry-run
```

Without `--yes`, deletion requires typing the exact confirmation phrase. Use `--yes` only after the user has explicitly approved the displayed candidates. This deletes finished Pods, not Jobs or output files.

### Analyzed checkpoints

Checkpoint deletion is a separate SSH filesystem operation:

1. List all active training and evaluation Jobs.
2. Build an allowlist of output directories still being written or read.
3. Preserve base/pretrained checkpoints and every allowlisted directory.
4. Dry-run the exact `.pth` candidate list and report count plus total GiB.
5. Delete only the reviewed candidate list after explicit authorization.
6. Recount remaining `.pth` files and compare directory usage.

Never use a broad recursive delete on the output root. Keep logs, configs, metrics, CSV/JSON, and evaluation results unless the user separately asks to remove them.

## Failure handling

- Authentication failure: stop repeated prompts; refresh interactively and retry a read-only query.
- Existing Job name: do not resubmit; inspect its status.
- Pending Pod: inspect `queue` and `events`; do not cancel unrelated Jobs.
- Failed Job: inspect events and rank-0 logs before changing config.
- Submit timeout: query by exact name before deciding whether submission failed.
- Cleanup uncertainty: stop at dry-run and ask for scope confirmation.
