#!/bin/bash
################################################################################
# hpc/fetch.sh — Rsync results from HPC to local working directory, print summary
#
# Usage:
#   bash hpc/fetch.sh <results_subdir> [job_id] [extra_dir ...]
#
#   results_subdir  relative path under the HPC project root, e.g. results/12345
#   job_id          optional; included in warning JSON if summary.json is absent
#   extra_dir       optional additional relative subdirs to fetch (e.g. `logs
#                   benchmarks`) for jobs whose output does not live under
#                   results/<JOBID>/ — array benchmark / precompute sbatches
#                   write to logs/ and benchmarks/ instead.
#
# Behaviour:
#   1. Rsyncs <HPC_PROJECT>/<dir>/ → ./<dir>/ for results_subdir and each
#      extra_dir. A dir absent on the remote is skipped with a notice
#      (not an error).
#   2. If <results_subdir>/summary.json exists locally after the sync, prints
#      it to stdout so callers (wait_and_fetch.sh, Claude Code) can parse it.
#   3. If summary.json is absent, prints a warning JSON to stdout instead.
#
# The HPC project root is ${HPC_PROJECTS}/<canonical-project-name>, where the
# project name is resolved from the *primary* git worktree (see
# resolve_project_name() in config.sh) — so a fetch invoked from a secondary
# worktree still targets the canonical HPC checkout rather than a
# worktree-slug path that does not exist on HPC.
#
# Called automatically by wait_and_fetch.sh on job completion or failure.
# May also be called manually to re-fetch results for a finished job.
################################################################################

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

RESULTS_DIR="${1:?Usage: fetch.sh <results_subdir> [job_id] [extra_dir ...]}"
JOB_ID="${2:-unknown}"
EXTRA_DIRS=("${@:3}")

# ── Derive HPC project root (worktree-aware; see config.sh) ──────────────────
PROJECT_ROOT="$(cd "${HPC_DIR}/.." && pwd)"
PROJECT_NAME="$(resolve_project_name "$PROJECT_ROOT")"
HPC_PROJECT="${HPC_PROJECTS}/${PROJECT_NAME}"

# ── Fetch each requested subdir (results_subdir first, then any extras) ──────
for d in "$RESULTS_DIR" ${EXTRA_DIRS[@]+"${EXTRA_DIRS[@]}"}; do
  remote_path="${HPC_PROJECT}/${d}"
  if ! ssh "$REMOTE" "test -d '${remote_path}'" 2>/dev/null; then
    echo "==> skip: ${REMOTE}:${remote_path}/ (absent on remote)"
    continue
  fi
  echo "==> Fetching: ${REMOTE}:${remote_path}/  →  ./${d}/"
  mkdir -p "./${d}"
  rsync -avz "${REMOTE}:${remote_path}/" "./${d}/"
done

# ── Print summary.json if present ─────────────────────────────────────────────
SUMMARY="${RESULTS_DIR}/summary.json"

if [[ -f "$SUMMARY" ]]; then
  echo "==> summary.json:"
  cat "$SUMMARY"
else
  echo "{\"warning\":\"No summary.json found in ${RESULTS_DIR}\",\"job_id\":\"${JOB_ID}\",\"hpc_path\":\"${HPC_PROJECT}/${RESULTS_DIR}\"}"
fi
