#!/bin/bash
#SBATCH --job-name=ci_pr
#SBATCH --output=logs/ci_pr_%j/console.log
#SBATCH --error=logs/ci_pr_%j/console.log
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#
# hpc-toolkit ci_pr.sbatch — Pattern A′ HPC PR CI gate (CPU shard).
#
# Runs $TEST_CMD (from hpc/project.conf) against the SHA exported by
# hpc/ci_pr.sh. Posts pending → success|failure to GitHub via
# hpc/post_pr_status.py. Project-specific extra gates (entropy scanners,
# manifest validation, math-docstring lint, etc.) MAY live in a sibling
# file `hpc/ci_pr_extras.sh` — when present, it is sourced near the end
# of this script so its exit code factors into the overall verdict.
#
# IMPORTANT: --partition= / --account= are injected by hpc/submit.sh.
# Do NOT add #SBATCH --partition / --account lines here.
#
# Expected environment from ci_pr.sh --export:
#   CI_BRANCH  — feature branch name being gated (for log clarity)
#   CI_SHA     — 40-char commit SHA to post statuses against
#   CI_GH_REPO — <owner>/<repo> form for the gh API

set -uo pipefail   # NOT -e: stage failures must NOT abort; we accumulate
                   # the verdict in OVERALL.

CI_BRANCH="${CI_BRANCH:?ERROR: CI_BRANCH not exported}"
CI_SHA="${CI_SHA:?ERROR: CI_SHA not exported}"
CI_GH_REPO="${CI_GH_REPO:?ERROR: CI_GH_REPO not exported}"
export GH_REPO="$CI_GH_REPO"

PROJECT_ROOT="$PWD"

# Source hpc/config.sh (which sources project.conf) for CONDA_PREFIX_HPC,
# CONDA_ENV, TEST_CMD, etc.
# shellcheck disable=SC1091
. "${PROJECT_ROOT}/hpc/config.sh"

LOG_DIR="${PROJECT_ROOT}/logs/ci_pr_${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_DIR"

echo "===== hpc/ci_pr.sbatch ====="
echo "  job_id:    ${SLURM_JOB_ID:-local}"
echo "  branch:    ${CI_BRANCH}"
echo "  sha:       ${CI_SHA}"
echo "  repo:      ${CI_GH_REPO}"
echo "  cluster:   $(hostname -f 2>/dev/null || hostname)"
echo "  conda:     ${CONDA_ENV} @ ${CONDA_PREFIX_HPC}"
echo "  started:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# ── Activate conda ────────────────────────────────────────────────────────────

# shellcheck disable=SC1091
if [[ -f "${CONDA_PREFIX_HPC}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_PREFIX_HPC}/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found at ${CONDA_PREFIX_HPC}/etc/profile.d/conda.sh" >&2
  python3 "${PROJECT_ROOT}/hpc/post_pr_status.py" \
    --sha "$CI_SHA" --state error --context "${BRANCH_PROTECTION_CONTEXT:-hpc-ci}" \
    --description "conda missing on $(hostname)" || true
  exit 1
fi
conda activate "${CONDA_ENV}"

# Ensure `gh` is on PATH (operator's gh binary lives at ~/.local/bin/gh).
export PATH="$HOME/.local/bin:$PATH"

# Refresh the editable install — keeps deps in sync with the pushed branch.
pip install -e "${PROJECT_ROOT}" > "${LOG_DIR}/install.log" 2>&1 || {
  echo "ERROR: pip install -e failed; see ${LOG_DIR}/install.log" >&2
  python3 "${PROJECT_ROOT}/hpc/post_pr_status.py" \
    --sha "$CI_SHA" --state error --context "${BRANCH_PROTECTION_CONTEXT:-hpc-ci}" \
    --description "pip install -e failed on $(hostname)" || true
  exit 1
}

# Sanity: HEAD should match CI_SHA. If the shared-WT topology raced us,
# log loudly but continue (we still test what's on disk).
ACTUAL_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ "$ACTUAL_HEAD" != "$CI_SHA" ]]; then
  echo "WARNING: HEAD ${ACTUAL_HEAD:0:8} != CI_SHA ${CI_SHA:0:8} — possible shared-WT race" >&2
fi

# ── Run the project's test command ────────────────────────────────────────────
# TEST_CMD is a single shell-word from hpc/project.conf (e.g.
# `pytest -m "unit" -q`). bash -c "$TEST_CMD" parses the quoting correctly;
# eval would also work but is more error-prone with operator-supplied strings.

echo "----- running TEST_CMD -----"
echo "  cmd: ${TEST_CMD}"
TEST_STARTED=$(date +%s)
bash -c "${TEST_CMD}" > "${LOG_DIR}/test.log" 2>&1
TEST_RC=$?
TEST_ELAPSED=$(( $(date +%s) - TEST_STARTED ))

if [[ $TEST_RC -eq 0 ]]; then
  echo "  ok  TEST_CMD passed in ${TEST_ELAPSED}s"
else
  echo "  FAIL  TEST_CMD rc=${TEST_RC} after ${TEST_ELAPSED}s — see ${LOG_DIR}/test.log"
  tail -40 "${LOG_DIR}/test.log" | sed 's/^/    /'
fi

OVERALL=$TEST_RC
FAILED_STAGES=()
[[ $TEST_RC -ne 0 ]] && FAILED_STAGES+=("test_cmd")

# ── Optional project-specific extras (entropy scanners, manifest validation, …)
# Project may provide hpc/ci_pr_extras.sh that runs additional gates against
# the same shell environment. Its exit code is added to OVERALL.

if [ -f "${PROJECT_ROOT}/hpc/ci_pr_extras.sh" ]; then
  echo "----- running hpc/ci_pr_extras.sh -----"
  # shellcheck source=../ci_pr_extras.sh
  . "${PROJECT_ROOT}/hpc/ci_pr_extras.sh"
  EXTRAS_RC=$?
  if [[ $EXTRAS_RC -ne 0 ]]; then
    echo "  FAIL  ci_pr_extras.sh rc=${EXTRAS_RC}"
    OVERALL=$EXTRAS_RC
    FAILED_STAGES+=("extras")
  else
    echo "  ok  ci_pr_extras.sh passed"
  fi
fi

# ── Write summary.json ────────────────────────────────────────────────────────

python3 - <<PY > "${LOG_DIR}/summary.json"
import json
data = {
    "branch": "${CI_BRANCH}",
    "sha": "${CI_SHA}",
    "job_id": "${SLURM_JOB_ID:-local}",
    "cluster_host": "$(hostname -f 2>/dev/null || hostname)",
    "test_cmd": ${TEST_CMD@Q},
    "test_rc": ${TEST_RC},
    "test_elapsed_s": ${TEST_ELAPSED},
    "overall_rc": ${OVERALL},
    "failed_stages": [$(printf '"%s",' "${FAILED_STAGES[@]}" | sed 's/,$//')],
}
print(json.dumps(data, indent=2))
PY

echo
echo "===== summary ====="
cat "${LOG_DIR}/summary.json"
echo

# ── Post terminal status to GitHub ────────────────────────────────────────────

if [[ $OVERALL -eq 0 ]]; then
  STATE="success"
  DESC="all stages passed"
else
  STATE="failure"
  F=${#FAILED_STAGES[@]}
  DESC="${F} stage(s) failed: ${FAILED_STAGES[*]:0:3}"
fi

python3 "${PROJECT_ROOT}/hpc/post_pr_status.py" \
  --sha "$CI_SHA" \
  --state "$STATE" \
  --context "${BRANCH_PROTECTION_CONTEXT:-hpc-ci}" \
  --description "$DESC" \
  --repo "$CI_GH_REPO" || {
    echo "WARNING: post_pr_status.py failed; status not updated on GitHub" >&2
}

echo "===== finished ====="
echo "  ended:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  verdict:   ${STATE}"
echo "  log_dir:   ${LOG_DIR}"
exit "$OVERALL"
