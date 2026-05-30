#!/bin/bash
#SBATCH --job-name=ci_pr_gpu
#SBATCH --output=logs/ci_pr_gpu_%j/console.log
#SBATCH --error=logs/ci_pr_gpu_%j/console.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#
# hpc-toolkit ci_pr_gpu.sbatch — Pattern A′ HPC PR CI gate (GPU shard).
#
# Runs $TEST_CMD_GPU (from hpc/project.conf) against the SHA exported by
# hpc/ci_pr.sh. Posts to a SEPARATE status context (default `hpc-ci-gpu`)
# so it doesn't race the CPU shard's `hpc-ci` post. Submitted only when
# ci_pr.sh receives `--gpu` (typically: PRs touching GPU code paths).
#
# Submission contract: --partition / --account injected by hpc/submit.sh.
# Do NOT add #SBATCH --partition/--account here.

set -uo pipefail

CI_BRANCH="${CI_BRANCH:?ERROR: CI_BRANCH not exported}"
CI_SHA="${CI_SHA:?ERROR: CI_SHA not exported}"
CI_GH_REPO="${CI_GH_REPO:?ERROR: CI_GH_REPO not exported}"
export GH_REPO="$CI_GH_REPO"

PROJECT_ROOT="$PWD"

# shellcheck disable=SC1091
. "${PROJECT_ROOT}/hpc/config.sh"

LOG_DIR="${PROJECT_ROOT}/logs/ci_pr_gpu_${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_DIR"

# Status context for GPU shard. Defaults to "<BRANCH_PROTECTION_CONTEXT>-gpu";
# operator can override via project.conf's GPU_CONTEXT.
GPU_CONTEXT="${GPU_CONTEXT:-${BRANCH_PROTECTION_CONTEXT:-hpc-ci}-gpu}"

echo "===== hpc/ci_pr_gpu.sbatch ====="
echo "  job_id:    ${SLURM_JOB_ID:-local}"
echo "  branch:    ${CI_BRANCH}"
echo "  sha:       ${CI_SHA}"
echo "  gpu_node:  $(hostname -f 2>/dev/null || hostname)"
echo "  context:   ${GPU_CONTEXT}"
nvidia-smi -L 2>&1 | head -5 || echo "(nvidia-smi unavailable)"
echo "  started:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Activate conda ────────────────────────────────────────────────────────────

# shellcheck disable=SC1091
if [[ -f "${CONDA_PREFIX_HPC}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_PREFIX_HPC}/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found at ${CONDA_PREFIX_HPC}/etc/profile.d/conda.sh" >&2
  python3 "${PROJECT_ROOT}/hpc/post_pr_status.py" \
    --sha "$CI_SHA" --state error --context "$GPU_CONTEXT" \
    --description "conda missing on $(hostname)" || true
  exit 1
fi
conda activate "${CONDA_ENV}"
export PATH="$HOME/.local/bin:$PATH"

pip install -e "${PROJECT_ROOT}" > "${LOG_DIR}/install.log" 2>&1 || {
  echo "ERROR: pip install -e failed; see ${LOG_DIR}/install.log" >&2
  python3 "${PROJECT_ROOT}/hpc/post_pr_status.py" \
    --sha "$CI_SHA" --state error --context "$GPU_CONTEXT" \
    --description "pip install -e failed on $(hostname)" || true
  exit 1
}

# ── Run the project's GPU test command ────────────────────────────────────────

if [[ -z "${TEST_CMD_GPU:-}" ]]; then
  echo "[ci_pr_gpu] TEST_CMD_GPU is empty in hpc/project.conf — nothing to run; treating as pass"
  RC=0
else
  echo "----- running TEST_CMD_GPU -----"
  echo "  cmd: ${TEST_CMD_GPU}"
  bash -c "${TEST_CMD_GPU}" > "${LOG_DIR}/pytest.log" 2>&1
  RC=$?
  if [[ $RC -eq 5 ]]; then
    # pytest exit 5 = no tests collected; treat as pass.
    echo "[ci_pr_gpu] no GPU-marked tests collected; treating as pass"
    RC=0
  fi
fi

# ── Post terminal status ──────────────────────────────────────────────────────

if [[ $RC -eq 0 ]]; then
  STATE="success"; DESC="gpu shard passed"
else
  STATE="failure"; DESC="gpu shard failed (rc=${RC}); see ${LOG_DIR}"
fi

python3 "${PROJECT_ROOT}/hpc/post_pr_status.py" \
  --sha "$CI_SHA" \
  --state "$STATE" \
  --context "$GPU_CONTEXT" \
  --description "$DESC" \
  --repo "$CI_GH_REPO" || echo "WARNING: status post failed" >&2

echo "===== finished ====="
echo "  ended:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  verdict:   ${STATE}"
exit $RC
