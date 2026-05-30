#!/bin/bash
################################################################################
# hpc/ci_pr.sh — MacBook-side driver for the HPC-based PR CI gate.
#
# Substrate for SPEC_development_workflow.md Pattern A′ (the HPC-CI variant of
# Pattern A, post 2026-05-25 Actions back-out). Replaces .github/workflows/ci.yml
# without using GitHub Actions runner minutes.
#
# Usage:
#   bash hpc/ci_pr.sh <branch>                       # Auto: load-balance + CPU
#   bash hpc/ci_pr.sh <branch> --gpu                 # Force a GPU shard too
#   bash hpc/ci_pr.sh <branch> --cluster=chpc        # Force CHPC (no LB)
#   bash hpc/ci_pr.sh <branch> --cluster=ris         # Force RIS (no LB)
#   bash hpc/ci_pr.sh <branch> --sha=<sha>           # Override target SHA
#   bash hpc/ci_pr.sh <branch> --dry-run             # Show plan, no submit
#
# What it does:
#   1. Resolves <branch> to a SHA (default: tip of origin/<branch>).
#   2. Probes CHPC + RIS queue depths via hpc/queue_depth.sh (skipped if
#      --cluster is forced).
#   3. Picks the cluster with the shorter CPU queue. Ties broken by config-
#      file preference (default: CHPC). For GPU runs, RIS preferred when both
#      are idle (per memory: RIS GPU gives 3-4x SST speedup).
#   4. Pushes the branch to that cluster via hpc/push.sh --integration-test.
#   5. Submits hpc/ci_pr.sbatch (CPU) and optionally hpc/ci_pr_gpu.sbatch
#      (GPU) on the chosen cluster via hpc/submit.sh.
#   6. Posts an initial "pending" commit status to GitHub via gh API.
#   7. Prints JOB_ID, log location, polling command, and how to verify in
#      the PR's Checks panel.
#
# The sbatch script itself posts the terminal status (success / failure) via
# hpc/post_pr_status.py when it exits. This driver returns once the submit
# is confirmed; the actual CI runs asynchronously.
#
# Authentication assumed:
#   - SSH to login3.chpc.wustl.edu and c2-login-001 (operator's keys).
#   - gh CLI authenticated on BOTH HPC hosts (one-time setup; see
#     hpc/post_pr_status.py header).
#
# Exit codes:
#   0  — submission successful (sbatch JOB_ID printed)
#   1  — argument parse error
#   2  — branch resolution / push failure
#   3  — sbatch submission failure
#   4  — gh status post failure (non-fatal: the SLURM job continues)
################################################################################

set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HPC_DIR}/.." && pwd)"

# Source config.sh (which sources project.conf) for PROJECT_NAME,
# CONDA_ENV, DEFAULT_BRANCH, GH_REPO, SINGLE_CLUSTER, etc. Required
# under `set -u`; without it the post-Phase-0 refactor references
# below trip unbound-variable.
# shellcheck source=config.sh
. "${HPC_DIR}/config.sh"

# ── Argument parsing ──────────────────────────────────────────────────────────

BRANCH=""
FORCE_CLUSTER=""
GPU_SHARD=false
OVERRIDE_SHA=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster=*) FORCE_CLUSTER="${1#*=}" ;;
    --cluster)   shift; FORCE_CLUSTER="${1:?--cluster requires chpc|ris}" ;;
    --gpu)       GPU_SHARD=true ;;
    --sha=*)     OVERRIDE_SHA="${1#*=}" ;;
    --sha)       shift; OVERRIDE_SHA="${1:?--sha requires 40-char hex}" ;;
    --dry-run)   DRY_RUN=true ;;
    -h|--help)
      sed -n '3,40p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    -*)
      echo "ERROR: unknown flag '$1'" >&2
      exit 1
      ;;
    *)
      if [[ -z "$BRANCH" ]]; then
        BRANCH="$1"
      else
        echo "ERROR: unexpected positional arg '$1'" >&2
        exit 1
      fi
      ;;
  esac
  shift
done

if [[ -z "$BRANCH" ]]; then
  echo "Usage: ci_pr.sh <branch> [--cluster=chpc|ris] [--gpu] [--sha=<sha>] [--dry-run]" >&2
  exit 1
fi

# ── Branch / SHA resolution ───────────────────────────────────────────────────

cd "$PROJECT_ROOT"

# Sanity: refuse to gate the default branch; this script is for PR branches.
if [[ "$BRANCH" == "${DEFAULT_BRANCH}" ]]; then
  echo "REFUSED: ci_pr.sh is for PR gates, not ${DEFAULT_BRANCH}. Pattern A′ gates PR branches; ${DEFAULT_BRANCH} is post-merge." >&2
  exit 1
fi

if [[ -n "$OVERRIDE_SHA" ]]; then
  TARGET_SHA="$OVERRIDE_SHA"
else
  # Prefer origin/<branch> SHA so the gate matches what GitHub sees.
  if git rev-parse --verify "origin/${BRANCH}" >/dev/null 2>&1; then
    TARGET_SHA="$(git rev-parse "origin/${BRANCH}")"
  elif git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    TARGET_SHA="$(git rev-parse "$BRANCH")"
    echo "[ci_pr] WARNING: origin/${BRANCH} not found; using local ref tip ${TARGET_SHA:0:8}" >&2
  else
    echo "ERROR: branch '${BRANCH}' not found locally or on origin" >&2
    exit 2
  fi
fi

echo "[ci_pr] branch=${BRANCH} sha=${TARGET_SHA:0:8} gpu=${GPU_SHARD} dry-run=${DRY_RUN}"

# ── Cluster selection ─────────────────────────────────────────────────────────
# Selection precedence (first match wins):
#   1. --cluster=<name> CLI flag (operator override)
#   2. SINGLE_CLUSTER from hpc/project.conf (chpc-only / ris-only projects;
#      brainsmash* set this to "chpc")
#   3. Queue-depth load balancing across both clusters (analytic_signal_sst
#      default — SINGLE_CLUSTER empty)

if [[ -n "$FORCE_CLUSTER" ]]; then
  case "$FORCE_CLUSTER" in
    chpc|ris) CLUSTER="$FORCE_CLUSTER" ;;
    *) echo "ERROR: --cluster must be chpc or ris, got '${FORCE_CLUSTER}'" >&2; exit 1 ;;
  esac
  echo "[ci_pr] cluster=${CLUSTER} (forced via --cluster)"
elif [[ -n "${SINGLE_CLUSTER:-}" ]]; then
  case "$SINGLE_CLUSTER" in
    chpc|ris) CLUSTER="$SINGLE_CLUSTER" ;;
    *) echo "ERROR: SINGLE_CLUSTER in project.conf must be chpc or ris, got '${SINGLE_CLUSTER}'" >&2; exit 1 ;;
  esac
  echo "[ci_pr] cluster=${CLUSTER} (project.conf SINGLE_CLUSTER; load-balancing skipped)"
else
  CHPC_DEPTH=$(bash "${HPC_DIR}/queue_depth.sh" chpc cpu)
  RIS_DEPTH=$(bash "${HPC_DIR}/queue_depth.sh" ris cpu)
  echo "[ci_pr] queue depths: chpc=${CHPC_DEPTH} ris=${RIS_DEPTH}"

  # Tie-break preference: CHPC for CPU (tier1_cpu free + closer to /ceph data
  # cache). Override via $CI_PR_CLUSTER_PREF env var if needed.
  PREF="${CI_PR_CLUSTER_PREF:-chpc}"
  if [[ "$CHPC_DEPTH" -lt "$RIS_DEPTH" ]]; then
    CLUSTER="chpc"
  elif [[ "$RIS_DEPTH" -lt "$CHPC_DEPTH" ]]; then
    CLUSTER="ris"
  else
    CLUSTER="$PREF"
  fi
  echo "[ci_pr] cluster=${CLUSTER} (chosen by queue-depth + pref=${PREF})"
fi

# GPU shard cluster: RIS preferred when GPU is requested (memory:
# ris_gpu_favored_for_precompute — 3-4x speedup over CHPC free_gpu).
GPU_CLUSTER=""
if [[ "$GPU_SHARD" == "true" ]]; then
  if [[ -n "$FORCE_CLUSTER" ]]; then
    GPU_CLUSTER="$FORCE_CLUSTER"
  else
    RIS_GPU_DEPTH=$(bash "${HPC_DIR}/queue_depth.sh" ris gpu)
    CHPC_GPU_DEPTH=$(bash "${HPC_DIR}/queue_depth.sh" chpc gpu)
    echo "[ci_pr] gpu queue depths: chpc=${CHPC_GPU_DEPTH} ris=${RIS_GPU_DEPTH}"
    # Prefer RIS even at slight depth disadvantage (≤2 jobs queue tolerance).
    if [[ $((RIS_GPU_DEPTH - 2)) -le "$CHPC_GPU_DEPTH" ]]; then
      GPU_CLUSTER="ris"
    else
      GPU_CLUSTER="chpc"
    fi
  fi
  echo "[ci_pr] gpu_cluster=${GPU_CLUSTER}"
fi

# ── Dry-run preview ───────────────────────────────────────────────────────────

if [[ "$DRY_RUN" == "true" ]]; then
  echo
  echo "Would push branch '${BRANCH}' to cluster '${CLUSTER}' (bare repo, --integration-test)."
  if [[ -n "$GPU_CLUSTER" && "$GPU_CLUSTER" != "$CLUSTER" ]]; then
    echo "Would push branch '${BRANCH}' to cluster '${GPU_CLUSTER}' (bare repo, --integration-test)."
  fi
  echo "Would push branch '${BRANCH}' to origin (idempotent — required for status post)."
  echo "Would submit hpc/ci_pr.sbatch on cluster '${CLUSTER}'."
  if [[ -n "$GPU_CLUSTER" ]]; then
    echo "Would also submit hpc/ci_pr_gpu.sbatch on cluster '${GPU_CLUSTER}'."
  fi
  echo "Would post commit status (AFTER pushes + submits): state=pending sha=${TARGET_SHA:0:8} context=hpc-ci"
  exit 0
fi

# ── Push branch to chosen cluster(s) ──────────────────────────────────────────

echo "[ci_pr] pushing branch to ${CLUSTER}..."
# SKIP_HPC_CI=1 stops the pre-push auto-fire from running ci_pr.sh again
# (we ARE ci_pr.sh — would be infinite recursion otherwise). Belt-and-
# suspenders with hooks/pre-push's $1=="origin" check, but explicit at
# the call site so the protection survives any future hook refactor.
# No `| tail -5` so the operator sees live output (tail buffers until EOF).
SKIP_HPC_CI=1 HPC_TARGET="$CLUSTER" bash "${HPC_DIR}/push.sh" --source-only \
  --integration-test "$BRANCH" --project "$PROJECT_NAME" 2>&1 || {
    echo "ERROR: push to ${CLUSTER} failed" >&2
    exit 2
}

if [[ -n "$GPU_CLUSTER" && "$GPU_CLUSTER" != "$CLUSTER" ]]; then
  echo "[ci_pr] pushing branch to ${GPU_CLUSTER} for GPU shard..."
  SKIP_HPC_CI=1 HPC_TARGET="$GPU_CLUSTER" bash "${HPC_DIR}/push.sh" --source-only \
    --integration-test "$BRANCH" --project "$PROJECT_NAME" 2>&1 || {
      echo "ERROR: push to ${GPU_CLUSTER} failed" >&2
      exit 2
  }
fi

# Idempotent push to GitHub origin: the SHA MUST be on origin before any
# commit-status post against that SHA, or GitHub responds 422 "No commit
# found for SHA". hpc/push.sh only reaches the CHPC/RIS bare repos; if
# ci_pr.sh is invoked directly (not via the pre-push hook that already
# pushed to origin), origin may be behind. `git push origin <branch>`
# is a no-op when up-to-date.
if git remote get-url origin >/dev/null 2>&1; then
  echo "[ci_pr] ensuring origin/${BRANCH} carries ${TARGET_SHA:0:8} (idempotent)..."
  git push origin "$BRANCH" 2>&1 || {
    echo "[ci_pr] WARNING: git push origin ${BRANCH} failed; the pending status post may 422" >&2
  }
fi

# ── Submit CPU sbatch ─────────────────────────────────────────────────────────

echo "[ci_pr] submitting hpc/ci_pr.sbatch on ${CLUSTER}..."
# submit.sh's SSH `cd ${HPC_PROJECT} && sbatch ... ${SCRIPT}` resolves
# ${SCRIPT} relative to the HPC project root — pass the repo-relative
# path, not the MacBook absolute one. ("sbatch: Unable to open file
# <macbook-absolute-path>" is the symptom of getting this wrong.)
SUBMIT_OUT=$(HPC_TARGET="$CLUSTER" bash "${HPC_DIR}/submit.sh" hpc/ci_pr.sbatch \
  --export="ALL,CI_BRANCH=${BRANCH},CI_SHA=${TARGET_SHA},CI_GH_REPO=${GH_REPO}" \
  --job-name="ci_pr_${BRANCH//\//_}_${TARGET_SHA:0:8}" \
  2>&1) || {
    echo "ERROR: sbatch submission failed:" >&2
    echo "$SUBMIT_OUT" >&2
    exit 3
}
CPU_JOB_ID=$(echo "$SUBMIT_OUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['job_id'])" 2>/dev/null || true)
echo "[ci_pr] CPU JOB_ID=${CPU_JOB_ID:-unknown}"

# ── Optional GPU sbatch ───────────────────────────────────────────────────────

GPU_JOB_ID=""
if [[ -n "$GPU_CLUSTER" ]]; then
  echo "[ci_pr] submitting hpc/ci_pr_gpu.sbatch on ${GPU_CLUSTER}..."
  GPU_SUBMIT_OUT=$(HPC_TARGET="$GPU_CLUSTER" bash "${HPC_DIR}/submit.sh" hpc/ci_pr_gpu.sbatch \
    --export="ALL,CI_BRANCH=${BRANCH},CI_SHA=${TARGET_SHA},CI_GH_REPO=${GH_REPO}" \
    --job-name="ci_pr_gpu_${BRANCH//\//_}_${TARGET_SHA:0:8}" \
    2>&1) || {
      echo "ERROR: GPU sbatch submission failed:" >&2
      echo "$GPU_SUBMIT_OUT" >&2
      # Don't abort the CPU job; just skip GPU.
      GPU_JOB_ID=""
  }
  if [[ -z "$GPU_JOB_ID" ]]; then
    GPU_JOB_ID=$(echo "$GPU_SUBMIT_OUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['job_id'])" 2>/dev/null || true)
  fi
  echo "[ci_pr] GPU JOB_ID=${GPU_JOB_ID:-unknown}"
fi

# ── Post initial 'pending' status ─────────────────────────────────────────────

# The MacBook posts the pending state immediately (the SLURM job's prolog could
# also post, but doing it here gives instant feedback in the PR UI). The
# sbatch's epilog posts the terminal state. Local gh CLI must be authenticated
# (it is, per operator's existing setup).
#
# Order matters: this post runs AFTER the HPC bare-repo push, the optional
# origin push (above), and both sbatch submissions. That ensures the SHA
# already exists on origin (the idempotent `git push origin` above), so the
# GitHub Statuses API doesn't 422 "No commit found for SHA". A 422 here is
# only cosmetic (the sbatch epilog still posts the terminal status), but the
# warning the operator sees is noisy.

TARGET_URL_DESC="logs/ci_pr_${CPU_JOB_ID:-pending}/ on ${CLUSTER}"
DESC="queued: cpu_job=${CPU_JOB_ID:-?}"
if [[ -n "$GPU_CLUSTER" ]]; then
  DESC="${DESC} gpu_job=${GPU_JOB_ID:-?}"
fi
if ! python3 "${HPC_DIR}/post_pr_status.py" \
    --sha "$TARGET_SHA" \
    --state pending \
    --context hpc-ci \
    --description "${DESC}" \
    2>&1; then
  echo "[ci_pr] WARNING: failed to post pending status (sbatch will still run; status will be posted from HPC)" >&2
fi

# ── Summary ───────────────────────────────────────────────────────────────────

cat <<EOF

[ci_pr] submitted.
  branch:           ${BRANCH}
  sha:              ${TARGET_SHA}
  cpu_job:          ${CPU_JOB_ID:-unknown} on ${CLUSTER}
  gpu_job:          ${GPU_JOB_ID:-N/A} on ${GPU_CLUSTER:-N/A}
  pending_status:   posted (visible in PR Checks panel as 'hpc-ci')
  poll_command:     bash hpc/status.sh ${CPU_JOB_ID:-} ${GPU_JOB_ID:-}
  log_target:       ${TARGET_URL_DESC}

The SLURM job will post the terminal status (success / failure) when it exits.
PR will become mergeable when 'hpc-ci' goes green (assuming branch protection
requires that context — see ../specs/infrastructure/SPEC_development_workflow.md
Pattern A′).
EOF
