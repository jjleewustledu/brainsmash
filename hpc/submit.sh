#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SCRIPT="${1:?Usage: submit.sh <slurm_script> [extra sbatch args]}"; shift
SUBMITTED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Determine project root relative to this script (analytic_signal_sst root).
PROJECT_ROOT="$(cd "${HPC_DIR}/.." && pwd)"

# Resolve the canonical (primary-worktree) project name. A secondary
# worktree's basename carries a branch slug that matches no HPC checkout
# path; see resolve_project_name() in config.sh.
PROJECT_NAME="$(resolve_project_name "$PROJECT_ROOT")"
HPC_PROJECT="${HPC_PROJECTS}/${PROJECT_NAME}"

# Detect GPU script to select appropriate partition
if grep -q '^#SBATCH.*--gres=gpu' "$SCRIPT" 2>/dev/null; then
  SBATCH_PARTITION="${PARTITION_GPU}"
else
  SBATCH_PARTITION="${PARTITION_CPU}"
fi

# Inject container arguments when support is enabled and image is configured
CONTAINER_ARGS=""
if [ "$CONTAINER_SUPPORT" = "true" ] && [ -n "$CONTAINER_IMAGE" ]; then
  CONTAINER_ARGS="--container-image=${CONTAINER_IMAGE}"
  if [ -n "$CONTAINER_MOUNTS" ]; then
    CONTAINER_ARGS="${CONTAINER_ARGS} --container-mounts=${CONTAINER_MOUNTS}"
  fi
fi

# Inject cluster-specific partition and account as sbatch overrides
# (command-line args override #SBATCH directives in the script)
# shellcheck disable=SC2029
JOB_ID=$(ssh "$REMOTE" \
  "cd ${HPC_PROJECT} && sbatch --partition=${SBATCH_PARTITION} --account=${SLURM_ACCOUNT} ${CONTAINER_ARGS} $* ${SCRIPT}" \
  | awk '{print $NF}')

if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
  echo '{"error":"sbatch did not return a numeric job ID","raw":"'"$JOB_ID"'"}' >&2
  exit 1
fi

# Detect array job by grepping the script for #SBATCH --array
if grep -q '^#SBATCH.*--array' "$SCRIPT" 2>/dev/null; then
  IS_ARRAY="true"
else
  IS_ARRAY="false"
fi

# -------------------------------------------------------------------------
# Post-submission partition audit (lesson from jobs 6634306 / 6638323
# where the on-disk script's partition diverged from what SLURM actually
# ran). For each successful submission, verify the controller-side
# Partition for the new job matches the partition we requested. Mismatch
# indicates upstream drift between hpc/config.sh and the cluster's actual
# partition naming, or a wrapper bypass.
# -------------------------------------------------------------------------
mkdir -p "${PROJECT_ROOT}/logs"
AUDIT_LOG="${PROJECT_ROOT}/logs/submit_audit_${JOB_ID}.json"

# sacct may not record a brand-new job for a few seconds; allow brief retry.
ACTUAL_PARTITION=""
for _ in 1 2 3 4 5; do
  ACTUAL_PARTITION=$(ssh "$REMOTE" \
    "sacct -j ${JOB_ID} -X --format=Partition --noheader 2>/dev/null | head -1 | awk '{print \$1}'" \
    2>/dev/null || true)
  if [ -n "$ACTUAL_PARTITION" ]; then break; fi
  sleep 2
done

if [ -z "$ACTUAL_PARTITION" ]; then
  AUDIT_STATUS="warning"
  AUDIT_MSG="sacct did not report a Partition for ${JOB_ID} after 5 retries; cannot verify"
elif [ "$ACTUAL_PARTITION" = "$SBATCH_PARTITION" ]; then
  AUDIT_STATUS="ok"
  AUDIT_MSG="partition verified: ${ACTUAL_PARTITION}"
else
  AUDIT_STATUS="mismatch"
  AUDIT_MSG="requested=${SBATCH_PARTITION} but slurmctld assigned partition=${ACTUAL_PARTITION}"
  echo "WARNING (submit.sh): ${AUDIT_MSG}" >&2
  echo "  Inspect hpc/config.sh::PARTITION_CPU / PARTITION_GPU and the cluster's partition list (sinfo -h -o '%P')." >&2
fi

cat > "$AUDIT_LOG" <<EOF
{"job_id":"${JOB_ID}","script":"${SCRIPT}","submitted_at":"${SUBMITTED_AT}","remote":"${REMOTE}","project":"${PROJECT_NAME}","is_array":${IS_ARRAY},"requested_partition":"${SBATCH_PARTITION}","actual_partition":"${ACTUAL_PARTITION}","slurm_account":"${SLURM_ACCOUNT}","audit_status":"${AUDIT_STATUS}","audit_message":"${AUDIT_MSG}"}
EOF

echo "{\"job_id\":\"${JOB_ID}\",\"script\":\"${SCRIPT}\",\"submitted_at\":\"${SUBMITTED_AT}\",\"remote\":\"${REMOTE}\",\"project\":\"${PROJECT_NAME}\",\"is_array\":${IS_ARRAY},\"requested_partition\":\"${SBATCH_PARTITION}\",\"actual_partition\":\"${ACTUAL_PARTITION}\",\"audit_status\":\"${AUDIT_STATUS}\",\"audit_log\":\"${AUDIT_LOG}\"}"
