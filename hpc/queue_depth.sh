#!/bin/bash
################################################################################
# hpc/queue_depth.sh — Report SLURM queue depth for a cluster.
#
# Used by hpc/ci_pr.sh load-balancer to choose between CHPC and RIS based on
# current contention. Output is a single integer (count of the operator's
# *pending* jobs on the target cluster — running jobs are excluded because
# they no longer compete for queue slots). 0 means idle; higher numbers mean
# longer expected wait.
#
# Usage:
#   bash hpc/queue_depth.sh chpc      # → 4
#   bash hpc/queue_depth.sh ris       # → 0
#   bash hpc/queue_depth.sh chpc cpu  # → 2 (filter by partition flavor)
#   bash hpc/queue_depth.sh ris gpu   # → 1
#
# Filter: when a second argument is supplied, only count jobs on partitions
# matching that flavor (cpu | gpu | short). Otherwise count all queued.
#
# State filter (2026-05-26):
#   The squeue invocation passes `-t PD` so RUNNING jobs are excluded. A
#   running job on cluster X does NOT block the operator's next submission
#   to X — only pending jobs do. Counting both used to produce ties whenever
#   the operator had exactly one running job on each cluster, which the
#   ci_pr.sh tiebreaker then resolved to CHPC by default; the result was a
#   new submission joining whatever CHPC backlog already existed instead of
#   the idle RIS partition. The PD-only count makes the probe measure
#   "blocking depth," which is the right signal for routing decisions.
#
# The probe uses an SSH connection per call. ControlPersist in ~/.ssh/config
# amortizes this across multiple calls (e.g., when ci_pr.sh probes both
# clusters back-to-back). When invoked from a nohup-backgrounded auto-fire
# context (hooks/pre-push's Pattern A′ HPC CI auto-fire), no ControlPersist
# socket exists in the detached subshell's environment — every probe is a
# cold-start handshake.
#
# Resilience (2026-05-28):
#   - ConnectTimeout=10 (was 5) — 5 s is insufficient margin for the
#     cold-start handshake from a detached subshell, especially when an
#     ssh-agent key needs to be selected from multiple candidates.
#   - Single retry with 2 s backoff before falling back to the sentinel —
#     protects against transient DNS / network blips that don't repeat.
#   - SSH stderr + sentinel-trigger context appended to
#     ${HOME}/.analytic_signal_sst/logs/queue_depth_errors.log on each
#     fallback, so the next sentinel firing has forensic evidence (the
#     pre-2026-05-28 code silenced the error and the operator had to
#     reverse-engineer the cause from sparse hints).
#
# Exit code: 0 on success (integer printed to stdout). Non-zero if SSH fails
# or the cluster is unreachable; in that case stdout prints a high sentinel
# (999) so the caller naturally picks the OTHER cluster.
################################################################################

set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source config.sh (which sources project.conf) for $DOTFOLDER. The
# HPC_TARGET detection runs eagerly; we override it in the per-cluster
# subshell below.
# shellcheck source=config.sh
. "${HPC_DIR}/config.sh" >/dev/null 2>&1 || true

CLUSTER="${1:?Usage: queue_depth.sh <chpc|ris> [<cpu|gpu|short>]}"
FLAVOR="${2:-all}"

case "$CLUSTER" in
  chpc|ris) ;;
  *)
    echo "ERROR: unknown cluster '$CLUSTER' (use chpc or ris)" >&2
    exit 2
    ;;
esac

# Resolve REMOTE (user@host) via hpc/config.sh — SPEC §6.4 forbids
# internal hostname literals in shell scripts. Subshell scope so this
# script's HPC_TARGET (if any) is not perturbed.
REMOTE="$(
  HPC_TARGET="$CLUSTER" bash -c "source '${HPC_DIR}/config.sh' >/dev/null 2>&1 && echo \"\$REMOTE\""
)"
if [[ -z "$REMOTE" ]]; then
  echo "ERROR: failed to resolve REMOTE for cluster '$CLUSTER' via hpc/config.sh" >&2
  exit 2
fi

# Per-cluster partition mapping (mirrors hpc/config.sh).
case "$CLUSTER:$FLAVOR" in
  chpc:cpu)   PARTITION_FILTER="tier1_cpu|free_cpu|general-cpu|general-short" ;;
  chpc:gpu)   PARTITION_FILTER="free_gpu" ;;
  chpc:short) PARTITION_FILTER="free_cpu|tier1_cpu" ;;
  ris:cpu)    PARTITION_FILTER="general-cpu" ;;
  ris:gpu)    PARTITION_FILTER="general-gpu" ;;
  ris:short)  PARTITION_FILTER="general-short" ;;
  *:all)      PARTITION_FILTER=".*" ;;
  *)
    echo "ERROR: unknown flavor '$FLAVOR' (use cpu|gpu|short|all)" >&2
    exit 2
    ;;
esac

# squeue: list this user's *pending* jobs; filter by partition; count rows.
# -h: no header. -u jjlee: this user only. -t PD: PENDING state only
# (running jobs don't block the next submission). -o "%P": partition column.
ERR_LOG="${DOTFOLDER}/logs/queue_depth_errors.log"
mkdir -p "$(dirname "$ERR_LOG")" 2>/dev/null || true

# Per-call SSH stderr trace — written only to the forensic log when the
# sentinel actually fires, so successful probes don't pollute the log with
# benign remote-shell startup banners (LMOD, MOTD, etc.).
SSH_TRACE="$(mktemp -t queue_depth_ssh.XXXXXX)"
trap 'rm -f "$SSH_TRACE"' EXIT

_probe_once() {
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE" \
    "squeue -u jjlee -h -t PD -o '%P' 2>/dev/null | grep -E '^(${PARTITION_FILTER})\$' | wc -l"
}

# First attempt; on any failure, DEPTH="" and we fall through to retry.
DEPTH="$(_probe_once 2>>"$SSH_TRACE")" || DEPTH=""

# Single retry with 2 s backoff if the first attempt produced empty or
# non-numeric output (transient DNS / network blip).
if [[ -z "$DEPTH" ]] || ! [[ "$DEPTH" =~ ^[0-9]+$ ]]; then
  printf '\n[%s] retry: cluster=%s flavor=%s prev_depth=%q\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CLUSTER" "$FLAVOR" "$DEPTH" >> "$SSH_TRACE"
  sleep 2
  DEPTH="$(_probe_once 2>>"$SSH_TRACE")" || DEPTH=""
fi

# Final fallback: still empty or non-numeric → sentinel 999. Dump the
# accumulated SSH trace to the forensic log so the next debugging session
# has concrete evidence (timestamp + remote shell stderr from both
# attempts).
if [[ -z "$DEPTH" ]] || ! [[ "$DEPTH" =~ ^[0-9]+$ ]]; then
  {
    printf '[%s] sentinel-999: cluster=%s flavor=%s final_depth=%q\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CLUSTER" "$FLAVOR" "$DEPTH"
    echo '--- ssh trace (both attempts) ---'
    cat "$SSH_TRACE"
    echo '--- end ssh trace ---'
  } >> "$ERR_LOG"
  DEPTH=999
fi

echo "$DEPTH"
