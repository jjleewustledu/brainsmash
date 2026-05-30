#!/bin/bash
################################################################################
# hpc/fanout_main.sh — Mirror origin/main to chpc, ris, and sandbox remotes.
#
# Idempotent. Safe to invoke from cron, launchd, manual shell, or a wrapper
# script. Designed as the post-merge "fan-out" automation under Pattern A′
# (specs/infrastructure/SPEC_development_workflow.md §9), replacing the manual
# four-command sequence the operator previously ran by hand after every
# squash-merge.
#
# Usage:
#   bash hpc/fanout_main.sh                 # quiet poll mode (good for cron)
#   bash hpc/fanout_main.sh --now           # human-eyes mode: more verbose
#   bash hpc/fanout_main.sh --from-launchd  # tagged for launchd's log files
#   bash hpc/fanout_main.sh --dry-run       # show plan, no pushes
#
# Behaviour:
#   1. git fetch origin (quiet).
#   2. If local `main` == origin/main AND every HPC remote's `main` ==
#      origin/main: log "nothing to fan out" and exit 0.
#   3. Otherwise, fast-forward local `main` from origin/main. Abort with a
#      loud error if it would NOT be fast-forward (means local diverged;
#      operator must intervene). Local HEAD on a non-main branch stays
#      where it is — we use `git fetch origin main:main` style so the
#      working tree isn't touched.
#   4. For each remote whose `main` lags origin/main: push with
#      `--no-verify --force-with-lease`. --no-verify is appropriate because
#      the SHA on `main` already passed pre-merge gating; --force-with-lease
#      is defensive (refuses if the HPC bare's tip drifted from what we
#      last fetched, never blind-overwrites).
#   5. Log each push outcome to ~/.analytic_signal_sst/logs/fanout.log.
#   6. Exit 0 on no-op or all-succeed; non-zero if any push failed. The
#      launchd job will retry on the next 10-minute tick.
#
# Remotes targeted (configurable via FANOUT_REMOTES env var; default list
# below): chpc, ris, sandbox. Any remote not present locally is silently
# skipped — the script is fail-safe across machines that don't have all
# three configured.
#
# Concurrency: per-process flock on a sentinel file at
# ~/.analytic_signal_sst/locks/fanout.lock so two concurrent invocations
# (e.g., launchd tick + operator manual run + scripts/merge_pr.sh wrapper
# all firing within seconds) don't race.
#
# Spec: SPEC_development_workflow.md §9 (post-merge fan-out); Pattern A′.
################################################################################

set -uo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────

MODE="quiet"
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --now)           MODE="now" ;;
    --from-launchd)  MODE="launchd" ;;
    --dry-run)       DRY_RUN=true ;;
    -h|--help)
      sed -n '3,38p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg '$arg'" >&2
      exit 1
      ;;
  esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────

# Repo root is the parent of this script's directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Source config.sh (which sources project.conf) for DOTFOLDER,
# DEFAULT_BRANCH, FANOUT_REMOTES. Required under `set -u`.
# shellcheck source=config.sh
. "${REPO_ROOT}/hpc/config.sh"

LOG_DIR="${DOTFOLDER}/logs"
LOG_FILE="${LOG_DIR}/fanout.log"
LOCK_DIR="${DOTFOLDER}/locks"
LOCK_FILE="${LOCK_DIR}/fanout.lock"
mkdir -p "$LOG_DIR" "$LOCK_DIR"

# Default remote set comes from hpc/project.conf ($FANOUT_REMOTES); the
# operator can still override at runtime via an explicit FANOUT_REMOTES
# env var, which takes precedence over the conf value via the source order.
read -ra REMOTES <<< "${FANOUT_REMOTES}"

# ── Logging ───────────────────────────────────────────────────────────────────

_log() {
  local level="$1"; shift
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local line="${ts} ${MODE} ${level} $*"
  echo "$line" >> "$LOG_FILE"
  if [[ "$MODE" != "quiet" ]] || [[ "$level" != "INFO" ]]; then
    echo "$line"
  fi
}

# ── Mutual exclusion (fcntl flock via Python — macOS has no /usr/bin/flock) ───

exec 9>"$LOCK_FILE"
if ! python3 -c "import fcntl,sys; fcntl.flock(int(sys.argv[1]), fcntl.LOCK_EX | fcntl.LOCK_NB)" 9 2>/dev/null; then
  _log INFO "another fanout already running (lock held); skipping this tick"
  exit 0
fi

# ── Sanity: SSH agent must have keys loaded for HPC pushes ────────────────────

_have_ssh_agent_keys() {
  # 1 = keys available; 0 = no agent or no keys.
  ssh-add -l >/dev/null 2>&1
}

# ── Fetch origin ──────────────────────────────────────────────────────────────

if ! git fetch --quiet origin "${DEFAULT_BRANCH}" 2>>"$LOG_FILE"; then
  _log ERROR "git fetch origin ${DEFAULT_BRANCH} failed; check network / origin access"
  exit 2
fi

# ── Capture SHAs ──────────────────────────────────────────────────────────────
# Variable names retain the historical _MAIN suffix for readability across
# the log corpus; the value is the SHA of ${DEFAULT_BRANCH} on the named
# remote.

ORIGIN_MAIN="$(git rev-parse "origin/${DEFAULT_BRANCH}")"
LOCAL_MAIN="$(git rev-parse "${DEFAULT_BRANCH}" 2>/dev/null || echo unknown)"

# ── Fast-forward local default branch from origin (without touching HEAD) ─────

if [[ "$LOCAL_MAIN" != "$ORIGIN_MAIN" ]]; then
  if [[ "$LOCAL_MAIN" == "unknown" ]]; then
    _log ERROR "local '${DEFAULT_BRANCH}' ref missing; cannot proceed"
    exit 3
  fi
  # Use merge-base to check fast-forward eligibility. If the merge-base
  # equals local tip, then origin is ahead of local — safe FF.
  MB="$(git merge-base "$LOCAL_MAIN" "$ORIGIN_MAIN")"
  if [[ "$MB" != "$LOCAL_MAIN" ]]; then
    _log ERROR "local '${DEFAULT_BRANCH}' (${LOCAL_MAIN:0:8}) has diverged from origin/${DEFAULT_BRANCH} (${ORIGIN_MAIN:0:8}); manual reconciliation needed"
    exit 4
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    _log INFO "[dry-run] would fast-forward ${DEFAULT_BRANCH}: ${LOCAL_MAIN:0:8} → ${ORIGIN_MAIN:0:8}"
  else
    # Update the ref directly so we don't perturb any checked-out branch.
    git update-ref "refs/heads/${DEFAULT_BRANCH}" "$ORIGIN_MAIN" "$LOCAL_MAIN" 2>>"$LOG_FILE" || {
      _log ERROR "git update-ref refs/heads/${DEFAULT_BRANCH} failed"
      exit 5
    }
    _log INFO "fast-forwarded ${DEFAULT_BRANCH}: ${LOCAL_MAIN:0:8} → ${ORIGIN_MAIN:0:8}"
  fi
else
  _log INFO "local ${DEFAULT_BRANCH} already at origin/${DEFAULT_BRANCH} (${ORIGIN_MAIN:0:8})"
fi

# ── Per-remote fan-out ────────────────────────────────────────────────────────

_remote_needs_push() {
  # Returns 0 if the remote's default branch lags origin's; 1 otherwise.
  local remote="$1"
  # Pull the remote's current default-branch tip (post-fetch) into the
  # comparison. `git ls-remote` is the source of truth —
  # refs/remotes/<remote>/<branch> may be stale if we haven't fetched
  # that remote in this run.
  local remote_main
  remote_main="$(git ls-remote --heads "$remote" "${DEFAULT_BRANCH}" 2>/dev/null | awk '{print $1}' | head -1)"
  if [[ -z "$remote_main" ]]; then
    echo "missing"
    return 0
  fi
  if [[ "$remote_main" == "$ORIGIN_MAIN" ]]; then
    echo "current"
    return 1
  fi
  echo "$remote_main"
  return 0
}

_push_to_remote() {
  local remote="$1"
  local expected_tip="$2"   # SHA we last saw on the remote, for --force-with-lease
  # --no-verify: the SHA already passed pre-merge gating; no point re-running
  #              local pre-push tests on the post-merge fan-out.
  # --force-with-lease: refuses if the remote's tip drifted from what we
  #                     just observed. Never blind-overwrites.
  local lease_arg=""
  if [[ -n "$expected_tip" ]] && [[ "$expected_tip" != "missing" ]]; then
    lease_arg="--force-with-lease=refs/heads/${DEFAULT_BRANCH}:${expected_tip}"
  fi
  git push --no-verify $lease_arg "$remote" "${ORIGIN_MAIN}:refs/heads/${DEFAULT_BRANCH}" 2>>"$LOG_FILE"
}

ANY_FAIL=0
PUSHED_COUNT=0
for remote in "${REMOTES[@]}"; do
  if ! git remote get-url "$remote" >/dev/null 2>&1; then
    _log INFO "remote '$remote' not configured locally; skipping"
    continue
  fi

  # SSH-based remotes need an agent with loaded keys; sandbox is local file path.
  if [[ "$remote" != "sandbox" ]] && ! _have_ssh_agent_keys; then
    _log ERROR "remote '$remote' needs SSH but no keys in ssh-agent (run: ssh-add ~/.ssh/id_rsa)"
    ANY_FAIL=1
    continue
  fi

  remote_state="$(_remote_needs_push "$remote")"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    _log INFO "remote '$remote' already at ${ORIGIN_MAIN:0:8}"
    continue
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    _log INFO "[dry-run] would push to '$remote' (current tip: ${remote_state:0:8} → ${ORIGIN_MAIN:0:8})"
    continue
  fi

  _log INFO "pushing to '$remote' (${remote_state:0:8} → ${ORIGIN_MAIN:0:8})"
  if _push_to_remote "$remote" "$remote_state"; then
    _log INFO "✓ '$remote' updated to ${ORIGIN_MAIN:0:8}"
    PUSHED_COUNT=$((PUSHED_COUNT + 1))
  else
    _log ERROR "✗ push to '$remote' FAILED (rc=$?). Will retry on next fanout tick."
    ANY_FAIL=1
  fi
done

# ── Summary + exit ────────────────────────────────────────────────────────────

if [[ "$DRY_RUN" == "true" ]]; then
  _log INFO "dry-run complete (no changes made)"
  exit 0
fi

if [[ $ANY_FAIL -ne 0 ]]; then
  _log ERROR "fanout finished with ${PUSHED_COUNT} successful push(es) and at least one failure"
  exit 6
fi

if [[ $PUSHED_COUNT -eq 0 ]]; then
  _log INFO "nothing to fan out (all remotes current at ${ORIGIN_MAIN:0:8})"
else
  _log INFO "fanout complete: pushed to ${PUSHED_COUNT} remote(s) at ${ORIGIN_MAIN:0:8}"
fi
exit 0
