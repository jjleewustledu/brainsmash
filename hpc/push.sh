#!/bin/bash
################################################################################
# hpc/push.sh — Sync all projects in sync_manifest.conf to HPC
#
# Source: git push to 'hpc' remote (bare repo + post-receive checkout hook)
# Config: rsync only (dotfolders have no git repo)
#
# Usage:
#   bash hpc/push.sh                         # sync everything
#   bash hpc/push.sh --source-only           # git repos only, skip rsync
#   bash hpc/push.sh --config-only           # rsync dotfolders only, skip git
#   bash hpc/push.sh --dry-run               # show what would happen, no changes
#   bash hpc/push.sh --project <name>        # sync one project by partial name match
#
# Multiple flags may be combined, e.g.:
#   bash hpc/push.sh --dry-run --source-only
#   bash hpc/push.sh --project brainsmash-adapter --dry-run
#
# Prerequisites:
#   - SSH key loaded:         ssh-add ~/.ssh/id_rsa (or equivalent)
#   - 'hpc' remote present:   bash hpc/setup_remotes.sh  (one-time)
#   - SSH ControlPersist:     see ~/.ssh/config (reuses connections within session)
#
# Manifest format (hpc/sync_manifest.conf):
#   TYPE|LOCAL_PATH|REMOTE_PATH|BRANCH
#   TYPE    = git | git_glob | rsync
#   BRANCH  = branch name, or 'auto' to read from local git symbolic-ref
#   REMOTE_PATH = path on HPC — may contain ${HPC_PROJECTS}, ${HPC_SCRATCH},
#                 or ~ (expanded at runtime from hpc/config.sh)
#
# Note on topology:
#   MacBook --[push hpc]--> HPC bare repo --[post-receive]--> HPC working copy
#   MacBook --[push origin]--> GitHub  (independent; use normally for backup)
#   The post-receive hook handles checkout; no 'git pull' is needed on HPC.
################################################################################

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# ── Mutual exclusion (MULTI_LOOP_PLAN.md B4) ──────────────────────────────────
# Two concurrent invocations of push.sh against the SAME HPC target would
# rsync to overlapping destination trees (partial-file corruption) and
# git-push to the same remote branches (non-fast-forward rejection, or
# concurrent post-receive hooks on the HPC bare repo). Per-target advisory
# fcntl.flock on a sentinel file serializes invocations against one target
# while permitting parallel pushes to different targets (chpc + ris).
#
# The lock is acquired via a short Python subprocess but survives the
# subprocess exit because the parent shell still references the open file
# description on fd 9. The lock is released when fd 9 is closed — which
# happens automatically on parent shell exit, including SIGKILL. No
# stale-lock recovery code is needed.
#
# Python's fcntl.flock is used instead of the BSD flock(1) binary because
# macOS does not ship /usr/bin/flock and we don't want a Homebrew
# dependency on the operator path. Python is already a hard-dep of the
# project.

PUSH_LOCK_DIR="${DOTFOLDER}/locks"
PUSH_LOCK_FILE="${PUSH_LOCK_DIR}/push-${HPC_TARGET}.lock"
mkdir -p "${PUSH_LOCK_DIR}"
exec 9>"${PUSH_LOCK_FILE}"

if ! python3 -c "import fcntl,sys; fcntl.flock(int(sys.argv[1]), fcntl.LOCK_EX | fcntl.LOCK_NB)" 9 2>/dev/null; then
    echo "[hpc/push] waiting for another push.sh against ${HPC_TARGET} (lock: ${PUSH_LOCK_FILE})..."
    python3 -c "import fcntl,sys; fcntl.flock(int(sys.argv[1]), fcntl.LOCK_EX)" 9
fi
echo "[hpc/push] acquired ${HPC_TARGET} lock (held until exit)"

# ── Argument parsing ──────────────────────────────────────────────────────────

DRY_RUN=false
FILTER="all"          # all | source | config
PROJECT_MATCH=""
INTEGRATION_TEST_BRANCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)      DRY_RUN=true ;;
    --source-only)  FILTER="source" ;;
    --config-only)  FILTER="config" ;;
    --project)
      [[ -n "${2:-}" ]] || { echo "ERROR: --project requires an argument" >&2; exit 1; }
      PROJECT_MATCH="$2"; shift ;;
    --integration-test)
      # Pattern A (SPEC_development_workflow.md §6.2): non-main branches
      # are pushed to HPC ONLY when this flag is present, naming the
      # specific branch being integration-tested. Catches accidental
      # WIP-branch pushes that should have gone through CI first.
      [[ -n "${2:-}" ]] || { echo "ERROR: --integration-test requires a branch argument" >&2; exit 1; }
      INTEGRATION_TEST_BRANCH="$2"; shift ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      echo "Usage: push.sh [--dry-run] [--source-only|--config-only] [--project <name>] [--integration-test <branch>]" >&2
      exit 1 ;;
  esac
  shift
done

$DRY_RUN && echo "==> DRY RUN — no changes will be made."

require_ssh

RSYNC_BASE_OPTS=(
    -avz
    --exclude='.DS_Store' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo'
    # Security: never sync secrets or credentials (SPEC-HPC-B §8)
    --exclude='.env' --exclude='.env.*'
    --exclude='*.pem' --exclude='*.key'
    --exclude='credentials.json' --exclude='secrets.yaml'
    --exclude='id_rsa' --exclude='id_ed25519'
    --exclude='id_ecdsa' --exclude='id_dsa'
    --exclude='*.p12' --exclude='*.pfx'
    --exclude='known_hosts' --exclude='authorized_keys'
    --exclude='.ssh'
)
$DRY_RUN && RSYNC_BASE_OPTS+=(--dry-run)

# ── Branch resolution ─────────────────────────────────────────────────────────
# Returns the branch name to push for a given repo.
# If manifest says 'auto' (or is empty), reads the local symbolic-ref.
# Falls back to ${DEFAULT_BRANCH} (from hpc/project.conf) if symbolic-ref
# fails (e.g. empty repo). The fallback only matters for newly cloned
# bare repos that haven't received any pushes yet; in normal operation
# the symbolic-ref query succeeds.

get_branch() {
  local repo_path="$1"
  local manifest_branch="${2:-auto}"

  if [[ "$manifest_branch" == "auto" || -z "$manifest_branch" ]]; then
    git -C "$repo_path" symbolic-ref --short HEAD 2>/dev/null || echo "${DEFAULT_BRANCH}"
  else
    echo "$manifest_branch"
  fi
}

# ── Project name filter ───────────────────────────────────────────────────────
# Returns 0 (match/pass) or 1 (skip) given a repo label.
# An empty PROJECT_MATCH passes everything.

matches_project_filter() {
  local label="$1"
  [[ -z "$PROJECT_MATCH" || "$label" == *"$PROJECT_MATCH"* ]]
}

# ── git push helper ───────────────────────────────────────────────────────────
# Pushes a single git repo to its 'hpc' remote.
# The post-receive hook on HPC runs 'git checkout -f <branch>' into the
# working directory — no explicit pull is needed after the push.

push_git_repo() {
  local local_path="$1"
  local remote_path="$2"    # informational only; bare repo URL is in .git/config
  local branch="$3"
  local label="${local_path##*/}"

  matches_project_filter "$label" || return 0

  echo ""
  echo "── git: ${label}  [branch: ${branch}] ──────────────────────────────────"

  if [[ ! -d "$local_path/.git" ]]; then
    echo "  ERROR: not a git repository: ${local_path}" >&2
    return 1
  fi

  # Pattern A main-only enforcement (SPEC_development_workflow.md §6.2).
  # Pushes to HPC are restricted to `main` by default; non-`main` branches
  # require an explicit `--integration-test <branch>` flag. This makes
  # HPC mirrors of `main` always reflect the canonical merged code (so
  # SLURM submissions always run against what's reviewed and merged),
  # and prevents accidental WIP-branch fan-out via routine `hpc/push.sh`.
  #
  # Resolve push_ref — what we will actually push to the remote:
  #
  #   --integration-test <X> supplied → push <X>'s ref by name (works from
  #     any active branch, including main; the auto-fire from
  #     hooks/pre-push runs from the primary on main and must be able to
  #     push a feature branch by name).
  #
  #   else, active branch is main → push main (routine fan-out).
  #
  #   else (non-main active branch, no --integration-test) → REFUSED
  #     (accidental-WIP guard).
  #
  # Pre-2026-05-28: this code skipped the integration-test handling
  # entirely when the active branch was main, silently no-op'ing the
  # --integration-test arg and pushing main. The auto-fire from primary
  # on main therefore never pushed feature branches; every per-job CI
  # prolog tried `git worktree add <feature-sha>` against a bare repo
  # that lacked the SHA, and failed. PRs #53, #54, #55, #56 in the
  # 2026-05-27/28 session all silently prolog-failed on this path until
  # the operator manually pushed the branch ref to each bare repo.
  local push_ref
  if [[ -n "$INTEGRATION_TEST_BRANCH" ]]; then
    if ! git -C "$local_path" show-ref --verify --quiet "refs/heads/${INTEGRATION_TEST_BRANCH}"; then
      echo "  REFUSED: --integration-test '${INTEGRATION_TEST_BRANCH}' does not exist as a local branch." >&2
      echo "  Available branches (first 10):" >&2
      git -C "$local_path" for-each-ref --format='    %(refname:short)' refs/heads/ | head -10 >&2
      return 1
    fi
    push_ref="$INTEGRATION_TEST_BRANCH"
    if [[ "$push_ref" != "$branch" ]]; then
      echo "  NOTE: --integration-test ${push_ref} — pushing the named ref directly" >&2
      echo "  (active branch is '${branch}'; cross-branch push from primary is the" >&2
      echo "   intended path for hooks/pre-push's Pattern A′ HPC-CI auto-fire)." >&2
    else
      echo "  NOTE: --integration-test ${push_ref} — pushing non-${DEFAULT_BRANCH} branch for HPC integration testing." >&2
      echo "  Reminder: this is transient. Pop the branch after the PR squash-merges." >&2
    fi
  elif [[ "$branch" == "${DEFAULT_BRANCH}" ]]; then
    push_ref="${DEFAULT_BRANCH}"
  else
    echo "  REFUSED: branch '${branch}' is not '${DEFAULT_BRANCH}' and --integration-test was not supplied." >&2
    echo "" >&2
    echo "  Pattern A routing (SPEC_development_workflow.md §6.2):" >&2
    echo "    • hpc/push.sh defaults to ${DEFAULT_BRANCH}-only fan-out." >&2
    echo "    • Feature branches reach HPC only when explicitly integration-tested." >&2
    echo "" >&2
    echo "  To push this branch for /hpc-integration:" >&2
    echo "    bash hpc/push.sh --integration-test ${branch}" >&2
    echo "" >&2
    echo "  To push ${DEFAULT_BRANCH} (from the primary clone):" >&2
    echo "    git checkout ${DEFAULT_BRANCH} && git pull origin ${DEFAULT_BRANCH} && bash hpc/push.sh" >&2
    return 1
  fi

  cd "$local_path"

  # Prefer the per-target remote ($HPC_TARGET, e.g. 'chpc' or 'ris').
  # Fall back to the legacy 'hpc' remote during migration.
  local remote_name="${LOCAL_REMOTE_NAME}"
  if ! git remote get-url "$remote_name" &>/dev/null; then
    if git remote get-url hpc &>/dev/null; then
      echo "  NOTE: '${remote_name}' remote not configured; falling back to legacy 'hpc'."
      echo "        Run:  bash hpc/setup_remotes.sh   # to add the per-target remote"
      remote_name="hpc"
    else
      echo "  WARNING: neither '${LOCAL_REMOTE_NAME}' nor 'hpc' remote configured."
      echo "  Run:  bash hpc/setup_remotes.sh"
      return 0
    fi
  fi

  # Dirty / behind checks apply only when pushing the active branch
  # (push_ref == branch). For cross-branch ref-by-name pushes, the
  # active checkout's state is irrelevant to what's being pushed.
  if [[ "$push_ref" == "$branch" ]]; then
    # Warn about uncommitted changes — they won't be pushed
    local dirty
    dirty=$(git status --porcelain 2>/dev/null) || true
    if [[ -n "$dirty" ]]; then
      echo "  WARNING: uncommitted changes present — only the last commit will be pushed:"
      git status --short | sed 's/^/    /'
    fi

    # Warn if local branch is behind origin
    git fetch origin "$branch" --quiet 2>/dev/null || true
    local behind
    behind=$(git rev-list --count "HEAD..origin/${branch}" 2>/dev/null || echo 0)
    if [[ "$behind" -gt 0 ]]; then
      echo "  WARNING: local branch is ${behind} commit(s) behind origin/${branch}."
      echo "  Consider: git pull origin ${branch}"
    fi
  fi

  # Build the refspec. When push_ref matches the active branch, use HEAD
  # (preserves the legacy behavior and the dirty-warn semantics above).
  # Otherwise push the named ref by-name; this works regardless of which
  # branch is checked out, since git pushes from the object store + ref
  # database, not from the working tree.
  local refspec
  if [[ "$push_ref" == "$branch" ]]; then
    refspec="HEAD:${push_ref}"
  else
    refspec="refs/heads/${push_ref}:refs/heads/${push_ref}"
  fi

  if $DRY_RUN; then
    echo "  [dry-run] git push ${remote_name} ${refspec}"
    echo "  [dry-run] → HPC checkout triggered by post-receive hook: ${remote_path}"
  else
    git push "$remote_name" "$refspec"
    echo "  Pushed ${push_ref} → ${remote_name}/${push_ref}"
    echo "  HPC working copy updated via post-receive hook: ${remote_path}"
  fi
}

# ── rsync helper ──────────────────────────────────────────────────────────────
# Syncs a local directory to an absolute path on HPC.
# Used for dotfolders that have no git repo.

push_rsync() {
  local local_path="$1"
  local remote_path="$2"
  local extra_opts="${3:-}"
  local label="${local_path##*/}"

  matches_project_filter "$label" || return 0

  echo ""
  echo "── rsync: ${label} ──────────────────────────────────────────────────────"

  if [[ ! -d "$local_path" ]]; then
    echo "  ERROR: local path does not exist: ${local_path}" >&2
    return 1
  fi

  # Build per-entry option array from space-separated string
  local -a entry_opts=()
  if [[ -n "$extra_opts" ]]; then
    read -ra entry_opts <<< "$extra_opts"
  fi

  # Trailing slash on source = sync contents, not the directory itself
  rsync "${RSYNC_BASE_OPTS[@]}" "${entry_opts[@]+"${entry_opts[@]}"}" \
    "${local_path}/" \
    "${REMOTE}:${remote_path}/"
}

# ── Manifest processing ───────────────────────────────────────────────────────
# Reads sync_manifest.conf line by line.
# Fields: TYPE | LOCAL_PATH | REMOTE_PATH | BRANCH_OR_OPTIONS
#   git/git_glob: 4th field = branch (default: auto)
#   rsync:        4th field = extra rsync flags (optional)

echo ""
echo "==> Target: ${HPC_TARGET} (${REMOTE_HOST})"
echo "    Reading manifest: ${MANIFEST}"
echo "    Filter: ${FILTER}${PROJECT_MATCH:+, project match: ${PROJECT_MATCH}}"

while IFS='|' read -r TYPE LOCAL REMOTE_PATH FIELD4 || [[ -n "${TYPE:-}" ]]; do

  # Skip blank lines and comments
  [[ -z "${TYPE// }" || "${TYPE}" =~ ^[[:space:]]*# ]] && continue

  # Strip any trailing whitespace from all fields
  TYPE="${TYPE%"${TYPE##*[![:space:]]}"}"
  LOCAL="${LOCAL%"${LOCAL##*[![:space:]]}"}"
  REMOTE_PATH="${REMOTE_PATH%"${REMOTE_PATH##*[![:space:]]}"}"
  FIELD4="${FIELD4:-}"
  FIELD4="${FIELD4%"${FIELD4##*[![:space:]]}"}"

  # Expand ~ in local path
  LOCAL="${LOCAL/#\~/$HOME}"

  # Expand config variables in remote path
  REMOTE_PATH="${REMOTE_PATH//\$\{HPC_PROJECTS\}/${HPC_PROJECTS}}"
  REMOTE_PATH="${REMOTE_PATH//\$\{HPC_SCRATCH\}/${HPC_SCRATCH}}"

  case "$TYPE" in

    git)
      [[ "$FILTER" == "config" ]] && continue
      local_branch="${FIELD4:-auto}"
      RESOLVED=$(get_branch "$LOCAL" "$local_branch")
      push_git_repo "$LOCAL" "$REMOTE_PATH" "$RESOLVED"
      ;;

    git_glob)
      [[ "$FILTER" == "config" ]] && continue
      local_branch="${FIELD4:-auto}"
      # Expand glob; iterate only directories that are git repos
      shopt -s nullglob
      for repo_path in $LOCAL; do
        [[ -d "$repo_path/.git" ]] || continue
        repo_name="${repo_path##*/}"
        RESOLVED=$(get_branch "$repo_path" "$local_branch")
        push_git_repo "$repo_path" "${REMOTE_PATH}/${repo_name}" "$RESOLVED"
      done
      shopt -u nullglob
      ;;

    rsync)
      [[ "$FILTER" == "source" ]] && continue
      push_rsync "$LOCAL" "$REMOTE_PATH" "$FIELD4"
      ;;

    *)
      echo "WARNING: Unknown type '${TYPE}' in manifest — skipping." >&2
      ;;

  esac

done < "$MANIFEST"

echo ""
echo "==> Push complete."
