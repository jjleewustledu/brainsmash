#!/bin/bash
################################################################################
# hpc/setup_remotes.sh — One-time setup of HPC bare repos and local 'hpc' remotes
#
# For each git repo in sync_manifest.conf this script:
#   1. Creates a bare repo on HPC at <REMOTE_PATH>.git
#   2. Installs a post-receive hook that checks out the working tree to
#      <REMOTE_PATH> (the non-.git working directory) on every push
#   3. Initialises <REMOTE_PATH> as a working directory pointing at the bare repo
#   4. Adds a per-target remote (named after HPC_TARGET, e.g. 'chpc' or 'ris')
#      to the local repo on this MacBook. Both clusters can coexist as
#      parallel local remotes; supervisor-finalize.sh and /commit fan out to
#      every configured HPC-class remote.
#
# rsync-only entries (dotfolders) are skipped — they have no git repo.
#
# Safe to re-run:
#   - Bare repos that already exist are left untouched (hook is always refreshed)
#   - Local per-target remotes that already exist are reported and left untouched
#   - A --repair flag re-adds a misconfigured per-target remote, and migrates
#     a legacy 'hpc' remote to the per-target name if it points at the same URL
#
# Usage:
#   bash hpc/setup_remotes.sh               # set up all git repos in manifest
#   bash hpc/setup_remotes.sh --dry-run     # show what would happen, no changes
#   bash hpc/setup_remotes.sh --project <n> # set up one repo by partial name match
#   bash hpc/setup_remotes.sh --repair      # remove and re-add any 'hpc' remote
#   bash hpc/setup_remotes.sh --verify      # check setup without modifying anything
#
# After running:
#   bash hpc/push.sh     # perform the first push to deploy source to HPC
#
# Background — topology this creates:
#   MacBook --[git push hpc]--> /scratch/jjlee/PycharmProjects/<repo>.git  (bare)
#                                    |
#                              post-receive hook
#                                    |
#                                    v
#                          /scratch/jjlee/PycharmProjects/<repo>/  (working copy)
#
#   GitHub remains the 'origin' remote on both MacBook and HPC, used
#   independently for backup and collaboration. push.sh never touches origin.
#
# Prerequisites:
#   - SSH key loaded:  ssh-add ~/.ssh/id_rsa  (or equivalent)
#   - SSH ControlPersist configured in ~/.ssh/config (avoids repeated auth)
################################################################################

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# ── Argument parsing ──────────────────────────────────────────────────────────

DRY_RUN=false
VERIFY=false
REPAIR=false
PROJECT_MATCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=true ;;
    --verify)   VERIFY=true ;;
    --repair)   REPAIR=true ;;
    --project)
      [[ -n "${2:-}" ]] || { echo "ERROR: --project requires an argument" >&2; exit 1; }
      PROJECT_MATCH="$2"; shift ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      echo "Usage: setup_remotes.sh [--dry-run] [--verify] [--repair] [--project <name>]" >&2
      exit 1 ;;
  esac
  shift
done

$DRY_RUN && echo "==> DRY RUN — no changes will be made."
$VERIFY  && echo "==> VERIFY mode — checking setup without modifying anything."

require_ssh

# ── Helpers ───────────────────────────────────────────────────────────────────

matches_project_filter() {
  local label="$1"
  [[ -z "$PROJECT_MATCH" || "$label" == *"$PROJECT_MATCH"* ]]
}

get_branch() {
  local repo_path="$1"
  local manifest_branch="${2:-auto}"
  if [[ "$manifest_branch" == "auto" || -z "$manifest_branch" ]]; then
    git -C "$repo_path" symbolic-ref --short HEAD 2>/dev/null || echo "main"
  else
    echo "$manifest_branch"
  fi
}

# ── Core setup function ───────────────────────────────────────────────────────
# Sets up one git repo end-to-end.
#
# Arguments:
#   local_path   — absolute local path on MacBook (~ already expanded)
#   remote_path  — absolute path on HPC for the WORKING directory
#   branch       — already-resolved branch name (not 'auto')

setup_repo() {
  local local_path="$1"
  local remote_path="$2"
  local branch="$3"
  local label="${local_path##*/}"
  local bare_path="${remote_path}.git"

  matches_project_filter "$label" || return 0

  echo ""
  echo "━━ ${label}  [branch: ${branch}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "   Local:      ${local_path}"
  echo "   HPC bare:   ${REMOTE}:${bare_path}"
  echo "   HPC work:   ${REMOTE}:${remote_path}"

  # ── Validate local repo ───────────────────────────────────────────────────
  if [[ ! -d "$local_path/.git" ]]; then
    echo "   ERROR: not a git repository: ${local_path}" >&2
    return 1
  fi

  if ! git -C "$local_path" show-ref --verify --quiet "refs/heads/${branch}" 2>/dev/null; then
    echo "   ERROR: branch '${branch}' does not exist in ${label}" >&2
    echo "   Available branches: $(git -C "$local_path" branch --format='%(refname:short)' | tr '\n' ' ')" >&2
    return 1
  fi

  # Warn if repo has no origin (brainsmash-inference case)
  if ! git -C "$local_path" remote get-url origin &>/dev/null; then
    echo "   WARNING: no 'origin' remote — repo has no GitHub backup."
  fi

  # ── VERIFY mode: check state, report, return ──────────────────────────────
  if $VERIFY; then
    # Check bare repo exists on HPC (SC2029: client-side expansion is intentional)
    # shellcheck disable=SC2029
    if ssh "$REMOTE" "[[ -d '${bare_path}' ]]" 2>/dev/null; then
      echo "   [OK]    HPC bare repo exists"
    else
      echo "   [MISS]  HPC bare repo missing: ${bare_path}"
    fi

    # Check post-receive hook
    # shellcheck disable=SC2029
    if ssh "$REMOTE" "[[ -x '${bare_path}/hooks/post-receive' ]]" 2>/dev/null; then
      echo "   [OK]    post-receive hook installed"
    else
      echo "   [MISS]  post-receive hook missing or not executable"
    fi

    # Check working directory on HPC
    # shellcheck disable=SC2029
    if ssh "$REMOTE" "[[ -d '${remote_path}' ]]" 2>/dev/null; then
      echo "   [OK]    HPC working directory exists"
    else
      echo "   [MISS]  HPC working directory missing: ${remote_path}"
    fi

    # Check local per-target remote (e.g. 'chpc' or 'ris')
    local remote_name="${LOCAL_REMOTE_NAME}"
    local expected_url="${REMOTE}:${bare_path}"
    if git -C "$local_path" remote get-url "$remote_name" &>/dev/null; then
      local current_url
      current_url=$(git -C "$local_path" remote get-url "$remote_name")
      if [[ "$current_url" == "$expected_url" ]]; then
        echo "   [OK]    local '${remote_name}' remote → ${current_url}"
      else
        echo "   [WARN]  local '${remote_name}' remote URL mismatch:"
        echo "           expected: ${expected_url}"
        echo "           actual:   ${current_url}"
        echo "           Run with --repair to fix."
      fi
    else
      echo "   [MISS]  local '${remote_name}' remote not configured"
    fi

    # Report legacy 'hpc' remote (informational; migrated by --repair only)
    if [[ "$remote_name" != "hpc" ]] && git -C "$local_path" remote get-url hpc &>/dev/null; then
      local legacy_url
      legacy_url=$(git -C "$local_path" remote get-url hpc)
      if [[ "$legacy_url" == "$expected_url" ]]; then
        echo "   [NOTE]  legacy 'hpc' remote present (same URL) — safe to remove after migration"
      else
        echo "   [NOTE]  legacy 'hpc' remote present → ${legacy_url}"
      fi
    fi

    return 0
  fi

  # ── DRY RUN: show planned actions ────────────────────────────────────────
  if $DRY_RUN; then
    echo "   [dry-run] ssh ${REMOTE} 'mkdir -p ${remote_path}'"
    echo "   [dry-run] ssh ${REMOTE} 'git init --bare ${bare_path}'  (if absent)"
    echo "   [dry-run] ssh ${REMOTE} 'install post-receive hook → ${bare_path}/hooks/post-receive'"
    echo "   [dry-run] ssh ${REMOTE} 'mkdir -p ${remote_path} && git init (if absent)'"
    echo "   [dry-run] git -C ${local_path} remote add ${LOCAL_REMOTE_NAME} ${REMOTE}:${bare_path}  (if absent)"
    return 0
  fi

  # ── Create/verify bare repo and working directory on HPC ─────────────────
  # Single SSH call to minimise round-trips (ControlPersist makes each call
  # fast, but one heredoc is cleaner for multi-step remote logic).
  # SC2087: unquoted heredoc is intentional — we expand local vars (bare_path,
  # remote_path, branch) on the client side before sending to the remote.
  # shellcheck disable=SC2087
  ssh "$REMOTE" bash <<REMOTE_SCRIPT
set -euo pipefail

BARE="${bare_path}"
WORK="${remote_path}"
BRANCH="${branch}"

# 1. Bare repo
if [[ ! -d "\${BARE}" ]]; then
  git init --bare "\${BARE}"
  echo "   Created bare repo: \${BARE}"
else
  echo "   Bare repo already exists: \${BARE}"
fi

# 2. Post-receive hook — always refreshed so branch name stays current
cat > "\${BARE}/hooks/post-receive" <<'HOOK'
#!/bin/bash
# Deployed by hpc/setup_remotes.sh — do not edit manually.
# Updates the adjacent working directory when commits are pushed to the bare repo.
#
# Strategy: if the working directory has its own .git (e.g. it was cloned from
# origin), fetch from the bare repo and reset --hard. This keeps the working
# copy's .git/HEAD consistent with the file contents. Falls back to the
# GIT_WORK_TREE approach for directories without their own .git.
BARE_DIR="\$(cd "\${GIT_DIR}" && pwd)"
WORK_DIR="\${BARE_DIR%.git}"
WORK_DIR="\${WORK_DIR%/}"
mkdir -p "\${WORK_DIR}"
while read OLD_REV NEW_REV REF; do
  PUSHED_BRANCH="\${REF##refs/heads/}"
  if [ -d "\${WORK_DIR}/.git" ]; then
    # Working copy is its own repo — fetch + reset to keep .git consistent.
    # Must unset GIT_DIR (set by git to the bare repo) so git -C uses the
    # working copy's own .git directory.
    (
      unset GIT_DIR GIT_WORK_TREE
      cd "\${WORK_DIR}"
      if ! git remote get-url bare-source &>/dev/null; then
        git remote add bare-source "\${BARE_DIR}"
      fi
      git fetch --quiet bare-source "\${PUSHED_BRANCH}"
      git reset --hard "bare-source/\${PUSHED_BRANCH}"
    )
    echo "post-receive: fetched + reset \${PUSHED_BRANCH} → \${WORK_DIR}"
  else
    # No .git in working dir — classic GIT_WORK_TREE checkout
    GIT_WORK_TREE="\${WORK_DIR}" GIT_DIR="\${GIT_DIR}" git checkout -f "\${PUSHED_BRANCH}"
    echo "post-receive: checked out \${PUSHED_BRANCH} → \${WORK_DIR}"
  fi
done
HOOK
chmod +x "\${BARE}/hooks/post-receive"
echo "   post-receive hook installed: \${BARE}/hooks/post-receive"

# 3. Working directory — initialise only if it has no git identity yet
mkdir -p "\${WORK}"
if [[ ! -f "\${WORK}/.git" && ! -d "\${WORK}/.git" ]]; then
  git -C "\${WORK}" init --quiet
  git -C "\${WORK}" remote add hpc "\${BARE}"
  echo "   Initialised working directory: \${WORK}"
else
  echo "   Working directory already initialised: \${WORK}"
fi
REMOTE_SCRIPT

  # ── Local per-target remote (e.g. 'chpc' or 'ris') ───────────────────────
  local remote_name="${LOCAL_REMOTE_NAME}"
  local expected_url="${REMOTE}:${bare_path}"

  if git -C "$local_path" remote get-url "$remote_name" &>/dev/null; then
    local current_url
    current_url=$(git -C "$local_path" remote get-url "$remote_name")

    if [[ "$current_url" == "$expected_url" ]]; then
      echo "   Local '${remote_name}' remote already correct: ${current_url}"
    elif $REPAIR; then
      git -C "$local_path" remote remove "$remote_name"
      git -C "$local_path" remote add "$remote_name" "$expected_url"
      echo "   Repaired local '${remote_name}' remote: ${current_url} → ${expected_url}"
    else
      echo "   WARNING: local '${remote_name}' remote URL mismatch:"
      echo "     expected: ${expected_url}"
      echo "     actual:   ${current_url}"
      echo "   Run with --repair to update it."
    fi
  else
    git -C "$local_path" remote add "$remote_name" "$expected_url"
    echo "   Added local '${remote_name}' remote → ${expected_url}"
  fi

  # ── Legacy 'hpc' remote migration ────────────────────────────────────────
  # If a legacy 'hpc' remote exists and points at this same bare repo (i.e.
  # this is the cluster it was originally set up for), --repair removes it
  # to avoid double-pushing through both 'hpc' and the per-target name.
  if [[ "$remote_name" != "hpc" ]] && git -C "$local_path" remote get-url hpc &>/dev/null; then
    local legacy_url
    legacy_url=$(git -C "$local_path" remote get-url hpc)
    if [[ "$legacy_url" == "$expected_url" ]]; then
      if $REPAIR; then
        git -C "$local_path" remote remove hpc
        echo "   Removed legacy 'hpc' remote (now superseded by '${remote_name}')"
      else
        echo "   NOTE: legacy 'hpc' remote points at this same URL."
        echo "         Run with --repair to remove it, or:  git remote remove hpc"
      fi
    else
      # Legacy remote points at a different cluster — leave it alone, the
      # operator will run setup_remotes.sh for that cluster separately.
      echo "   NOTE: legacy 'hpc' remote → ${legacy_url}  (different cluster, left intact)"
    fi
  fi
}

# ── Manifest processing ───────────────────────────────────────────────────────

echo ""
echo "==> Reading manifest: ${MANIFEST}"
echo "    Remote: ${REMOTE}"
[[ -n "$PROJECT_MATCH" ]] && echo "    Project filter: ${PROJECT_MATCH}"
echo ""

SETUP_COUNT=0
SKIP_COUNT=0

while IFS='|' read -r TYPE LOCAL REMOTE_PATH BRANCH || [[ -n "${TYPE:-}" ]]; do

  # Skip blank lines and comments
  [[ -z "${TYPE// }" || "${TYPE}" =~ ^[[:space:]]*# ]] && continue

  # Strip trailing whitespace from all fields
  TYPE="${TYPE%"${TYPE##*[![:space:]]}"}"
  LOCAL="${LOCAL%"${LOCAL##*[![:space:]]}"}"
  REMOTE_PATH="${REMOTE_PATH%"${REMOTE_PATH##*[![:space:]]}"}"
  BRANCH="${BRANCH:-auto}"
  BRANCH="${BRANCH%"${BRANCH##*[![:space:]]}"}"

  # Expand ~ in local path
  LOCAL="${LOCAL/#\~/$HOME}"

  # Expand config variables in remote path (matches push.sh behavior)
  REMOTE_PATH="${REMOTE_PATH//\$\{HPC_PROJECTS\}/${HPC_PROJECTS}}"
  REMOTE_PATH="${REMOTE_PATH//\$\{HPC_SCRATCH\}/${HPC_SCRATCH}}"

  case "$TYPE" in

    git)
      RESOLVED=$(get_branch "$LOCAL" "$BRANCH")
      setup_repo "$LOCAL" "$REMOTE_PATH" "$RESOLVED"
      (( SETUP_COUNT++ )) || true
      ;;

    git_glob)
      shopt -s nullglob
      MATCHED=0
      for repo_path in $LOCAL; do
        [[ -d "$repo_path/.git" ]] || continue
        repo_name="${repo_path##*/}"
        RESOLVED=$(get_branch "$repo_path" "$BRANCH")
        setup_repo "$repo_path" "${REMOTE_PATH}/${repo_name}" "$RESOLVED"
        (( SETUP_COUNT++ )) || true
        (( MATCHED++ ))  || true
      done
      shopt -u nullglob
      if [[ $MATCHED -eq 0 ]]; then
        echo ""
        echo "   WARNING: git_glob '${LOCAL}' matched no git repositories." >&2
      fi
      ;;

    rsync)
      label="${LOCAL##*/}"
      if matches_project_filter "$label"; then
        echo ""
        echo "━━ ${label}  [rsync — no git setup required] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        (( SKIP_COUNT++ )) || true
      fi
      ;;

    *)
      echo "WARNING: Unknown type '${TYPE}' in manifest — skipping." >&2
      ;;

  esac

done < "$MANIFEST"

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if $VERIFY; then
  echo "==> Verification complete.  ${SETUP_COUNT} repo(s) checked, ${SKIP_COUNT} rsync entry(s) skipped."
elif $DRY_RUN; then
  echo "==> Dry run complete.  ${SETUP_COUNT} repo(s) would be configured, ${SKIP_COUNT} rsync entry(s) skipped."
else
  echo "==> Setup complete.  ${SETUP_COUNT} repo(s) configured, ${SKIP_COUNT} rsync entry(s) skipped."
  echo ""
  echo "Next step — first push to deploy source to HPC:"
  echo "  bash hpc/push.sh --dry-run   # preview"
  echo "  bash hpc/push.sh             # deploy"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
