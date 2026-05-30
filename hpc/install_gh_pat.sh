#!/bin/bash
################################################################################
# hpc/install_gh_pat.sh — Install a GitHub Personal Access Token on CHPC + RIS
# for use by hpc/ci_pr.sbatch's post-back to the Statuses API.
#
# Wraps the strip-whitespace + pipe-to-gh-auth + verify dance into one
# auditable procedure. Used periodically when the PAT expires (fine-grained
# PATs max ~1 year for "Commit statuses: Read and write" scope).
#
# Usage:
#   pbpaste | bash hpc/install_gh_pat.sh                  # both clusters
#   pbpaste | bash hpc/install_gh_pat.sh --cluster=chpc   # CHPC only
#   pbpaste | bash hpc/install_gh_pat.sh --cluster=ris    # RIS only
#
#   # Read PAT from a file instead of clipboard:
#   bash hpc/install_gh_pat.sh < ~/path/to/pat.txt
#
# What it does, per requested cluster:
#   1. Logs out any existing github.com auth (idempotent; ignores "not logged
#      in" error).
#   2. Pipes the cleaned PAT into `gh auth login --hostname github.com
#      --git-protocol https --with-token`.
#   3. Verifies via `gh auth status` (token stored).
#   4. Verifies via `gh api user` (token actually works against the GitHub
#      API — exercises the same code path hpc/post_pr_status.py uses).
#
# Token hygiene:
#   - The PAT is read from stdin once, into a local shell variable.
#   - It is piped to ssh stdin per cluster; the local variable is unset
#     immediately after the last ssh call.
#   - The PAT is never echoed, never logged, never written to disk on the
#     MacBook side. On the cluster side gh stores it in ~/.config/gh/hosts.yml.
#   - Set permissions on hosts.yml are gh's responsibility (typically 0600).
#
# Assumes:
#   - SSH access to login3.chpc.wustl.edu and c2-login-001 (operator's keys).
#   - `gh` installed at ~/.local/bin/gh on each cluster.
#
# Exit codes:
#   0  — all requested clusters authenticated and verified.
#   1  — argument parse error or empty PAT on stdin.
#   2  — PAT prefix validation failed (not github_pat_… or ghp_…).
#   3  — at least one cluster's auth or verification failed.
#
# Spec: SPEC_development_workflow.md Pattern A′; SPEC_continuous_integration.md §0a.
################################################################################

set -uo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────

CLUSTER_FILTER="both"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster=*) CLUSTER_FILTER="${1#*=}" ;;
    --cluster)   shift; CLUSTER_FILTER="${1:?--cluster requires chpc|ris|both}" ;;
    -h|--help)
      sed -n '3,36p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg '$1'" >&2
      echo "Usage: pbpaste | bash hpc/install_gh_pat.sh [--cluster=chpc|ris|both]" >&2
      exit 1
      ;;
  esac
  shift
done

case "$CLUSTER_FILTER" in
  chpc|ris|both) ;;
  *)
    echo "ERROR: --cluster must be chpc, ris, or both (got '${CLUSTER_FILTER}')" >&2
    exit 1
    ;;
esac

# ── Read + clean + validate the PAT ───────────────────────────────────────────

if [[ -t 0 ]]; then
  echo "ERROR: PAT must be piped on stdin (no terminal input)." >&2
  echo "Usage: pbpaste | bash hpc/install_gh_pat.sh [--cluster=...]" >&2
  exit 1
fi

# Strip ALL whitespace — including \r and \n which would otherwise wind up
# in the HTTP Authorization header and produce "invalid header field value
# for Authorization" from gh's post-login validation step.
PAT="$(tr -d '[:space:]')"

if [[ -z "$PAT" ]]; then
  echo "ERROR: empty PAT on stdin." >&2
  exit 1
fi

# Validate the PAT prefix. github_pat_… is fine-grained; ghp_… is classic.
case "$PAT" in
  github_pat_*|ghp_*) ;;
  *)
    # Print the first 8 chars only so the operator can see the prefix
    # without leaking the body in transcripts.
    echo "ERROR: PAT does not look like a GitHub token (prefix: ${PAT:0:8}…)." >&2
    echo "  Expected prefix: 'github_pat_' (fine-grained) or 'ghp_' (classic)." >&2
    exit 2
    ;;
esac

# ── Per-cluster installer ─────────────────────────────────────────────────────

# Resolve each cluster's REMOTE (user@host) via hpc/config.sh so internal
# hostnames stay in a single source of truth (SPEC §6.4). config.sh detects
# HPC_TARGET from env, so we override it per iteration in a subshell to
# avoid re-running auto-detection and to leave the parent shell's HPC_TARGET
# (if any) untouched.
HPC_DIR_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A CLUSTERS=()
for _target in chpc ris; do
  CLUSTERS[$_target]="$(
    HPC_TARGET="$_target" bash -c "source '${HPC_DIR_LOCAL}/config.sh' >/dev/null 2>&1 && echo \"\$REMOTE\""
  )"
done
unset _target

if [[ "$CLUSTER_FILTER" == "both" ]]; then
  TARGETS=(chpc ris)
else
  TARGETS=("$CLUSTER_FILTER")
fi

# Track per-cluster verdicts so we can report at the end.
declare -A RESULTS

install_on_cluster() {
  local name="$1"
  local remote="${CLUSTERS[$name]}"
  echo "═══ ${name} (${remote}) ═══"

  # Step 1: logout any existing auth (idempotent; ignore non-zero if not logged in).
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote" \
    '~/.local/bin/gh auth logout --hostname github.com 2>/dev/null || true' >/dev/null
  echo "  logout: done (or no prior auth)"

  # Step 2: pipe the cleaned PAT to `gh auth login --with-token`.
  # The PAT goes to ssh's stdin → ssh's remote stdin → gh's stdin.
  if ! printf '%s' "$PAT" | ssh -o BatchMode=yes -o ConnectTimeout=15 "$remote" \
      '~/.local/bin/gh auth login --hostname github.com --git-protocol https --with-token' \
      2>&1 | sed 's/^/  login: /'; then
    echo "  ✗ login failed on ${name}"
    RESULTS[$name]=fail
    return
  fi

  # Step 3: verify storage via `gh auth status`.
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote" \
      '~/.local/bin/gh auth status --hostname github.com' 2>&1 | sed 's/^/  status: /'; then
    echo "  ✗ gh auth status failed on ${name}"
    RESULTS[$name]=fail
    return
  fi

  # Step 4: actively exercise the API. This catches a scope-mismatch token
  # that stores fine but can't reach /user — same code path
  # hpc/post_pr_status.py hits when it POSTs to /repos/.../statuses/<sha>.
  local user_check
  user_check="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote" \
    '~/.local/bin/gh api user --jq .login' 2>&1)"
  if [[ -z "$user_check" ]] || [[ "$user_check" == *"error"* ]] || [[ "$user_check" == *"HTTP"* ]]; then
    echo "  ✗ gh api user failed on ${name}: ${user_check}"
    RESULTS[$name]=fail
    return
  fi
  echo "  api: gh api user → ${user_check}"

  echo "  ✓ ${name} OK"
  RESULTS[$name]=ok
}

for cluster in "${TARGETS[@]}"; do
  install_on_cluster "$cluster"
  echo
done

# ── Drop the PAT from memory ──────────────────────────────────────────────────

PAT=""
unset PAT

# ── Aggregate verdict ─────────────────────────────────────────────────────────

ANY_FAIL=0
for cluster in "${TARGETS[@]}"; do
  if [[ "${RESULTS[$cluster]:-fail}" != "ok" ]]; then
    ANY_FAIL=1
  fi
done

echo "═══ summary ═══"
for cluster in "${TARGETS[@]}"; do
  printf '  %-5s %s\n' "$cluster" "${RESULTS[$cluster]:-fail}"
done

if [[ $ANY_FAIL -ne 0 ]]; then
  echo
  echo "✗ at least one cluster failed. Review the per-cluster output above."
  echo "  Common causes:"
  echo "    - PAT scope insufficient (need 'Commit statuses: Read and write'"
  echo "      on the analytic-signal-sst repo; or 'repo:status' for classic PAT)"
  echo "    - PAT expired"
  echo "    - gh binary missing at ~/.local/bin/gh on the cluster"
  exit 3
fi

echo
echo "✓ all requested clusters authenticated and verified."
echo "  Next: confirm hpc/ci_pr.sh can post a status by running"
echo "    bash hpc/ci_pr.sh <some-feature-branch>"
echo "  against a PR and watching for 'hpc-ci' in the PR's Checks panel."
