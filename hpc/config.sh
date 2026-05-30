# shellcheck shell=bash
# Sourced by all hpc/ scripts — variables are used by the sourcing script.
# shellcheck disable=SC2034

REMOTE_USER="jjlee"
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${HPC_DIR}/sync_manifest.conf"

# Per-project values (PROJECT_NAME, CONDA_ENV, DEFAULT_BRANCH, DOTFOLDER,
# SINGLE_CLUSTER, BRANCH_PROTECTION_CONTEXT, TEST_CMD, TEST_CMD_GPU). The
# toolkit reads these from hpc/project.conf rather than hardcoding the
# analytic_signal_sst names so the same scripts can be reused across
# sibling projects (brainsmash*). See hpc/project.conf for the schema.
#
# The source is guarded so config.sh remains usable in test fixtures
# (tests/test_fetch_sh.py, tests/test_submit_sh.py) that build minimal
# fake projects with only the scripts they exercise. Consumers' code
# uses ${VAR:-default} so missing values fall back to the prior
# hardcoded defaults — i.e. config.sh without project.conf behaves
# identically to pre-Phase-0 config.sh.
if [ -f "${HPC_DIR}/project.conf" ]; then
    # shellcheck source=project.conf
    . "${HPC_DIR}/project.conf"
fi

# ---------------------------------------------------------------------------
# Multi-cluster dispatch
#
# HPC_TARGET selects the active cluster.  Set explicitly or let auto-detect
# choose.  Valid values: "chpc", "ris".
# ---------------------------------------------------------------------------

_detect_hpc_target() {
  # Priority chain mirrors Python Config._detect_host_profile().
  # On-cluster: resolved instantly via filesystem/hostname (steps 2–3).
  # MacBook: the sticky preference file (step 4) is consulted BEFORE the
  # SSH probe (step 5) so detection is deterministic — the probe order is
  # otherwise a race (whichever cluster's login node answers first wins).
  # Pin a per-machine default with:
  #   echo ris > "${DOTFOLDER}/.hpc_target"
  # Neither cluster is "the" default: CHPC and RIS are both first-class;
  # RIS GPUs give a ~3-4x SST speedup, so RIS is often preferred for GPU
  # precompute. The fix for ambiguous detection is to make the resolved
  # target visible, not to hard-code a winner.

  # 1. Explicit env var (highest priority)
  if [ -n "${HPC_TARGET:-}" ]; then echo "$HPC_TARGET"; return; fi

  # 2. Filesystem signature (robust on login + compute nodes)
  if [ -d "/scratch/jjlee" ]; then echo "chpc"; return; fi
  if [ -d "/storage3/fs1/jjlee" ]; then echo "ris"; return; fi

  # 3. Hostname
  local fqdn
  fqdn="$(hostname -f 2>/dev/null || hostname)"
  case "$fqdn" in
    *.chpc.wustl.edu) echo "chpc"; return ;;
    c2-login*|*.ris.wustl.edu) echo "ris"; return ;;
  esac

  # 4. Sticky preference file — the operator's per-machine default.
  #    Consulted BEFORE the SSH probe so the MacBook resolves
  #    deterministically instead of racing the two login nodes.
  local pref_file="${DOTFOLDER}/.hpc_target"
  if [ -f "$pref_file" ]; then
    local pref
    pref="$(tr -d '[:space:]' < "$pref_file")"
    if [ -n "$pref" ]; then
      echo "[hpc] HPC_TARGET not set — using '${pref}' from ${pref_file}." >&2
      echo "$pref"; return
    fi
  fi

  # 4.5. SINGLE_CLUSTER from hpc/project.conf — a project.conf-pinned
  #      cluster wins over the SSH probe race but yields to explicit env
  #      var, on-cluster detection, hostname, and operator sticky-pref file.
  #      $SINGLE_CLUSTER is sourced from project.conf at the top of
  #      config.sh, so it's defined here if and only if the project pinned
  #      a single cluster.
  if [ -n "${SINGLE_CLUSTER:-}" ]; then
    case "$SINGLE_CLUSTER" in
      chpc|ris)
        echo "[hpc] HPC_TARGET not set — using '${SINGLE_CLUSTER}' from project.conf SINGLE_CLUSTER." >&2
        echo "$SINGLE_CLUSTER"; return
        ;;
      *)
        echo "[hpc] WARNING: ignoring invalid SINGLE_CLUSTER='${SINGLE_CLUSTER}' (must be chpc or ris)" >&2
        ;;
    esac
  fi

  # 5. SSH probe (last resort: no env var, not on a cluster, no sticky
  #    file, no SINGLE_CLUSTER pin). The probe order is NOT authoritative —
  #    whichever login node answers first wins — so this path announces
  #    itself loudly.
  if ssh -o ConnectTimeout=3 -o BatchMode=yes c2-login-001 true 2>/dev/null; then
    echo "[hpc] HPC_TARGET auto-detected as 'ris' via SSH probe — order is a race; create ${pref_file} to pin." >&2
    echo "ris"; return
  fi
  if ssh -o ConnectTimeout=3 -o BatchMode=yes login3.chpc.wustl.edu true 2>/dev/null; then
    echo "[hpc] HPC_TARGET auto-detected as 'chpc' via SSH probe — order is a race; create ${pref_file} to pin." >&2
    echo "chpc"; return
  fi

  # 6. Fail explicitly — never silently guess wrong
  echo "ERROR: Cannot detect HPC target. Set HPC_TARGET=chpc|ris or create ${DOTFOLDER}/.hpc_target" >&2
  exit 1
}
HPC_TARGET="$(_detect_hpc_target)"

case "$HPC_TARGET" in
  chpc)
    REMOTE_HOST="login3.chpc.wustl.edu"
    HPC_SCRATCH="/scratch/jjlee"
    HPC_PROJECTS="${HPC_SCRATCH}/PycharmProjects"
    SLURM_ACCOUNT="joshua_shimony"
    # v5 baseline (2026-05-12, per SPEC_v5_generation.md):
    #   AnalyticSignalHCP_nofilt/ renamed to *_v3_quarantine_2026-05-12/.
    #   Canonical outputs live under data/AnalyticSignalHCP/ (relocated
    #   from the parent's AnalyticSignalHCP/ on 2026-05-19; see
    #   docs/playbook_2026-05-19_data_quarantine_cleanup.md §5.2).
    CEPH_DATA_ROOT="/ceph/chpc/shared/joshua_shimony_group/jjlee/data/AnalyticSignalHCP"
    # Per-cluster staging directory for SLURM precompute output. CHPC writes
    # to fast /scratch first, then the playbook's promote step rsyncs to /ceph
    # (CEPH_DATA_ROOT). RIS has no /scratch ↔ /ceph split, so on RIS the
    # staging dir equals CEPH_DATA_ROOT (see ris) stanza below).
    PRECOMPUTE_STAGING_DIR="${HPC_SCRATCH}/Singularity/AnalyticSignalHCP/precomputed_391training"
    PARTITION_CPU="tier1_cpu"
    PARTITION_GPU="free_gpu"
    PARTITION_SHORT="free_cpu"
    # Conda lives at /home/jjlee/miniconda3 on CHPC (real directory, not a
    # symlink to /scratch). CLAUDE.md claims ~/miniconda3 → /scratch/jjlee/
    # miniconda3 but that symlink doesn't exist on the live cluster as of
    # 2026-05-17; CHPC smoke 6719629 failed at sbatch line 49 with
    # "No such file or directory" because $HPC_SCRATCH/miniconda3 doesn't
    # exist. Use an absolute cluster path to match the eager-expansion
    # pattern used elsewhere in this stanza (HPC_SCRATCH, HPC_PROJECTS,
    # CEPH_DATA_ROOT) — `${HOME}` would resolve differently when this
    # script is sourced on MacBook vs inside an sbatch on CHPC.
    CONDA_PREFIX_HPC="/home/jjlee/miniconda3"
    CONTAINER_SUPPORT=false
    CONTAINER_IMAGE=""
    CONTAINER_MOUNTS=""
    ;;
  ris)
    REMOTE_HOST="c2-login-001"  # SSH config alias; no load balancer
    HPC_SCRATCH="/storage3/fs1/jjlee/Active"
    HPC_PROJECTS="${HPC_SCRATCH}/PycharmProjects"
    SLURM_ACCOUNT="compute2-jjlee"
    # v5 baseline — see chpc note above.
    CEPH_DATA_ROOT="${HPC_SCRATCH}/data/AnalyticSignalHCP"
    # RIS has no /scratch ↔ /ceph split: precompute writes directly to the
    # canonical location, no promote step.
    PRECOMPUTE_STAGING_DIR="${CEPH_DATA_ROOT}/precomputed_391training"
    PARTITION_CPU="general-cpu"
    PARTITION_GPU="general-gpu"
    PARTITION_SHORT="general-short"
    CONDA_PREFIX_HPC="${HPC_SCRATCH}/miniforge3"
    CONTAINER_SUPPORT=true
    CONTAINER_IMAGE=""  # Set to registry reference when ready (e.g., docker.io/...)
    CONTAINER_MOUNTS="/storage3/fs1/jjlee/Active:/storage3/fs1/jjlee/Active,/scratch2/fs1/jjlee:/scratch2/fs1/jjlee"
    ;;
  *)
    echo "ERROR: Unknown HPC_TARGET='${HPC_TARGET}'. Use 'chpc' or 'ris'." >&2
    exit 1
    ;;
esac
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# ---------------------------------------------------------------------------
# Git remote naming
#
# Each HPC target gets its own local git remote (named after HPC_TARGET, e.g.
# `chpc` or `ris`) so both clusters can coexist as parallel push destinations.
# The legacy `hpc` remote — created before the dual-target split — is still
# honoured by the fan-out logic in supervisor-finalize.sh and /commit during
# migration. Once a target's per-name remote is set up, the legacy `hpc`
# remote may be removed manually:  git remote remove hpc
# ---------------------------------------------------------------------------

LOCAL_REMOTE_NAME="${HPC_TARGET}"

# All known HPC-class remote names, ordered for fan-out iteration.
# Add new clusters here as they come online.
HPC_REMOTE_NAMES=(chpc ris hpc)

# ---------------------------------------------------------------------------
# Globus endpoints (used by hpc/globus_sync.sh for bulk data transfer
# between MacBook, CHPC, and RIS). UUIDs are not secrets; commit-safe.
#
# Look up endpoint UUIDs once via:
#   globus endpoint search 'CHPC' --filter-owner-id <your-globus-id>
#   globus endpoint search 'RIS WashU'
#   globus endpoint search 'twistor'           # local GCP collection
# Env vars override the defaults below per-shell.
# ---------------------------------------------------------------------------

# CHPC — "Collection of datasets for RCIF_CHPC" (Mapped Collection, GCS)
GLOBUS_ENDPOINT_CHPC="${GLOBUS_ENDPOINT_CHPC:-4e6887fc-b5da-4283-93cb-dbfb7e2409d2}"

# RIS — High-Assurance Mapped Collection over Storage1 GPFS (POSIX, GCS)
GLOBUS_ENDPOINT_RIS="${GLOBUS_ENDPOINT_RIS:-b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4}"

# MacBook (twistor) — Globus Connect Personal collection on this machine
GLOBUS_ENDPOINT_MACBOOK="${GLOBUS_ENDPOINT_MACBOOK:-222f1acc-23dd-11f1-847b-0affe9efaee5}"

# Path prefixes on each endpoint — used to compose absolute Globus paths
# from project-relative subpaths (e.g. AnalyticSignalHCP_nofilt/precomputed/).
#
# CHPC: RCIF's Globus collection exposes ONLY the /ceph filesystem (not
#   /scratch), and the collection is rooted AT /ceph — so collection-relative
#   paths drop the /ceph/ prefix. The underlying POSIX path on a CHPC node
#   for the same data is /ceph/chpc/shared/joshua_shimony_group/jjlee/...
#
#   Project data on /scratch/jjlee/... (e.g. Singularity/AnalyticSignalHCP/
#   precomputed/) is NOT reachable via Globus directly — it must first be
#   staged to /ceph/chpc/shared/joshua_shimony_group/jjlee/... on the CHPC
#   side (rsync/cp on a CHPC node), then Globus can pick it up. The base
#   path below uses the collection-relative form so relative subpaths like
#   "AnalyticSignalHCP_nofilt/cache/" resolve correctly through Globus.
#
# RIS: the collection's restricted root is "/" POSIX with ACL-gated
#   visibility scoped to the user's home tree. Globus-relative paths
#   MUST be the FULL POSIX path including the /storage3/fs1/jjlee/Active
#   prefix; collection-relative paths that drop the prefix return 403
#   EndpointPermissionDenied (verified 2026-05-15 against b9545fe1's
#   `globus ls`). The base below holds that POSIX prefix; `--path foo/bar/`
#   composes to "/storage3/fs1/jjlee/Active/foo/bar/" which Globus then
#   resolves to the same POSIX path on the cluster.
#   Prior comment claimed the root was at /storage3/fs1/jjlee/Active and
#   the base was empty — that was incorrect and reproducibly produced 403s.
#
# MacBook (GCP): Globus Connect Personal serves only the paths listed in
#   ~/.globusonline/lta/config-paths (the "Access" tab in the GCP UI). By
#   default GCP exposes ~/ which Globus surfaces as /~/. The prefix below
#   matches whatever absolute paths you've allow-listed in GCP — set it
#   to /~ if only the home folder is exposed.
GLOBUS_PATH_CHPC="/chpc/shared/joshua_shimony_group/jjlee"
GLOBUS_PATH_RIS="/storage3/fs1/jjlee/Active"
GLOBUS_PATH_MACBOOK="/Users/jjlee"

require_ssh() {
  if ! ssh -q -o BatchMode=yes "$REMOTE" "exit" 2>/dev/null; then
    echo "ERROR: Cannot reach ${REMOTE}. Try: ssh-add ~/.ssh/id_rsa" >&2
    exit 1
  fi
}

# Returns 0 if the named git remote exists in the cwd repo, 1 otherwise.
remote_exists() {
  git config --get "remote.${1}.url" >/dev/null 2>&1
}

# Resolve the canonical project name from the *primary* git worktree.
#
# Under a secondary git worktree, $1's basename carries a branch slug
# (e.g. analytic_signal_sst-<branch>) that matches no HPC path — only the
# primary clone is synced via hpc/push.sh, so submit.sh / fetch.sh must
# target ${HPC_PROJECTS}/analytic_signal_sst regardless of which worktree
# they are invoked from. The primary worktree is always the first
# `worktree` entry of `git worktree list --porcelain`. Falls back to the
# directory basename outside a git repo (e.g. a tarball deployment).
#
# Usage: PROJECT_NAME="$(resolve_project_name "$PROJECT_ROOT")"
resolve_project_name() {
  local project_root="$1" primary_wt
  primary_wt="$(git -C "$project_root" worktree list --porcelain 2>/dev/null \
    | awk '$1=="worktree" && !seen {print $2; seen=1}')" || primary_wt=""
  if [ -n "$primary_wt" ]; then
    basename "$primary_wt"
  else
    basename "$project_root"
  fi
}
