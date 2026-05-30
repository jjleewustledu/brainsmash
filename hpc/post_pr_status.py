#!/usr/bin/env python3
"""Post a GitHub commit status from an HPC job (Pattern A′ replacement for Actions).

Usage:
    python hpc/post_pr_status.py \
        --sha <40-char-sha> \
        --state pending|success|failure|error \
        --context hpc-ci \
        --description "<≤140-char human-readable>" \
        --target-url <link-to-log-tree>

This script is the substrate of the HPC-driven PR gate per
``SPEC_development_workflow.md`` Pattern A′. The HPC SLURM job posts:

1. ``pending`` at job start (via the sbatch prolog).
2. ``success`` or ``failure`` at job end (via the sbatch epilog or
   aggregator job).

The GitHub Statuses API endpoint is free — it does NOT trigger Actions
runs and does NOT accumulate Actions minutes. Authentication is via the
``gh`` CLI (``gh auth status`` must succeed on the calling host); the
script delegates to ``gh api`` rather than using a raw token, so token
storage stays in ``gh``'s normal location (``~/.config/gh/hosts.yml`` on
CHPC/RIS).

Operator setup (one-time per HPC host, or whenever the PAT expires —
fine-grained PATs typically max ~1 year for "Commit statuses:
Read and write"):

    # Generate a fine-grained PAT on github.com/settings/tokens?type=beta
    # with: Repository access → Only select repositories → analytic-signal-sst;
    # Permissions → Commit statuses: Read and write (nothing else).
    # Copy the github_pat_… string, then on the MacBook:

    pbpaste | bash hpc/install_gh_pat.sh

That helper script:
  - Strips whitespace from the PAT (the trailing newline from pbpaste
    is otherwise rejected by gh's HTTP layer as an invalid Authorization
    header value).
  - Validates the prefix (must be github_pat_… or ghp_…).
  - For each cluster (CHPC + RIS by default; --cluster=chpc|ris to scope
    to one), pipes the cleaned PAT to ``gh auth login --with-token``
    over SSH.
  - Verifies via ``gh auth status`` AND ``gh api user`` (the latter
    exercises the actual API call this module makes).
  - Reports per-cluster success/failure.

Classic PATs require only ``repo:status`` scope (not full ``repo``).
Fine-grained PATs are preferred for least-privilege.

Exit code: 0 on successful POST, 1 on any error. The HPC job's overall
status is independent — a failed status-post should NOT mask a failed
test run. Tests should fail-loud via their own log paths; the
commit-status is purely a UI signal to GitHub.

Spec reference: ``SPEC_development_workflow.md`` Pattern A′, ``SPEC_continuous_integration.md`` §2.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

VALID_STATES = ("pending", "success", "failure", "error")
DEFAULT_CONTEXT = "hpc-ci"
DEFAULT_REPO = "jjleewustledu/analytic-signal-sst"


def _post_status(
    *,
    sha: str,
    state: str,
    context: str,
    description: str,
    target_url: str | None,
    repo: str,
) -> int:
    """POST to /repos/{owner}/{repo}/statuses/{sha} via the gh CLI.

    Parameters
    ----------
    sha : str
        Full 40-char commit SHA to attach the status to.
    state : str
        One of ``pending``, ``success``, ``failure``, ``error``.
    context : str
        Status context (appears in the PR's Checks panel). Must match
        the name registered in branch protection's required-status-checks
        for the gate to enforce. Default ``hpc-ci``.
    description : str
        Short human-readable summary (≤140 chars). Truncated if longer.
    target_url : str, optional
        URL the operator can click to see the run log. CHPC/RIS log dir
        is not GitHub-accessible, so this typically points at a local
        path string the operator can paste into ``ssh`` or ``rsync``,
        or to a future log-aggregator service.
    repo : str
        ``<owner>/<repo>`` form. Default ``jjleewustledu/analytic-signal-sst``.

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    if state not in VALID_STATES:
        print(f"ERROR: state must be one of {VALID_STATES}, got {state!r}", file=sys.stderr)
        return 1
    if len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha.lower()):
        print(f"ERROR: sha must be a 40-char hex string, got {sha!r}", file=sys.stderr)
        return 1
    # GitHub truncates description at 140; pre-truncate so we never see a 422.
    if len(description) > 140:
        description = description[:137] + "..."

    body = {"state": state, "context": context, "description": description}
    if target_url:
        body["target_url"] = target_url

    cmd = [
        "gh",
        "api",
        "-X",
        "POST",
        f"/repos/{repo}/statuses/{sha}",
        "--input",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        input=json.dumps(body),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        print(
            f"ERROR: gh api failed (rc={proc.returncode}):\n  stderr: {proc.stderr.strip()}",
            file=sys.stderr,
        )
        return 1
    # Successful response is a JSON object with an "id" field.
    try:
        result = json.loads(proc.stdout)
        status_id = result.get("id")
        print(
            f"[post_pr_status] OK — context={context} state={state} sha={sha[:8]} id={status_id}"
        )
    except json.JSONDecodeError:
        print(
            f"[post_pr_status] OK — context={context} state={state} sha={sha[:8]} (raw: {proc.stdout.strip()[:80]})"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post a GitHub commit status from HPC (Pattern A′).",
    )
    parser.add_argument("--sha", required=True, help="40-char commit SHA")
    parser.add_argument(
        "--state",
        required=True,
        choices=VALID_STATES,
        help="Status state to post",
    )
    parser.add_argument(
        "--context",
        default=DEFAULT_CONTEXT,
        help=f"Status context label (default: {DEFAULT_CONTEXT})",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Short human-readable summary (≤140 chars)",
    )
    parser.add_argument(
        "--target-url",
        default=None,
        help="URL to a log/run page the operator can click",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GH_REPO", DEFAULT_REPO),
        help=f"<owner>/<repo> form (default from $GH_REPO or {DEFAULT_REPO})",
    )
    args = parser.parse_args()
    return _post_status(
        sha=args.sha,
        state=args.state,
        context=args.context,
        description=args.description or f"hpc-ci {args.state}",
        target_url=args.target_url,
        repo=args.repo,
    )


if __name__ == "__main__":
    sys.exit(main())
