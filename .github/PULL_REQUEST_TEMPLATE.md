<!--
PR template — minimal hpc-toolkit baseline.

Branch name SHOULD start with feature/, fix/, experiment/, or chore/.
Fill in both sections before marking the PR Ready-for-review; the body
becomes the merged commit message on squash-merge.
-->

## Summary

<!--
1-3 paragraphs. Why this change matters. What it does, not just how.
A future operator skimming `git log main` should be able to reconstruct
the motivation from this body alone.
-->


## CI checklist

- [ ] `hpc-ci` status check green on the PR head commit
      (submit via `bash hpc/ci_pr.sh <branch>` — Pattern A′)
- [ ] `hpc-ci-gpu` status green if `--gpu` shard was needed (optional;
      required only for GPU-touching PRs)
- [ ] `CHANGELOG.md` entry added under `[Unreleased]` (if the project
      tracks one)
