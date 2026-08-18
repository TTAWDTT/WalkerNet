# Open-source Release Checklist

## Current tree

Before every public release:

1. Run `python scripts/release/audit_public_tree.py`.
2. Run `python -m pytest -q`.
3. Confirm that raw data, caches, checkpoints, logs, and `outputs/` are not
   tracked.
4. Confirm that every README command uses public relative paths or explicit
   `/path/to/...` placeholders.
5. Publish model/data artifacts separately with checksums and licenses.

## Git history

Removing a private path from the latest commit does not remove it from earlier
commits. The existing research history contains machine-specific experiment
paths. Before making the repository public, choose one of:

- create a new public repository from a reviewed squash commit; or
- rewrite all refs with `git filter-repo`, coordinate with collaborators,
  force-push, and require everyone to clone again.

The squash-release approach is recommended because it preserves the private
research repository while exposing only the reviewed public tree.

Do not make the current historical repository public until this step is done.
