# Contributing

1. Create a focused branch and keep unrelated experiment output out of Git.
2. Add or update tests for behavioral changes.
3. Run `python scripts/release/audit_public_tree.py` and
   `python -m pytest -q`.
4. Document changes to tensor shapes, normalization, climatology, loss, or
   evaluation definitions.
5. Use relative paths in public configs and explicit command-line arguments
   for machine-specific paths.

Generated data and checkpoints belong in external artifact storage. Only
small, selected figures and aggregate tables belong in `docs/assets/`.
