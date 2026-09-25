## Summary

<!-- What does this change, and why? Link the issue it resolves, e.g. "Closes #12". -->

## Checklist

- [ ] Tests cover the change, and `uv run pytest -n auto` passes.
- [ ] `uv run ruff check .`, `uv run ruff format --check .` and `pyright` pass.
- [ ] The docstrings, READMEs and `docs/` of the changed packages are up to date, and `uv run --group docs python docs/build_all.py` passes.
- [ ] The `CHANGELOG.md` of each changed published package has an entry under `Unreleased`.
