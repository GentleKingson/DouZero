# Packaging and Python support

> P01 establishes the Python support policy and consolidates project metadata
> under `pyproject.toml` (PEP 621).

## Supported Python

**Minimum: Python 3.11.** Tested on 3.11, 3.12, and 3.13 (mandatory CI matrix).

Rationale:
- Python 3.9 is end-of-life.
- Python 3.10 leaves maintenance around 2026-10 (imminent).
- Python 3.11 is maintained until around 2027-10.
- Current PyTorch requires Python `>=3.10`.

Python 3.14 is not yet in the mandatory matrix; it will be added (initially as
a non-blocking smoke) once `rlcard` / `GitPython` compatibility on 3.14 is
confirmed. See `docs/reproducibility.md` → "CI Python matrix".

## Where metadata lives

As of P01, `pyproject.toml` is the **single source of truth** for project
metadata (PEP 621 `[project]` table): name, dynamic version, description,
`requires-python`, license (SPDX `Apache-2.0`), classifiers, and dependencies.

- **Version** is dynamic, read from `douzero/_version.__version__`, so there is
  one source of truth shared with `import douzero` runtime code.
- **`setup.py`** is retained as a thin shim (`setuptools.setup()` with no
  arguments) for backward compatibility with tooling that invokes it directly.
  Do **not** re-declare metadata there — duplicate declarations can conflict
  with `pyproject.toml`.
- **License** uses the PEP 639 SPDX expression `license = "Apache-2.0"`; the
  legacy `License :: OSI Approved :: Apache Software License` classifier was
  removed because setuptools ≥ 83 forbids mixing the two.

## Runtime dependencies

Declared in `pyproject.toml` `[project].dependencies` (and mirrored in
`requirements.txt` for source installs):

| Dependency | Why |
|---|---|
| `torch` | models, training, evaluation |
| `rlcard` | the rule-based RLCard baseline agent |
| `GitPython` | run-metadata stamping in `file_writer.py` (training logging) |
| `pyyaml` | `--config <yaml>` loading (`douzero.config`) |

`GitPython` is imported **lazily** inside `gather_metadata()` (training only),
so `import douzero.dmc` and `train.py --help` work without it at import time —
but it is a declared dependency because actual training uses it. The legacy
`gitdb2` line was removed from `requirements.txt` (GitPython already depends
on the maintained `gitdb`).

## Wheel install and import contract

A plain `pip install dist/*.whl` must make the package importable. Verified by
the `Building` CI workflow, which builds the wheel, installs it, and imports
`douzero`, `douzero.dmc`, `douzero.env`, `douzero.evaluation` from a directory
**outside** the source tree (asserting the import resolves to the installed
wheel, not the checkout):

```bash
python -m pip install --upgrade pip build
python -m build
python -m pip install dist/*.whl
cd "$RUNNER_TEMP"
python -c "import douzero, douzero.dmc, douzero.env, douzero.evaluation"
```

## Build requirements

`[build-system]` requires `setuptools>=77` (for PEP 621 support) and `wheel`.
