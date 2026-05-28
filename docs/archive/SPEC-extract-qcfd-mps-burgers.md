# SPEC — Extract `murali_burgers` into standalone repo `q8020-mps-burgers`

Self-contained handoff for an agent that has not seen the prior
conversation. Phase 0 (capture-commit of axequalsb) is already done.
This spec covers Phases 1–4. Phases 5 (push to GitHub) and 6 (delete
from axequalsb) are explicitly **out of scope** and will be done by the
user after they validate Phase 4 with q8020 sweeps.

## 0. Context

`/Users/agallojr/proj/src/q8020/q8020-cfd-axequalsb/src/murali_burgers/`
is a self-contained Burgers / Cole-Hopf quantum-circuit pipeline that
has outgrown its host repo (`q8020-cfd-axequalsb`). It needs its own
top-level workspace member so it can have its own deps, its own GitHub
repo, and its own release cadence.

Source root after extraction:
`/Users/agallojr/proj/src/q8020/q8020-mps-burgers/`

## 1. Decisions (do not relitigate)

- **Repo name:** `q8020-mps-burgers`
- **Layout:** flat — files in `src/` directly, no `__init__.py`, no
  package nesting. The current code already imports as bare modules
  (`from burgers_cole_hopf import ...`); flat preserves that with
  zero import rewrites in the modules themselves.
- **Specs/docs/post-procs:** in `docs/` (sibling of `src/`)
- **Tests:** in `tests/` (sibling of `src/` and `docs/`) — split out of
  the old `analysis/` directory
- **Run droppings to delete:** `q8020_analysis_0.json`,
  `q8020_artifacts_0.json`, `q8020_case_0.json`,
  `q8020_results_0.json` currently sitting in
  `src/murali_burgers/`. Do NOT carry these forward.
- **Python:** 3.12 only (matches axequalsb)

## 2. Target layout

```
q8020-mps-burgers/
├── .python-version           # contents: "3.12\n"
├── pyproject.toml
├── README.md                 # short, points at docs/
├── src/
│   ├── burgers_classical.py
│   ├── burgers_cole_hopf.py
│   ├── burgers_cole_hopf_circuit.py
│   ├── burgers_encoding.py
│   ├── burgers_mpo.py
│   ├── burgers_mps.py
│   ├── burgers_nonlinear.py
│   ├── burgers_sign_recovery.py
│   ├── burgers_solver.py
│   ├── burgers_tebd.py
│   └── burgers_trotter.py
├── docs/
│   ├── *.md                  # all specs/reviews/handoffs from analysis/
│   ├── plot_*.py             # all post-processors
│   └── *.png, *.gif          # existing figures
├── tests/
│   └── test_burgers.py
│   └── test_cole_hopf_circuit.py
│   └── test_mpo.py
│   └── test_nonlinear.py
│   └── test_encoding_switch.py   # NOTE: does not yet exist; not your job
└── input/
    └── burgers_quantum.toml
```

## 3. `pyproject.toml`

Start from `q8020-cfd-axequalsb/pyproject.toml` and produce:

```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "q8020-mps-burgers"
version = "0.1.0"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "qiskit>=2.0",
    "qiskit_aer",
    "matplotlib",
    "numpy",
    "scipy",
    "quimb",
    "numba>=0.59",
    "pytest",
    "q8020_cfd_metautil @ git+https://github.com/Q8020-CFD/q8020-cfd-metautil",
    "q8020_cfd_qutil @ git+https://github.com/Q8020-CFD/q8020-cfd-qutil",
]

[tool.uv.sources]
q8020_cfd_metautil = { workspace = true }
q8020_cfd_qutil = { workspace = true }

[tool.setuptools]
package-dir = {"" = "src"}
py-modules = [
    "burgers_classical",
    "burgers_cole_hopf",
    "burgers_cole_hopf_circuit",
    "burgers_encoding",
    "burgers_mpo",
    "burgers_mps",
    "burgers_nonlinear",
    "burgers_sign_recovery",
    "burgers_solver",
    "burgers_tebd",
    "burgers_trotter",
]

[tool.black]
line-length = 88

[tool.ruff]
line-length = 88

[tool.pylint.messages_control]
disable = [
    "C0103",
    "C0305",
    "C0303",
    "C0301",
    "W0621",
]
```

Notes:
- `q8020_cfd_qutil` is required: `burgers_trotter.py` imports
  `q8020_cfd_qutil.backend.get_backend` at lines 260 and 736
  (lazy imports for hardware paths). Do NOT drop it.
- `q8020_cfd_metautil` is required: `burgers_solver.py` imports it at
  module top (lines 26-27); plot_*.py post-procs use it heavily.
- `qrisp` and `quantum_linear_solvers`: confirmed unused by
  murali_burgers. Drop them.
- `scipy` and `pytest` are surfaced explicitly (axequalsb relied on
  transitive resolution).

## 4. Workspace wire-up

Edit `/Users/agallojr/proj/src/q8020/pyproject.toml`:

- Add `"q8020-mps-burgers"` to `[tool.uv.workspace] members`
- Add `q8020-mps-burgers = { workspace = true }` under
  `[tool.uv.sources]`

Then run `uv sync` from `/Users/agallojr/proj/src/q8020/`. It must
succeed. Do NOT modify the workspace `pyproject.toml` in any other way.

## 5. Step-by-step execution

### 5.1 Create skeleton

```
mkdir -p /Users/agallojr/proj/src/q8020/q8020-mps-burgers/{src,docs,tests,input}
```

Write `.python-version` (single line: `3.12`), `pyproject.toml` (per
§3), and a 5-line `README.md` that says what the package is and points
at `docs/HANDOFF-burgers-pipeline.md`.

### 5.2 Move source modules

Copy these eleven files from
`q8020-cfd-axequalsb/src/murali_burgers/*.py` to
`q8020-mps-burgers/src/`:

`burgers_classical.py`, `burgers_cole_hopf.py`,
`burgers_cole_hopf_circuit.py`, `burgers_encoding.py`,
`burgers_mpo.py`, `burgers_mps.py`, `burgers_nonlinear.py`,
`burgers_sign_recovery.py`, `burgers_solver.py`, `burgers_tebd.py`,
`burgers_trotter.py`.

**Do not modify import statements in these files.** The bare-import
style (`from burgers_cole_hopf import ...`) keeps working when `src/`
is on `sys.path`, which `burgers_solver.py` arranges via the standard
script-relative `sys.path.insert` it already does. Verify by reading
the top of `burgers_solver.py` — if no path mangling exists, add:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
```

near the top of `burgers_solver.py` only. (The other modules import
each other through this same path once the entry script puts `src/`
on the path.)

### 5.3 Move docs

From `q8020-cfd-axequalsb/src/murali_burgers/analysis/`:
- All `*.md` → `q8020-mps-burgers/docs/`
- All `plot_*.py` and `animate_*.py` → `q8020-mps-burgers/docs/`
- All `*.png`, `*.gif` → `q8020-mps-burgers/docs/`

### 5.4 Move tests

From `q8020-cfd-axequalsb/src/murali_burgers/analysis/`:
- All `test_*.py` → `q8020-mps-burgers/tests/`

Then update each `test_*.py`'s `sys.path` insertion to point at
`../src` instead of the old `murali_burgers/`. Concretely the
existing pattern is something like:

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

After the move it should be:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

Verify by reading each test file post-edit and running `pytest tests/`
from the new repo root. All four existing test files
(`test_burgers.py`, `test_cole_hopf_circuit.py`, `test_mpo.py`,
`test_nonlinear.py`) must pass.

### 5.5 Move input TOML

Copy `q8020-cfd-axequalsb/input/burgers_quantum.toml` to
`q8020-mps-burgers/input/burgers_quantum.toml`.

Rewrite path references inside it (currently 14 lines, see grep
results in §7 below). All `_script` and `_group_postproc` paths of the
form:

- `./q8020-cfd-axequalsb/src/murali_burgers/burgers_solver.py`
  → `./q8020-mps-burgers/src/burgers_solver.py`
- `./q8020-cfd-axequalsb/src/murali_burgers/analysis/plot_*.py`
  → `./q8020-mps-burgers/docs/plot_*.py`

Verify with: `grep -n "murali_burgers\|axequalsb" input/burgers_quantum.toml`
must return empty.

### 5.6 Delete run droppings

Do NOT carry forward. They were never meant to be checked in:
- `q8020_analysis_0.json`
- `q8020_artifacts_0.json`
- `q8020_case_0.json`
- `q8020_results_0.json`

These exist in the SOURCE directory only. After the copy to the new
repo, do NOT include them in the new `src/`. (Don't delete them from
axequalsb yet — that's Phase 6, user's call.)

### 5.7 Workspace registration

Per §4. After `uv sync` succeeds, sanity-check by running from the
workspace root:

```
cd /Users/agallojr/proj/src/q8020
uv run python -c "import burgers_solver" \
  --with q8020-mps-burgers
```

(or equivalent — the goal is "the new package is importable through
the workspace").

### 5.8 Smoke test

From `/Users/agallojr/proj/src/q8020/q8020-mps-burgers/`:

1. `pytest tests/` — all four existing test files pass.
2. `python src/burgers_solver.py --help` — argparse output renders, no
   import errors. `--encoding` flag is present.

If both pass, Phase 4 is done. Stop and report. The user will then
drive q8020 sweeps against the new repo to validate end-to-end before
Phase 5/6.

## 6. Out of scope

- Pushing the new repo to GitHub (Phase 5 — user does this)
- Deleting `murali_burgers/` from axequalsb (Phase 6 — user does this
  after validation)
- Writing the missing `test_encoding_switch.py` (separate task, see
  `docs/SPEC-encoding-switch.md` §5)
- Refactoring the flat layout into a real package
- Renaming the `burgers_*` modules
- Changing any code behavior

## 7. Verification grep targets

After all moves, these must hold inside `q8020-mps-burgers/`:

- `grep -rn "murali_burgers" .` → empty
- `grep -rn "q8020-cfd-axequalsb" .` → empty
- `grep -rn "from burgers_" src/` → matches existing inter-module imports
- `grep -rn "from q8020_cfd_metautil\|from q8020_cfd_qutil" .` →
  same hits as before the move (none added, none lost)

The 14 path references in `burgers_quantum.toml` that need rewriting
were at lines 21, 128, 143, 298, 311, 336, 348, 360, 376, 396, 410,
448, 463, 498 in the source file — use these as a checklist.

## 8. Acceptance

- New repo skeleton at `q8020-mps-burgers/` matches §2 layout exactly
- `pyproject.toml` matches §3
- Workspace `pyproject.toml` updated per §4 and `uv sync` is clean
- `pytest tests/` from new repo: 4 test files all green
- `python src/burgers_solver.py --help` from new repo: clean
- All §7 grep checks pass
- axequalsb is untouched (Phase 6 is the user's job, not yours)
