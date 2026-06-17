"""Animate every run listed in a datasets JSON onto one set of axes.

Unlike plot_method_compare.py (which globs a single sweep dir and keys
series by method), this reads a curated dataset descriptor -- a group of
runs with shared physics but different runtime params (target, backend,
shots) -- and draws each run as its own labelled line in one movie.

Usage:
    plot_dataset_movie.py <dataset.json> [--out movie.gif] [--fps 2]

The dataset JSON shape (see analysis/.../smooth_movie_q3.json):
    {
      "name": "...", "global": {...},
      "runs": [
        {"label": "...", "method": "...", "path": "<rel-to-json>", ...},
        ...
      ]
    }

Each run's solution curves come from a q8020_artifacts_*.json holding
{"grid": [...], "solution_steps": {"0": [...], ...}}.  Hardware runs
store theirs under <path>/method_compare/<method>/; sim/reference runs
store theirs directly in <path>.  Both layouts are resolved here.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np


def _shock_pct(g: dict) -> float | None:
    """Effective % of inviscid shock time the run reached.  Uses the
    dataset's explicit value if present, else derives it from the global
    physics: t_end = n_steps * cfl * dx, t_shock = 1/max|du0/dx| on the
    N=2^q grid for the sine IC u0 = A*sin(2*pi*x)."""
    if "shock_pct_effective" in g:
        return float(g["shock_pct_effective"])
    try:
        q = int(g["q"])
        cfl = float(g["cfl"])
        n_steps = int(g["n_steps"])
        amp = float(g["ic_amplitude"])
    except (KeyError, TypeError, ValueError):
        return None
    n = 2 ** q
    dx = 1.0 / n
    x = np.arange(n) * dx
    u0 = amp * np.sin(2 * np.pi * x)
    max_grad = float(np.max(np.abs(np.gradient(u0, dx))))
    if max_grad <= 0:
        return None
    t_shock = 1.0 / max_grad
    t_end = n_steps * cfl * dx
    return 100.0 * t_end / t_shock


def _find_artifacts(run_dir: Path, method: str) -> Path | None:
    """Locate the artifacts JSON for a run, trying the direct case dir
    first, then the hardware method_compare/<method>/ subdir."""
    candidates = [
        run_dir,
        run_dir / "method_compare" / method,
    ]
    for d in candidates:
        hits = sorted(d.glob("q8020_artifacts_*.json"))
        if hits:
            return hits[0]
    return None


def _load_run(run: dict, base: Path) -> dict | None:
    """Read a run's grid + per-step snapshots; None if unavailable."""
    run_dir = (base / run["path"]).resolve()
    art = _find_artifacts(run_dir, run.get("method", ""))
    if art is None:
        print(f"  {run['label']}: no artifacts under {run_dir}",
              file=sys.stderr)
        return None
    data = json.loads(art.read_text())
    grid = np.array(data.get("grid", []))
    steps = {
        int(k): np.array(v)
        for k, v in data.get("solution_steps", {}).items()
    }
    if grid.size == 0 or not steps:
        print(f"  {run['label']}: empty grid/steps", file=sys.stderr)
        return None
    return {"label": run["label"], "grid": grid, "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="path to the dataset JSON")
    ap.add_argument("--out", default=None, help="output GIF path")
    ap.add_argument("--fps", type=int, default=2, help="frames per second")
    args = ap.parse_args()

    ds_path = Path(args.dataset).resolve()
    ds = json.loads(ds_path.read_text())
    base = ds_path.parent

    runs = []
    for r in ds.get("runs", []):
        loaded = _load_run(r, base)
        if loaded is not None:
            runs.append(loaded)
    if not runs:
        print("No usable runs found.", file=sys.stderr)
        sys.exit(1)

    # Common frame timeline = union of every run's step keys.
    all_steps = sorted({s for run in runs for s in run["steps"]})

    # Identify the reference run (the error baseline) from the dataset.
    # Error panel measures every OTHER run's L2 distance to it; the
    # reference itself is drawn in the profile panel but not the error
    # panel (its error is zero by definition).
    ref_label = None
    for r in ds.get("runs", []):
        if r.get("method") == "ftcs_reference" or r.get("target") == "reference":
            ref_label = r["label"]
            break
    ref_run = next((r for r in runs if r["label"] == ref_label), None)

    def _nearest_step(run: dict, step: int) -> np.ndarray:
        keys = sorted(run["steps"])
        k = min(keys, key=lambda kk: abs(kk - step))
        return run["steps"][k]

    # Precompute per-step L2 error of each non-reference run vs the
    # reference, on the reference's grid.
    err_runs = [r for r in runs if r["label"] != ref_label]
    err_curve: dict[str, list[float]] = {}
    if ref_run is not None:
        for run in err_runs:
            errs = []
            for s in all_steps:
                diff = _nearest_step(run, s) - _nearest_step(ref_run, s)
                errs.append(float(np.sqrt(np.mean(diff ** 2))))
            err_curve[run["label"]] = errs

    # y-limits across every snapshot, with a small margin.
    all_vals = np.concatenate(
        [v for run in runs for v in run["steps"].values()]
    )
    ymin, ymax = float(all_vals.min()), float(all_vals.max())
    pad = 0.08 * (ymax - ymin or 1.0)
    ymin, ymax = ymin - pad, ymax + pad

    have_err = bool(err_curve)
    if have_err:
        fig, (ax, ax_err) = plt.subplots(
            2, 1, figsize=(9, 9), height_ratios=[2, 1],
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax_err = None

    cmap = plt.get_cmap("tab10")
    color = {run["label"]: cmap(i % 10) for i, run in enumerate(runs)}

    # ── profile panel ──────────────────────────────────────────────
    lines = {}
    for run in runs:
        (ln,) = ax.plot(
            [], [], marker="o", ms=4, lw=1.8,
            color=color[run["label"]], label=run["label"],
        )
        lines[run["label"]] = ln

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    title = ax.set_title("")
    g = ds.get("global", {})
    shock = _shock_pct(g)
    shock_str = f", to shock {shock:.0f}%" if shock is not None else ""
    subtitle = (
        f"{ds.get('name', ds_path.stem)}  "
        f"(q={g.get('q', '?')}, nu={g.get('nu', '?')}, "
        f"A={g.get('ic_amplitude', '?')}, cfl={g.get('cfl', '?')}, "
        f"{g.get('bc', '?')}{shock_str})"
    )

    # ── error panel ────────────────────────────────────────────────
    err_lines = {}
    if have_err:
        emax = max(max(v) for v in err_curve.values())
        for run in err_runs:
            (ln,) = ax_err.plot(
                [], [], marker="o", ms=4, lw=1.8,
                color=color[run["label"]], label=run["label"],
            )
            err_lines[run["label"]] = ln
        ax_err.set_xlim(all_steps[0], all_steps[-1])
        ax_err.set_ylim(0.0, emax * 1.08 or 1.0)
        ax_err.set_xlabel("step")
        ax_err.set_ylabel(f"L2 error vs {ref_label}")
        ax_err.grid(True, alpha=0.3)

    def update(frame_idx: int):
        step = all_steps[frame_idx]
        for run in runs:
            y = _nearest_step(run, step)
            lines[run["label"]].set_data(run["grid"], y)
        title.set_text(f"{subtitle}\nstep {step}/{all_steps[-1]}")
        artists = list(lines.values()) + [title]
        if have_err:
            xs = all_steps[: frame_idx + 1]
            for label, ln in err_lines.items():
                ln.set_data(xs, err_curve[label][: frame_idx + 1])
            artists += list(err_lines.values())
        return artists

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    anim = manimation.FuncAnimation(
        fig, update, frames=len(all_steps), interval=1000 / args.fps,
        blit=False,
    )

    out = Path(args.out) if args.out else ds_path.with_suffix(".gif")
    anim.save(str(out), writer=manimation.PillowWriter(fps=args.fps))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
