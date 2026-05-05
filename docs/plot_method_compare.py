"""Method comparison animation: FTCS vs Cole-Hopf vs LBM.

Reads sweep output from method_compare_q5_* runs and produces an
animated GIF overlaying all three methods on the same axes.

Two invocation modes:
  1. Group postproc:  python plot_method_compare.py <group_postproc.json>
  2. Manual:          python plot_method_compare.py --sweep-dir ~/q8020/<id>
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from q8020_cfd_metautil.harvest import harvest_metadata
from q8020_cfd_metautil.metakeys import _walk_case_dirs


# ── helpers ───────────────────────────────────────────────────────────

def _extract_case_params(meta: dict) -> dict | None:
    cases = meta.get('case', [])
    if not cases:
        return None
    for c in cases:
        if c.get('_source') == 'solver':
            return c
    for c in cases:
        params = c.get('params', {})
        if params:
            return {
                k.lstrip('-'): v for k, v in params.items()
                if not k.startswith('_')
            }
    return cases[0]


def _find_solver_entry(entries: list[dict], key: str) -> dict:
    for e in entries:
        if key in e:
            return e
    return {}


def _load_cases(sweep_dir: Path) -> list[dict]:
    case_dirs, no_meta = _walk_case_dirs(sweep_dir)
    out = []
    for cd in case_dirs + no_meta:
        cd = Path(cd)
        if not cd.is_dir():
            continue
        meta, _, _ = harvest_metadata(cd, read_only=True)
        if meta.get('case'):
            out.append(meta)
    return out


# ── main ──────────────────────────────────────────────────────────────

METHOD_STYLE = {
    'shift': {
        'color': '#1f77b4', 'ls': '-', 'lw': 1.5,
        'label': 'FTCS (shift-operator)',
    },
    'quantum_exact': {
        'color': '#ff7f0e', 'ls': '-', 'lw': 1.8,
        'label': 'Quantum exact (expm)',
    },
    'quantum_circuit': {
        'color': '#9467bd', 'ls': '-', 'lw': 1.5,
        'label': 'Pauli-Trotter (circuit)',
    },
    'cole_hopf': {
        'color': '#2ca02c', 'ls': '-', 'lw': 2.0,
        'label': 'Cole-Hopf (MPS)',
    },
    'qlbm': {
        'color': '#d62728', 'ls': '--', 'lw': 1.8,
        'label': 'LBM (D1Q3 BGK)',
    },
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('postproc_json', nargs='?', default=None)
    p.add_argument('--sweep-dir', type=Path, default=None)
    p.add_argument('--outfile', default=None)
    p.add_argument('--fps', type=int, default=8)
    args = p.parse_args()

    if args.postproc_json and Path(args.postproc_json).is_file():
        with open(args.postproc_json, encoding='utf-8') as f:
            pp = json.load(f)
        run_dir = Path(pp['run_dir'])
        all_cases = _load_cases(run_dir)
        outfile = args.outfile or str(
            run_dir / 'method_compare.gif',
        )
    elif args.sweep_dir:
        sd = args.sweep_dir.expanduser().resolve()
        all_cases = _load_cases(sd)
        outfile = args.outfile or 'method_compare.gif'
    else:
        print(
            "Provide either a postproc JSON or --sweep-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Group cases by method
    by_method: dict[str, dict] = {}
    for c in all_cases:
        cm = _extract_case_params(c)
        if cm is None:
            continue
        m = cm.get('method', '')
        if m in METHOD_STYLE:
            by_method[m] = c

    if len(by_method) < 2:
        print(
            f"Need at least 2 methods, found: {list(by_method)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Methods found: {list(by_method.keys())}",
        file=sys.stderr,
    )

    # Extract solution steps and grid from each method
    frames: dict[str, list[np.ndarray]] = {}
    step_keys_common = None
    x = None
    anchor_meta = None

    for m, case in by_method.items():
        art = _find_solver_entry(
            case.get('artifacts', []), 'solution_steps',
        )
        sol_steps = art.get('solution_steps', {})
        g = np.array(art.get('grid', []))
        if not sol_steps or g.size == 0:
            print(f"  {m}: no solution_steps, skipping", file=sys.stderr)
            continue

        keys = sorted(sol_steps.keys(), key=int)
        frames[m] = [np.array(sol_steps[k]) for k in keys]

        if step_keys_common is None:
            step_keys_common = keys
            x = g
            anchor_meta = _extract_case_params(case) or {}
        else:
            # Intersect to common step keys
            step_keys_common = [
                k for k in step_keys_common if k in keys
            ]

    if not frames or step_keys_common is None or x is None:
        print("No usable frames.", file=sys.stderr)
        sys.exit(1)

    # Trim frames to common steps
    for m in list(frames.keys()):
        art = _find_solver_entry(
            by_method[m].get('artifacts', []), 'solution_steps',
        )
        sol_steps = art.get('solution_steps', {})
        frames[m] = [np.array(sol_steps[k]) for k in step_keys_common]

    dt = float(anchor_meta.get('dt', 0.0))
    nu = float(anchor_meta.get('nu', 1e-2))
    q_val = anchor_meta.get('q', '?')
    n_steps_total = int(
        anchor_meta.get('n_steps', len(step_keys_common) - 1),
    )
    u0 = frames[next(iter(frames))][0].copy()

    # Recompute Godunov baseline from IC + params (not stored per-step).
    from burgers_classical import solve_burgers_godunov
    bc = anchor_meta.get('bc', 'periodic')
    print('  Computing Godunov reference ...', file=sys.stderr)
    sols_godunov = solve_burgers_godunov(
        u0, x, nu, dt, n_steps_total, bc=bc,
    )
    frames_ref = [sols_godunov[int(k)] for k in step_keys_common]

    # Truncate at divergence
    amp0 = max(np.max(np.abs(u0)), 1.0)
    amp_limit = 10.0 * amp0
    valid_end = len(step_keys_common)
    for i in range(len(step_keys_common)):
        for m in frames:
            f = frames[m][i]
            if not np.all(np.isfinite(f)):
                valid_end = i
                break
            if np.max(np.abs(f)) > amp_limit:
                valid_end = i
                break
        else:
            continue
        break
    if valid_end == 0:
        print("No valid frames.", file=sys.stderr)
        sys.exit(1)
    step_keys_common = step_keys_common[:valid_end]
    for m in frames:
        frames[m] = frames[m][:valid_end]
    frames_ref = frames_ref[:valid_end]

    # Build animation
    all_vals = [np.abs(np.array(fs)) for fs in frames.values()]
    all_vals.append(np.abs(np.array(frames_ref)))
    ymax = max(np.max(v) for v in all_vals) * 1.15
    times = np.array([int(k) * dt for k in step_keys_common])

    fig, (ax_u, ax_err) = plt.subplots(
        2, 1, figsize=(10, 6.5),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"Method Comparison: q={q_val} (N={2**int(q_val)}), "
        f"$\\nu$={nu:.0e}, {bc}, sine IC",
        fontsize=12, fontweight="bold",
    )

    ax_u.set_xlim(x[0], x[-1] + x[1] - x[0])
    ax_u.set_ylim(-ymax, ymax)
    ax_u.set_ylabel("u(x, t)")
    ax_u.grid(alpha=0.2)

    # IC + Godunov reference lines
    ax_u.plot(x, u0, 'k--', alpha=0.15, lw=1, label='IC')
    ref_line, = ax_u.plot(
        [], [], 'k-', lw=2.5, alpha=0.35, label='Godunov (reference)',
    )

    # Method lines
    lines: dict[str, plt.Line2D] = {}
    for m, sty in METHOD_STYLE.items():
        if m not in frames:
            continue
        ln, = ax_u.plot(
            [], [], ls=sty['ls'], color=sty['color'],
            lw=sty['lw'], label=sty['label'],
        )
        lines[m] = ln

    ax_u.legend(loc='lower left', fontsize=9)
    time_text = ax_u.text(
        0.02, 0.95, '', transform=ax_u.transAxes,
        fontsize=10, va='top', fontfamily='monospace',
    )

    # Error panel: each method vs Godunov
    err_lines: dict[str, plt.Line2D] = {}
    errors: dict[str, np.ndarray] = {}
    for m in frames:
        sty = METHOD_STYLE.get(m)
        if sty is None:
            continue
        err = np.zeros(len(step_keys_common))
        for i in range(len(step_keys_common)):
            norm_ref = np.linalg.norm(frames_ref[i])
            if norm_ref > 1e-15:
                err[i] = np.linalg.norm(
                    frames[m][i] - frames_ref[i],
                ) / norm_ref
        errors[m] = err
        ln, = ax_err.plot(
            [], [], ls=sty['ls'], color=sty['color'], lw=1.5,
            label=f'{sty["label"]} vs Godunov',
        )
        err_lines[m] = ln

    ax_err.set_xlim(times[0], times[-1])
    all_errs = np.concatenate(
        [e[e > 0] for e in errors.values()]
    ) if errors else np.array([1e-6])
    if all_errs.size == 0:
        all_errs = np.array([1e-6])
    ax_err.set_yscale('log')
    ax_err.set_ylim(
        max(all_errs.min() * 0.5, 1e-12),
        all_errs.max() * 3.0,
    )
    ax_err.set_xlabel("Time")
    ax_err.set_ylabel("Rel. L2 error vs Godunov")
    ax_err.legend(loc='upper left', fontsize=8)
    ax_err.grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    du0dx = np.gradient(u0, x[1] - x[0])
    max_grad = np.max(np.abs(du0dx))
    t_shock = 1.0 / max_grad if max_grad > 0 else 1.0

    def update(frame_idx):
        step = int(step_keys_common[frame_idx])
        t = step * dt
        t_pct = 100.0 * t / t_shock
        ref_line.set_data(x, frames_ref[frame_idx])
        for m, ln in lines.items():
            ln.set_data(x, frames[m][frame_idx])
        for m, ln in err_lines.items():
            ln.set_data(
                times[:frame_idx + 1],
                errors[m][:frame_idx + 1],
            )
        time_text.set_text(
            f"step {step}/{n_steps_total}  "
            f"t={t:.4f} ({t_pct:.0f}% T_shock)"
        )
        return (ref_line,) + tuple(lines.values()) + \
            tuple(err_lines.values()) + (time_text,)

    anim = manimation.FuncAnimation(
        fig, update, frames=len(step_keys_common),
        interval=1000 / args.fps, blit=False,
    )
    writer = manimation.PillowWriter(fps=args.fps)
    anim.save(outfile, writer=writer, dpi=120)
    plt.close(fig)
    print(
        f"Saved animation to {outfile}  "
        f"({len(step_keys_common)} frames @ {args.fps} fps)",
    )


if __name__ == '__main__':
    main()
