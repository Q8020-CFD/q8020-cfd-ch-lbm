"""Cole-Hopf circuit time evolution animation.

Reads sweep output from cole_hopf_circuit runs, animates the quantum
circuit solution vs classical FTCS baseline frame-by-frame.

Two invocation modes:
  1. Group postproc (called by sweeper):
       python plot_cole_hopf_circuit_evolution.py <group_postproc.json>
  2. Manual:
       python plot_cole_hopf_circuit_evolution.py --sweep-dir ~/q8020/<id>
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from q8020_cfd_metautil.harvest import harvest_metadata
from q8020_cfd_metautil.metakeys import _walk_case_dirs


def _find_solver_entry(entries: list[dict], required_key: str) -> dict:
    for e in entries:
        if required_key in e:
            return e
    return {}


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
            return {k.lstrip('-'): v for k, v in params.items()
                    if not k.startswith('_')}
    return cases[0]


def _load_group_params(sweep_dir: Path) -> dict:
    """Read --params from the _group_postproc_*.json in sweep_dir."""
    import glob as _glob
    jsons = _glob.glob(str(sweep_dir / '_group_postproc_*.json'))
    if jsons:
        with open(jsons[0], encoding='utf-8') as f:
            gp = json.load(f)
        return gp.get('params', {})
    return {}


def _load_cases_from_sweep(sweep_dir: Path) -> list[dict]:
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('postproc_json', nargs='?', default=None)
    p.add_argument('--sweep-dir', type=Path, default=None)
    p.add_argument('--outfile', default=None)
    p.add_argument('--fps', type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.postproc_json and Path(args.postproc_json).is_file():
        with open(args.postproc_json, encoding='utf-8') as f:
            pp = json.load(f)
        run_dir = Path(pp['run_dir'])
        all_cases = _load_cases_from_sweep(run_dir)
        group_params = _load_group_params(run_dir)
        outfile = args.outfile or str(
            run_dir / 'cole_hopf_circuit_evolution.gif',
        )
    elif args.sweep_dir:
        sd = args.sweep_dir.expanduser().resolve()
        all_cases = _load_cases_from_sweep(sd)
        group_params = _load_group_params(sd)
        outfile = args.outfile or 'cole_hopf_circuit_evolution.gif'
    else:
        print(
            "Provide either a postproc JSON or --sweep-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Find cole_hopf_circuit cases
    candidates: list[dict] = []
    for c in all_cases:
        cm = _extract_case_params(c)
        if cm is None:
            continue
        if cm.get('method') != 'cole_hopf_circuit':
            continue
        candidates.append(c)

    if not candidates:
        print("No cole_hopf_circuit cases found.", file=sys.stderr)
        sys.exit(1)

    # Pick the best anchor: highest bond_dim (most accurate MPS prep),
    # then most solution_steps as tiebreaker.
    def _anchor_key(c: dict) -> tuple[int, int]:
        cm = _extract_case_params(c) or {}
        bd = cm.get('bond_dim') or 0
        if bd is None:
            bd = 999  # None = full rank, best possible
        art = _find_solver_entry(
            c.get('artifacts', []), 'solution_steps',
        )
        n = len(art.get('solution_steps', {}))
        return (int(bd), n)

    anchor = max(candidates, key=_anchor_key)
    anchor_meta = _extract_case_params(anchor) or {}
    art = _find_solver_entry(
        anchor.get('artifacts', []), 'solution_steps',
    )
    sol_steps = art.get('solution_steps', {})
    x = np.array(art.get('grid', []))

    if not sol_steps or x.size == 0:
        print(
            "Anchor case has no solution_steps/grid.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Also grab results for classical baseline
    res = _find_solver_entry(
        anchor.get('results', []), 'u_final_classical',
    )
    u_cl_final = np.array(res.get('u_final_classical', []))

    step_keys = sorted(sol_steps.keys(), key=int)
    frames_q = [np.array(sol_steps[k]) for k in step_keys]

    # Recompute classical FTCS + Cole-Hopf MPS for per-step overlay
    from burgers_classical import solve_burgers, source_term_sine
    from burgers_cole_hopf import run_cole_hopf_simulation

    dt = float(anchor_meta.get('dt', 0.0))
    nu = float(anchor_meta.get('nu', 1e-2))
    bc = anchor_meta.get('bc', 'periodic')
    source = anchor_meta.get('source', 'sine')
    n_steps_total = int(
        anchor_meta.get('n_steps', len(step_keys) - 1),
    )
    u0 = frames_q[0].copy()

    source_fn = source_term_sine if source == 'sine' else None
    forced = source_fn is not None
    sols_classical = solve_burgers(
        u0, x, nu, dt, n_steps_total,
        source_fn=source_fn, bc=bc,
    )
    frames_cl = [sols_classical[int(k)] for k in step_keys]

    # Cole-Hopf MPS: exact same physics as circuit, no circuit error.
    # Skipped when forcing is active — run_cole_hopf_simulation has no
    # source_fn parameter, so the curve would solve a different PDE
    # (unforced) and mislead the comparison.
    if not forced:
        print('  Computing Cole-Hopf MPS reference ...', file=sys.stderr)
        sols_mps, _ = run_cole_hopf_simulation(
            u0, x, nu, dt, n_steps_total, bc=bc,
        )
        frames_mps = [sols_mps[int(k)] for k in step_keys]
    else:
        print(
            '  Skipping Cole-Hopf MPS reference: forced run '
            '(unforced classical-CH would mislead).',
            file=sys.stderr,
        )
        frames_mps = None

    # Truncate at divergence
    amp0 = max(np.max(np.abs(u0)), 1.0)
    amp_limit = 10.0 * amp0
    valid_end = len(step_keys)
    for i, (fq, fc) in enumerate(zip(frames_q, frames_cl)):
        finite = (np.all(np.isfinite(fq))
                  and np.all(np.isfinite(fc)))
        bounded = (np.max(np.abs(fq)) < amp_limit
                   and np.max(np.abs(fc)) < amp_limit)
        if not (finite and bounded):
            valid_end = i
            break
    if valid_end == 0:
        print("No valid frames.", file=sys.stderr)
        sys.exit(1)
    step_keys = step_keys[:valid_end]
    frames_q = frames_q[:valid_end]
    frames_cl = frames_cl[:valid_end]
    if frames_mps is not None:
        frames_mps = frames_mps[:valid_end]

    # Animation
    def _fmax(fs):
        vals = [np.abs(f[np.isfinite(f)]).max() for f in fs
                if np.any(np.isfinite(f))]
        return max(vals) if vals else 1.0

    _frame_lists = [frames_q, frames_cl]
    if frames_mps is not None:
        _frame_lists.append(frames_mps)
    ymax = max(_fmax(fs) for fs in _frame_lists) * 1.15

    q_val = anchor_meta.get('q', '?')
    propagator = (
        anchor_meta.get('propagator')
        or group_params.get('--propagator', 'qft-diagonal')
    )
    shots = anchor_meta.get('shots', 0)
    shots_label = (
        f'{int(shots) // 1000}k shots' if shots else 'statevector'
    )

    fig, (ax_u, ax_err) = plt.subplots(
        2, 1, figsize=(9, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"Cole-Hopf Circuit: q={q_val} (N={2**int(q_val)}), "
        f"$\\nu$={nu:.0e}, {propagator}, {shots_label}",
        fontsize=12, fontweight="bold",
    )

    ax_u.set_xlim(x[0], x[-1] + x[1] - x[0])
    ax_u.set_ylim(-ymax, ymax)
    ax_u.set_ylabel("u(x, t)")
    ax_u.grid(alpha=0.2)

    ic_line, = ax_u.plot(
        x, u0, 'k--', alpha=0.2, lw=1, label='IC',
    )
    if frames_mps is not None:
        mps_line, = ax_u.plot(
            [], [], '-', color='#2ca02c', lw=2.2,
            label='Exact classical (tensor net)',
        )
    else:
        mps_line = None
    cl_line, = ax_u.plot(
        [], [], 'b-', lw=1.2, alpha=0.5, label='FTCS (num. diffusive)',
    )
    q_line, = ax_u.plot(
        [], [], '--', color='#d62728', lw=1.8,
        label=f'Cole-Hopf circuit ({propagator})',
    )
    ax_u.legend(loc='lower left', fontsize=9)
    time_text = ax_u.text(
        0.02, 0.95, '', transform=ax_u.transAxes,
        fontsize=10, va='top', fontfamily='monospace',
    )

    # Error panel: vs MPS (circuit-only error, unforced runs only) and vs FTCS
    err_vs_mps = np.zeros(len(step_keys)) if frames_mps is not None else None
    err_vs_ftcs = np.zeros(len(step_keys))
    for i in range(len(step_keys)):
        norm_cl = np.linalg.norm(frames_cl[i])
        if frames_mps is not None:
            norm_mps = np.linalg.norm(frames_mps[i])
            if norm_mps > 1e-15:
                err_vs_mps[i] = np.linalg.norm(
                    frames_q[i] - frames_mps[i],
                ) / norm_mps
        if norm_cl > 1e-15:
            err_vs_ftcs[i] = np.linalg.norm(
                frames_q[i] - frames_cl[i],
            ) / norm_cl
    times = np.array([int(k) * dt for k in step_keys])
    if frames_mps is not None:
        line_err_mps, = ax_err.plot(
            [], [], '-', color='#2ca02c', lw=1.5,
            label='vs exact classical (circuit err)',
        )
    else:
        line_err_mps = None
    line_err_ftcs, = ax_err.plot(
        [], [], 'b-', lw=1, alpha=0.5,
        label='vs FTCS (scheme diff)',
    )
    ax_err.set_xlim(times[0], times[-1])
    err_series = [err_vs_ftcs]
    if err_vs_mps is not None:
        err_series.append(err_vs_mps)
    err_pos = np.concatenate([s[s > 0] for s in err_series]) \
        if any((s > 0).any() for s in err_series) else np.array([1e-6])
    err_lo = max(err_pos.min() * 0.5, 1e-12)
    err_hi = err_pos.max() * 2.0
    ax_err.set_yscale('log')
    ax_err.set_ylim(err_lo, err_hi)
    ax_err.set_xlabel("Time")
    ax_err.set_ylabel("Rel. L2 error")
    ax_err.legend(loc='upper left', fontsize=8)
    ax_err.grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    du0dx = np.gradient(u0, x[1] - x[0])
    max_grad = np.max(np.abs(du0dx))
    t_shock = 1.0 / max_grad if max_grad > 0 else 1.0

    def update(frame_idx):
        step = int(step_keys[frame_idx])
        t = step * dt
        t_pct = 100.0 * t / t_shock
        cl_line.set_data(x, frames_cl[frame_idx])
        q_line.set_data(x, frames_q[frame_idx])
        artists = [cl_line, q_line, time_text, line_err_ftcs]
        if mps_line is not None:
            mps_line.set_data(x, frames_mps[frame_idx])
            line_err_mps.set_data(
                times[:frame_idx + 1], err_vs_mps[:frame_idx + 1],
            )
            err_text = f"  err_vs_MPS={err_vs_mps[frame_idx]:.2e}"
            artists.extend([mps_line, line_err_mps])
        else:
            err_text = f"  err_vs_FTCS={err_vs_ftcs[frame_idx]:.2e}"
        time_text.set_text(
            f"step {step}/{n_steps_total}  "
            f"t={t:.4f} ({t_pct:.0f}% T_shock)" + err_text
        )
        line_err_ftcs.set_data(
            times[:frame_idx + 1], err_vs_ftcs[:frame_idx + 1],
        )
        return tuple(artists)

    anim = manimation.FuncAnimation(
        fig, update, frames=len(step_keys),
        interval=1000 / args.fps, blit=False,
    )
    writer = manimation.PillowWriter(fps=args.fps)
    anim.save(outfile, writer=writer, dpi=120)
    plt.close(fig)
    print(
        f"Saved animation to {outfile}  "
        f"({len(step_keys)} frames @ {args.fps} fps)",
    )


if __name__ == '__main__':
    main()
