"""Hadamard-test time evolution animation: sine wave steepening into a shock.

Reads a sweep that varies shock_pct (simulation end-time) with all cases
using sign_recovery=hadamard_test, picks the longest-running case (most
complete trajectory), and animates its per-step snapshots as a GIF.

The quantum (Hadamard-recovered) and classical solutions are overlaid
frame-by-frame so sign recovery fidelity is visible throughout evolution.

Two invocation modes:
  1. Group postproc (called by sweeper):
       python plot_hadamard_time_evolution.py <group_postproc.json>
  2. Manual:
       python plot_hadamard_time_evolution.py --sweep-dir ~/q8020/<run_id>
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
    p.add_argument('--fps', type=int, default=8, help='Animation frames/sec')
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.postproc_json and Path(args.postproc_json).is_file():
        with open(args.postproc_json, encoding='utf-8') as f:
            pp = json.load(f)
        run_dir = Path(pp['run_dir'])
        all_cases = _load_cases_from_sweep(run_dir)
        outfile = args.outfile or str(run_dir / 'hadamard_evolution.gif')
    elif args.sweep_dir:
        all_cases = _load_cases_from_sweep(
            args.sweep_dir.expanduser().resolve()
        )
        outfile = args.outfile or 'hadamard_evolution.gif'
    else:
        print("Provide either a postproc JSON or --sweep-dir.", file=sys.stderr)
        sys.exit(1)

    # Collect Hadamard-test cases
    candidates: list[dict] = []
    for c in all_cases:
        cm = _extract_case_params(c)
        if cm is None:
            continue
        if cm.get('method') != 'quantum_circuit':
            continue
        ana = _find_solver_entry(c.get('analysis', []), 'sign_recovery')
        sr = (ana.get('sign_recovery')
              or cm.get('sign_recovery')
              or cm.get('sign-recovery', 'none'))
        if sr != 'hadamard_test':
            continue
        candidates.append(c)

    if not candidates:
        print("No Hadamard-test cases found.", file=sys.stderr)
        sys.exit(1)

    # Pick the case with the longest trajectory (most solution_steps entries)
    def _nsteps(c: dict) -> int:
        art = _find_solver_entry(c.get('artifacts', []), 'solution_steps')
        return len(art.get('solution_steps', {}))

    anchor = max(candidates, key=_nsteps)
    anchor_meta = _extract_case_params(anchor) or {}
    art = _find_solver_entry(anchor.get('artifacts', []), 'solution_steps')
    sol_steps = art.get('solution_steps', {})
    x = np.array(art.get('grid', []))

    if not sol_steps or x.size == 0:
        print("Anchor case has no solution_steps/grid.", file=sys.stderr)
        sys.exit(1)

    # Sort steps numerically
    step_keys = sorted(sol_steps.keys(), key=int)
    frames_q = [np.array(sol_steps[k]) for k in step_keys]

    # For classical reference per step: recompute with the same BC to match
    # the quantum path exactly.  Using the IC from step 0.
    from burgers_classical import solve_burgers
    dt = float(anchor_meta.get('dt', 0.0))
    nu = float(anchor_meta.get('nu', 1e-4))
    bc = anchor_meta.get('bc', 'periodic')
    n_steps_total = int(anchor_meta.get('n_steps', len(step_keys) - 1))
    u0 = frames_q[0].copy()

    # Synthetic source term (sine source was used for these cases)
    def _src(xx, tt):
        return np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * tt)

    sols_classical = solve_burgers(
        u0, x, nu, dt, n_steps_total, source_fn=_src, bc=bc,
    )
    # Map step index -> classical snapshot
    frames_cl = [sols_classical[int(k)] for k in step_keys]

    # Classical FTCS can blow up past shock formation (nu=1e-4, CFL=0.1 is
    # marginally stable for near-discontinuous profiles).  Truncate the
    # animation at the first frame where either solution contains a
    # non-finite value OR amplitudes exceed a reasonable bound (the
    # solution can grow to 1e+82 before hitting Inf, squashing all
    # earlier frames to flat lines).
    amp0 = max(np.max(np.abs(u0)), 1.0)
    amp_limit = 10.0 * amp0
    valid_end = len(step_keys)
    for i, (fq, fc) in enumerate(zip(frames_q, frames_cl)):
        finite = np.all(np.isfinite(fq)) and np.all(np.isfinite(fc))
        bounded = (np.max(np.abs(fq)) < amp_limit
                   and np.max(np.abs(fc)) < amp_limit)
        if not (finite and bounded):
            valid_end = i
            break
    if valid_end < len(step_keys):
        print(
            f"[warn] solution diverged at step {step_keys[valid_end]}; "
            f"truncating animation to {valid_end} frames.",
            file=sys.stderr,
        )
    if valid_end == 0:
        print("No valid frames (classical diverged immediately).",
              file=sys.stderr)
        sys.exit(1)
    step_keys = step_keys[:valid_end]
    frames_q = frames_q[:valid_end]
    frames_cl = frames_cl[:valid_end]

    # Plot setup -- ymax from finite values only
    def _fmax(fs):
        vals = [np.abs(f[np.isfinite(f)]).max() for f in fs
                if np.any(np.isfinite(f))]
        return max(vals) if vals else 1.0

    ymax = max(_fmax(frames_q), _fmax(frames_cl)) * 1.15

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel('x')
    ax.set_ylabel('velocity u(x, t)')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.grid(alpha=0.15)

    q_val = anchor_meta.get('q', '?')
    shots = anchor_meta.get('shots', 0)
    shots_label = f'{int(shots) // 1000}k shots' if shots else 'statevector'
    ic_name = anchor_meta.get('ic', 'sine')
    n_modes = anchor_meta.get('ic_modes', '')
    ic_label = ic_name if not n_modes else f'{ic_name} ({n_modes} modes)'

    ic_line, = ax.plot(x, u0, 'k--', alpha=0.25, linewidth=1,
                       label=f'IC: {ic_label}')
    cl_line, = ax.plot([], [], 'b-', linewidth=2.0,
                       label='Classical FTCS baseline')
    q_line, = ax.plot([], [], '--', color='#d62728', linewidth=1.8,
                      label=f'Quantum (Hadamard test, {shots_label})')
    ax.legend(loc='lower left', fontsize=9)
    title = ax.set_title('')

    def init():
        cl_line.set_data([], [])
        q_line.set_data([], [])
        title.set_text('')
        return cl_line, q_line, title

    du0dx = np.gradient(u0, x[1] - x[0])
    max_grad = np.max(np.abs(du0dx))
    t_shock = 1.0 / max_grad if max_grad > 0 else 1.0 / (2.0 * np.pi)

    def update(frame_idx):
        step = int(step_keys[frame_idx])
        t = step * dt
        t_pct = 100.0 * t / t_shock
        cl_line.set_data(x, frames_cl[frame_idx])
        q_line.set_data(x, frames_q[frame_idx])
        cl = frames_cl[frame_idx]
        if np.all(np.isfinite(cl)):
            norm_cl = np.linalg.norm(cl)
            err = np.linalg.norm(frames_q[frame_idx] - cl) / max(norm_cl, 1e-15)
        else:
            err = float("nan")
        title.set_text(
            f'IC={ic_label}  |  q={q_val} (N={2**int(q_val)})  |  BC={bc}\n'
            f'step {step}/{n_steps_total}  |  t={t:.4f} ({t_pct:.0f}% of T_shock)'
            f'  |  Rel. L2 error = {err:.2e}'
        )
        return cl_line, q_line, title

    anim = manimation.FuncAnimation(
        fig, update, init_func=init, frames=len(step_keys),
        interval=1000 / args.fps, blit=False,
    )

    writer = manimation.PillowWriter(fps=args.fps)
    anim.save(outfile, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved animation to {outfile}  ({len(step_keys)} frames @ {args.fps} fps)")


if __name__ == '__main__':
    main()
