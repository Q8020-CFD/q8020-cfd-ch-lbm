"""Pure-quantum pathway comparison: Pauli-Trotter vs Cole-Hopf circuit.

Overlays both pure-quantum methods + classical FTCS reference in
a single animated GIF.  Designed for the pq_compare_q5_* TOML groups.

Invocation modes:
  1. Group postproc (called by sweeper for pq_compare_q5_cole_hopf):
       python plot_pq_compare.py <group_postproc.json>
     The script auto-discovers the sibling pq_compare_q5_trotter case
     in the same sweep run_dir.

  2. Manual:
       python plot_pq_compare.py \
           --trotter-dir ~/q8020/<run>/pq_compare_q5_trotter/<exp> \
           --ch-dir ~/q8020/<run>/pq_compare_q5_cole_hopf/<exp>
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


def _find_entry(entries: list[dict], key: str) -> dict:
    for e in entries:
        if key in e:
            return e
    return {}


def _extract_params(meta: dict) -> dict | None:
    # Prefer the solver-emitted case (has dt, n_steps, etc.)
    for c in meta.get('case', []):
        if c.get('_source') == 'solver':
            return c
    # Fall back to sweep params
    for c in meta.get('case', []):
        params = c.get('params', {})
        if params:
            return {k.lstrip('-'): v for k, v in params.items()
                    if not k.startswith('_')}
    cases = meta.get('case', [])
    return cases[0] if cases else None


def _load_case(case_dir: Path) -> dict | None:
    """Load metadata from a single case directory."""
    meta, _, _ = harvest_metadata(case_dir, read_only=True)
    return meta if meta.get('case') else None


def _find_sibling_cases(run_dir: Path, prefix: str) -> dict[str, dict]:
    """Find all case dirs in run_dir whose method is relevant.

    The sweeper stores cases as flat experiment-ID dirs (not nested
    under group names), so we scan all case dirs and key by method.
    """
    cases: dict[str, dict] = {}
    case_dirs, no_meta = _walk_case_dirs(run_dir)
    for cd in case_dirs + no_meta:
        cd = Path(cd)
        if not cd.is_dir():
            continue
        meta = _load_case(cd)
        if not meta:
            continue
        p = _extract_params(meta) or {}
        method = p.get('method', '')
        if method:
            cases[method] = meta
    return cases


def _extract_frames(meta: dict) -> tuple[np.ndarray, dict[int, np.ndarray], dict]:
    """Return (x_grid, {step: u_array}, params_dict) from metadata."""
    art = _find_entry(meta.get('artifacts', []), 'solution_steps')
    sol = art.get('solution_steps', {})
    x = np.array(art.get('grid', []))
    params = _extract_params(meta) or {}
    frames = {int(k): np.array(v) for k, v in sol.items()}
    return x, frames, params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('postproc_json', nargs='?', default=None)
    parser.add_argument('--trotter-dir', type=Path, default=None)
    parser.add_argument('--ch-dir', type=Path, default=None)
    parser.add_argument('--outfile', default=None)
    parser.add_argument('--fps', type=int, default=8)
    args = parser.parse_args()

    trotter_meta = None
    ch_meta = None
    outfile = args.outfile or 'pq_compare.gif'

    if args.postproc_json and Path(args.postproc_json).is_file():
        with open(args.postproc_json, encoding='utf-8') as f:
            pp = json.load(f)
        run_dir = Path(pp['run_dir'])
        outfile = args.outfile or str(run_dir / 'pq_compare.gif')

        # Find sibling cases (keyed by method)
        siblings = _find_sibling_cases(run_dir, 'pq_compare')
        trotter_meta = siblings.get('quantum_circuit')
        ch_meta = siblings.get('cole_hopf_circuit')

    if args.trotter_dir:
        trotter_meta = _load_case(args.trotter_dir.expanduser().resolve())
    if args.ch_dir:
        ch_meta = _load_case(args.ch_dir.expanduser().resolve())

    if trotter_meta is None and ch_meta is None:
        print("No cases found. Provide postproc JSON or --trotter-dir/--ch-dir.",
              file=sys.stderr)
        sys.exit(1)

    # Extract data from whichever cases we have
    datasets = {}  # label -> (x, frames_dict, params)
    if trotter_meta:
        datasets['Pauli-Trotter'] = _extract_frames(trotter_meta)
    if ch_meta:
        datasets['Cole-Hopf (LCU)'] = _extract_frames(ch_meta)

    if not datasets:
        print("No valid solution data found.", file=sys.stderr)
        sys.exit(1)

    # Use first dataset for shared params
    first_label = next(iter(datasets))
    x, _, ref_params = datasets[first_label]
    dt = float(ref_params.get('dt', 0.0))
    nu = float(ref_params.get('nu', 1e-2))
    bc = ref_params.get('bc', 'dirichlet')
    n_steps = int(ref_params.get('n_steps', 0))
    shots = int(ref_params.get('shots', 0))
    q = int(ref_params.get('q', 0)) or int(np.log2(len(x)))

    # Compute classical references
    from burgers_classical import solve_burgers, solve_burgers_godunov
    u0 = None
    for label, (xg, frames, _) in datasets.items():
        if 0 in frames:
            u0 = frames[0].copy()
            x = xg
            break
    if u0 is None:
        print("No step-0 IC found.", file=sys.stderr)
        sys.exit(1)

    sols_godunov_list = solve_burgers_godunov(
        u0, x, nu, dt, n_steps, source_fn=None, bc=bc,
    )
    sols_classical = {i: sols_godunov_list[i]
                      for i in range(len(sols_godunov_list))}
    sols_ftcs_list = solve_burgers(
        u0, x, nu, dt, n_steps, source_fn=None, bc=bc,
    )
    sols_ftcs = {i: sols_ftcs_list[i]
                 for i in range(len(sols_ftcs_list))}
    # Track FTCS divergence step
    ftcs_diverged: int | None = None
    for i in range(len(sols_ftcs_list)):
        if not np.all(np.isfinite(sols_ftcs_list[i])):
            ftcs_diverged = i
            break

    # Unify step keys across all datasets
    all_step_sets = [set(frames.keys()) for _, frames, _ in datasets.values()]
    common_steps = sorted(set.intersection(*all_step_sets) if all_step_sets
                          else set())
    if not common_steps:
        # Fall back to union
        common_steps = sorted(set.union(*all_step_sets))

    # Per-method divergence tracking (don't let one method kill others)
    amp0 = max(np.max(np.abs(u0)), 1.0)
    amp_limit = 10.0 * amp0
    diverged: dict[str, int | None] = {lab: None for lab in datasets}
    valid_steps = []
    for s in common_steps:
        # Track per-method divergence
        for label, (_, frames, _) in datasets.items():
            if diverged[label] is None and s in frames:
                u = frames[s]
                if not (np.all(np.isfinite(u))
                        and np.max(np.abs(u)) < amp_limit):
                    diverged[label] = s
        # Stop only when classical blows up
        if s <= n_steps:
            cl = sols_classical.get(s)
            if cl is not None and not (np.all(np.isfinite(cl))
                                       and np.max(np.abs(cl)) < amp_limit):
                break
        valid_steps.append(s)
    if not valid_steps:
        print("No valid frames.", file=sys.stderr)
        sys.exit(1)

    # Y-axis range
    def _ymax(frames, steps):
        vals = [np.abs(frames[s]).max() for s in steps if s in frames
                and np.any(np.isfinite(frames[s]))]
        return max(vals) if vals else 1.0

    all_ymax = []
    for label, (_, f, _) in datasets.items():
        div_s = diverged[label]
        safe = [s for s in valid_steps if div_s is None or s < div_s]
        all_ymax.append(_ymax(f, safe) if safe else 1.0)
    cl_ymax = max(np.abs(sols_classical.get(s, np.zeros_like(u0))).max()
                  for s in valid_steps if s in sols_classical)
    ymax = min(max(max(all_ymax), cl_ymax) * 1.15, 2.0 * amp0)

    # Colors for methods
    colors = {
        'Pauli-Trotter': '#d62728',       # red
        'Cole-Hopf (LCU)': '#9467bd',     # purple
    }

    # Shock time for percentage display
    du0dx = np.gradient(u0, x[1] - x[0])
    max_grad = np.max(np.abs(du0dx))
    t_shock = 1.0 / max_grad if max_grad > 0 else 1.0

    # Build figure
    fig, (ax_u, ax_err) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    N = len(u0)
    shots_label = "SV" if shots == 0 else f"{shots:,} shots"
    fig.suptitle(
        f"Pure Quantum Pathways: q={q} (N={N}), "
        f"$\\nu$={nu:.0e}, {bc} BC, {shots_label}",
        fontsize=12, fontweight="bold",
    )

    ax_u.set_xlim(x[0], x[-1] + (x[1] - x[0]))
    ax_u.set_ylim(-ymax, ymax)
    ax_u.set_ylabel("u(x, t)")
    ax_u.grid(alpha=0.2)

    # IC
    ax_u.plot(x, u0, 'k--', alpha=0.2, lw=1, label='IC')

    # Classical Godunov (reference)
    cl_line, = ax_u.plot([], [], 'b-', lw=1.5, alpha=0.6,
                         label='Classical Godunov')

    # Classical FTCS (overlay, disappears at blowup)
    ftcs_line, = ax_u.plot([], [], '-', color='#2ca02c', lw=1.0,
                           alpha=0.5, label='Classical FTCS')

    # Quantum method lines
    q_lines = {}
    for label in datasets:
        c = colors.get(label, '#333333')
        line, = ax_u.plot([], [], '--', color=c, lw=2.0, label=label)
        q_lines[label] = line

    ax_u.legend(loc='lower left', fontsize=9)
    time_text = ax_u.text(0.02, 0.95, '', transform=ax_u.transAxes,
                          fontsize=10, va='top', fontfamily='monospace')

    # Error panel
    times = np.array([s * dt for s in valid_steps])
    err_data = {}
    for label, (_, frames, _) in datasets.items():
        errs = np.zeros(len(valid_steps))
        for i, s in enumerate(valid_steps):
            cl = sols_classical.get(s)
            if cl is not None and s in frames:
                norm_cl = np.linalg.norm(cl)
                if norm_cl > 1e-15:
                    errs[i] = np.linalg.norm(frames[s] - cl) / norm_cl
        err_data[label] = errs

    err_lines = {}
    for label in datasets:
        c = colors.get(label, '#333333')
        line, = ax_err.plot([], [], '-', color=c, lw=1.5,
                            label=f'{label} vs Godunov')
        err_lines[label] = line

    ax_err.set_xlim(times[0], times[-1])
    all_errs = np.concatenate([e[e > 0] for e in err_data.values()]
                               ) if err_data else np.array([1e-6])
    if all_errs.size == 0:
        all_errs = np.array([1e-6])
    ax_err.set_yscale('log')
    ax_err.set_ylim(
        max(all_errs.min() * 0.5, 1e-12),
        min(all_errs.max() * 2, 10.0),
    )
    ax_err.set_xlabel("Time")
    ax_err.set_ylabel("Rel. L2 error vs Godunov")
    ax_err.legend(loc='upper left', fontsize=8)
    ax_err.grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def update(idx):
        s = valid_steps[idx]
        t = s * dt
        pct = 100.0 * t / t_shock

        # Classical Godunov
        cl = sols_classical.get(s, np.zeros_like(u0))
        cl_line.set_data(x, cl)

        # Classical FTCS (hide after divergence)
        if ftcs_diverged is not None and s >= ftcs_diverged:
            ftcs_line.set_data([], [])
        else:
            ftcs_line.set_data(x, sols_ftcs.get(s, np.zeros_like(u0)))

        # Quantum methods (stop updating after divergence)
        parts = []
        for label, (_, frames, _) in datasets.items():
            div_step = diverged.get(label)
            if div_step is not None and s >= div_step:
                q_lines[label].set_data([], [])
                parts.append(f"{label}: DIVERGED")
            elif s in frames:
                q_lines[label].set_data(x, frames[s])
                e = err_data[label][idx]
                parts.append(f"{label}: {e:.2e}")
            else:
                e = err_data[label][idx]
                parts.append(f"{label}: {e:.2e}")

        # Error lines
        for label in datasets:
            err_lines[label].set_data(times[:idx+1], err_data[label][:idx+1])

        time_text.set_text(
            f"step {s}/{n_steps}  t={t:.4f} ({pct:.0f}% T_shock)\n"
            + "  ".join(parts)
        )
        return tuple([cl_line, ftcs_line, time_text]
                     + list(q_lines.values())
                     + list(err_lines.values()))

    anim = manimation.FuncAnimation(
        fig, update, frames=len(valid_steps),
        interval=1000 / args.fps, blit=False,
    )
    writer = manimation.PillowWriter(fps=args.fps)
    anim.save(outfile, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved {outfile}  ({len(valid_steps)} frames @ {args.fps} fps)")


if __name__ == '__main__':
    main()
