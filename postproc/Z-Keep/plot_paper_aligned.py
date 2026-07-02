"""Paper-aligned wave comparison: classical vs quantum circuit at q=5,6.

Reads q8020 sweep output and produces a multi-panel figure showing IC,
classical solution, and quantum circuit solution with error annotation.

Two invocation modes:
  1. Group postproc (called by sweeper):
       python plot_paper_aligned.py <group_postproc.json>
  2. Manual:
       python plot_paper_aligned.py --sweep-dir ~/q8020/<run_id>
"""

import matplotlib

matplotlib.use('Agg')

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from q8020_cfd_metautil.harvest import harvest_metadata
from q8020_cfd_metautil.metakeys import _walk_case_dirs


def _extract_case_params(meta: dict) -> dict | None:
    """Get flat case params from a metadata dict.

    The harvester returns a list of case entries.  Entries sourced by the
    solver have flat keys (method, q, bc, ...).  Entries sourced by the
    sweeper nest them under ``params`` with ``--`` prefixes.  We prefer
    the solver entry; fall back to unwrapping the sweeper entry.
    """
    cases = meta.get('case', [])
    if not cases:
        return None
    # Prefer solver-sourced entry
    for c in cases:
        if c.get('_source') == 'solver':
            return c
    # Fall back: unwrap sweeper params
    for c in cases:
        params = c.get('params', {})
        if params:
            return {
                k.lstrip('-'): v for k, v in params.items()
                if not k.startswith('_')
            }
    return cases[0]


def _find_solver_entry(entries: list[dict], required_key: str) -> dict:
    """Find the entry in a metadata list that has the required key."""
    for e in entries:
        if required_key in e:
            return e
    return {}


def _load_cases_from_dirs(case_dirs: list[Path]) -> list[dict]:
    out = []
    for cd in case_dirs:
        cd = Path(cd)
        if not cd.is_dir():
            continue
        meta, _, _ = harvest_metadata(cd, read_only=True)
        if meta.get('case'):
            out.append(meta)
    return out


def _load_cases_from_sweep(sweep_dir: Path) -> list[dict]:
    case_dirs, no_meta = _walk_case_dirs(sweep_dir)
    return _load_cases_from_dirs(case_dirs + no_meta)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        'postproc_json', nargs='?', default=None,
        help='Group postproc JSON (passed by sweeper)',
    )
    p.add_argument('--sweep-dir', type=Path, default=None,
                   help='Path to sweep output directory (manual mode)')
    p.add_argument('--outfile', default=None,
                   help='Output PNG path (default: run_dir/paper_aligned_comparison.png)')
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.postproc_json and Path(args.postproc_json).is_file():
        with open(args.postproc_json, encoding='utf-8') as f:
            pp = json.load(f)
        run_dir = Path(pp['run_dir'])
        case_dirs = [Path(d) for d in pp.get('case_dirs', [])]
        # Also scan the full run_dir so we pick up classical baselines
        all_cases = _load_cases_from_sweep(run_dir)
        if not all_cases:
            all_cases = _load_cases_from_dirs(case_dirs)
        outfile = args.outfile or str(
            run_dir / 'paper_aligned_comparison.png'
        )
    elif args.sweep_dir:
        sweep_dir = args.sweep_dir.expanduser().resolve()
        all_cases = _load_cases_from_sweep(sweep_dir)
        outfile = args.outfile or 'paper_aligned_comparison.png'
    else:
        print("Provide either a postproc JSON or --sweep-dir.",
              file=sys.stderr)
        sys.exit(1)

    # Bucket by q value, separate classical vs quantum_circuit
    classical: dict[int, dict] = {}
    quantum: dict[int, list[dict]] = defaultdict(list)

    for c in all_cases:
        case_meta = _extract_case_params(c)
        if case_meta is None:
            continue
        q_val = case_meta.get('q')
        bc = case_meta.get('bc', 'periodic')
        if bc != 'dirichlet':
            continue
        method = case_meta.get('method')
        if method == 'shift':
            classical[q_val] = c
        elif method in ('quantum_circuit', 'quantum_exact', 'mps'):
            quantum[q_val].append(c)

    q_vals = sorted(set(classical.keys()) & set(quantum.keys()))
    if not q_vals:
        print("No matching Dirichlet classical+quantum pairs found.",
              file=sys.stderr)
        sys.exit(1)

    n_panels = len(q_vals)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5),
                             squeeze=False)

    for col, q_val in enumerate(q_vals):
        ax = axes[0, col]
        c_cl = classical[q_val]
        q_cases = quantum[q_val]

        cl_results = _find_solver_entry(
            c_cl.get('results', []), 'u_initial',
        )
        cl_artifacts = _find_solver_entry(
            c_cl.get('artifacts', []), 'grid',
        )
        if not cl_results or not cl_artifacts:
            ax.set_title(f'q={q_val} — no classical data')
            continue

        x = np.array(cl_artifacts['grid'])
        u0 = np.array(cl_results['u_initial'])
        u_cl = np.array(cl_results['u_final_classical'])
        N = len(x)

        ax.plot(x, u0, 'k--', alpha=0.3, linewidth=1, label='IC')
        ax.plot(x, u_cl, 'b-', linewidth=2.2, label='Classical (FTCS)')

        cmap = plt.colormaps['plasma']
        colors = cmap(np.linspace(0.2, 0.8, max(len(q_cases), 1)))
        for c_q, color in zip(q_cases, colors):
            if not c_q.get('results') or not c_q.get('analysis'):
                continue
            cm = _extract_case_params(c_q) or {}
            shots = cm.get('shots', 0)
            meth = cm.get('method', '?')
            bd = cm.get('bond_dim') or cm.get('bond-dim')
            q_res = _find_solver_entry(
                c_q.get('results', []), 'u_final_method',
            )
            q_ana = _find_solver_entry(
                c_q.get('analysis', []), 'final_error_epsilon',
            )
            if not q_res:
                continue
            u_q = np.array(q_res['u_final_method'])
            eps = q_ana.get('final_error_epsilon', float('nan'))
            t_wall = q_ana.get('method_wall_time_s', float('nan'))
            tag = meth.upper()
            if bd is not None:
                tag += f' χ={bd}'
            if shots:
                tag += f' {shots // 1000}k shots'
            label = f'{tag}\n  ε={eps:.2e}, {t_wall:.1f}s'
            ax.plot(x, u_q, '--', color=color, linewidth=1.8, label=label)

        cl_meta = _extract_case_params(c_cl) or {}
        shock_pct = cl_meta.get('shock_pct', '?')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, T)')
        ax.set_title(
            f'q={q_val}  N={N}  '
            f'Dirichlet BC  {shock_pct}% shock',
            fontsize=11,
        )
        ax.set_xlim(0, 1.0)
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(alpha=0.15)

    fig.suptitle(
        'Paper-Aligned: Classical vs Quantum Circuit  (Meena et al. AIAA 2026)',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=180, bbox_inches='tight')
    print(f"Saved to {outfile}")


if __name__ == '__main__':
    main()
