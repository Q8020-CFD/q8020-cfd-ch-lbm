"""Sign recovery comparison: four options on a single figure.

One panel per sign recovery strategy, all at the same q/shots/BC.
Clearly shows sign-loss artifact (Option 1) and progressive correction.

Two invocation modes:
  1. Group postproc (called by sweeper):
       python plot_sign_recovery_comparison.py <group_postproc.json>
  2. Manual:
       python plot_sign_recovery_comparison.py --sweep-dir ~/q8020/<run_id>
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from q8020_cfd_metautil.harvest import harvest_metadata
from q8020_cfd_metautil.metakeys import _walk_case_dirs


_OPTION_ORDER = ["none", "classical_oracle", "hadamard_test", "dual_rail"]
_OPTION_LABELS = {
    "none":             "Option 1 — None (as-is)",
    "classical_oracle": "Option 3 — Classical oracle",
    "hadamard_test":    "Option 2 — Hadamard test / interferometric",
    "dual_rail":        "Option 4 — Dual-rail encoding",
}
_OPTION_COLORS = {
    "none":             "#d62728",
    "classical_oracle": "#2ca02c",
    "hadamard_test":    "#1f77b4",
    "dual_rail":        "#9467bd",
}


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
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.postproc_json and Path(args.postproc_json).is_file():
        with open(args.postproc_json, encoding='utf-8') as f:
            pp = json.load(f)
        run_dir = Path(pp['run_dir'])
        all_cases = _load_cases_from_sweep(run_dir)
        outfile = args.outfile or str(
            run_dir / 'sign_recovery_comparison.png'
        )
    elif args.sweep_dir:
        all_cases = _load_cases_from_sweep(
            args.sweep_dir.expanduser().resolve()
        )
        outfile = args.outfile or 'sign_recovery_comparison.png'
    else:
        print("Provide either a postproc JSON or --sweep-dir.", file=sys.stderr)
        sys.exit(1)

    # Collect sign-recovery quantum cases.  Classical baseline is read from
    # u_final_classical inside each case's results (always written by solver).
    by_option: dict[str, dict] = {}

    for c in all_cases:
        cm = _extract_case_params(c)
        if cm is None:
            continue
        if cm.get('bc', 'periodic') != 'dirichlet':
            continue
        if cm.get('method') != 'quantum_circuit':
            continue
        # sign_recovery is written to the analysis fragment by the solver;
        # fall back to sweeper case params (--sign-recovery key) if absent.
        ana = _find_solver_entry(c.get('analysis', []), 'sign_recovery')
        sr = (ana.get('sign_recovery')
              or cm.get('sign_recovery')
              or cm.get('sign-recovery', 'none'))
        if sr not in by_option:
            by_option[sr] = c

    if not by_option:
        print("No Dirichlet quantum_circuit cases found.", file=sys.stderr)
        sys.exit(1)

    # Use the first available case to extract grid and classical reference
    anchor = next(iter(by_option.values()))
    anchor_res = _find_solver_entry(anchor.get('results', []), 'u_initial')
    anchor_art = _find_solver_entry(anchor.get('artifacts', []), 'grid')
    if not anchor_res or not anchor_art:
        print("Anchor case missing results/artifacts.", file=sys.stderr)
        sys.exit(1)

    x = np.array(anchor_art['grid'])
    u0 = np.array(anchor_res['u_initial'])
    u_cl = np.array(anchor_res['u_final_classical'])

    # Canonical panel order; include any option actually present
    options = [o for o in _OPTION_ORDER if o in by_option]
    options += [o for o in by_option if o not in _OPTION_ORDER]

    n = len(options)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0), squeeze=False,
                             sharey=True)

    for col, opt in enumerate(options):
        ax = axes[0, col]
        c_q = by_option[opt]

        ax.plot(x, u0, 'k--', alpha=0.25, linewidth=1, label='IC')
        ax.plot(x, u_cl, 'b-', linewidth=2.0, label='Classical')
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.8,
                   alpha=0.6, label='shock front x=0.5')

        q_res = _find_solver_entry(c_q.get('results', []), 'u_final_method')
        q_ana = _find_solver_entry(
            c_q.get('analysis', []), 'final_error_epsilon'
        )
        if q_res:
            u_q = np.array(q_res['u_final_method'])
            eps = q_ana.get('final_error_epsilon', float('nan'))
            shots = (_extract_case_params(c_q) or {}).get('shots', 0)
            shots_label = f'{shots // 1000}k shots' if shots else 'SV'
            color = _OPTION_COLORS.get(opt, '#ff7f0e')
            ax.plot(x, u_q, '--', color=color, linewidth=1.8,
                    label=f'Quantum  ε={eps:.2e}\n{shots_label}')

        ax.set_xlim(0, 1.0)
        ax.set_xlabel('x')
        if col == 0:
            ax.set_ylabel('u(x, T)')
        ax.set_title(_OPTION_LABELS.get(opt, opt), fontsize=10)
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(alpha=0.15)

    anchor_meta = _extract_case_params(anchor) or {}
    q_val = anchor_meta.get('q', '?')
    shock_pct = anchor_meta.get('shock_pct', '?')
    fig.suptitle(
        f'Sign Recovery Comparison  q={q_val}  Dirichlet BC  {shock_pct}% shock'
        '  (Meena et al. AIAA 2026)',
        fontsize=12, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=180, bbox_inches='tight')
    print(f"Saved to {outfile}")


if __name__ == '__main__':
    main()
