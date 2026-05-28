"""3-panel MPS bond-dim sweep: simulation error, encoding fidelity, wall time.

Reads from a q8020 sweep output directory. Finds all method='mps' cases,
groups by q, and plots error/fidelity/wall-time vs bond dimension.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from q8020_cfd_metautil.compare import load_metadata
from q8020_cfd_metautil.harvest import harvest_metadata
from q8020_cfd_metautil.metakeys import _walk_case_dirs

from burgers_mps import mps_fidelity


def _load_all_cases(sweep_dir: Path) -> list[dict]:
    case_dirs, _ = _walk_case_dirs(sweep_dir)
    out = []
    for cd in case_dirs:
        try:
            meta = load_metadata(cd)
        except FileNotFoundError:
            meta, _, _ = harvest_metadata(cd, read_only=True)
        if meta.get('case'):
            out.append(meta)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweep-dir', type=Path, required=True,
                   help='Path to sweep output directory')
    p.add_argument('--outfile', default='mps_bond_sweep.png')
    p.add_argument('--noshow', action='store_true', default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sweep_dir = args.sweep_dir.expanduser().resolve()

    mps_cases = [
        c for c in _load_all_cases(sweep_dir)
        if c['case'][0].get('method') == 'mps'
    ]
    if not mps_cases:
        print("No MPS cases found in sweep dir.", file=sys.stderr)
        sys.exit(1)

    by_q: dict[int, list[dict]] = defaultdict(list)
    for c in mps_cases:
        by_q[c['case'][0]['q']].append(c)
    for q_val in by_q:
        by_q[q_val].sort(
            key=lambda c: c['case'][0].get('bond_dim') or 999999
        )

    n_q = len(by_q)
    fig, axes = plt.subplots(n_q, 3, figsize=(14, 4 * n_q), squeeze=False)

    for row, q_val in enumerate(sorted(by_q)):
        cases = by_q[q_val]
        bond_dims = [c['case'][0].get('bond_dim') for c in cases]
        x_labels = [str(bd) if bd else 'full' for bd in bond_dims]
        x_pos = np.arange(len(bond_dims))

        errors = [
            c['analysis'][0].get('final_error_epsilon', np.nan)
            if c.get('analysis') else np.nan
            for c in cases
        ]
        wall_times = [
            c['analysis'][0].get('method_wall_time_s', np.nan)
            if c.get('analysis') else np.nan
            for c in cases
        ]

        ic_fids, final_fids = [], []
        for c, bd in zip(cases, bond_dims):
            if c.get('results'):
                u0 = np.array(c['results'][0]['u_initial'])
                u_cl = np.array(c['results'][0]['u_final_classical'])
                _, fic, _ = mps_fidelity(u0, bond_dim=bd)
                _, ffin, _ = mps_fidelity(u_cl, bond_dim=bd)
            else:
                fic = ffin = np.nan
            ic_fids.append(fic)
            final_fids.append(ffin)

        ax_e, ax_f, ax_t = axes[row]

        ax_e.semilogy(x_pos, errors, 'o-', color='C0')
        ax_e.set_xticks(x_pos)
        ax_e.set_xticklabels(x_labels)
        ax_e.set_xlabel('Bond dimension χ')
        ax_e.set_ylabel('Simulation error ε')
        ax_e.set_title(f'q={q_val}: Simulation Error')

        ax_f.plot(x_pos, ic_fids, 's-', color='C2', label='IC')
        ax_f.plot(x_pos, final_fids, '^-', color='C3', label='Final state')
        ax_f.set_xticks(x_pos)
        ax_f.set_xticklabels(x_labels)
        ax_f.set_xlabel('Bond dimension χ')
        ax_f.set_ylabel('Fidelity |⟨ψ|ψ_MPS⟩|²')
        ax_f.set_title(f'q={q_val}: Encoding Fidelity')
        ax_f.legend(fontsize=8)

        ax_t.bar(x_labels, wall_times, color='C4')
        ax_t.set_xlabel('Bond dimension χ')
        ax_t.set_ylabel('Wall time (s)')
        ax_t.set_title(f'q={q_val}: Wall Time')

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.outfile}")


if __name__ == '__main__':
    main()
