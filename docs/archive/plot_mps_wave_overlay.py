"""Wave comparison: IC, classical baseline, MPS at various bond dimensions.

Reads from a q8020 sweep output directory. Finds MPS cases for the given
q value and overlays waveforms on a single panel.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from q8020_cfd_metautil.compare import load_metadata
from q8020_cfd_metautil.harvest import harvest_metadata
from q8020_cfd_metautil.metakeys import _walk_case_dirs


def _load_all_cases(sweep_dir: Path) -> list[dict]:
    case_dirs, no_meta = _walk_case_dirs(sweep_dir)
    all_dirs = case_dirs + no_meta
    out = []
    for cd in all_dirs:
        # Always harvest from raw fragments; assembled metadata
        # may be missing fields.
        meta, _, _ = harvest_metadata(cd, read_only=True)
        if meta.get('case'):
            out.append(meta)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweep-dir', type=Path, required=True,
                   help='Path to sweep output directory')
    p.add_argument('--q', type=int, required=True,
                   help='Number of qubits (q value to plot)')
    p.add_argument('--outfile', default='mps_wave_overlay.png')
    p.add_argument('--noshow', action='store_true', default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sweep_dir = args.sweep_dir.expanduser().resolve()

    all_cases = _load_all_cases(sweep_dir)
    mps_cases = [
        c for c in all_cases
        if c['case'][0].get('method') == 'mps'
        and c['case'][0].get('q') == args.q
    ]
    if not mps_cases:
        print(f"No MPS cases found for q={args.q}.", file=sys.stderr)
        sys.exit(1)

    mps_cases.sort(key=lambda c: c['case'][0].get('bond_dim') or 0)

    anchor = next(
        (c for c in mps_cases if c.get('results') and c.get('artifacts')),
        None,
    )
    if anchor is None:
        print("No cases with results+artifacts found.", file=sys.stderr)
        sys.exit(1)

    x = np.array(anchor['artifacts'][0]['grid'])
    u0 = np.array(anchor['results'][0]['u_initial'])
    u_classical = np.array(anchor['results'][0]['u_final_classical'])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, u0, 'k--', alpha=0.35, label='IC')
    ax.plot(x, u_classical, 'b-', linewidth=2.5, label='Classical')

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(mps_cases)))
    for c, color in zip(mps_cases, colors):
        if not c.get('results') or not c.get('analysis'):
            continue
        bd = c['case'][0].get('bond_dim')
        bd_label = str(bd) if bd else 'full'
        u_mps = np.array(c['results'][0]['u_final_method'])
        eps = c['analysis'][0].get('final_error_epsilon', float('nan'))
        ax.plot(x, u_mps, '--', color=color, linewidth=1.5,
                label=f'χ={bd_label}  ε={eps:.2e}')

    shock_pct = anchor['case'][0].get('shock_pct')
    title = f'MPS Wave Overlay  q={args.q}'
    if shock_pct is not None:
        title += f'  shock={shock_pct}%'
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, T)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.outfile}")


if __name__ == '__main__':
    main()
