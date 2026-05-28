"""Shots study: error vs shots count, wave comparison at different shot counts.

Reads from a q8020 sweep output directory. Finds quantum_circuit cases with
shots > 0, groups by q, and shows how finite sampling noise affects results.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_all_cases(sweep_dir: Path) -> list[dict]:
    """Read raw q8020 fragment files from each sub-directory."""
    import json
    out = []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir():
            continue
        case_f = d / 'q8020_case_0.json'
        if not case_f.exists():
            continue
        entry: dict = {'case': [json.loads(case_f.read_text())]}
        results_f = d / 'q8020_results_0.json'
        if results_f.exists():
            entry['results'] = [json.loads(results_f.read_text())]
        analysis_f = d / 'q8020_analysis_0.json'
        if analysis_f.exists():
            entry['analysis'] = [json.loads(analysis_f.read_text())]
        artifacts_f = d / 'q8020_artifacts_0.json'
        if artifacts_f.exists():
            entry['artifacts'] = [json.loads(artifacts_f.read_text())]
        out.append(entry)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweep-dir', type=Path, required=True,
                   help='Path to sweep output directory')
    p.add_argument('--outfile', default='mps_shots_study.png')
    p.add_argument('--noshow', action='store_true', default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sweep_dir = args.sweep_dir.expanduser().resolve()

    all_cases = _load_all_cases(sweep_dir)
    shot_cases = [
        c for c in all_cases
        if c['case'][0].get('method') in ('quantum_circuit', 'mps')
        and (c['case'][0].get('shots') or 0) >= 100000
    ]
    if not shot_cases:
        print("No finite-shots cases found.", file=sys.stderr)
        sys.exit(1)

    # Group by (q, bond_dim)
    by_q_bd: dict[tuple[int, int | None], list[dict]] = defaultdict(list)
    for c in shot_cases:
        key = (c['case'][0]['q'], c['case'][0].get('bond_dim'))
        by_q_bd[key].append(c)
    for key in by_q_bd:
        by_q_bd[key].sort(key=lambda c: c['case'][0].get('shots', 0))

    groups = sorted(by_q_bd.keys())
    n_groups = len(groups)
    fig, axes = plt.subplots(
        n_groups, 1, figsize=(7, 4 * n_groups), squeeze=False,
    )

    for row, (q_val, bd) in enumerate(groups):
        cases = by_q_bd[(q_val, bd)]
        bd_label = str(bd) if bd else 'full'

        ax = axes[row, 0]

        anchor = next(
            (c for c in cases if c.get('results') and c.get('artifacts')),
            None,
        )
        if anchor is not None:
            x = np.array(anchor['artifacts'][0]['grid'])
            u0 = np.array(anchor['results'][0]['u_initial'])
            u_cl = np.array(anchor['results'][0]['u_final_classical'])
            ax.plot(x, np.abs(u0), 'k--', alpha=0.35, label='|IC|')
            ax.plot(x, np.abs(u_cl), 'b-', linewidth=2.5,
                    label='|Classical|')

            colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(cases)))
            for c, color in zip(cases, colors):
                if not c.get('results') or not c.get('analysis'):
                    continue
                shots = c['case'][0].get('shots', 0)
                u_q = np.array(c['results'][0]['u_final_method'])
                eps = c['analysis'][0].get(
                    'final_error_epsilon', float('nan'),
                )
                ax.plot(
                    x, u_q, '--', color=color, linewidth=1.5,
                    label=f'{shots:,} shots  ε={eps:.2e}',
                )

        ax.set_xlim(0, 0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('|u(x, T)|')
        ax.set_title(
            f'q={q_val}, χ={bd_label}: Wave Comparison by Shots',
        )
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.outfile}")


if __name__ == '__main__':
    main()
