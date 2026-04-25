"""4-panel quantum circuit scaling: CX gates, depth, timing, error vs q.

Reads from a q8020 sweep output directory. Finds statevector quantum_circuit
cases across q values and plots scaling of circuit metrics and timing.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from q8020_cfd_metautil.compare import load_metadata
from q8020_cfd_metautil.harvest import harvest_metadata
from q8020_cfd_metautil.metakeys import _walk_case_dirs


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
    p.add_argument('--outfile', default='mps_scaling.png')
    p.add_argument('--noshow', action='store_true', default=True)
    return p.parse_args()


def _avg_step_metric(cases: list[dict], field: str) -> float:
    vals = []
    for c in cases:
        if not c.get('analysis'):
            continue
        metrics = c['analysis'][0].get('per_step_metrics', [])
        if metrics:
            vals.append(float(np.mean([m.get(field, 0) for m in metrics])))
    return float(np.mean(vals)) if vals else float('nan')


def _avg_cx(cases: list[dict]) -> float:
    vals = []
    for c in cases:
        if not c.get('analysis'):
            continue
        metrics = c['analysis'][0].get('per_step_metrics', [])
        if metrics:
            cx_per_step = [m.get('gate_counts', {}).get('cx', 0) for m in metrics]
            vals.append(float(np.mean(cx_per_step)))
    return float(np.mean(vals)) if vals else float('nan')


def _mean_analysis(cases: list[dict], field: str) -> float:
    vals = [
        c['analysis'][0].get(field, float('nan'))
        for c in cases if c.get('analysis')
    ]
    return float(np.nanmean(vals)) if vals else float('nan')


_TIMING_FIELDS = [
    ('pauli_decomp_time_s', 'Pauli decomp'),
    ('circuit_construction_time_s', 'Circuit build'),
    ('transpilation_time_s', 'Transpile'),
    ('execution_time_s', 'Execute'),
]


def _total_timing(cases: list[dict]) -> dict[str, float]:
    out = {}
    for field, label in _TIMING_FIELDS:
        vals = []
        for c in cases:
            if not c.get('analysis'):
                continue
            metrics = c['analysis'][0].get('per_step_metrics', [])
            if metrics:
                vals.append(sum(m.get(field, 0) for m in metrics))
        out[label] = float(np.mean(vals)) if vals else 0.0
    return out


def main() -> None:
    args = _parse_args()
    sweep_dir = args.sweep_dir.expanduser().resolve()

    all_cases = _load_all_cases(sweep_dir)
    qc_cases = [
        c for c in all_cases
        if c['case'][0].get('method') == 'quantum_circuit'
        and not c['case'][0].get('shots', 0)
    ]
    if not qc_cases:
        print("No statevector quantum_circuit cases found.", file=sys.stderr)
        sys.exit(1)

    by_q: dict[int, list[dict]] = defaultdict(list)
    for c in qc_cases:
        by_q[c['case'][0]['q']].append(c)
    qs = sorted(by_q)

    avg_cx = [_avg_cx(by_q[q]) for q in qs]
    avg_depths = [_avg_step_metric(by_q[q], 'circuit_depth') for q in qs]
    n_pauli = [_mean_analysis(by_q[q], 'n_pauli_terms') for q in qs]
    errors = [_mean_analysis(by_q[q], 'final_error_epsilon') for q in qs]
    timing_by_q = [_total_timing(by_q[q]) for q in qs]
    timing_labels = [label for _, label in _TIMING_FIELDS]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_cx, ax_depth, ax_time, ax_err = axes.flat

    # Top-left: CX gate count vs q
    ax_cx.semilogy(qs, avg_cx, 'o-', color='C0')
    ax_cx.set_xlabel('q (qubits)')
    ax_cx.set_ylabel('Avg CX gates per step')
    ax_cx.set_title('CX Gate Count vs q')
    ax_cx.set_xticks(qs)

    # Top-right: circuit depth vs q
    ax_depth.semilogy(qs, avg_depths, 's-', color='C1')
    ax_depth.set_xlabel('q (qubits)')
    ax_depth.set_ylabel('Avg circuit depth per step')
    ax_depth.set_title('Circuit Depth vs q')
    ax_depth.set_xticks(qs)

    # Bottom-left: timing breakdown stacked bar vs q
    bottoms = np.zeros(len(qs))
    for i, label in enumerate(timing_labels):
        vals = np.array([t.get(label, 0) for t in timing_by_q])
        ax_time.bar(qs, vals, bottom=bottoms, label=label, color=f'C{i + 2}')
        bottoms += vals
    ax_time.set_xlabel('q (qubits)')
    ax_time.set_ylabel('Total wall time (s)')
    ax_time.set_title('Timing Breakdown vs q')
    ax_time.set_xticks(qs)
    ax_time.legend(fontsize=8)

    # Bottom-right: final error vs q
    ax_err.semilogy(qs, errors, '^-', color='C6')
    ax_err.set_xlabel('q (qubits)')
    ax_err.set_ylabel('Final error ε')
    ax_err.set_title('Simulation Error vs q')
    ax_err.set_xticks(qs)

    plt.suptitle('Quantum Circuit Scaling', y=1.01)
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.outfile}")


if __name__ == '__main__':
    main()
