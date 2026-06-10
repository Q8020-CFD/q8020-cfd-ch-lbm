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
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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


def _case_id(meta: dict) -> str | None:
    """TOML section name for this case (unique disambiguator)."""
    for c in meta.get('case', []):
        if c.get('case_id'):
            return c['case_id']
        p = c.get('params')
        if isinstance(p, dict) and p.get('_case_id'):
            return p['_case_id']
    return None


def _case_flag(meta: dict, cli_flag: str) -> str | None:
    """Best-effort fetch of a CLI flag value (e.g. --evolution-mode)."""
    bare = cli_flag.lstrip('-')
    for c in meta.get('case', []):
        p = c.get('params')
        if isinstance(p, dict):
            for k in (cli_flag, bare, bare.replace('-', '_')):
                if k in p:
                    return p[k]
        for fld in ('command', 'args'):
            v = c.get(fld)
            toks: list[str] = []
            if isinstance(v, list):
                for t in v:
                    toks += str(t).split()
            elif isinstance(v, str):
                toks = v.split()
            if cli_flag in toks:
                i = toks.index(cli_flag)
                if i + 1 < len(toks):
                    return toks[i + 1]
    return None


def _analysis(meta: dict) -> dict:
    """The harvested 'analysis' fragment carrying the resource metrics
    (n_qubits, avg_circuit_depth, n_circuits_executed, runtime ...)."""
    frags = meta.get('analysis', [])
    for a in frags:
        if isinstance(a, dict) and (
            a.get('n_qubits') is not None
            or a.get('avg_circuit_depth') is not None
            or a.get('n_circuits_executed')
        ):
            return a
    for a in frags:
        if isinstance(a, dict):
            return a
    return {}


# Flags a sweep may vary; the first that differs across the resource-bearing
# cases becomes the x-axis of the resource panel.
SWEEP_FLAGS = ['--nu', '--ic-amplitude', '--q', '--fock-qubits']


def _detect_swept(cands: list[dict]) -> tuple[str | None, list[str | None]]:
    for flag in SWEEP_FLAGS:
        vals = [_case_flag(e['case'], flag) for e in cands]
        if any(v is None for v in vals):
            continue
        if len(set(vals)) > 1:
            return flag, vals
    return None, [None] * len(cands)


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
    'ftcs_reference': {
        'color': '#000000', 'ls': '-', 'lw': 2.5,
        'label': 'FTCS (reference)',
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
        'color': '#2ca02c', 'ls': '--', 'lw': 1.5,
        'label': 'Cole-Hopf classical (MPS)',
    },
    'cole_hopf_circuit': {
        'color': '#17becf', 'ls': '-', 'lw': 2.0,
        'label': 'Cole-Hopf',
    },
    'lbm': {
        'color': '#d62728', 'ls': '--', 'lw': 1.5,
        'label': 'LBM classical (D1Q3 BGK)',
    },
    'qlbm_circuit': {
        'color': '#d62728', 'ls': '-', 'lw': 2.0,
        'label': 'QLBM',
    },
    'direct_lcu': {
        'color': '#8c564b', 'ls': '-', 'lw': 2.0,
        'label': 'Direct-u LCU (conservative)',
    },
}


# ── Circuit resources accumulate per measure-reprepare segment ─────────
# Each segment runs a circuit, so depth / CX / circuit-count / runtime grow
# segment-by-segment.  per_step_metrics carries one entry per segment, tagged
# with the timestep it ends on, so we can (a) animate cumulative curves vs
# time in the movie and (b) bar the final totals in the standalone PNG.
# Width (# qubits) is the only constant.

def _sm_depth(m: dict):
    after = (m.get('transpile') or {}).get('after') or {}
    return m.get('circuit_depth', after.get('depth'))


def _sm_cx(m: dict):
    after = (m.get('transpile') or {}).get('after') or {}
    g = m.get('gate_counts', after.get('gate_counts')) or {}
    return g.get('cx')


def _sm_circuits(m: dict):
    return m.get('n_circuits', 0) or 0


def _sm_runtime(m: dict):
    tr = (m.get('transpile') or {}).get('wall_time', 0) or 0
    ex = (m.get('execute') or {}).get('wall_time', 0) or 0
    return (
        (m.get('execution_time_s') or ex)
        + (m.get('transpilation_time_s') or tr)
        + (m.get('circuit_construction_time_s') or 0)
    )


# Time-varying metrics, animated as cumulative curves: (title, increment fn).
CUM_METRICS = [
    ('cumulative depth', _sm_depth),
    ('cumulative CX', _sm_cx),
    ('# circuits (cumulative)', _sm_circuits),
    ('sim runtime (s, cumulative)', _sm_runtime),
]


def _sval_sort_key(v):
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, str(v))


def _psm(case: dict) -> list:
    return _analysis(case).get('per_step_metrics') or []


def _series_total(case: dict, fn) -> float:
    return float(sum((fn(m) or 0.0) for m in _psm(case)))


def _series_native(case: dict, fn, dt: float):
    """Native (segment_time, cumulative) points at the series' OWN segment
    cadence -- no resampling to a coarser timeline, so a fine-cadence method
    keeps full resolution.  A (t=0, 0) baseline is prepended so every series
    starts at the common origin and rises at its own first segment."""
    pairs = sorted(
        (m.get('step'), fn(m)) for m in _psm(case) if m.get('step') is not None
    )
    if not pairs:
        return np.array([]), np.array([])
    t = np.array([0.0] + [p[0] * dt for p in pairs])
    cum = np.concatenate(([0.0], np.cumsum([(p[1] or 0.0) for p in pairs])))
    return t, cum


def setup_resource_curves(res_axes, series, key_style, dt, t_end, animated):
    """Cumulative-resource artists, each drawn at its OWN segment cadence.
    Coarse series (few segments, e.g. QLBM) get a marked staircase; dense
    series (e.g. CH) get a smooth line.  Curves start at the first segment so
    they sit left.  Returns {(metric_idx, key): (line, t_arr, cum)}."""
    seg_counts = {
        s['key']: len([m for m in _psm(s['case']) if m.get('step') is not None])
        for s in series
    }
    max_pts = max(seg_counts.values(), default=1)
    first_t: list[float] = []
    artists = {}
    for mi, (title, fn) in enumerate(CUM_METRICS):
        ax = res_axes[mi]
        ymax = 0.0
        for s in series:
            t, cum = _series_native(s['case'], fn, dt)
            if t.size:
                ymax = max(ymax, float(cum.max()))
                first_t.append(float(t[0]))
            sty = key_style[s['key']]
            coarse = seg_counts[s['key']] < 0.6 * max_pts
            ln, = ax.plot(
                [] if animated else t,
                [] if animated else cum,
                color=sty['color'], ls=sty['ls'], lw=1.6,
                drawstyle='steps-post' if coarse else 'default',
                marker='o' if coarse else None, ms=3,
                label=sty['label'],
            )
            artists[(mi, s['key'])] = (ln, t, cum)
        ax.set_title(title, fontsize=9)
        ax.set_yscale('log')
        if ymax > 0:
            ymin_pos = min(
                (float(cum[cum > 0].min()) if np.any(cum > 0) else ymax)
                for _, (_, _, cum) in
                ((k, artists[k]) for k in artists if k[0] == mi)
            ) if any(k[0] == mi for k in artists) else ymax
            ax.set_ylim(ymin_pos * 0.5, ymax * 3.0)
        else:
            ax.set_ylim(0.1, 1.0)
        ax.set_xlabel("simulation time", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
    left = min(first_t) if first_t else 0.0   # start at the first segment
    for ax in res_axes:
        ax.set_xlim(left, t_end)
    if res_axes:
        res_axes[0].legend(fontsize=7, loc='upper left')
    return artists


def plot_resource_panel(candidates: list[dict], out_path: Path) -> None:
    """Standalone PNG: final accumulated resource totals as grouped bars."""
    circ = [
        e for e in candidates
        if _analysis(e['case']).get('n_qubits') is not None
    ]
    if not circ or not any(_psm(e['case']) for e in circ):
        print("No circuit per-step metrics; skipping resource panel.",
              file=sys.stderr)
        return
    swept_flag, _ = _detect_swept(circ)
    for e in circ:
        e['sval'] = (
            _case_flag(e['case'], swept_flag) if swept_flag else e['method']
        )
    methods = sorted({e['method'] for e in circ})
    cats = sorted({e['sval'] for e in circ}, key=_sval_sort_key)
    by = {(e['method'], e['sval']): e['case'] for e in circ}

    # (title, per-segment fn or 'nq' for the constant width, is_int)
    bar_metrics = [
        ('# qubits (width)', 'nq', True),
        ('# circuits (total)', _sm_circuits, True),
        ('cumulative depth', _sm_depth, False),
        ('cumulative CX', _sm_cx, False),
        ('sim runtime (s)', _sm_runtime, False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    bw = 0.8 / max(len(methods), 1)
    xpos = np.arange(len(cats))
    for ax, (title, fn, is_int) in zip(axes, bar_metrics):
        for mi, method in enumerate(methods):
            ys = []
            for c in cats:
                case = by.get((method, c))
                if case is None:
                    ys.append(None)
                elif fn == 'nq':
                    ys.append(_analysis(case).get('n_qubits'))
                else:
                    ys.append(_series_total(case, fn))
            xs = xpos + (mi - (len(methods) - 1) / 2) * bw
            xs_p = [x for x, y in zip(xs, ys) if y is not None]
            ys_p = [y for y in ys if y is not None]
            if not ys_p:
                continue
            sty = METHOD_STYLE.get(method, {})
            bars = ax.bar(
                xs_p, ys_p, bw,
                color=sty.get('color', f'C{mi}'),
                label=sty.get('label', method),
            )
            ax.bar_label(
                bars,
                labels=[f"{y:.0f}" if is_int else f"{y:.3g}" for y in ys_p],
                fontsize=6, padding=1,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(c) for c in cats], fontsize=8)
        if swept_flag:
            ax.set_xlabel(swept_flag.lstrip('-'), fontsize=8)
        ax.grid(axis='y', ls=':', alpha=0.4)
    axes[-1].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc='lower center',
            ncol=len(labels), fontsize=9, frameon=False,
        )
    sweep_txt = f" vs {swept_flag.lstrip('-')}" if swept_flag else ""
    fig.suptitle(f"Accumulated circuit resources{sweep_txt}", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved resource panel to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('postproc_json', nargs='?', default=None)
    p.add_argument('--sweep-dir', type=Path, default=None)
    p.add_argument('--outfile', default=None)
    p.add_argument('--fps', type=int, default=8)
    p.add_argument(
        '--frames', type=int, default=0,
        help="Force an exact frame count, evenly spaced over [0, n_steps] "
             "and resampled per method from nearest snapshot. 0 (default) "
             "uses the union of all methods' stored snapshot steps.",
    )
    p.add_argument(
        '--anchor', default=None,
        help="Method name (e.g. qlbm_circuit) whose stored snapshot steps "
             "define the timeline; every other method is sampled onto "
             "exactly those steps. Use to match a coarse method's genuine "
             "cadence without padding it or downsampling the others' solve. "
             "Takes precedence over --frames.",
    )
    p.add_argument(
        '--ref', default='ftcs',
        help="DEPRECATED / ignored -- superseded by --reference (the "
             "reference is now a stored FTCS case, ftcs_reference).  Kept "
             "only so existing TOMLs that pass --ref still parse.",
    )
    p.add_argument(
        '--hide', default='',
        help="Comma-separated method names to omit from the plot "
             "(e.g. 'lbm,shift').",
    )
    p.add_argument(
        '--no-ic', action='store_true',
        help="Do not draw the initial-condition line.",
    )
    p.add_argument(
        '--no-resources', action='store_true',
        help="Skip the circuit resource-utilization panel "
             "(<outfile>_resources.png).",
    )
    p.add_argument(
        '--reference', default='ftcs_reference',
        help="Method name of the case whose stored solution_steps are the "
             "classical reference (error baseline + reference curve). The "
             "series is rendered as the reference, not as a method line. "
             "Must be present in the sweep (run it as its own TOML case).",
    )
    args = p.parse_args()
    hidden = {m.strip() for m in args.hide.split(',') if m.strip()}

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

    # Collect candidate series.  A single sweep may contain several cases
    # with the SAME method (e.g. measure_reprepare vs single cole_hopf_circuit);
    # each becomes its own series with a unique key so they overlay as
    # distinct lines instead of overwriting one another.
    candidates: list[dict] = []
    for c in all_cases:
        cm = _extract_case_params(c)
        if cm is None:
            continue
        m = cm.get('method', '')
        if m not in METHOD_STYLE:
            continue
        candidates.append({
            'method': m, 'case': c, 'params': cm,
            'case_id': _case_id(c),
            'evo': _case_flag(c, '--evolution-mode'),
        })

    # Assign a series key per candidate.  Methods with a single case keep
    # the bare method name (so existing sweeps/styles are unchanged);
    # collisions are disambiguated by evolution-mode when distinct, else
    # by case_id.
    counts = Counter(e['method'] for e in candidates)
    by_method_group: dict[str, list[dict]] = {}
    for e in candidates:
        by_method_group.setdefault(e['method'], []).append(e)
    LS_CYCLE = ['-', '--', ':', '-.']
    key_style: dict[str, dict] = {}
    for m, es in by_method_group.items():
        base = METHOD_STYLE[m]
        if counts[m] == 1:
            es[0]['key'] = m
            key_style[m] = dict(base)
            continue
        evos = [e['evo'] for e in es]
        use_evo = all(evos) and len(set(evos)) == len(es)
        for i, e in enumerate(es):
            tag = e['evo'] if use_evo else (e['case_id'] or f'v{i}')
            key = f"{m} ({tag})"
            e['key'] = key
            st = dict(base)
            st['ls'] = LS_CYCLE[i % len(LS_CYCLE)]
            st['label'] = f"{base['label']} [{tag}]"
            key_style[key] = st

    # Series key -> case meta, for pulling per-step resource metrics later.
    key_case = {e['key']: e['case'] for e in candidates if e.get('key')}

    if len(candidates) < 2:
        print(
            f"Need at least 2 series, found: "
            f"{[e['key'] for e in candidates]}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Series found: {[e['key'] for e in candidates]}",
        file=sys.stderr,
    )

    # Extract each series' stored snapshots, keyed by integer step.
    method_snaps: dict[str, dict[int, np.ndarray]] = {}
    key_method: dict[str, str] = {}
    key_caseid: dict[str, str | None] = {}
    x = None
    anchor_meta = None

    for e in candidates:
        key, case = e['key'], e['case']
        art = _find_solver_entry(
            case.get('artifacts', []), 'solution_steps',
        )
        sol_steps = art.get('solution_steps', {})
        g = np.array(art.get('grid', []))
        if not sol_steps or g.size == 0:
            print(f"  {key}: no solution_steps, skipping", file=sys.stderr)
            continue

        method_snaps[key] = {
            int(k): np.array(v) for k, v in sol_steps.items()
        }
        key_method[key] = e['method']
        key_caseid[key] = e['case_id']
        if x is None:
            x = g
            anchor_meta = e['params'] or {}

    if not method_snaps or x is None:
        print("No usable frames.", file=sys.stderr)
        sys.exit(1)

    # Capture the FTCS run before hiding, in case it is the reference.
    # Drop series the user asked to hide (kept above for reference use).
    # A token matches by series key, method name, or case_id.
    if hidden:
        method_snaps = {
            k: s for k, s in method_snaps.items()
            if not (hidden & {k, key_method[k], key_caseid.get(k)})
        }
        if not method_snaps:
            print(
                f"All series hidden by --hide={sorted(hidden)}.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Pull out the reference series.  The classical reference is a stored
    # case (run once as its own TOML group) -- it is the error baseline and
    # the reference curve, NOT a method line, so remove it from method_snaps
    # and keep its snapshots separately.
    ref_keys = [
        k for k in method_snaps
        if args.reference in {k, key_method[k], key_caseid.get(k)}
    ]
    if not ref_keys:
        print(
            f"--reference '{args.reference}' not among series "
            f"{list(method_snaps)}; run it as its own case "
            f"(e.g. --method ftcs_reference).",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(ref_keys) > 1:
        print(
            f"--reference '{args.reference}' is ambiguous across "
            f"{ref_keys}; pass a full series key.",
            file=sys.stderr,
        )
        sys.exit(1)
    ref_key = ref_keys[0]
    ref_snaps = method_snaps.pop(ref_key)
    ref_label = key_style.get(ref_key, {}).get('label', 'FTCS (reference)')
    if not method_snaps:
        print(
            f"Only the reference series '{ref_key}' present; "
            "nothing to compare against it.",
            file=sys.stderr,
        )
        sys.exit(1)

    dt = float(anchor_meta.get('dt', 0.0))
    nu = float(anchor_meta.get('nu', 1e-2))
    q_val = anchor_meta.get('q', '?')

    # Build ONE common frame timeline and resample every method onto it.
    # Intersecting raw step keys is brittle: any method whose snapshot
    # cadence differs (segment boundaries) or that drops a diverged step
    # collapses the whole animation.  Instead, take the union of all
    # methods' step keys as the timeline (or an explicit evenly-spaced
    # --frames count over [0, n_steps]) and fill each method's curve from
    # its NEAREST stored snapshot.  Result: a consistent frame count with
    # every method drawn in every frame, independent of save cadence.
    union_keys = sorted(set().union(*(s.keys() for s in method_snaps.values())))
    n_steps_total = int(anchor_meta.get('n_steps', union_keys[-1]))

    if args.anchor:
        cand = [
            k for k in method_snaps
            if args.anchor in {k, key_method[k], key_caseid.get(k)}
        ]
        if not cand:
            print(
                f"--anchor '{args.anchor}' not among series "
                f"{list(method_snaps)}",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(cand) > 1:
            print(
                f"--anchor '{args.anchor}' is ambiguous across "
                f"{cand}; pass a full series key.",
                file=sys.stderr,
            )
            sys.exit(1)
        step_keys_common = sorted(method_snaps[cand[0]].keys())
    elif args.frames and args.frames > 0:
        step_keys_common = sorted(set(
            int(round(s))
            for s in np.linspace(0, n_steps_total, args.frames)
        ))
    else:
        step_keys_common = union_keys

    def _nearest(snaps: dict[int, np.ndarray], step: int) -> np.ndarray:
        keys = np.fromiter(snaps.keys(), dtype=int)
        return snaps[int(keys[np.argmin(np.abs(keys - step))])]

    frames: dict[str, list[np.ndarray]] = {
        m: [_nearest(snaps, s) for s in step_keys_common]
        for m, snaps in method_snaps.items()
    }
    u0 = frames[next(iter(frames))][0].copy()

    # Reference curve: the stored ftcs_reference case, resampled onto the
    # common timeline from its nearest snapshot (same q-grid, so no spatial
    # interpolation).  No recomputation -- the reference is rendered, not
    # regenerated, here.
    bc = anchor_meta.get('bc', 'periodic')
    print(
        f"  Reference series: {ref_key} "
        f"({len(ref_snaps)} stored snapshots)",
        file=sys.stderr,
    )
    frames_ref = [_nearest(ref_snaps, s) for s in step_keys_common]

    # Per-method divergence masking.  A single diverged method must not
    # collapse the whole animation: instead of globally truncating at the
    # first bad step (which leaves one frame if any method blows up at its
    # first snapshot), mask each method independently.  From its first
    # non-finite / over-amplitude frame on, that method's curve is set to
    # NaN (so its line simply vanishes) while every healthy method keeps
    # animating over the full common step range.
    amp0 = max(np.max(np.abs(u0)), 1.0)
    amp_limit = 10.0 * amp0
    n_common = len(step_keys_common)
    nan_frame = np.full_like(np.asarray(u0, dtype=float), np.nan)
    diverged: dict[str, int] = {}
    for m in frames:
        for i in range(n_common):
            f = frames[m][i]
            if (not np.all(np.isfinite(f))) or np.max(np.abs(f)) > amp_limit:
                for j in range(i, n_common):
                    frames[m][j] = nan_frame.copy()
                diverged[m] = int(step_keys_common[i])
                break
    for m, step in diverged.items():
        print(
            f"  {m}: diverged at step {step} "
            f"(|u| > {amp_limit:.1f}); masked from there on",
            file=sys.stderr,
        )
    if all(diverged.get(m, n_common) == 0 for m in frames):
        print(
            "All methods diverged at the first step; nothing to animate.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build animation.  ymax is taken over finite values only so a brief
    # over-amplitude excursion (now masked) cannot blow up the y-scale.
    finite_maxes = []
    for fs in list(frames.values()) + [frames_ref]:
        arr = np.abs(np.asarray(fs, dtype=float))
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_maxes.append(float(arr.max()))
    ymax = (max(finite_maxes) if finite_maxes else 1.0) * 1.15
    times = np.array([int(k) * dt for k in step_keys_common])

    # Circuit-bearing series (those reporting n_qubits) get an animated
    # cumulative-resource block embedded to the right of the u/error panels.
    circ_keys = [
        k for k in frames
        if _analysis(key_case.get(k, {})).get('n_qubits') is not None
    ]
    embed_res = bool(circ_keys) and not args.no_resources
    if embed_res:
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.2)
        lg = gs[0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax_u = fig.add_subplot(lg[0])
        ax_err = fig.add_subplot(lg[1])
        rg = gs[1].subgridspec(2, 2, hspace=0.5, wspace=0.33)
        res_axes = [fig.add_subplot(rg[i // 2, i % 2]) for i in range(4)]
    else:
        fig, (ax_u, ax_err) = plt.subplots(
            2, 1, figsize=(10, 6.5),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        res_axes = []
    ic_name = anchor_meta.get('ic', '?')
    fig.suptitle(
        f"Method Comparison: q={q_val} (N={2**int(q_val)}), "
        f"$\\nu$={nu:.0e}, {bc}, {ic_name} IC",
        fontsize=12, fontweight="bold",
    )

    ax_u.set_xlim(x[0], x[-1] + x[1] - x[0])
    ax_u.set_ylim(-ymax, ymax)
    ax_u.set_ylabel("u(x, t)")
    ax_u.grid(alpha=0.2)

    # IC (optional) + reference line
    if not args.no_ic:
        ax_u.plot(x, u0, 'k--', alpha=0.15, lw=1, label='IC')
    ref_line, = ax_u.plot(
        [], [], 'k-', lw=2.5, alpha=0.35, label=ref_label,
    )

    # Series lines (one per disambiguated series key).
    lines: dict[str, plt.Line2D] = {}
    for key in frames:
        sty = key_style[key]
        label = sty['label']
        if key in diverged:
            label += f" (diverged @ step {diverged[key]})"
        ln, = ax_u.plot(
            [], [], ls=sty['ls'], color=sty['color'],
            lw=sty['lw'], label=label,
        )
        lines[key] = ln

    ax_u.legend(loc='lower left', fontsize=9)
    time_text = ax_u.text(
        0.02, 0.95, '', transform=ax_u.transAxes,
        fontsize=10, va='top', fontfamily='monospace',
    )

    # Error panel: each series vs the reference.
    err_lines: dict[str, plt.Line2D] = {}
    errors: dict[str, np.ndarray] = {}
    for key in frames:
        sty = key_style[key]
        err = np.full(len(step_keys_common), np.nan)
        for i in range(len(step_keys_common)):
            fm = frames[key][i]
            norm_ref = np.linalg.norm(frames_ref[i])
            if np.all(np.isfinite(fm)) and norm_ref > 1e-15:
                err[i] = np.linalg.norm(
                    fm - frames_ref[i],
                ) / norm_ref
        errors[key] = err
        ln, = ax_err.plot(
            [], [], ls=sty['ls'], color=sty['color'], lw=1.5,
            label=f'{sty["label"]} vs {ref_label.split()[0]}',
        )
        err_lines[key] = ln

    ax_err.set_xlim(times[0], times[-1])
    all_errs = np.concatenate(
        [e[np.isfinite(e) & (e > 0)] for e in errors.values()]
    ) if errors else np.array([1e-6])
    if all_errs.size == 0:
        all_errs = np.array([1e-6])
    ax_err.set_yscale('log')
    ax_err.set_ylim(
        max(all_errs.min() * 0.5, 1e-12),
        all_errs.max() * 3.0,
    )
    ax_err.set_xlabel("Time")
    ax_err.set_ylabel(f"Rel. L2 error vs {ref_label.split()[0]}")
    ax_err.legend(loc='upper left', fontsize=8)
    ax_err.grid(alpha=0.2)

    # Animated cumulative-resource curves (depth / CX / circuits / runtime),
    # growing per measure-reprepare segment.  Width is constant -> annotate.
    res_artists: dict = {}
    if res_axes:
        series = [{'key': k, 'case': key_case[k]} for k in circ_keys]
        res_artists = setup_resource_curves(
            res_axes, series, key_style, dt, times[-1], animated=True,
        )
        widths = {
            key_style[k]['label']: _analysis(key_case[k]).get('n_qubits')
            for k in circ_keys
        }
        res_axes[0].text(
            0.02, 0.86,
            "width: " + ", ".join(
                f"{lbl.split('[')[0].strip()} {n}q"
                for lbl, n in widths.items() if n is not None
            ),
            transform=res_axes[0].transAxes, fontsize=6, va='top',
            color='0.3',
        )

    if embed_res:
        # add_gridspec subfigures are not tight_layout-compatible; the grid
        # already carries its own spacing, so just reserve room for suptitle.
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.985)
    else:
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
        for (_, _), (ln, t_arr, cum) in res_artists.items():
            mask = t_arr <= t + 1e-9
            ln.set_data(t_arr[mask], cum[mask])
        return (ref_line,) + tuple(lines.values()) + \
            tuple(err_lines.values()) + \
            tuple(a[0] for a in res_artists.values()) + (time_text,)

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

    if not args.no_resources:
        res_path = Path(outfile).with_name(
            Path(outfile).stem + '_resources.png',
        )
        plot_resource_panel(candidates, res_path)


if __name__ == '__main__':
    main()
