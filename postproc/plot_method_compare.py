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
        'label': 'Cole-Hopf circuit (pure-quantum)',
    },
    'lbm': {
        'color': '#d62728', 'ls': '--', 'lw': 1.5,
        'label': 'LBM classical (D1Q3 BGK)',
    },
    'qlbm_circuit': {
        'color': '#d62728', 'ls': '-', 'lw': 2.0,
        'label': 'LBM circuit (hybrid, Option A)',
    },
    'direct_lcu': {
        'color': '#8c564b', 'ls': '-', 'lw': 2.0,
        'label': 'Direct-u LCU (conservative)',
    },
}


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
        '--ref', choices=['godunov', 'ftcs'], default='godunov',
        help="Single reference curve: 'godunov' (entropy-satisfying "
             "upwind + viscous diffusion, recomputed; default) or 'ftcs' "
             "(the shift-operator classical run).",
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
    ftcs_snaps = next(
        (s for k, s in method_snaps.items() if key_method[k] == 'shift'),
        None,
    )

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

    if args.ref == 'ftcs' and ftcs_snaps is None:
        print(
            "--ref ftcs requested but no 'shift' method in the sweep.",
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

    # Single reference curve.
    bc = anchor_meta.get('bc', 'periodic')
    if args.ref == 'ftcs':
        ref_label = 'FTCS (reference)'
        frames_ref = [_nearest(ftcs_snaps, int(k)) for k in step_keys_common]
    else:
        # Godunov: entropy-satisfying upwind flux + viscous diffusion,
        # recomputed from the IC (not stored per-step).
        from burgers_classical import solve_burgers_godunov
        ref_label = 'Godunov (reference)'
        print('  Computing Godunov reference ...', file=sys.stderr)
        sols_godunov = solve_burgers_godunov(
            u0, x, nu, dt, n_steps_total, bc=bc,
        )
        frames_ref = [sols_godunov[int(k)] for k in step_keys_common]

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

    fig, (ax_u, ax_err) = plt.subplots(
        2, 1, figsize=(10, 6.5),
        gridspec_kw={"height_ratios": [3, 1]},
    )
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
        return (ref_line,) + tuple(lines.values()) + \
            tuple(err_lines.values()) + (time_text,)

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


if __name__ == '__main__':
    main()
