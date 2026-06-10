# A-B Bakeoff TOMLs — CH vs QLBM

*Companion to OVERVIEW-burgers-solver.md.*

## Experiments at a glance (by importance; each runs at q=6/7/8)

| # | experiment | defining args | purpose |
|---|------------|---------------|---------|
| 1 | `shock` | A=0.4, nu=0.03, 90% t_shock | the sharpest A-B — carry both methods INTO a forming shock, where they differ most |
| 2 | `mutual_nu015` | nu=0.015, A=0.3 | CH at its resolvable floor — the crossover edge where the gap to QLBM closes |
| 3 | `show_qlbm_margin` | nu=0.025, A=0.5 | QLBM's win condition: peaked phi pushes CH to its edge, QLBM flat + cheaper/stabler |
| 4 | `show_ch_smooth` | nu=0.08, A=0.3 | CH's win condition: very smooth phi, exact-via-transform beats QLBM's scheme floor |
| 5 | `mutual_nu030` | nu=0.03, A=0.3 | comfortable mid-range reference; CH ahead, both healthy |
| 6 | `mutual_nu050` | nu=0.05, A=0.3 | easiest case, both excellent (sanity anchor) |
| 7 | `mutual_nu020` | nu=0.02, A=0.3 | fills the nu corner-map between mid-range and the floor |
| 8 | `show_ch_stress` | nu=0.04, A=0.6 | high-amplitude probe: how far CH holds before QLBM overtakes |

Each `burgers_ab_*.toml` is **one operating point, no swept variable** — a
single head-to-head of Cole-Hopf (`cole_hopf_circuit`), QLBM
(`qlbm_circuit`), classical LBM, all scored against the FTCS reference. The
postproc (`plot_method_compare.py`) animates the one point over time and
writes `method_compare.gif` + `method_compare_resources.png`.

## Naming and the q-family

Files are `burgers_ab_<set>_q<Q>_<point>.toml`, where `<set>` is `mutual` or
`show` and `q<Q>` is the grid exponent.  Every operating point exists at
**q=6, 7, 8** (the same physics at three resolutions), so the set is
7 points x 3 q-levels = 21 files.  Start with q6 (fast, proves plumbing),
escalate to q8 (paper-grade).  All 7 points stay QLBM-stable (tau>1) at
every q.

## Fixed across every file

| knob | value | why |
|------|-------|-----|
| `--q` | 6 / 7 / 8 | the resolution family (N = 64 / 128 / 256) |
| `--n-steps` | 100 / 200 / 400 | scales with N to hold the physical window fixed (t_end~0.156, ~30% t_shock at A=0.3) |
| `--cfl` | 0.1 | dt = cfl*dx; QLBM lands every 10 fine steps |
| `--segment-size` / `--save-every` | 10 / 10 | aligns CH/FTCS snapshots to the QLBM lattice cadence |
| `--shots` | 150000 | costly but... |
| `--evolution-mode` | measure_reprepare | both methods, k=1 incremental |
| IC / BC | sine / periodic | — |

## Best-vs-best knobs (each method at full strength)

The narrative comes from the **operating point** — both methods run their best settings everywhere.

- **CH**: `--propagator qft-diagonal`, `--bond-dim 8`, `--phi-modes 8`.
  bond-dim 8 (vs the smooth-phi default 4) gives CH headroom when phi
  sharpens at low nu / high A; phi-modes 8 low-passes shot noise before the
  log-derivative amplifies it.
- **QLBM**: `--fock-qubits 3` (qc=2 too coarse, qc=4 is 4x cost/site for
  marginal gain), `--qalb-collision-trotter-reps 0` (dense exact collision
  = best L2). For a hardware-honest depth metric instead, set reps=2 (the
  Trotter synthesis; ~1/reps^2 error) — at the cost of higher L2.

## The mutual-range fair-fight set (4 points x 3 q = 12 files)

A clean nu corner-map where **both methods are viable** (CH floor nu>=0.015;
QLBM tau>1 throughout, at every q). Fixed A=0.3; only nu moves.

| point (file stem `..._q<Q>_..`) | nu | CH peakedness phi_max/phi_min = exp(A/2*pi*nu) | expectation |
|------|------|------|------|
| `burgers_ab_mutual_q<Q>_nu050` | 0.05 | ~2.6x | both excellent; CH best L2 |
| `burgers_ab_mutual_q<Q>_nu030` | 0.03 | ~4.9x | CH still ahead |
| `burgers_ab_mutual_q<Q>_nu020` | 0.02 | ~11x | CH starting to feel it |
| `burgers_ab_mutual_q<Q>_nu015` | 0.015 | ~24x | CH near its floor; gap narrows |

## The showcase set (3 points x 3 q = 9 files)

Single points chosen to make one method's advantage visible. All stay
inside the mutual range (QLBM wins on margin/resources, never by sitting
where CH is broken).

| point (file stem `..._q<Q>_..`) | nu / A | shows |
|------|--------|-------|
| `burgers_ab_show_q<Q>_ch_smooth` | 0.08 / 0.3 | CH wins L2: very smooth phi (~1.8x), exact-via-transform beats QLBM's ~0.11 scheme floor |
| `burgers_ab_show_q<Q>_qlbm_margin` | 0.025 / 0.5 | QLBM wins: peaked phi (~24x) pushes CH to its edge, QLBM flat at its floor + cheaper/stabler |
| `burgers_ab_show_q<Q>_ch_stress` | 0.04 / 0.6 | high-amplitude CH stress (~11x): how far CH holds before QLBM overtakes |

## The shock challenge (1 point x 3 q = 3 files)

The only cases that run long enough to FORM the shock (~90% t_shock); all
the others stop in the smooth pre-shock regime (29-59% t_shock).  A=0.4,
nu=0.03 chosen so both methods stay viable THROUGH the shock: CH peakedness
~8.3x at the start (margin to sharpen), QLBM tau=2.42 at q6 (stable).  This
is where the two methods are stressed most and should differ most.

| point (file stem `..._q<Q>`) | A / nu | n_steps (q6/7/8) | shows |
|------|--------|------|------|
| `burgers_ab_shock_q<Q>` | 0.4 / 0.03 | 230 / 460 / 920 | both methods carried into a forming shock; where the A-B is sharpest |

Note `--n-steps` here is set to reach 90% t_shock at each q (not the fixed
~30%-window value the other sets use), so these runs are ~2x longer than the
same-q mutual/show cases.

## Per-method knobs are q-independent

The best-vs-best knobs (CH bd8/phi8, QLBM qc3/reps0) are the same at q=6, 7,
8 and need no retuning: CH's bond-dim and phi-modes track phi's smoothness
(set by nu/A, not grid size), and QLBM's fock-qubits and trotter-reps are
per-SITE quantities independent of the grid exponent.  Only `--n-steps`
scales with q.

## Runtime (plan accordingly)

CH is statevector-simulated; cost ~ 2^(q+bond+anc) * depth * #segments, and
n_steps scales with N, so cost climbs steeply with q:

| q | N | n_steps | CH/case (approx) | QLBM/case |
|---|---|---------|------------------|-----------|
| 6 | 64 | 100 | ~15 min | ~1-2 min |
| 7 | 128 | 200 | ~1-2 hr | ~5 min |
| 8 | 256 | 400 | ~6-8 hr | ~10-20 min |

FTCS is seconds at any q.  Start the q6 family (fast, proves alignment +
plotting), then escalate.  The full q8 set is ~45-55 CH-hours — run on a
cluster, or shorten `--n-steps` for a faster (shorter-window) pass.

## Running one

```
q8020-sweep q8020-mps-burgers/input/burgers_ab_mutual_q6_nu030.toml
```

No `--group` needed — each file's named groups all run, and the
`_group_postproc` on the file produces the comparison figure in the run dir.
