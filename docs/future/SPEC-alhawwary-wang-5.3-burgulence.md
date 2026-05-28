# SPEC: Alhawwary & Wang (2018) §5.3 — Decaying Burgers Turbulence

Target reproduction: Fig. 22 energy-spectrum comparison at t=0.1 for the
decaying Burgulence case. Source: Alhawwary & Wang, JCP 373 (2018) 835-862,
§5.3 "Resolution for the Burgers turbulence".

This spec defines the case only. Implementation work is scoped in the
"Gaps & tasks" section but left for a follow-up PR.

## 1. Governing equation

Viscous Burgers in conservative form:

    du/dt + d(f(u))/dx = gamma * d2u/dx2,    f(u) = u^2 / 2

with `gamma = 2e-4` (paper Eq. 79). Periodic BC.

## 2. Domain and grid

- Spatial domain: `x in [0, 2*pi]`, periodic.
- Nominal grid: `N_n = 1201` nodes for FD/CD baselines; DG uses
  `N_e = 200` with p=5 (same ~1200 DOFs). For our classical `shift`
  solver we use `N = N_n` nodes uniformly on `[0, 2*pi)`.
- Since our quantum paths require `N = 2^q`, this case runs only on the
  `--method shift` classical path (or is relaxed to `N = 1024` / `2048`
  if we want to pad for a later quantum comparison; paper result is
  insensitive at the wavenumbers of interest).

## 3. Initial condition

Energy spectrum (paper Eqs. 80-81):

    E(k, 0) = A * k^4 * rho^5 * exp(-k^2 * rho^2)
    A       = 2 / (3 * sqrt(pi))
    rho     = 10                  # peak at k = 13

Physical field (paper Eqs. 82-83):

    v(x) = sum_{j=0..n_k} sqrt(2 * E(k_j)) * cos(k_j * x + 2*pi * Phi(k_j))
           + v_m

- `Phi(k)` uniform in `[0, 1]`, seeded. Real-field constraint
  `Phi(-k) = -Phi(k)` is enforced implicitly by the cosine form.
- `k_j = j` (integer wavenumber; domain length `2*pi` makes integers
  the natural Fourier basis).
- `k_max = 2048`, so `n_k = 2048`.
- `v_m = 75` — selected to give turbulence intensity
  `u_rms / v_m ~ 0.67%`. The mean advects the field quickly across
  the domain; fluctuations evolve on the slow timescale.

Ensemble: **64 independent samples**, differing only in the random-phase
seed. All schemes being compared MUST use the same 64 seeds to keep
the comparison fair (paper explicitly states this).

## 4. Time integration

- Scheme: our classical `shift` path uses forward Euler; the paper uses
  RK4. Forward Euler is adequate at the CFLs used here because the
  effective wave speed `v_m = 75` dominates; diffusion is small
  (`gamma = 2e-4`). If stability bites, upgrade to RK3/RK4 — out of
  scope for this spec.
- CFL: paper defines `CFL = |u_max| * dt / dx`, where `|u_max|` is the
  max eigenvalue of the **initial** solution (near-constant across the
  run because fluctuations are tiny). We must scale `dt` by
  `max|u_0|` rather than assuming wave speed 1 (current solver does
  the latter — see Gaps).
- Final time: `t = 0.1`. Paper notes this is "long enough that the
  spectrum reaches a statistically steady state" in the resolved band.

## 5. Diagnostic

Primary: **ensemble-averaged energy spectrum** `E(k)` at `t = 0.1`,
plotted log-log vs. `k`, compared against:

1. The analytic initial spectrum `E(k, 0)` (sanity — should match at t=0).
2. The `E ~ k^{-2}` reference slope (decaying Burgulence theory).
3. DNS-like reference: DGp5 with `N_e = 4096` is the paper's gold
   reference. For our purposes a well-resolved `shift` run at
   `N = 4096` or `8192` substitutes.

Computation:

    u_hat(k) = FFT(u(x, t=0.1)) / N          # unbiased DFT normalization
    E_s(k)   = 0.5 * |u_hat(k)|^2             # per-sample
    E(k)     = mean over 64 samples of E_s(k)

Report on `k in [1, N/2]`. Subtract the `v_m` DC component before
FFT, or just drop `k=0` from the plot.

Secondary: total kinetic energy `integral 0.5 * u'^2 dx` vs. `t`
(optional; sanity check on decay).

## 6. Acceptance criteria

1. At `t = 0` the ensemble-averaged `E(k)` matches the analytical
   `E(k, 0)` within 5% for `k in [1, 50]` (Monte-Carlo noise floor
   sets the bound for 64 samples).
2. At `t = 0.1` the ensemble-averaged `E(k)` exhibits a visible
   inertial range on a log-log plot whose slope matches `-2` within
   `+-0.2` over at least half a decade of `k`.
3. The high-`k` rolloff location is consistent across repeated runs
   with fresh seeds — i.e., the result is a property of the scheme,
   not of one unlucky draw.

## 7. Gaps vs. current murali_burgers code

Working from `burgers_solver.py`, `burgers_classical.py`, and the
nonlinear RHS in `burgers_nonlinear.py`:

| Need | Status |
|------|--------|
| Domain length `L` as parameter (default `2*pi`) | Hardcoded `[0, 1]` in solver L154-162. Needs `--L` arg plumbed through grid. |
| Burgulence IC per Eqs. 80-83 | Missing. `initial_condition_multimode` uses `sin(k*pi*x)` Dirichlet basis with `max\|u\|=1` normalization — not this case. |
| CFL scaled by `max\|u_0\|` | Missing. Solver sets `dt = cfl * dx` assuming speed 1. With `v_m = 75` this is off by 75x. |
| Nonzero mean velocity | Supported implicitly (RHS is translation-invariant for periodic BC), but untested. |
| Ensemble driver (64 seeds) | Use sweeper with `--ic-seed` variation. Post-hoc averaging step is new. |
| Energy-spectrum post-processor | Missing. Reads `artifacts.json` solution snapshots, FFTs, averages, plots. |
| `k^{-2}` reference overlay | New plot script (sibling to `plot_paper_aligned.py`). |

## 8. Out of scope for this spec

- DG, FD6, CD6 scheme comparisons (paper's main point). We only run
  our shift-operator classical baseline; paper comparisons require
  separate solver implementations.
- Quantum-path reproduction. Secondary follow-up once the classical
  baseline matches.
- BR2 diffusion flux (DG-specific).
- RK3/RK4 upgrade.

## 9. References

- Alhawwary, M.; Wang, Z.J. *Fourier analysis and evaluation of DG, FD
  and compact difference methods for conservation laws.* JCP 373
  (2018) 835-862. DOI: 10.1016/j.jcp.2018.07.018.
- Paper §5.3 and Fig. 22 are the specific target.
- Paper refs [45, 66] for the IC form (earlier Burgulence studies).
