# REVIEW — SPEC-source-forcing.md

Review of [SPEC-source-forcing.md](SPEC-source-forcing.md). Findings to
fold into the spec before implementation begins. Written for an agent
that has not seen the prior conversation.

## Critical: sign error in V derivation

[SPEC-source-forcing.md:49](SPEC-source-forcing.md:49) gives

```
V(x,t) = −(1/(2ν)) · ∫_0^x g(x', t) dx'   + C(t)
```

which implies `V_x = −g/(2ν)`. This is the wrong sign.

**Re-derivation.** Substitute `φ_t = ν φ_xx − V·φ` and
`u = −2ν (ln φ)_x` into forced Burgers `u_t + u u_x = ν u_xx + g`:

```
u_t = −2ν (φ_t/φ)_x = −2ν ((ν φ_xx + S)/φ)_x   where S = −V·φ
    = −2ν² (φ_xx/φ)_x  −  2ν (S/φ)_x
```

The first term equals `ν u_xx − u u_x` (standard unforced Cole-Hopf
identity). So

```
u_t + u u_x − ν u_xx = −2ν (S/φ)_x = +2ν V_x
```

Setting this equal to `g`:

```
V_x = +g/(2ν)
V(x,t) = +(1/(2ν)) · ∫_0^x g(x', t) dx' + C(t)
```

**Consequence.** [SPEC-source-forcing.md:76](SPEC-source-forcing.md:76)
gives `V(x,t) = +cos(2πx)·cos(2πt)/(4πν)` (gauge-fixed). With the
correct sign it should be **negative**:

```
V(x,t) = −cos(2πx)·cos(2πt)/(4πν)
```

The implementation snippet at [SPEC-source-forcing.md:200](SPEC-source-forcing.md:200)
(`V_raw = -G / (2.0 * nu)`) inherits the same sign error. Fix by
flipping to `V_raw = +G / (2.0 * nu)`. The propagator math
`M_n = ν·L − diag(V_n)` at [SPEC-source-forcing.md:62](SPEC-source-forcing.md:62)
and [SPEC-source-forcing.md:229](SPEC-source-forcing.md:229) is then
self-consistent and doesn't change.

If shipped as written, the propagator runs with sign-flipped forcing
and the forced result diverges from the FTCS reference. Test #5
([SPEC-source-forcing.md:344](SPEC-source-forcing.md:344)) would catch
it, but the spec should be correct on its face — fix here, not in
debug.

Test #2 ([SPEC-source-forcing.md:331](SPEC-source-forcing.md:331))
must update its analytical comparison to the negative form.

## Lost optimization on the unforced path

[SPEC-source-forcing.md:239-267](SPEC-source-forcing.md:239)
restructures `heat_dense_block_full_circuit` to a per-step build loop
unconditionally. Today's path builds one `step_qc` and inlines it
`N_steps` times — eigendecomposition runs once. The proposed
restructure drops that optimization **even when `source_fn is None`**,
i.e., for every existing run.

Eigendecomposition of an N×N dense matrix is non-trivial at large
N_steps; the regression is real, not theoretical.

**Fix.** Keep the build-once branch when `source_fn is None`; only
enter the per-step loop when forcing is active:

```python
if source_fn is None:
    step_qc = heat_dense_block_step_circuit(q, nu, dt, L_box, bc=bc,
                                            encoding=encoding, V=None)
    for step_idx in range(N_steps):
        # existing inline + remap-ancilla-bit code
        ...
else:
    for step_idx in range(N_steps):
        t_mid = (step_idx + 0.5) * dt
        V_n = potential_from_source(source_fn, x, t_mid, nu, bc=bc)
        step_qc = heat_dense_block_step_circuit(q, nu, dt, L_box, bc=bc,
                                                encoding=encoding, V=V_n)
        # same inline + remap code
        ...
```

A small `if`; preserves perf for every existing run.

The same split applies to the SV path
([SPEC-source-forcing.md:272-283](SPEC-source-forcing.md:272), §5.6)
— make the structure explicit there too rather than "same shape as
5.5".

## Test #1 wording is too strong

[SPEC-source-forcing.md:327-330](SPEC-source-forcing.md:327): "the
dense-block path with V=None must produce **bitwise-identical**
circuits to the pre-change code."

If §5.5 always rebuilds per step, "bitwise-identical" doesn't hold —
new `QuantumCircuit` objects each step, even if logically equivalent.

**Fix.** Loosen to "SV output identical to 1e-12" or "transpiled
circuit logically equivalent (same gate sequence, same parameters)."
Combined with the build-once fix above, true bitwise identity
**can** survive — say so explicitly, and structure the test around
which property is being asserted.

## Smaller items

### `permute_to_encoding` on a diagonal vector

[SPEC-source-forcing.md:227-228](SPEC-source-forcing.md:227) calls
`permute_to_encoding(V, q, encoding)` on a 1D vector. Verify the
helper in `burgers_encoding.py` accepts a vector, not only an
operator. If it only takes operators, the workaround is
`permute_operator(np.diag(V), q, encoding).diagonal()` — correct but
wasteful (O(N²) memory for an O(N) operation). Either confirm vector
support or add a `permute_diagonal_to_encoding` helper.

### SV path is hand-waved

§5.6 ([SPEC-source-forcing.md:272-283](SPEC-source-forcing.md:272))
says the SV branch needs "the same shape" as §5.5. Future maintainers
will re-derive that mapping inconsistently. Spell out the SV-path
restructure with the same fidelity as the shots path, including the
build-once-when-unforced branch above.

### Strang-splitting language in non-goals

[SPEC-source-forcing.md:99-106](SPEC-source-forcing.md:99) lists
qft-diagonal+source as a non-goal because it "needs Strang splitting"
and §3 explicitly excludes "classical operator splitting (the
steerage path)." Risk of misreading: Strang splitting *as a circuit*
(V block as a separate quantum layer between heat layers, no
decode/re-encode) is **still pure-quantum**.

Clarify: classical-evaluate-and-reinject is what's excluded;
circuit-level Strang for qft-diagonal is just deferred (which §10
already says correctly). Tighten §3 wording so the two non-goals don't
appear to forbid the same thing for different reasons.

### Dispatch kwargs hand-off hazard

[SPEC-source-forcing.md:146](SPEC-source-forcing.md:146):
`# ... whatever new kwargs the shots-backend spec adds`. If both
specs land in either order without coordination, the dispatch
signature will conflict.

**Fix.** Either (a) sequence: shots-backend lands first, this spec
references its final signature, or (b) pin: "as of 2026-04-25 the
args are `[u0, x, nu, dt, n_steps, bc, propagator, shots,
snapshot_interval, bond_dim, encoding]`; verify against current
HEAD before merging." Either makes a future conflict visible
instead of silent.

### Test #6 acceptance is unverified

[SPEC-source-forcing.md:348-350](SPEC-source-forcing.md:348):
"relative L2 < 0.05" at q=5, shots=50000. Plausible but no estimate
backs it. Add a one-paragraph noise budget:

```
σ_per_bin ≈ 1/√(P_succ · shots / N_bins)
At P_succ ≈ 0.9, shots=50k, N_bins=32: σ_per_bin ≈ 0.027
L2 over 32 bins, normalized: ≈ σ_per_bin (i.i.d. assumption)
Plus Strang-split error in dt: ≈ O(dt²)
Combined: 0.05 is realistic; tighten to 0.03 if measurements support it
```

A wrong gate is worse than a missing one — either a number with
back-up or "TODO: calibrate after first run."

### Gauge test bound

[SPEC-source-forcing.md:336-339](SPEC-source-forcing.md:336): adding
constant `c` to V multiplies φ by `exp(−c·dt·N_steps)`. For large
`|c|·T` this overflows or underflows at the SV layer. Bound `|c|`
explicitly in the test (e.g., `|c| ≤ 1`) so the test doesn't flake
on numerics.

## What's solid (do not change)

- Core idea: heat + diagonal time-dependent potential, dense-block
  only, defer qft-diagonal. Right scope for one parcel.
- Per-step midpoint `V_n` at `t_n + dt/2`: correct second-order
  choice, doesn't oversell.
- Gauge-fix to spatial mean-zero V: correct numerical hygiene,
  correctly justified as physically inert.
- §10 future-work list: honest about what's deferred and why.
- §9 implementation order (classical helper + unit tests first, then
  plumbing): fastest path to trust.

## Action items for tomorrow

1. **Fix the sign** in [SPEC-source-forcing.md:49](SPEC-source-forcing.md:49),
   [:76](SPEC-source-forcing.md:76), [:200](SPEC-source-forcing.md:200).
   Update test #2's expected analytical form.
2. **Preserve unforced fast path** in §5.5 and §5.6 with the
   `if source_fn is None` branch above.
3. **Loosen test #1 wording** to match the build-once-when-unforced
   structure.
4. **Spell out SV-path restructure** in §5.6 instead of "same shape."
5. **Verify `permute_to_encoding` vector support**; add helper if
   needed.
6. **Clarify §3 vs §10** Strang-splitting language.
7. **Pin or sequence** the dispatch signature to avoid shots-backend
   merge conflict.
8. **Add noise budget** for test #6 acceptance, or mark as TBD.
9. **Bound `|c|`** in gauge test #3.

Items 1, 2, 3 are load-bearing. The rest are quality and should land
together but won't break the run if deferred.
