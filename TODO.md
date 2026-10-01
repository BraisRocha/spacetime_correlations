# TODO / Deferred items

Items identified during code reviews that we have consciously chosen not to
fix right now, but which should be revisited.

## Physics / modelling

- **FoV-boundary handling needs a package-wide review**
  Near the edge of the field of view, flare generation (e.g.
  `Flare.generate` / `Flare.generate_in_window`, which sample a Gaussian
  cluster around the centre) can place events *outside* the FoV — a flare
  centred close to the visible-sky boundary may scatter events into
  directions the observatory never sees. This is not currently guarded.
  More broadly, the whole package needs a careful pass to understand and
  make consistent how the FoV boundaries are handled (event generation,
  acceptance/exposure evaluation, window containment, and any implicit
  visible-declination assumptions). Decide on the intended behaviour at the
  edge and enforce it uniformly.

- **Declination-only exposure ignores the leftover partial sidereal day**
  The closed form for the exposure per solid angle over the run assumes
  `T_obs` is a whole number of sidereal days. With the real `t0`/`tf` there
  is a leftover partial day, during which directions whose RA crosses the
  meridian collect extra exposure, so the exact result depends weakly on RA.
  `max_norm_cumul_exposure` (per direction) keeps this term; the
  declination-only form used for whole-sky sampling drops it. The size is
  at most `1 sidereal day / T_obs`: ~3e-4 for 10 yr, but ~3% for the 1-month
  run in `sampling_diagnostic.py`. Revisit if short runs matter.

## Ideas for the future

- **Particle-nature weighting in the estimator**
  The estimator is currently purely statistical. Incorporating the probability
  that a cosmic ray is a neutral particle (e.g. a photon) could increase
  sensitivity. The Polish group is reportedly already working on photon
  probability weights in their method. Extending our estimator in this
  direction is a natural next step once the current pipeline is stable.

- **Time-dependent effective area A(t)**
  Model bad periods, tanks offline and array growth as a factor
  `A(t)/A_max` on the acceptance.

  *Done:* per-event thinning. `detection_probability` takes an optional
  `area` callable; `detect_times` forwards it. No real `A(t)` supplied
  yet, so the factor is 1.

  *Left:* the cumulative exposure. `cumulative_integral` is closed-form
  only because the integrand is sinusoidal in the hour angle; `A(t)` is a
  function of *time*, so it breaks the primitive. If `A(t)` is piecewise
  constant (the realistic case) it stays analytic — split `[t0, t]` at the
  steps and weight each segment by `A_k/A_max`. If it is smooth, it needs
  numerical quadrature. Same applies to `P` if it depends on `theta`.

  *Whole-sky spatial sampling:* the exposure per solid angle over the run
  splits into `(1/T_obs) ∫ A(t)/A_max dt` times a function of declination
  only, **provided `A(t)` varies slowly compared to a sidereal day**. The
  area factor is then a single constant that cancels in the declination
  thinning and in `relative_window_exposure`. If `A(t)` ever has structure
  on sidereal-day timescales, this no longer splits: the exposure per solid
  angle must be integrated numerically and picks up an RA dependence.

## Current homework

- **Update `run_blind_search.py` to the new whole-sky pipeline**
  `EventSample.full_sky` now requires an `exposure_model` (signature
  `full_sky(n_total, exposure_model, t0, tf, rng)`), so the script fails
  at its `full_sky` call until it is updated. The script is unfinished and
  has never been used; when it is picked up again, pass the exposure model
  it already builds and check the rest of it against the new code.

- **Effect of T_obs on signal significance**
  Produce a plot analogous to Fig. 1 of the paper comparing two scenarios:
  `(T_obs = 10 yr, flare duration = 1 day)` vs.
  `(T_obs = 1 yr,  flare duration = 1 day)`.
  As the signal fraction increases the significance should grow, but results
  must be penalised for the number of tested intervals: if `T_obs` is divided
  into 10 windows, p-values should be multiplied by 10.



## THINGS OF MY OWN
- **Why doesn't `_subset()` change `expected_n`? — answered, follow-up open**
  Answered: it *is* done, but separately, in `EventSample.select_subsample()`.
  What remains open is why it has to be separate — couldn't those lines live
  in `_subset()` itself?

## Found while reviewing the new inject_flare bkg-removal (2026-09-11)

- **A stale test describing the old removal formula**
  It still passes, but no longer tests what the code does:
  - `test_inject_flare_requires_expected_n`
    (tests/test_event_sample.py:439) asserts that a sample without
    `expected_n` raises. The new code never reads `expected_n` in
    overdensity mode — it requires `window`. The test only passes because
    the bare-constructor sample happens to have *both* unset. Rename to
    `test_inject_flare_overdensity_requires_window` and assert on the
    window, otherwise a future regression here goes undetected.
  Separately: injecting into a `full_sky` sample in overdensity mode (the
  `AttributeError` that motivated the guard) has no test at all. Worth
  adding one.

- **Minor: the `min(n_removed, n_bkg)` clip biases the removal low**
  In overdensity mode `n_removed` is clipped at the number of available
  background events. When the flare is large relative to the in-window
  background, the realised mean removal sits below `mu_removed`. Harmless in
  the small-window / large-`n_total` regime we work in, and already
  documented in the docstring Notes, but it is a real (small) bias if the
  regime ever changes.

## `lambda_estimator` duplicate-exposure crash in grid_p50 (fixed 2026-09-22, one loose end)

A job of the 800-cell `grid_p50` run died on the guard at statistics.py:382,
`duplicate (or non-increasing) exposure values`. Two flare events had landed
on *exactly* the same exposure, so their spacing was zero.

### Cause

`Flare._accumulate_events` rebuilt its times through `.jd`, collapsing
astropy's two-number time representation into a single float64. A Julian Date
is ~2.46e6, and a float64 only carries ~16 digits, so once the date part is
stored there are only ~9 digits left for the fraction of a day — a resolution
of 40 us. Every flare time was snapped onto that grid. Flare exposure is a
function of time alone, so two events in the same 40 us bin got identical
exposure. A flare packs many events into a short window, hence the risk went
as intensity^2 / duration and the short-duration, high-intensity cells were
the ones that died. (Background exposure is sampled directly in exposure
space, never from times, so it was never at risk.)

### Fix — both parts were needed

- **Times** (flare.py:475-482): concatenate `jd1` and `jd2` separately instead
  of going through `.jd`. Resolution 40 us -> ~5 ps. Measured on 12 flares of
  3000 events at duration 0.001 d: **19 ties -> 0**.
- **Estimator** (statistics.py:390): use `-np.expm1(-x)` instead of
  `1 - np.exp(-x)`. This was *required*, not optional: restoring the time
  precision pushes typical spacings below `x = 2**-53`, where `1 - exp(-x)`
  underflows to exactly 0.0 and Lambda becomes `+inf` — silently, because the
  duplicate guard only catches `delta <= 0`. Fixing the times alone would
  have turned a loud `ValueError` into a wrong `p_value = 0`. Verified:
  agrees with the old form to 6.5e-15 over 200 realizations, and returns
  39.41 instead of `inf` at x = 7.7e-18.

Note earlier runs are no longer bit-reproducible against the pre-fix code —
worth recording in the run metadata.

### Loose end

The tie-rate model predicted ~0.10 failing cells across the 800-cell grid and
we saw 1. Plausible as a fluctuation, but the dead job was never identified.
If it died at *long* duration the 40 us mechanism does not explain it and
there is a second cause still unfound. Recover its `(duration, intensity)`
from the Condor logs to close this properly.

## Failure handling: "too few events" is not distinguished from real errors (2026-09-30)

"Too few events in the window" and genuine bugs both raise `ValueError`,
and the two MC scripts handle that differently. Problem only; no solution
agreed yet.

### The raise sites

- `EventSample.in_window` (event_sample.py:255): `ValueError` if the
  Poisson draw gives `n_sample == 0`. `n_sample == 1` passes through.
- `lambda_estimator` (statistics.py:354-387): `RuntimeError` if exposure is
  not set; `ValueError` for `n_sample < 2`, for non-finite exposures
  (forgotten `assign_cumul_exposure`), and for duplicate exposures
  (quantisation bug, see previous section).

So `ValueError` mixes one *normal* outcome (too few events) with two
*pathologies* (a caller mistake, a numerics bug).

### How the scripts react

- `run_grid_p50.py:264` catches only `RuntimeError` and then `continue`s,
  i.e. discards the realization and draws a new one until `n_simulations`
  successes. Any `ValueError` escapes and kills the whole Condor job.
- `run_blind_search.py:320` catches `(ValueError, RuntimeError)`, logs it,
  and records NaN for the window. Bugs and routine skips look identical.

### Why "too few events" is not rare

1.05 deg window, `n_total = 1e5`, Auger site:

| dec  | expected_n | P(n < 2) |
|------|-----------:|---------:|
| -35  | 15.4       | 3e-6     |
| 0    | 9.2        | 1e-3     |
| +15  | 5.1        | 4e-2     |
| +24  | 1.3        | 0.61     |
| >=28 | 0 (outside FoV) | 1   |

Negligible at the zenith, but the normal outcome across a large part of
the sky in a blind search.

### Why simply widening grid_p50's `except` is not neutral

Its loop throws away failed realizations and draws new ones, so p50 becomes
conditional on the realization being computable. For the duplicate-exposure
failures those were the high-Lambda realizations, so p50 would be biased low
with no sign of it. For too-few-events at high dec the conditioning changes
what p50 means. The narrow `except` currently makes these failures loud,
which is accidentally the safer behaviour.
