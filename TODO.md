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

- **Marginal Lambda p-value and windows with fewer than 2 events**
  Lambda needs `n >= 2`. The marginal functions in `statistics.py` do not
  agree on what happens below that:
  - `lambda_marginal_pdf` and `lambda_marginal_sf` put windows with
    `n < 2` at a point mass `Lambda = 0`, of probability
    `P(N < 2) = exp(-mu) (1 + mu)` (written in the `pdf` Notes). The `sf`
    sum starts at `n = 2` and is not renormalised, so it returns
    `P(Lambda >= x and N >= 2)` and tends to `P(N >= 2)`, not 1, as
    `x -> 0`.
  - `lambda_marginal_rvs` discards draws with `N < 2`, so it samples Lambda
    *conditional* on `N >= 2`.

  They differ by the factor `P(N >= 2)`. At the median of `rvs`, `sf`
  gives 0.132 / 0.401 / 0.498 for `mu` = 1 / 3 / 8, instead of 0.5.

  Which one is right depends on what is done with windows where `n < 2`:
  - kept, with `Lambda = 0` and p-value 1: `sf` is right and `rvs` is not;
  - dropped (the blind-search plan): the null is conditional, so the
    p-value should be `sf / P(N >= 2)`. The current `sf` gives p-values
    too small by `P(N >= 2)` (0.80 at `mu = 3`), overstating significance.

  Negligible for `mu >~ 10`. It matters in low-exposure windows near the
  FoV edge (`mu ~ 3` in the last HEALPix ring, nside 32, `n_total = 5e5`).
  A possible fix for the dropped case is to subtract `log P(N >= 2)` in
  `lambda_marginal_logsf`. Check which scripts ran at small `mu` before
  changing it: `lambda_marginal_sf` is used by `run_null`,
  `run_compare_bg_signal`, `run_targeted_search` and the plotting scripts.

- **HEALPix windows at the FoV edge: exposure at the pixel centre**
  `HealpixWindow.expected_n_in_window` takes the exposure towards the
  pixel as `Omega_pix * eps(centre)`. Near the FoV edge
  (`dec_max = lat + theta_max = 24.85 deg` at Auger) `spatial_exposure`
  drops steeply to 0, and the centre value stops being the pixel average.
  Ratio of centre value to pixel average at nside 32: 1.004 at dec 22.0,
  1.016 at 23.3, and 1.06 at 24.6 (the last ring whose centre is inside
  the FoV). The expected count is therefore too high, which is
  conservative. On the last ring `mu ~ 3` for `n_total = 5e5`, so the
  bias is ~0.2 events against a Poisson spread of ~1.7. Kept as is for
  now. `run_blind_search_null.py` skips the pixels that straddle the FoV
  edge, i.e. that worst ring; the outermost ring it tests is off by ~1.6%.

  If it needs fixing: `spatial_exposure` depends on declination only, so
  average it over the sub-pixels of the window at a finer nside. In NEST
  ordering, the children of pixel `p` at `nside * 2^k` are
  `p * 4^k ... (p + 1) * 4^k - 1`. That matches how `EventSample.full_sky`
  draws the events.

  Pixels whose centre lies outside the FoV are dropped. The ring just past
  the edge has a mean `spatial_exposure` of ~1e-4, against ~0.3 in the
  bulk, so the events lost there are negligible.

- **Whole-sky direction sampling ignores A(t)**
  `EventSample.full_sky` thins isotropic candidate directions with
  `ExposureModel.detect_directions`, keeping each one with probability
  `spatial_exposure / max over the sky`. `spatial_exposure` is the
  closed form for a constant area: the integral of `H cos(theta)` over one
  sidereal day, a function of declination only. With `A = A(t)` the
  integrand is no longer periodic, and the exposure per solid angle over
  the run,

      dE/dOmega(tf) = ∫_{t0}^{tf} H(theta_max - theta(t')) cos(theta(t')) A(t') dt'

  (`_directional_exposure_tf`), can also depend on RA: two directions at
  the same declination are in the FoV at different sidereal hours, and see
  `A(t)` at different moments. The expected counts
  (`relative_window_exposure`) use this exact value; the full-sky samples
  do not, so simulated null samples and expected counts can disagree.

  Not needed to analyse data (mock samples): there `spatial_exposure` is
  only used as a FoV test (`> 0`), which is geometric and unaffected.

  *Plan.* Computing `dE/dOmega(tf)` exactly per candidate (~0.15-0.2 s
  each, ~2e6 candidates for 1e6 events) would take days per sample.
  Instead:
  1. First measure how much `dE/dOmega(tf)` varies with RA at fixed
     declination (a few declinations, every ~10 deg in RA). If far below
     the Poisson fluctuations of a pixel, keeping the declination-only
     form is enough and only the docstring needs updating.
  2. Otherwise, compute `dE/dOmega(tf)` once at the centres of the
     nside-32 HEALPix pixels in the FoV (~10,500, ~25-35 min on one core,
     independent pixels so it parallelises) and save it to a file
     (~100 KB) with what it was computed for: `t0`, `tf`, `theta_max`,
     nside and the A(t) file. Check these on reading.
  3. Store the ratio of that value to the declination-only closed form.
     The steep fall at the FoV edge depends on declination only and is
     exact in the closed form; the ratio is smooth, so it interpolates
     well. For each candidate: closed form at its own declination times
     the ratio interpolated with `healpy.get_interp_val` (bilinear, 4
     nearest pixel centres). Check the interpolation error against the
     exact integral at directions off the grid.
  4. The same file holds the values the expected counts need at nside 32
     (`relative_window_exposure` at the pixel centres), so reading them
     from it saves one of the two passes over the A(t) steps per pixel in
     the blind search (~half its run time). Only useful if all datasets
     share a common `[t0, tf]`: `run_blind_search_mock.py` currently takes
     `t0`, `tf` from each sample.

  Also update the `spatial_exposure` docstring: it is the constant-area
  (geometric) profile, and its normalisation is no longer that of
  `norm_cumul_exposure` when `A` varies.

- **Mock events in minutes with no active cells (`dataset_stcwg_vert`)**
  `dataset_0` (and `dataset_1`, which shares its arrival times to the
  second) has 5 events, all in 2009, in minutes where the hexagon files
  give 0 cells of every type (6T5, 5T5, 4T5), outside any bad period, and
  surrounded by minutes with hundreds of cells (e.g. 2009-08-02 14:48 UTC).
  Our A(t) is 0 there, so the model gives them zero detection probability
  and a flat cumulative exposure. The production apparently did not treat
  those minutes as A = 0 (another version of the files, or a coarser time
  resolution?). Ask whoever ran it. Harmless for now (5 in ~2e5 events; it
  only breaks Lambda if two fall in the same such minute and pixel).

- **Efficiency `P` in the detection probability but not in the exposure**
  `ExposureModel.detection_probability` (and `detect_times`, which thins
  events with it) can take an efficiency `P(t)`, but the cumulative
  exposure (`norm_cumul_exposure`, `cumul_exposure`,
  `relative_window_exposure`) always assumes `P = 1`. Events thinned with
  a `P < 1` then carry exposures computed as if `P = 1`. No analysis uses a
  `P` other than 1 yet, so it has no effect now. If one is used, `P` has to
  enter the exposure integral too: piecewise constant in time it fits the
  step integral as `A(t)` does; if it depends on the direction (`theta`) it
  breaks the closed form.

## Ideas for the future

- **Particle-nature weighting in the estimator**
  The estimator is currently purely statistical. Incorporating the probability
  that a cosmic ray is a neutral particle (e.g. a photon) could increase
  sensitivity. The Polish group is reportedly already working on photon
  probability weights in their method. Extending our estimator in this
  direction is a natural next step once the current pipeline is stable.

## Current homework

- **Blind search with flares: decisions before injecting signal**
  `run_blind_search_null.py` scans a background-only full-sky sample over a
  `HealpixGrid`. Adding flares needs these decisions first:
  1. *Positions.* The old script took catalogue sources. For random
     positions: uniform in solid angle over the FoV band, or weighted by
     exposure? How many flares, and can they overlap?
  2. *Multiplicity.* The old `mu_flare = S/N * sqrt(expected_n)` used the
     bare sky fraction of a 1.05 deg cap, the same for every flare. With
     exposure, `expected_n` of the pixel holding the flare ranges from ~3
     to ~125 (nside 32, `n_total = 5e5`). Fix the S/N (so `n_flare` varies
     with position) or fix `n_flare`?
  3. *Detection thinning.* `Flare._accumulate_events` thins every candidate
     with the detection probability at the flare *centre*. Events up to a
     few sigma away are kept even when they arrive from outside the
     instantaneous FoV: 7-10% of events for 1 h flares, 0.4-7% for 1-30 d
     flares (sigma = 1 deg; the larger values near the FoV edge). The
     full-sky background uses each event's own direction. Thin flare
     events at their own direction in `generate`?
  4. *`no_overdensity` on a full-sky sample.* `inject_flare` removes its
     background events at random from the *whole sky*
     (event_sample.py:906), so `n_total` is kept but the pixel holding the
     flare gains ~`n_flare` events. In the blind search a flare is then a
     spatial *and* temporal excess, which the Poisson test also sees. A
     temporal-only signal per pixel (the meaning of `no_overdensity` in the
     targeted search) would need the removal to be local to the flare.
  5. *Flare exposure.* Flare events carry the exposure evaluated at the
     flare centre at generation time, while the background of a pixel is
     sampled for the pixel centre. `EventSample` does not store event
     times, so it cannot be re-evaluated per window. Expected to be small
     (all events of one flare shift by about the same amount); accept, or
     store times?

  Also: a flare at a random position is split between neighbouring pixels.
  At nside 32 the median S/N is ~0.81 of that of a 1.05 deg cap centred on
  the flare (worst 10%: ~0.67).

- **Effect of T_obs on signal significance**
  Produce a plot analogous to Fig. 1 of the paper comparing two scenarios:
  `(T_obs = 10 yr, flare duration = 1 day)` vs.
  `(T_obs = 1 yr,  flare duration = 1 day)`.
  As the signal fraction increases the significance should grow, but results
  must be penalised for the number of tested intervals: if `T_obs` is divided
  into 10 windows, p-values should be multiplied by 10.



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
- `run_blind_search_null.py` catches only `ValueError` and records NaN for
  the window. It tells the two cases apart by the window's event count:
  `n < 2` is counted as a routine drop, while `n >= 2` is logged with its
  traceback and counted separately as a failure (a bug). `RuntimeError`
  is not caught.

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
