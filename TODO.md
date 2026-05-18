# TODO / Deferred items

Items identified during code reviews that we have consciously chosen not to
fix right now, but which should be revisited.

## Physics / modelling

- **`sample_directional_exposure` ("free maximum exposure" sampling)**
  `ExposureModel.sample_directional_exposure` oversamples uniformly on
  `[0, factor * max_exposure / Γ]` and keeps the first `n_events` after
  sorting. Intended to avoid a bias near `max_exposure`. Worth revisiting
  to confirm the resulting distribution matches what we want analytically.
  Perhaps a test script could be written in order to have a tool to check
  it at any moment.

## Code quality

- **Statistical-validation tests** (not yet in `tests/`)
  The deterministic / contract part of the test suite is in place under
  `tests/` (see `test_skywindow.py`, `test_event_sample.py`,
  `test_flare.py`, `test_exposure.py`, `test_statistics.py`,
  `test_observatory.py`).  Three statistical validations remain to be
  added as proper pass/fail tests, with seeded RNG + a numeric tolerance.
  Visual versions of (some of) these already exist as Monte-Carlo
  diagnostics; the pytest versions will assert on a statistic rather than
  produce a plot.

  - **Per-window isotropy validation** — with `EventSample.in_window`
    generating background-only events, verify that the realised
    p-value distribution is uniform on `[0, 1]` (KS test against
    `Uniform(0, 1)`).  Also resolve whether a Poisson or a binomial
    draw is more appropriate given the absence of sampling bias.
  - **Exposure-weighted end-to-end check** — now that
    `SkyWindow.expected_n_in_window` folds in `omega(delta_centre)` and
    `EventSample.in_window` threads `exposure_model` through, verify
    that the realised event count vs. declination tracks the expected
    count within Poisson tolerance.
  - **Effect of the significance-function cut** — add a test that
    quantifies whether the cut in the significance function affects
    the result.

- **Log-handler / file-handle housekeeping**
  `setup_logger` now resets handlers on each call; consider also surfacing a
  `close_logger` helper for very long-running pipelines.

## Documentation

- **Module-level docstrings**
  Several files (`event_sample.py`, `flare.py`, `exposure.py`) have no
  top-level summary. Add when the API stabilises.

## Open API decisions (pending review)

- **`sample_directional_exposure` — asymmetric input handling**
  Currently in `ExposureModel.sample_directional_exposure`:
    * `expected_exposure_rate <= 0`  → raises `ValueError` (setup error).
    * `max_dir_exposure <= 0`        → silently returns an empty array
      (physically meaningful "always-invisible" regime, `A + B <= 0`).
    * `n_events == 0` / `mu_expanded == 0` → silently returns an empty array.
  The current behaviour kept the asymmetry on the grounds that the silent
  branches encode genuine physical edge cases while the raising branch flags
  caller bugs. Decision to revisit: keep this (status quo) or unify the
  three branches (e.g. all raise, or all return empty).

- **`EventSample.has_exposure` semantics with partly-NaN arrays**
  Right after `inject_flare()` and before a follow-up
  `assign_directional_exposure()`, the exposure array is allocated but the
  non-flare slots are still `NaN`. Two interpretations:
    * **(a)** Tighten `has_exposure` to require all-finite values
      (`np.all(np.isfinite(self.exposure))`), i.e. "fully populated".
    * **(b)** Leave `has_exposure` as the structural "array allocated?"
      check (current behaviour) and rely on the NaN guard in
      `lambda_estimator` to catch partly-filled arrays at the use site.
  Currently using (b). Decision to revisit: keep (b), or switch to (a) if
  we ever want `has_exposure` to mean "ready for analysis".


## Ideas for the future

- **Particle-nature weighting in the estimator**
  The estimator is currently purely statistical. Incorporating the probability
  that a cosmic ray is a neutral particle (e.g. a photon) could increase
  sensitivity. The Polish group is reportedly already working on photon
  probability weights in their method. Extending our estimator in this
  direction is a natural next step once the current pipeline is stable.

- **Time-dependent effective area A(t)**
  The directional exposure model should eventually account for effects that
  modulate the detector's effective area over time: bad periods, tanks going
  offline, planned extensions of the array, etc. These can all be parametrised
  as a time-dependent factor `A(t)` multiplying the geometric acceptance, and
  incorporated into the temporal sampling naturally.

## Current homework

- **Effect of T_obs on signal significance**
  Produce a plot analogous to Fig. 1 of the paper comparing two scenarios:
  `(T_obs = 10 yr, flare duration = 1 day)` vs.
  `(T_obs = 1 yr,  flare duration = 1 day)`.
  As the signal fraction increases the significance should grow, but results
  must be penalised for the number of tested intervals: if `T_obs` is divided
  into 10 windows, p-values should be multiplied by 10.

- I want Claude to review thoroughly the Flare class. It would be better to
  break it into FlareModel and FlareSample classes? Are there inconsistencies
  among the class' script?

- **`run_scan_correlation.py` / `plot_scan_correlation.py` not migrated**
  The other four `scripts/montecarlo/` scripts and their corresponding
  `scripts/plots/` plotting scripts have been migrated to the new
  per-window pipeline (`EventSample.in_window`, exposure-weighted
  `expected_n`, renamed keys `n_events_* → n_sample_*`). The
  scan-correlation pair was deliberately left untouched: the old
  pipeline's three-case structure (`bkg / ST / T / S`) does not survive
  the new pipeline as-is — `inject_flare` on an in-window sample always
  *replaces* events, so the old `ST` (inject-into-parent → re-window,
  which *adds* events) and old `T` (inject-into-subsample, which
  replaces them) collapse to the same operation. A pipeline-level
  redesign is needed before migrating this pair.

- **Pre-migration MC `.npz` outputs are not readable by the new plot scripts**
  The renamed keys (`n_events_window → n_sample_window`,
  `n_events_bkg → n_sample_bkg`, `n_events_flare → n_sample_flare`,
  metadata `mu_window → expected_n`, etc.) mean the migrated plot
  scripts cannot consume runs produced before this migration. To view
  old results, either re-run the MC scripts or write a one-off
  conversion that copies the old keys to the new names.
