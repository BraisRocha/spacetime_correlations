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