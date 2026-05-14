# TODO / Deferred items

Items identified during code reviews that we have consciously chosen not to
fix right now, but which should be revisited.

## Physics / modelling

- **`sample_directional_exposure` ("free maximum exposure" sampling)**
  `ExposureModel.sample_directional_exposure` oversamples uniformly on
  `[0, factor * max_exposure / Γ]` and keeps the first `n_events` after
  sorting. Intended to avoid a bias near `max_exposure`. Worth revisiting
  to confirm the resulting distribution matches what we want analytically.

## Code quality

- **`EventSample` dual-pipeline state**
  The class currently contains two generations of methods that coexist but
  serve different pipelines and have not yet been reconciled:

  *Old full-sky pipeline* — generate a large isotropic sample over the whole
  sky, then carve out a window:
  - `assign_coordinates()` (full-sky isotropic sampling)
  - `select_subsample(window)` — filters events and sets `expected_n` using
    the uniform `SkyWindow.expected_n_in_window`. Note that
    `SkyWindow.expected_n_in_window` now supports an optional `exposure_model`
    argument for exposure-weighted counts, but `select_subsample` does not yet
    thread it through.
  - `generate_directional_exposure` / `assign_directional_exposure` — exposure
    sampling machinery tied to the full-sky subsample workflow.

  *New per-window pipeline* — sample directly within a window with a
  Poisson-drawn event count:
  - `assign_coordinates_in_window(window)` (spherical-cap uniform sampling)

  The old methods have been kept deliberately while the new pipeline matures.
  Once the per-window workflow is validated end-to-end, decide which old
  methods to adapt, deprecate, or remove.

- **Tests**
  No `tests/` directory yet. A small pytest suite covering at least:
  - `cumulative_directional_exposure` in the always-visible / always-invisible
    / partial-visibility branches,
  - the `lambda_estimator`, `spatial_estimator`, and the marginal/conditional
    Lambda PDFs,
  - the `EventSample` -> `Flare` -> `inject_flare` chain (round-trip masks /
    counts).
  This would make it much easier to catch silent statistical bugs.

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

- **Per-window event generation and isotropy validation**
  Migrate the simulation pipeline to draw `n ~ Poisson(expected_n_in_window)`
  and sample events directly within the window, rather than generating a
  full-sky sample and carving out a subset. Once implemented, verify that
  isotropy is correctly reproduced by checking that background p-values are
  uniformly distributed. It is also worth investigating whether a Poisson or
  a binomial draw is more appropriate given the absence of sampling bias.

- **Exposure-weighted pipeline**
  The per-window pipeline should weight the expected event count by the
  relative exposure `omega(delta)` at the window centre, as implemented in
  `SkyWindow.expected_n_in_window`. The solid-angle factor is already in
  place; the remaining step is to fold in the declination-dependent exposure
  and verify the end-to-end result. Time-dependent effects on the acceptance
  will be absorbed into the temporal sampling via the cumulative directional
  exposure.

- **Effect of T_obs on signal significance**
  Produce a plot analogous to Fig. 1 of the paper comparing two scenarios:
  `(T_obs = 10 yr, flare duration = 1 day)` vs.
  `(T_obs = 1 yr,  flare duration = 1 day)`.
  As the signal fraction increases the significance should grow, but results
  must be penalised for the number of tested intervals: if `T_obs` is divided
  into 10 windows, p-values should be multiplied by 10.
