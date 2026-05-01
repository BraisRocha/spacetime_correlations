# TODO / Deferred items

Items identified during code reviews that we have consciously chosen not to
fix right now, but which should be revisited.

## Physics / modelling

- **Zenith-angle cut in the directional exposure model**
  `ExposureModel.instantaneous_acceptance` (`spacetimecorr/exposure.py`) currently
  uses `a(t) = max(0, cos θ)` with no upper-zenith cut. Real Auger SD/HE
  acceptance is capped (e.g. 60° or 80°). To revisit when we want a more
  realistic acceptance.

- **`expected_n_in_window` assumes uniform sky**
  `EventSample.select_subsample` propagates `expected_n` based on a uniform
  sky-fraction (`SkyWindow.expected_n_in_window`). This ignores the
  declination dependence of Auger's directional exposure. Plan to replace
  with an exposure-weighted estimate in the near future.

- **`sample_directional_exposure` ("free maximum exposure" sampling)**
  `ExposureModel.sample_directional_exposure` oversamples uniformly on
  `[0, factor * max_exposure / Γ]` and keeps the first `n_events` after
  sorting. Intended to avoid a bias near `max_exposure`. Worth revisiting
  to confirm the resulting distribution matches what we want analytically.

## Code quality

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
