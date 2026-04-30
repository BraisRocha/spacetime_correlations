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

- **`tau_log_likelihood` is currently unused**
  Not exercised by any analysis or script at the moment. May be removed
  in a future cleanup if it stays unused.

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

- **Hour-angle bookkeeping in `cumulative_directional_exposure`**
  The boundary handling at `eta == h_star` is consistent but not documented.
  Add a short comment when revisited.

- **Log-handler / file-handle housekeeping**
  `setup_logger` now resets handlers on each call; consider also surfacing a
  `close_logger` helper for very long-running pipelines.

## Documentation

- **Module-level docstrings**
  Several files (`event_sample.py`, `flare.py`, `exposure.py`) have no
  top-level summary. Add when the API stabilises.

- **`spacetimecorr/__init__.py` docstring placement**
  Current docstring sits below `__all__` and is therefore dropped by Python.
  Move it to the top of the module on the next pass.
