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

## Code quality

- **Poisson vs. binomial draw in `EventSample.in_window`** — open
  question, separated from the now-resolved statistical-validation tests.
  The per-window sampler currently uses a Poisson draw with mean
  ``expected_n_in_window``.  Whether a binomial draw on the parent
  full-sky population is more appropriate given the absence of sampling
  bias is still to be settled.

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



## THINGS OF MY OWN
- **Why EventSample._subset() does not change expected_n?**
  I need to understand this. If you have a full_sky sample then the expected_n
  is the total number of events. However, when you make an in_window sample,
  you should compute the expected number of events for that specific region
  based on the covered fraction and the exposure model. It is also true 
  that for the full_sky pipeline no ExposureModel is given. This could be
  the reason behind as you need it to obtain the expected_n. This is worth 
  revisiting in case there is another reason why this is done this way.

- **EventSample.full_sky() pipeline needs ExposureModel implementation**
  Connected to the previous point, currently no exposure model is implemented in
  this pipeline. This generates a sample that doesn't follow Auger's spatial
  exposure. For the paper we are still only using the in_window() pipeline
  as is the one used for the targeted search. It is nevertheless crucial to 
  solve this problem as soon as possible.

- **OKAY, First point solved** 
  I see that the first point is solved in EvenSample.select_subsample(). 
  My question now is why this has to be done separately from the rest.
  Wouldn't be possible to include these lines in _subset()?

- **event_sample.py line 831: POTENTIAL FATAL BUG**
  In the method EventSample.inject_flare() could there be a fatal bug
  related to the injection of flares in windowed samples.
  When a flare is injected in these cases, a number of background events have to be removed to maintaing n_total constant. As we only 
  sample events inside the window, the number of removed events is 
  computed as a Poisson realization with 

  expected_number_of_removed_events = probability_of_an_events_lying_inside_the_window * n_flare

  The problem is hidden inside this probability. Currently, we are only
  computing it as 

  p = expected_n_in_window / n_total, but we are not taking into account the exposure of the region which accounts for the fact that there are regions with more density of events than others. I have to 
  check this with Miguel to make sure this is correct.

- **Deep check to exposure.py**
  I deep check to this module is necessary in order to see that all the expessions being used are correct. One concern I have right now is whether I am using the proper expresions as I dont know if at the end Im getting the correct units. For instance, the area of the observatory isnt being included anywhere and Im finding hard to believe that this is right as to get the exposure you need to add the area as a multiplicative factor of \omega(dec, ra).

  -**Very Important Change in the ExposureModel**
Change the units, take into account observatory's area (km²), target's surface (sr) and time (yr)