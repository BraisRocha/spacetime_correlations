"""
Reproducible, independent random-number streams by name.

Defines :class:`RNGManager`, the single entry point used across the
package to obtain ``numpy.random.Generator`` instances. Each logical
consumer (events, flare, exposure, …) requests a generator by name and
gets a deterministic child stream derived from
``(master_seed, blake2b(name))``.

Two properties matter for Monte-Carlo reproducibility:

- The same ``(seed, name)`` pair always yields the same stream, so a
  run is fully determined by the master seed.
- Streams for different names are independent of each other and of the
  order in which they are first requested, so adding or reordering
  consumers does not perturb existing streams.
"""

import numpy as np
import hashlib

class RNGManager:
    """
    Manage reproducible, independent RNG streams using named generators.

    Each name corresponds to a deterministic SeedSequence derived from:
      (master_seed, name)

    Calling get(name) multiple times returns the same Generator instance.
    Independent of call order across modules/files.
    """

    def __init__(self, seed: int = 42):
        """
        Parameters
        ----------
        seed : int, optional
            Master seed used to derive every named child stream. The same
            ``(seed, name)`` pair always yields the same stream.
        """
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise TypeError("Seed must be an integer.")
        self._seed = int(seed)
        self._rngs: dict[str, np.random.Generator] = {}

    def get(self, name: str) -> np.random.Generator:
        """
        Return the ``Generator`` associated with ``name``, creating it on
        first use.

        The child stream is derived from a ``SeedSequence`` whose spawn key
        is a stable hash of ``name``. Successive calls with the same
        ``name`` therefore return the same instance, regardless of the
        order in which different names are requested.

        Parameters
        ----------
        name : str
            Logical name of the stream (e.g. ``"events"``, ``"flare"``).

        Returns
        -------
        numpy.random.Generator
            Cached generator for the requested name.
        """
        if name not in self._rngs:
            # Turn the name into a stable uint32 using a hash
            digest = hashlib.blake2b(name.encode("utf-8"), digest_size=4).digest()
            key = int.from_bytes(digest, "little")

            child_ss = np.random.SeedSequence(self._seed, spawn_key=(key,))
            self._rngs[name] = np.random.default_rng(child_ss)

        return self._rngs[name]

    def names(self) -> tuple[str, ...]:
        """Return the tuple of stream names that have been instantiated so far."""
        return tuple(self._rngs.keys())