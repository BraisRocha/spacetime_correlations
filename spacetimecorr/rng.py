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
        if not isinstance(seed, int):
            raise TypeError("Seed must be an integer.")
        self._seed = seed
        self._rngs: dict[str, np.random.Generator] = {}
    
    def get(self, name: str) -> np.random.Generator:
        if name not in self._rngs:
            # Turn the name into a stable uint32 using a hash
            digest = hashlib.blake2b(name.encode("utf-8"), digest_size=4).digest()
            key = int.from_bytes(digest, "little")

            child_ss = np.random.SeedSequence(self._seed, spawn_key=(key,))
            self._rngs[name] = np.random.default_rng(child_ss)

        return self._rngs[name]

    def names(self):
        return tuple(self._rngs.keys())