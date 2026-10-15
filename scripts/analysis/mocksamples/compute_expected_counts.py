"""
Expected number of events per HEALPix pixel, from all the mock samples.

Every mock sample in ``inputs/mocksamples/mocksamples/`` is read in turn,
one at a time since each holds ~1.7 million events, and its events are
counted in every HEALPix pixel of the sky at resolution ``nside``, with no
FoV restriction. Across the datasets this gives, for every pixel, a
distribution of observed counts, and its mean is taken as the expected
count of the pixel.

Output, ``output/analysis/mocksamples/expected_counts_nside{nside}.npz``:

    nside        HEALPix resolution (RING ordering)
    datasets     (D,)       dataset names, in the order of the rows below
    n_total      (D,)       number of events in each dataset
    counts       (D, Npix)  observed count per dataset and pixel
    expected_n   (Npix,)    mean of ``counts`` over the datasets
"""

from pathlib import Path

import healpy as hp
import numpy as np

from tqdm import tqdm


def main(nside: int = 32) -> None:
    """Count the events of every mock sample per pixel and save their mean."""

    # The project root, found from this file's own location
    # (scripts/analysis/mocksamples/ is three levels below it).
    project_root = Path(__file__).resolve().parents[3]
    input_dir = project_root / "inputs" / "mocksamples" / "mocksamples"
    output_path = (
        project_root / "output" / "analysis" / "mocksamples"
        / f"expected_counts_nside{nside}.npz"
    )

    # dataset_0, dataset_1, ... in numerical order
    paths = sorted(
        input_dir.glob("dataset_*.dat"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    npix = hp.nside2npix(nside)
    counts = np.zeros((len(paths), npix), dtype=np.int64)
    n_total = np.zeros(len(paths), dtype=np.int64)

    for k, path in enumerate(tqdm(paths, desc="Datasets")):
        ra, dec = np.loadtxt(path, usecols=(2, 3), unpack=True)
        pixel = hp.ang2pix(nside, ra, dec, lonlat=True)
        counts[k] = np.bincount(pixel, minlength=npix)
        n_total[k] = ra.size

    expected_n = counts.mean(axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        nside=nside,
        datasets=np.array([p.stem for p in paths]),
        n_total=n_total,
        counts=counts,
        expected_n=expected_n,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main(nside=32)
