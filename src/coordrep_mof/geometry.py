"""Periodic crystal geometry validation and frozen element sets for MOF coordination priors."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

# Frozen broad metal definition, including post-transition metals, lanthanides, and actinides.
# Metalloids B/Si/Ge/As/Sb/Te and halogens are excluded; Po is included explicitly.
METALS = frozenset({
    3, 4, 11, 12, 13,
    *range(19, 32),
    *range(37, 51),
    *range(55, 85),
    *range(87, 117),
})

# Donor element histogram bins: O, N, S, P, C, F, Cl; remaining elements map to Other (bin 7).
DONOR_Z = (8, 7, 16, 15, 6, 9, 17)


def _numpy(value: Any, dtype: Any = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError("complex-valued geometry/features are not supported")
    return np.asarray(array, dtype=dtype)


def validate_geometry(z: Any, frac: Any, cell: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a single fully occupied periodic crystal structure, with row lattice vectors."""
    raw_z = _numpy(z)
    if raw_z.ndim != 1 or len(raw_z) == 0:
        raise ValueError("z must be a nonempty one-dimensional atomic-number array")
    if not np.issubdtype(raw_z.dtype, np.number) or np.iscomplexobj(raw_z):
        raise ValueError("atomic numbers must be real integers")
    if not np.isfinite(raw_z).all() or not np.equal(raw_z, np.floor(raw_z)).all():
        raise ValueError("atomic numbers must be finite integers")
    numbers = raw_z.astype(np.int64)
    if np.any((numbers < 1) | (numbers > 118)):
        raise ValueError("atomic numbers must lie in [1, 118]")
    coordinates = _numpy(frac, np.float64)
    lattice = _numpy(cell, np.float64)
    if coordinates.shape != (len(numbers), 3) or lattice.shape != (3, 3):
        raise ValueError("expected frac[N,3] and cell[3,3]")
    if not np.isfinite(coordinates).all() or not np.isfinite(lattice).all():
        raise ValueError("geometry must contain only finite numbers")
    singular_values = np.linalg.svd(lattice, compute_uv=False)
    if singular_values[-1] <= 1e-10 or singular_values[0] / singular_values[-1] > 1e10:
        raise ValueError("cell is singular or too ill-conditioned for safe image enumeration")
    return numbers, coordinates, lattice
