"""Coordination method v1: periodic covalent-radius contact prior + exact CN=2..6 Procrustes shape scores.

Method ID: ``periodic_covalent_radius_direction_procrustes_v1``.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from itertools import permutations
import json
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
import numpy as np
import torch

from .geometry import DONOR_Z, METALS, validate_geometry

ROOT = Path(__file__).resolve().parents[2]
METHOD_CONFIG_V1 = ROOT / "configs/coordination_method_v1.json"

TEMPLATE_BY_CN: dict[int, tuple[str, ...]] = {
    2: ("L",),
    3: ("TP",),
    4: ("Td", "SP"),
    5: ("TBP", "SPY"),
    6: ("Oh", "TPr"),
}
TEMPLATES = ("L", "TP", "Td", "SP", "TBP", "SPY", "Oh", "TPr")


def _templates() -> dict[str, np.ndarray]:
    a = np.sqrt(3.0) / 2.0
    raw = {
        "L": [[0, 0, 1], [0, 0, -1]],
        "TP": [[1, 0, 0], [-0.5, a, 0], [-0.5, -a, 0]],
        "Td": [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        "SP": [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        "TBP": [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-0.5, a, 0], [-0.5, -a, 0]],
        "SPY": [[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        "Oh": [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        "TPr": [[1, 0, 0.5], [-0.5, a, 0.5], [-0.5, -a, 0.5],
                [1, 0, -0.5], [-0.5, a, -0.5], [-0.5, -a, -0.5]],
    }
    result = {}
    for name, points in raw.items():
        array = np.asarray(points, dtype=np.float64)
        array /= np.linalg.norm(array, axis=1, keepdims=True)
        array.setflags(write=False)
        result[name] = array
    return result


TEMPLATE_DIRECTIONS = _templates()


@lru_cache(maxsize=8)
def _permuted_templates(name: str) -> np.ndarray:
    template = TEMPLATE_DIRECTIONS[name]
    indices = np.asarray(list(permutations(range(len(template)))), dtype=np.int64)
    result = template[indices]
    result.setflags(write=False)
    return result


def shape_scores(vectors: Any) -> dict[str, float]:
    """Exact all-permutation SO(3)+scale fit of unit donor directions for CN in {2..6}.

    For each template T and permutation p, maximize c = <U, T_p R> over R in SO(3).
    The optimal nonnegative scale is max(c, 0) / ||T||^2, giving:
        S(U, T) = 100 * (1 - max(c, 0)^2 / (||U||^2 ||T||^2)).
    Never subtracts the donor centroid (metal center remains the fixed origin).
    """
    value = np.asarray(vectors, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("shape vectors must be finite [CN,3] Cartesian displacements")
    lengths = np.linalg.norm(value, axis=1)
    if np.any(lengths <= 1e-10) or not np.isfinite(lengths).all():
        raise ValueError("shape vectors require finite positive donor distances")
    cn = len(value)
    if cn not in TEMPLATE_BY_CN:
        return {}
    directions = value / lengths[:, None]
    norm_u = float(np.sum(directions * directions))
    result = {}
    for name in TEMPLATE_BY_CN[cn]:
        template = TEMPLATE_DIRECTIONS[name]
        covariance = np.einsum("pni,nj->pij", _permuted_templates(name), directions)
        left, singular, right = np.linalg.svd(covariance)
        signs = np.where(np.linalg.det(left @ right) < 0, -1.0, 1.0)
        correlation = singular[:, :2].sum(axis=1) + signs * singular[:, 2]
        best = max(float(correlation.max()), 0.0)
        denominator = norm_u * float(np.sum(template * template))
        score = 100.0 * (1.0 - best * best / denominator)
        if not np.isfinite(score) or score < -1e-9 or score > 100.0 + 1e-9:
            raise ArithmeticError("invalid Procrustes shape score")
        result[name] = float(np.clip(score, 0.0, 100.0))
    return result


def _index(value: Any, n: int, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer site index")
    value = int(value)
    if value < 0 or value >= n:
        raise ValueError(f"{field} lies outside the atom table")
    return value


def encode_coordination(z: Any, frac: Any, cell: Any,
                        centers: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode schema cgc_coord_v1 slots [0:36] and four external quality masks."""
    numbers, coordinates, lattice = validate_geometry(z, frac, cell)
    n = len(numbers)
    output = np.zeros((n, 36), dtype=np.float64)
    quality = np.zeros((n, 4), dtype=np.float64)
    quality[:, 0] = [int(int(v) in METALS) for v in numbers]
    if not isinstance(centers, (list, tuple)):
        raise ValueError("centers must be a list of explicit center records")
    seen_centers = set()
    for record in centers:
        if not isinstance(record, dict) or "center" not in record:
            raise ValueError("each center record must contain center")
        i = _index(record["center"], n, "center")
        if i in seen_centers:
            raise ValueError(f"duplicate center record for site {i}")
        seen_centers.add(i)
        if int(numbers[i]) not in METALS:
            raise ValueError("coordination center must be in the frozen metal set")
        coord_valid = record.get("coord_valid", True)
        if not isinstance(coord_valid, (bool, np.bool_)):
            raise ValueError("coord_valid must be boolean")
        if not coord_valid:
            if record.get("donors") or record.get("shape_scores"):
                raise ValueError("failed coordination cannot also supply donor or shape data")
            continue
        if "donors" not in record or not isinstance(record["donors"], (list, tuple)):
            raise ValueError("valid center record requires an explicit donor list")
        donors = record["donors"]
        cn = len(donors)
        quality[i, 1] = 1
        if cn >= 2:
            output[i, min(cn, 10) - 2] = 1
        output[i, 9] = min(cn, 12) / 12
        output[i, 10] = float(cn <= 1)
        vectors = []
        donor_elements = []
        seen_donors = set()
        for donor in donors:
            if not isinstance(donor, dict) or "site" not in donor or "image" not in donor:
                raise ValueError("each donor needs site and integer image")
            j = _index(donor["site"], n, "donor.site")
            image = np.asarray(donor["image"])
            if image.shape != (3,) or not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
                raise ValueError("donor image must be three integers")
            if not np.isfinite(image).all() or not np.equal(image, np.floor(image)).all():
                raise ValueError("donor image must contain finite integers")
            image = image.astype(np.int64)
            identity = (j, *image.tolist())
            if identity in seen_donors:
                raise ValueError("duplicate donor (site,image)")
            seen_donors.add(identity)
            displacement = (coordinates[j] + image - coordinates[i]) @ lattice
            length = float(np.linalg.norm(displacement))
            if not np.isfinite(length) or length <= 1e-10:
                raise ValueError("zero or nonfinite donor displacement is invalid")
            vectors.append(displacement)
            donor_elements.append(int(numbers[j]))
        if cn:
            for element in donor_elements:
                bin_index = DONOR_Z.index(element) if element in DONOR_Z else 7
                output[i, 11 + bin_index] += 1 / cn
            output[i, 19] = len(set(donor_elements)) / cn
            vectors_arr = np.asarray(vectors, dtype=np.float64)
            lengths = np.linalg.norm(vectors_arr, axis=1)
            output[i, 31:34] = [lengths.mean(), lengths.std(ddof=0), lengths.max() / lengths.min()]
            directions = vectors_arr / lengths[:, None]
            q = directions.T @ directions / cn - np.eye(3) / 3
            output[i, 34:36] = [np.linalg.norm(q, ord="fro"), np.linalg.det(q)]
            quality[i, 2:] = 1

        scores = record.get("shape_scores")
        if scores is not None:
            expected = TEMPLATE_BY_CN.get(cn)
            if expected is None:
                raise ValueError("supplied shape scores have unsupported CN; omit scores for unsupported shapes")
            if not isinstance(scores, dict) or set(scores) != set(expected):
                raise ValueError(f"CN={cn} requires exactly shape templates {expected}")
            values = []
            for name in expected:
                value = scores[name]
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                    raise ValueError("shape scores must be real numeric values")
                value = float(value)
                if not np.isfinite(value) or value < 0:
                    raise ValueError("shape scores must be finite and nonnegative")
                values.append(value)
                output[i, 21 + TEMPLATES.index(name)] = np.exp(-0.1 * value)
            output[i, 20] = 1
            output[i, 29] = np.log1p(min(values))
            if len(values) == 2:
                output[i, 30] = np.tanh(values[1] - values[0])
    if not np.isfinite(output).all():
        raise ValueError("coordination features became nonfinite")
    if np.any(np.abs(output) > np.finfo(np.float32).max):
        raise ValueError("coordination features exceed float32 range")
    return torch.from_numpy(output.astype(np.float32)), torch.from_numpy(quality.astype(np.float32))


def _configuration(overrides: dict | None) -> dict:
    config = json.loads(METHOD_CONFIG_V1.read_text(encoding="utf-8"))
    if overrides is not None:
        if not isinstance(overrides, dict) or not set(overrides) <= set(config):
            raise ValueError("coordination configuration contains unknown fields")
        mutable = {"cutoff_factor", "audit_search_factor", "cutoff_tolerance_angstrom",
                   "near_cutoff_margin_angstrom", "max_nbins", "max_candidate_pairs"}
        for key, value in overrides.items():
            if key not in mutable and value != config[key]:
                raise ValueError(f"Unsupported change to fixed coordination method: {key}")
            config[key] = value
    for key in ("cutoff_factor", "audit_search_factor"):
        if (isinstance(config[key], bool) or not isinstance(config[key], (int, float))
                or not np.isfinite(config[key]) or not 0.5 <= config[key] <= 2):
            raise ValueError(f"Invalid {key}")
    for key in ("cutoff_tolerance_angstrom", "near_cutoff_margin_angstrom"):
        if (isinstance(config[key], bool) or not isinstance(config[key], (int, float))
                or not np.isfinite(config[key]) or not 0 <= config[key] <= 0.1):
            raise ValueError(f"Invalid {key}")
    for key in ("max_nbins", "max_candidate_pairs"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    return config


def extract_coordination(z: Any, frac: Any, cell: Any, config: dict | None = None
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return raw v1 features [N,36], quality masks [N,4], and geometric diagnostics."""
    numbers, coordinates, lattice = validate_geometry(z, frac, cell)
    settings = _configuration(config)
    metal_indices = [i for i, number in enumerate(numbers) if int(number) in METALS]
    allowed_donors = set(settings["donor_atomic_numbers"])
    radii = np.asarray(covalent_radii[numbers], dtype=np.float64)
    relevant = np.array([int(number) in METALS or int(number) in allowed_donors for number in numbers])
    if (not np.isfinite(radii[relevant]).all()
            or np.any(radii[relevant] <= settings["reject_ase_placeholder_radius_angstrom"])):
        raise ValueError("Relevant elements require supported covalent radii; ASE placeholders are rejected")
    factor = float(settings["cutoff_factor"])
    tolerance = float(settings["cutoff_tolerance_angstrom"])
    search_factor = max(factor, float(settings["audit_search_factor"]))
    diagnostics = {
        "method_id": settings["method_id"],
        "shape_method": settings["shape_method"],
        "config": settings,
        "num_atoms": len(numbers),
        "num_metal_centers": len(metal_indices),
        "donor_definition": "project_geometric_contact_prior_not_verified_chemical_bond",
        "centers": [],
        "candidate_search_factor": search_factor,
        "margin_scope": "accepted_and_rejected_allowed_contacts_within_candidate_search_only",
        "threshold_boundary_policy": "absolute_distance_tolerance; all_ties_retained",
        "covalent_radii_sha256": hashlib.sha256(np.asarray(covalent_radii, dtype="<f8").tobytes()).hexdigest(),
    }
    candidates = {i: [] for i in metal_indices}
    if metal_indices:
        atoms = Atoms(numbers=numbers, scaled_positions=coordinates, cell=lattice, pbc=True)
        first, second, shifts, distances = neighbor_list(
            "ijSd", atoms, search_factor * radii + tolerance,
            self_interaction=False, max_nbins=settings["max_nbins"])
        if len(first) > settings["max_candidate_pairs"]:
            raise ValueError("Periodic contact enumeration exceeds the configured candidate-pair budget")
        diagnostics["enumerated_pair_count"] = int(len(first))
        for i, j, shift, distance in zip(first, second, shifts, distances):
            i, j = int(i), int(j)
            if i not in candidates or int(numbers[j]) not in allowed_donors or int(numbers[j]) in METALS:
                continue
            if not np.isfinite(distance) or distance <= 1e-10:
                raise ValueError("Coincident/nonfinite candidate donor displacement")
            vector = (coordinates[j] + shift - coordinates[i]) @ lattice
            actual_distance = float(np.linalg.norm(vector))
            if not np.isclose(actual_distance, distance, atol=1e-8, rtol=1e-8):
                raise ArithmeticError("ASE periodic image and shared atom-table displacement disagree")
            cutoff = factor * float(radii[i] + radii[j])
            margin = cutoff - actual_distance
            candidates[i].append({
                "site": j,
                "image": [int(v) for v in shift],
                "atomic_number": int(numbers[j]),
                "vector": vector.tolist(),
                "distance": actual_distance,
                "cutoff_angstrom": cutoff,
                "signed_cutoff_margin_angstrom": margin,
                "accepted": actual_distance <= cutoff + tolerance,
            })
    else:
        diagnostics["enumerated_pair_count"] = 0
    records = []
    for i in metal_indices:
        nearby = sorted(candidates[i], key=lambda item: (item["site"], *item["image"]))
        donors = [item for item in nearby if item["accepted"]]
        vectors = np.asarray([item["vector"] for item in donors], dtype=np.float64).reshape(-1, 3)
        scores = shape_scores(vectors)
        record = {"center": i, "donors": [{"site": d["site"], "image": d["image"]} for d in donors]}
        if scores:
            record["shape_scores"] = scores
        records.append(record)
        cn = len(donors)
        state = "supported_shape" if scores else "zero_donor" if cn == 0 else "unsupported_shape_cn"
        margins = [abs(d["signed_cutoff_margin_angstrom"]) for d in nearby]
        diagnostics["centers"].append({
            "site": i,
            "atomic_number": int(numbers[i]),
            "cn": cn,
            "coord_valid": True,
            "shape_valid": bool(scores),
            "state": state,
            "donors": donors,
            "shape_scores": scores,
            "candidate_count_in_audit_search": len(nearby),
            "min_abs_cutoff_margin_audited_angstrom": min(margins) if margins else None,
            "near_cutoff_candidate_count": sum(m <= settings["near_cutoff_margin_angstrom"] for m in margins),
            "rejected_candidate_count": len(nearby) - cn,
        })
    features, quality = encode_coordination(numbers, coordinates, lattice, records)
    valid_shape_count = sum(row["shape_valid"] for row in diagnostics["centers"])
    diagnostics.update(
        shape_supported_centers=valid_shape_count,
        shape_supported_fraction_of_metals=valid_shape_count / len(metal_indices) if metal_indices else None,
        all_metal_centers_shape_supported=bool(metal_indices) and valid_shape_count == len(metal_indices),
        no_metal_centers=not bool(metal_indices),
        zero_donor_centers=sum(row["cn"] == 0 for row in diagnostics["centers"]),
    )
    return features.numpy(), quality.numpy(), diagnostics
