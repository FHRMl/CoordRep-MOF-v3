"""Coordination method v2 (D-006): exclude chelating carboxylate C contacts.

Method ID: ``periodic_covalent_radius_direction_procrustes_v2``.
Rule: identical to v1 except that on each metal center, any accepted C contact (Z=6) whose
Euclidean distance to an accepted O or N contact (Z in {7, 8}) of the *same* metal center is
< 1.45 A is excluded before computing CN, donor element fractions, radial/directional statistics,
and Procrustes shape scores.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import DONOR_Z, METALS, validate_geometry
from .v1 import TEMPLATE_BY_CN, TEMPLATES, encode_coordination, extract_coordination, shape_scores

ROOT = Path(__file__).resolve().parents[2]
METHOD_ID_V2 = "periodic_covalent_radius_direction_procrustes_v2"
METHOD_CONFIG_V2 = ROOT / "configs/coordination_method_v2.json"
DEFAULT_CARBOXYLATE_C_CUTOFF = 1.45


def load_v2_config(overrides: dict | None = None) -> dict:
    cfg = json.loads(METHOD_CONFIG_V2.read_text(encoding="utf-8"))
    if overrides:
        if not isinstance(overrides, dict) or not set(overrides) <= set(cfg):
            raise ValueError("Unknown coordination v2 configuration override")
        mutable = {"cutoff_factor", "audit_search_factor", "cutoff_tolerance_angstrom",
                   "near_cutoff_margin_angstrom", "max_nbins", "max_candidate_pairs"}
        for k, v in overrides.items():
            if k not in mutable and v != cfg[k]:
                raise ValueError(f"Unsupported change to fixed v2 coordination method: {k}")
            cfg[k] = v
    return cfg


def filter_carboxylate_c_donors(donors: list[dict], *,
                                max_distance_angstrom: float = DEFAULT_CARBOXYLATE_C_CUTOFF
                                ) -> tuple[list[dict], list[dict]]:
    """Separate accepted v1 donors into (retained_v2_donors, rejected_carboxylate_c_donors)."""
    if not isinstance(max_distance_angstrom, (int, float)) or not (0.5 <= float(max_distance_angstrom) <= 2.5):
        raise ValueError("Invalid carboxylate C-O/N bond cutoff distance")
    cutoff = float(max_distance_angstrom)
    on_vectors = [np.asarray(d["vector"], dtype=np.float64)
                  for d in donors if int(d["atomic_number"]) in (7, 8)]
    retained, rejected = [], []
    for d in donors:
        if int(d["atomic_number"]) == 6 and on_vectors:
            vec = np.asarray(d["vector"], dtype=np.float64)
            dists = [float(np.linalg.norm(vec - h)) for h in on_vectors]
            min_dist = min(dists)
            if min_dist < cutoff:
                item = dict(d)
                item["accepted_v1"] = True
                item["accepted"] = False
                item["rejected_reason"] = "carboxylate_c_within_1.45A_of_accepted_O_or_N"
                item["min_accepted_on_distance_angstrom"] = min_dist
                rejected.append(item)
                continue
        retained.append(d)
    return retained, rejected


def extract_coordination_v2(z: Any, frac: Any, cell: Any,
                            config: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract v2 coordination features [N,36], quality masks [N,4], and diagnostics from structure geometry."""
    numbers, coordinates, lattice = validate_geometry(z, frac, cell)
    settings = load_v2_config(config)
    v1_overrides = {k: settings[k] for k in ("cutoff_factor", "audit_search_factor",
                                             "cutoff_tolerance_angstrom", "near_cutoff_margin_angstrom",
                                             "max_nbins", "max_candidate_pairs")}
    _, _, v1_diag = extract_coordination(numbers, coordinates, lattice, config=v1_overrides)
    c_cutoff = float(settings["carboxylate_c_filter"]["max_distance_angstrom"])
    records = []
    centers_diag = []
    total_rejected_c = 0
    changed_centers = 0
    for center in v1_diag["centers"]:
        v1_donors = center["donors"]
        donors, rejected_c = filter_carboxylate_c_donors(v1_donors, max_distance_angstrom=c_cutoff)
        total_rejected_c += len(rejected_c)
        if rejected_c:
            changed_centers += 1
        vectors = np.asarray([d["vector"] for d in donors], dtype=np.float64).reshape(-1, 3)
        scores = shape_scores(vectors) if rejected_c else dict(center["shape_scores"])
        record = {"center": center["site"], "donors": [{"site": d["site"], "image": d["image"]} for d in donors]}
        if scores:
            record["shape_scores"] = scores
        records.append(record)
        cn = len(donors)
        state = "supported_shape" if scores else "zero_donor" if cn == 0 else "unsupported_shape_cn"
        c_diag = dict(center)
        c_diag.update(cn=cn, v1_cn=center["cn"], shape_valid=bool(scores), state=state,
                      donors=donors, shape_scores=scores,
                      rejected_carboxylate_c_donors=rejected_c,
                      rejected_carboxylate_c_count=len(rejected_c),
                      rejected_candidate_count=center["candidate_count_in_audit_search"] - cn)
        centers_diag.append(c_diag)
    features, quality = encode_coordination(numbers, coordinates, lattice, records)
    metal_count = v1_diag["num_metal_centers"]
    valid_shape_count = sum(row["shape_valid"] for row in centers_diag)
    diagnostics = dict(v1_diag)
    diagnostics.update(
        method_id=METHOD_ID_V2,
        config=settings,
        centers=centers_diag,
        shape_supported_centers=valid_shape_count,
        shape_supported_fraction_of_metals=valid_shape_count / metal_count if metal_count else None,
        all_metal_centers_shape_supported=bool(metal_count) and valid_shape_count == metal_count,
        zero_donor_centers=sum(row["cn"] == 0 for row in centers_diag),
        v2_rejected_carboxylate_c_total=total_rejected_c,
        v2_changed_centers_count=changed_centers,
    )
    return features.numpy(), quality.numpy(), diagnostics


def encode_center_from_diagnostics(center: dict, *,
                                   apply_v2_filter: bool = True,
                                   max_distance_angstrom: float = DEFAULT_CARBOXYLATE_C_CUTOFF
                                   ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Encode one metal center's [36] feature and [4] quality vectors from stored v1 diagnostics."""
    output = np.zeros(36, dtype=np.float64)
    quality = np.zeros(4, dtype=np.float64)
    if int(center["atomic_number"]) not in METALS:
        raise ValueError("Diagnostic center is not in METALS")
    quality[0] = 1.0
    if not center.get("coord_valid", True):
        return output.astype(np.float32), quality.astype(np.float32), {
            "cn_v1": 0, "cn_v2": 0, "rejected_c": 0, "shape_valid_v1": False, "shape_valid_v2": False,
            "state_v1": center.get("state", "failed"), "state_v2": center.get("state", "failed")}
    v1_donors = center["donors"]
    if apply_v2_filter:
        donors, rejected_c = filter_carboxylate_c_donors(v1_donors, max_distance_angstrom=max_distance_angstrom)
    else:
        donors, rejected_c = v1_donors, []
    cn = len(donors)
    quality[1] = 1.0
    if cn >= 2:
        output[min(cn, 10) - 2] = 1.0
    output[9] = min(cn, 12) / 12.0
    output[10] = float(cn <= 1)
    if cn:
        vectors = np.asarray([d["vector"] for d in donors], dtype=np.float64).reshape(cn, 3)
        donor_elements = [int(d["atomic_number"]) for d in donors]
        for element in donor_elements:
            bin_index = DONOR_Z.index(element) if element in DONOR_Z else 7
            output[11 + bin_index] += 1.0 / cn
        output[19] = len(set(donor_elements)) / cn
        lengths = np.linalg.norm(vectors, axis=1)
        output[31:34] = [lengths.mean(), lengths.std(ddof=0), lengths.max() / lengths.min()]
        directions = vectors / lengths[:, None]
        q = directions.T @ directions / cn - np.eye(3) / 3.0
        output[34:36] = [np.linalg.norm(q, ord="fro"), np.linalg.det(q)]
        quality[2:] = 1.0
        scores = shape_scores(vectors) if rejected_c else (center.get("shape_scores") or {})
        if scores:
            expected = TEMPLATE_BY_CN.get(cn)
            if expected is None or set(scores) != set(expected):
                raise ValueError(f"Invalid shape score keys for CN={cn}")
            values = []
            for name in expected:
                val = float(scores[name])
                values.append(val)
                output[21 + TEMPLATES.index(name)] = np.exp(-0.1 * val)
            output[20] = 1.0
            output[29] = np.log1p(min(values))
            if len(values) == 2:
                output[30] = np.tanh(values[1] - values[0])
    else:
        scores = {}
    state_v2 = "supported_shape" if scores else "zero_donor" if cn == 0 else "unsupported_shape_cn"
    info = {
        "site": int(center["site"]),
        "atomic_number": int(center["atomic_number"]),
        "cn_v1": len(v1_donors),
        "cn_v2": cn,
        "rejected_c": len(rejected_c),
        "c_donors_v1": sum(1 for d in v1_donors if int(d["atomic_number"]) == 6),
        "c_donors_v2": sum(1 for d in donors if int(d["atomic_number"]) == 6),
        "o_donors_v1": sum(1 for d in v1_donors if int(d["atomic_number"]) == 8),
        "o_donors_v2": sum(1 for d in donors if int(d["atomic_number"]) == 8),
        "n_donors_v1": sum(1 for d in v1_donors if int(d["atomic_number"]) == 7),
        "n_donors_v2": sum(1 for d in donors if int(d["atomic_number"]) == 7),
        "shape_valid_v1": bool(center.get("shape_valid", False)),
        "shape_valid_v2": bool(scores),
        "state_v1": center.get("state"),
        "state_v2": state_v2,
    }
    return output.astype(np.float32), quality.astype(np.float32), info
