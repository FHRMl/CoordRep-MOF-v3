"""Coordination method v3 (D-015): carboxylate-C filter (v2) + exact Procrustes shapes for CN=7,8,9.

Method ID: ``periodic_covalent_radius_direction_procrustes_v3``.
Rules:
1. Donor selection is identical to v2 (``periodic_covalent_radius_direction_procrustes_v2``):
   excludes any accepted C contact (Z=6) within < 1.45 A of an accepted O/N contact (Z in {7,8})
   of the same metal center.
2. Exact unit-direction Procrustes shape scores are extended from CN in {2..6} to CN in {2..9}
   by adding canonical two-template polyhedral pairs for high-coordination centers:
   - CN = 7: ``PBPY7`` (Pentagonal Bipyramid, D5h) vs ``OCF7`` (Capped Octahedron, C3v)
   - CN = 8: ``SAPR8`` (Archimedean Square Antiprism, D4d) vs ``CU8`` (Cube, Oh)
   - CN = 9: ``TCTPR9`` (Tricapped Trigonal Prism, D3h) vs ``CSAPR9`` (Capped Square Antiprism, C4v)
3. Exact O(3)/SO(3) symmetry quotient reduction (S_N / G_{O(3)}) is used for achiral templates,
   yielding bit-for-bit identical Procrustes scores to full N! enumeration at a fraction of the cost.
4. Batched GPU/CPU high-CN evaluation uses the closed-form Viète trigonometric eigenvalue solution
   for 3x3 symmetric positive semidefinite matrices M = H^T H (``_svals_sum_3x3``).
5. Schema compatibility (36 columns):
   - For CN in {2..6}: 100% identical to v2.
   - For CN in {7,8,9}: sets column 20 (``shape_valid``) = 1.0, column 29 = ``log1p(min(scores))``,
     and column 30 = ``tanh(scores[1] - scores[0])``, while keeping CN=2..6 specific template
     slots [21:29] at 0.0 (masked during training-only standardization).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .geometry import DONOR_Z, METALS, validate_geometry
from .v1 import TEMPLATE_DIRECTIONS as _V1_TEMPLATES, TEMPLATES as _V1_TEMPLATE_NAMES, extract_coordination
from .v2 import DEFAULT_CARBOXYLATE_C_CUTOFF, filter_carboxylate_c_donors

ROOT = Path(__file__).resolve().parents[2]
METHOD_ID_V3 = "periodic_covalent_radius_direction_procrustes_v3"
METHOD_CONFIG_V3 = ROOT / "configs/coordination_method_v3.json"

TEMPLATE_BY_CN_V3: dict[int, tuple[str, ...]] = {
    2: ("L",),
    3: ("TP",),
    4: ("Td", "SP"),
    5: ("TBP", "SPY"),
    6: ("Oh", "TPr"),
    7: ("PBPY7", "OCF7"),
    8: ("SAPR8", "CU8"),
    9: ("TCTPR9", "CSAPR9"),
}
TEMPLATES_V3 = (
    "L", "TP", "Td", "SP", "TBP", "SPY", "Oh", "TPr",
    "PBPY7", "OCF7", "SAPR8", "CU8", "TCTPR9", "CSAPR9",
)


def _high_cn_templates() -> dict[str, np.ndarray]:
    raw: dict[str, list[list[float]]] = {}

    # CN=7: PBPY7 (Pentagonal Bipyramid, D5h)
    pbpy = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    for k in range(5):
        phi = 2.0 * math.pi * k / 5.0
        pbpy.append([math.cos(phi), math.sin(phi), 0.0])
    raw["PBPY7"] = pbpy

    # CN=7: OCF7 (Monocapped Octahedron, C3v)
    ocf = [[0.0, 0.0, 1.0]]
    z1, r1 = 0.5, math.sqrt(0.75)
    z2, r2 = -1.0 / math.sqrt(3.0), math.sqrt(2.0 / 3.0)
    for k in range(3):
        phi1 = 2.0 * math.pi * k / 3.0
        phi2 = phi1 + math.pi / 3.0
        ocf.append([r1 * math.cos(phi1), r1 * math.sin(phi1), z1])
        ocf.append([r2 * math.cos(phi2), r2 * math.sin(phi2), z2])
    raw["OCF7"] = ocf

    # CN=8: SAPR8 (Archimedean uniform-edge Square Antiprism, D4d: z/r = 2**(-3/4))
    z_over_r = 2.0 ** (-0.75)
    r_sapr = 1.0 / math.sqrt(1.0 + z_over_r * z_over_r)
    z_sapr = z_over_r * r_sapr
    sapr = []
    for k in range(4):
        phi = 0.5 * math.pi * k
        sapr.append([r_sapr * math.cos(phi), r_sapr * math.sin(phi), z_sapr])
    for k in range(4):
        phi = 0.5 * math.pi * k + 0.25 * math.pi
        sapr.append([r_sapr * math.cos(phi), r_sapr * math.sin(phi), -z_sapr])
    raw["SAPR8"] = sapr

    # CN=8: CU8 (Cube, Oh)
    cu = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                cu.append([sx, sy, sz])
    raw["CU8"] = cu

    # CN=9: TCTPR9 (Tricapped Trigonal Prism, D3h: prism z/r = sqrt(3)/2 + 3 equatorial caps)
    zp_over_rp = math.sqrt(3.0) / 2.0
    rp = 1.0 / math.sqrt(1.0 + zp_over_rp * zp_over_rp)
    zp = zp_over_rp * rp
    tctpr = []
    for sz in (1.0, -1.0):
        for k in range(3):
            phi = 2.0 * math.pi * k / 3.0
            tctpr.append([rp * math.cos(phi), rp * math.sin(phi), sz * zp])
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0 + math.pi / 3.0
        tctpr.append([math.cos(phi), math.sin(phi), 0.0])
    raw["TCTPR9"] = tctpr

    # CN=9: CSAPR9 (Monocapped Square Antiprism, C4v)
    csapr = [[0.0, 0.0, 1.0]]
    zc1, rc1 = 1.0 / 3.0, math.sqrt(8.0 / 9.0)
    zc2, rc2 = -1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)
    for k in range(4):
        phi = 0.5 * math.pi * k
        csapr.append([rc1 * math.cos(phi), rc1 * math.sin(phi), zc1])
    for k in range(4):
        phi = 0.5 * math.pi * k + 0.25 * math.pi
        csapr.append([rc2 * math.cos(phi), rc2 * math.sin(phi), zc2])
    raw["CSAPR9"] = csapr

    result = dict(_V1_TEMPLATES)
    for name, pts in raw.items():
        arr = np.asarray(pts, dtype=np.float64)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        arr.setflags(write=False)
        result[name] = arr
    return result


TEMPLATE_DIRECTIONS_V3 = _high_cn_templates()


def _find_orthogonal_symmetry_group(template: np.ndarray) -> tuple[list[tuple[int, ...]], bool]:
    """Return all permutations g in S_N such that T[g] = T @ R_g for R_g in O(3), and whether det=-1 exists."""
    n = len(template)
    gram = template @ template.T
    basis_idx = None
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                sub = template[[i, j, k]]
                if abs(np.linalg.det(sub)) > 1e-5:
                    basis_idx = (i, j, k)
                    break
            if basis_idx is not None:
                break
        if basis_idx is not None:
            break

    if basis_idx is None:
        syms = []
        has_improper = True
        for p in permutations(range(n)):
            if np.allclose(gram[np.ix_(p, p)], gram, atol=1e-10):
                syms.append(p)
        return syms, has_improper

    b0, b1, b2 = basis_idx
    base_inv = np.linalg.inv(template[[b0, b1, b2]])
    syms = []
    has_improper = False
    for i in range(n):
        for j in range(n):
            if j == i or not math.isclose(gram[i, j], gram[b0, b1], abs_tol=1e-7):
                continue
            for k in range(n):
                if k in (i, j) or not (
                    math.isclose(gram[i, k], gram[b0, b2], abs_tol=1e-7)
                    and math.isclose(gram[j, k], gram[b1, b2], abs_tol=1e-7)
                ):
                    continue
                rot = base_inv @ template[[i, j, k]]
                if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-7):
                    continue
                mapped = template @ rot
                dists = np.linalg.norm(mapped[:, None, :] - template[None, :, :], axis=-1)
                perm = tuple(int(np.argmin(dists[r])) for r in range(n))
                if len(set(perm)) == n and float(np.max(dists[np.arange(n), perm])) < 1e-7:
                    if perm not in syms:
                        syms.append(perm)
                        if np.linalg.det(rot) < 0:
                            has_improper = True
    return syms, has_improper


@lru_cache(maxsize=16)
def _coset_permuted_templates(name: str) -> tuple[np.ndarray, bool]:
    """Return (T[coset_reps] of shape [K, CN, 3], is_achiral)."""
    template = TEMPLATE_DIRECTIONS_V3[name]
    n = len(template)
    if n <= 6:
        indices = np.asarray(list(permutations(range(n))), dtype=np.int64)
        res = template[indices]
        res.setflags(write=False)
        return res, False

    syms, has_improper = _find_orthogonal_symmetry_group(template)
    if not has_improper:
        raise RuntimeError(f"Expected achiral template with improper symmetry for {name}")
    seen_prefixes: set[tuple[int, int, int, int]] = set()
    canonical_prefixes: list[tuple[int, ...]] = []
    for p4 in permutations(range(n), 4):
        if p4 in seen_prefixes:
            continue
        orbit = {(g[p4[0]], g[p4[1]], g[p4[2]], g[p4[3]]) for g in syms}
        if len(orbit) == len(syms):
            canonical_prefixes.append(p4)
            seen_prefixes.update(orbit)
        else:
            for tail in permutations([x for x in range(n) if x not in p4]):
                full_p = p4 + tail
                full_orbit = [tuple(g[x] for x in full_p) for g in syms]
                rep = min(full_orbit)
                if full_p == rep:
                    canonical_prefixes.append(full_p)
            seen_prefixes.update(orbit)

    reps: list[tuple[int, ...]] = []
    all_set = set(range(n))
    for pref in canonical_prefixes:
        if len(pref) == n:
            reps.append(pref)
        else:
            rem = sorted(all_set - set(pref))
            for tail in permutations(rem):
                reps.append(pref + tail)

    indices = np.asarray(reps, dtype=np.int64)
    res = np.ascontiguousarray(template[indices], dtype=np.float64)
    res.setflags(write=False)
    return res, True


@lru_cache(maxsize=16)
def _flat_coset_matrix(name: str, device_str: str = "cpu") -> torch.Tensor:
    """Return reshaped template matrix [3*K, CN] on device for fast batched matmul."""
    cosets, _ = _coset_permuted_templates(name)
    mat = np.ascontiguousarray(np.transpose(cosets, (0, 2, 1)).reshape(-1, cosets.shape[1]), dtype=np.float64)
    return torch.from_numpy(mat).to(device=device_str, dtype=torch.float64)


def shape_scores_v3(vectors: Any) -> dict[str, float]:
    """Exact unit-direction Procrustes shape scores for CN in {2..9}."""
    value = np.asarray(vectors, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("shape vectors must be finite [CN,3] Cartesian displacements")
    lengths = np.linalg.norm(value, axis=1)
    if np.any(lengths <= 1e-10) or not np.isfinite(lengths).all():
        raise ValueError("shape vectors require finite positive donor distances")
    cn = len(value)
    if cn not in TEMPLATE_BY_CN_V3:
        return {}
    directions = value / lengths[:, None]
    norm_u = float(np.sum(directions * directions))
    result: dict[str, float] = {}
    for name in TEMPLATE_BY_CN_V3[cn]:
        template = TEMPLATE_DIRECTIONS_V3[name]
        cosets, is_achiral = _coset_permuted_templates(name)
        covariance = np.einsum("pni,nj->pij", cosets, directions)
        if is_achiral:
            singular = np.linalg.svd(covariance, compute_uv=False)
            best = max(float(singular.sum(axis=1).max()), 0.0)
        else:
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


def _svals_sum_3x3(h: torch.Tensor) -> torch.Tensor:
    """Analytical sum of singular values (sigma_1 + sigma_2 + sigma_3) for batched [..., 3, 3] matrices."""
    m = h.transpose(-1, -2) @ h
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m11, m12, m22 = m[..., 1, 1], m[..., 1, 2], m[..., 2, 2]
    q = (m00 + m11 + m22) / 3.0
    b00, b11, b22 = m00 - q, m11 - q, m22 - q
    p2 = (b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * (m01 * m01 + m02 * m02 + m12 * m12)) / 6.0
    p = torch.sqrt(p2.clamp_min(0.0))
    det_b = b00 * (b11 * b22 - m12 * m12) - m01 * (m01 * b22 - m12 * m02) + m02 * (m01 * m12 - b11 * m02)
    denom = (2.0 * p * p2).clamp_min(1e-30)
    r = torch.clamp(det_b / denom, -1.0, 1.0)
    phi = torch.acos(r) / 3.0
    two_p = 2.0 * p
    lam1 = q + two_p * torch.cos(phi)
    lam3 = q + two_p * torch.cos(phi + (2.0 * math.pi / 3.0))
    lam2 = 3.0 * q - lam1 - lam3
    return torch.sqrt(lam1.clamp_min(0.0)) + torch.sqrt(lam2.clamp_min(0.0)) + torch.sqrt(lam3.clamp_min(0.0))


def batched_high_cn_shape_scores(vectors_list: list[np.ndarray], cn: int, *,
                                 device: str | None = None,
                                 center_chunk: int = 128) -> list[dict[str, float]]:
    """Compute exact v3 Procrustes shape scores for a batch of centers all having ``cn`` in {7, 8, 9}."""
    if not vectors_list:
        return []
    if cn not in (7, 8, 9):
        return [shape_scores_v3(v) for v in vectors_list]
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arr = np.asarray(vectors_list, dtype=np.float64)
    if arr.shape != (len(vectors_list), cn, 3) or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite [{len(vectors_list)}, {cn}, 3] array")
    lengths = np.linalg.norm(arr, axis=2, keepdims=True)
    if np.any(lengths <= 1e-10):
        raise ValueError("Positive donor distances required")
    dirs = arr / lengths
    total = len(vectors_list)
    names = TEMPLATE_BY_CN_V3[cn]
    out: list[dict[str, float]] = [{} for _ in range(total)]
    denom = float(cn * cn)

    for name in names:
        flat_t = _flat_coset_matrix(name, dev)
        k_cosets = flat_t.shape[0] // 3
        coset_chunk = 8192
        for start in range(0, total, center_chunk):
            end = min(total, start + center_chunk)
            b = end - start
            u_chunk = torch.from_numpy(
                np.ascontiguousarray(np.transpose(dirs[start:end], (1, 0, 2)).reshape(cn, 3 * b))
            ).to(device=dev, dtype=torch.float64)
            best_corr = torch.zeros(b, device=dev, dtype=torch.float64)
            for c_start in range(0, k_cosets, coset_chunk):
                c_end = min(k_cosets, c_start + coset_chunk)
                ck = c_end - c_start
                t_sub = flat_t[3 * c_start : 3 * c_end]
                cov = (t_sub @ u_chunk).view(ck, 3, b, 3).permute(0, 2, 1, 3)
                svals = _svals_sum_3x3(cov)
                best_corr = torch.maximum(best_corr, svals.max(dim=0).values)
            best_np = best_corr.detach().cpu().numpy()
            scores_np = np.clip(100.0 * (1.0 - (best_np * best_np) / denom), 0.0, 100.0)
            for idx_local, sc in enumerate(scores_np):
                out[start + idx_local][name] = float(sc)
    return out


def load_v3_config(overrides: dict | None = None) -> dict:
    cfg = json.loads(METHOD_CONFIG_V3.read_text(encoding="utf-8"))
    if overrides:
        if not isinstance(overrides, dict) or not set(overrides) <= set(cfg):
            raise ValueError("Unknown coordination v3 configuration override")
        mutable = {"cutoff_factor", "audit_search_factor", "cutoff_tolerance_angstrom",
                   "near_cutoff_margin_angstrom", "max_nbins", "max_candidate_pairs"}
        for k, v in overrides.items():
            if k not in mutable and v != cfg[k]:
                raise ValueError(f"Unsupported change to fixed v3 coordination method: {k}")
            cfg[k] = v
    return cfg


def encode_coordination_v3(z: Any, frac: Any, cell: Any,
                           centers: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode schema cgc_coord_v3 slots [0:36] and four external quality masks (supporting CN 2..9)."""
    numbers, coordinates, lattice = validate_geometry(z, frac, cell)
    n = len(numbers)
    output = np.zeros((n, 36), dtype=np.float64)
    quality = np.zeros((n, 4), dtype=np.float64)
    quality[:, 0] = [int(int(v) in METALS) for v in numbers]
    if not isinstance(centers, (list, tuple)):
        raise ValueError("centers must be a list of explicit center records")
    seen_centers = set()
    for record in centers:
        i = int(record["center"])
        if i in seen_centers or not (0 <= i < n) or int(numbers[i]) not in METALS:
            raise ValueError(f"Invalid or duplicate metal center site {i}")
        seen_centers.add(i)
        if not record.get("coord_valid", True):
            continue
        donors = record["donors"]
        cn = len(donors)
        quality[i, 1] = 1.0
        if cn >= 2:
            output[i, min(cn, 10) - 2] = 1.0
        output[i, 9] = min(cn, 12) / 12.0
        output[i, 10] = float(cn <= 1)
        vectors = []
        donor_elements = []
        seen_donors = set()
        for donor in donors:
            j = int(donor["site"])
            image = np.asarray(donor["image"], dtype=np.int64)
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
                output[i, 11 + bin_index] += 1.0 / cn
            output[i, 19] = len(set(donor_elements)) / cn
            vectors_arr = np.asarray(vectors, dtype=np.float64)
            lengths = np.linalg.norm(vectors_arr, axis=1)
            output[i, 31:34] = [lengths.mean(), lengths.std(ddof=0), lengths.max() / lengths.min()]
            directions = vectors_arr / lengths[:, None]
            q = directions.T @ directions / cn - np.eye(3) / 3.0
            output[i, 34:36] = [np.linalg.norm(q, ord="fro"), np.linalg.det(q)]
            quality[i, 2:] = 1.0

        scores = record.get("shape_scores")
        if scores is not None and scores != {}:
            expected = TEMPLATE_BY_CN_V3.get(cn)
            if expected is None or set(scores) != set(expected):
                raise ValueError(f"CN={cn} requires exactly shape templates {expected}")
            values = []
            for name in expected:
                val = float(scores[name])
                if not np.isfinite(val) or val < 0:
                    raise ValueError("shape scores must be finite and nonnegative")
                values.append(val)
                if name in _V1_TEMPLATE_NAMES:
                    output[i, 21 + _V1_TEMPLATE_NAMES.index(name)] = np.exp(-0.1 * val)
            output[i, 20] = 1.0
            output[i, 29] = np.log1p(min(values))
            if len(values) == 2:
                output[i, 30] = np.tanh(values[1] - values[0])
    return torch.from_numpy(output.astype(np.float32)), torch.from_numpy(quality.astype(np.float32))


def extract_coordination_v3(z: Any, frac: Any, cell: Any,
                            config: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract v3 coordination features [N,36], quality masks [N,4], and diagnostics from structure geometry."""
    numbers, coordinates, lattice = validate_geometry(z, frac, cell)
    settings = load_v3_config(config)
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
        cn = len(donors)
        vectors = np.asarray([d["vector"] for d in donors], dtype=np.float64).reshape(-1, 3)
        if not rejected_c and cn in (2, 3, 4, 5, 6):
            scores = dict(center["shape_scores"])
        elif cn in TEMPLATE_BY_CN_V3:
            scores = shape_scores_v3(vectors)
        else:
            scores = {}
        if rejected_c or (cn in (7, 8, 9) and scores):
            changed_centers += 1
        record = {"center": center["site"], "donors": [{"site": d["site"], "image": d["image"]} for d in donors]}
        if scores:
            record["shape_scores"] = scores
        records.append(record)
        state = "supported_shape" if scores else "zero_donor" if cn == 0 else "unsupported_shape_cn"
        c_diag = dict(center)
        c_diag.update(cn=cn, v1_cn=center["cn"], shape_valid=bool(scores), state=state,
                      donors=donors, shape_scores=scores,
                      rejected_carboxylate_c_donors=rejected_c,
                      rejected_carboxylate_c_count=len(rejected_c),
                      rejected_candidate_count=center["candidate_count_in_audit_search"] - cn)
        centers_diag.append(c_diag)
    features, quality = encode_coordination_v3(numbers, coordinates, lattice, records)
    metal_count = v1_diag["num_metal_centers"]
    valid_shape_count = sum(row["shape_valid"] for row in centers_diag)
    diagnostics = dict(v1_diag)
    diagnostics.update(
        method_id=METHOD_ID_V3,
        config=settings,
        centers=centers_diag,
        shape_supported_centers=valid_shape_count,
        shape_supported_fraction_of_metals=valid_shape_count / metal_count if metal_count else None,
        all_metal_centers_shape_supported=bool(metal_count) and valid_shape_count == metal_count,
        zero_donor_centers=sum(row["cn"] == 0 for row in centers_diag),
        v3_rejected_carboxylate_c_total=total_rejected_c,
        v3_changed_centers_count=changed_centers,
    )
    return features.numpy(), quality.numpy(), diagnostics


def valid_coordination_channels_v3(coord_fea: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """Return boolean mask [N, 36] of active channels supporting shape_valid on CN 2..9."""
    if quality.device.type != "cpu" or coord_fea.device.type != "cpu":
        raise ValueError("Fit/transform raw coordination features on CPU")
    if coord_fea.ndim != 2 or coord_fea.shape[1] != 36 or quality.shape != (coord_fea.shape[0], 4):
        raise ValueError("Invalid coordination feature/mask dimensions")
    valid = torch.zeros_like(coord_fea, dtype=torch.bool)
    available = (quality[:, 0] == 1) & (quality[:, 1] == 1)
    valid[:, :20] = available[:, None]
    shape_valid = available & (coord_fea[:, 20] == 1)
    supported_cn = coord_fea[:, :8] == 1  # CN 2..9 (columns 0..7)
    if bool((shape_valid & (supported_cn.sum(1) != 1)).any()):
        raise ValueError("A valid shape requires exactly one raw supported-CN one-hot entry (CN 2..9)")
    for cn_column, start, stop in ((0, 21, 22), (1, 22, 23), (2, 23, 25), (3, 25, 27), (4, 27, 29)):
        valid[:, start:stop] = (shape_valid & supported_cn[:, cn_column])[:, None]
    valid[:, 29] = shape_valid
    valid[:, 30] = shape_valid & supported_cn[:, 2:8].any(1)  # Two-template CN 4..9
    valid[:, 31:34] = (available & (quality[:, 2] == 1))[:, None]
    valid[:, 34:36] = (available & (quality[:, 3] == 1))[:, None]
    return valid


@dataclass(frozen=True)
class CoordinationScalerV3:
    """Channel-selective standardization scaler supporting CN 2..9 shape masks."""
    mean: torch.Tensor
    scale: torch.Tensor
    count: torch.Tensor

    @classmethod
    def fit(cls, records: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> "CoordinationScalerV3":
        """Fit channel means and scales on training (coord_fea, quality) pairs only."""
        count = torch.zeros(36, dtype=torch.int64)
        total = torch.zeros(36, dtype=torch.float64)
        total_sq = torch.zeros(36, dtype=torch.float64)
        n_records = 0
        for coord_fea, quality in records:
            n_records += 1
            valid = valid_coordination_channels_v3(coord_fea, quality)
            x = coord_fea.double()
            masked = torch.where(valid, x, torch.zeros_like(x))
            count += valid.sum(0)
            total += masked.sum(0)
            total_sq += (masked * masked).sum(0)
        if n_records == 0:
            raise ValueError("Cannot fit CoordinationScalerV3 on empty training set")
        safe_count = count.clamp(min=1).double()
        mean = torch.where(count > 0, total / safe_count, torch.zeros_like(total))
        var = torch.where(count > 0, (total_sq / safe_count - mean * mean).clamp(min=0.0), torch.ones_like(total))
        scale = torch.sqrt(var)
        scale = torch.where((count == 0) | (scale < 1e-8), torch.ones_like(scale), scale)
        mean[20] = 0.0
        scale[20] = 1.0
        return cls(mean=mean, scale=scale, count=count)

    def transform(self, coord_fea: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        valid = valid_coordination_channels_v3(coord_fea, quality)
        result = torch.where(
            valid,
            (coord_fea.double() - self.mean) / self.scale,
            torch.zeros_like(coord_fea, dtype=torch.float64),
        )
        result[:, 20] = coord_fea[:, 20]
        return result.float()
