"""Unit tests for CoordRep-MOF (v1, v2, and v3 high-CN Procrustes descriptors)."""
from __future__ import annotations

from itertools import permutations
import unittest

import numpy as np
import torch

from coordrep_mof import (
    CoordinationScalerV3,
    TEMPLATE_BY_CN_V3,
    TEMPLATE_DIRECTIONS_V3,
    batched_high_cn_shape_scores,
    extract_coordination,
    extract_coordination_v2,
    extract_coordination_v3,
    shape_scores,
    shape_scores_v3,
)
from coordrep_mof.v3 import _coset_permuted_templates, _find_orthogonal_symmetry_group


def _random_so3(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _brute_force_so3_procrustes(directions: np.ndarray, template: np.ndarray) -> float:
    n = len(template)
    perms = np.asarray(list(permutations(range(n))), dtype=np.int64)
    cov = np.einsum("pni,nj->pij", template[perms], directions)
    u, s, vt = np.linalg.svd(cov)
    signs = np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)
    corr = s[:, :2].sum(axis=1) + signs * s[:, 2]
    best = max(float(corr.max()), 0.0)
    denom = float(np.sum(directions * directions) * np.sum(template * template))
    return float(np.clip(100.0 * (1.0 - best * best / denom), 0.0, 100.0))


class TestCoordRepMOFV3(unittest.TestCase):
    def test_high_cn_templates_unit_norm_and_symmetry_groups(self):
        expected_orders = {
            "PBPY7": 20,   # D5h
            "OCF7": 6,     # C3v
            "SAPR8": 16,   # D4d
            "CU8": 48,     # Oh
            "TCTPR9": 12,  # D3h
            "CSAPR9": 8,   # C4v
        }
        for name, expected_g in expected_orders.items():
            tpl = TEMPLATE_DIRECTIONS_V3[name]
            norms = np.linalg.norm(tpl, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-12)
            syms, has_improper = _find_orthogonal_symmetry_group(tpl)
            self.assertEqual(len(syms), expected_g, f"Unexpected |G_O(3)| for {name}")
            self.assertTrue(has_improper, f"Expected improper symmetry in {name}")
            cosets, is_achiral = _coset_permuted_templates(name)
            self.assertTrue(is_achiral)
            n_fact = int(np.prod(np.arange(1, len(tpl) + 1)))
            self.assertEqual(len(cosets) * expected_g, n_fact)

    def test_ideal_templates_zero_score_and_pairwise_separation(self):
        rng = np.random.default_rng(20261109)
        for cn in range(2, 10):
            names = TEMPLATE_BY_CN_V3[cn]
            for name in names:
                tpl = TEMPLATE_DIRECTIONS_V3[name]
                rot = _random_so3(rng)
                perm = rng.permutation(cn)
                radii = rng.uniform(1.8, 2.6, size=(cn, 1))
                distorted = (tpl[perm] @ rot) * radii
                scores = shape_scores_v3(distorted)
                self.assertAlmostEqual(scores[name], 0.0, places=9)
                for other in names:
                    if other != name:
                        self.assertGreater(scores[other], 1.5, f"{name} vs {other} not separated")

    def test_quotient_reduction_matches_brute_force_factorial(self):
        rng = np.random.default_rng(42)
        for cn in (7, 8):
            for _ in range(3):
                pts = rng.standard_normal((cn, 3))
                pts /= np.linalg.norm(pts, axis=1, keepdims=True)
                fast_scores = shape_scores_v3(pts)
                for name in TEMPLATE_BY_CN_V3[cn]:
                    exact = _brute_force_so3_procrustes(pts, TEMPLATE_DIRECTIONS_V3[name])
                    self.assertAlmostEqual(fast_scores[name], exact, places=9)

    def test_batched_analytical_3x3_matches_single_center_svd(self):
        rng = np.random.default_rng(99)
        for cn in (7, 8, 9):
            batch = [rng.standard_normal((cn, 3)) + 2.0 * TEMPLATE_DIRECTIONS_V3[TEMPLATE_BY_CN_V3[cn][0]]
                     for _ in range(5)]
            single = [shape_scores_v3(v) for v in batch]
            batched = batched_high_cn_shape_scores(batch, cn, device="cpu", center_chunk=2)
            for s1, s2 in zip(single, batched):
                for k in s1:
                    self.assertAlmostEqual(s1[k], s2[k], places=8)

    def test_zr8_uranyl7_and_la9_carboxylate_filter_pipeline(self):
        cell = np.diag([20.0, 20.0, 20.0])
        center = np.array([10.0, 10.0, 10.0])

        # 1. ZrO8 (SAPR8)
        o8 = center + 2.15 * TEMPLATE_DIRECTIONS_V3["SAPR8"]
        z_zr = [40] + [8] * 8
        frac_zr = np.vstack([center, o8]) / 20.0
        f_v1, _, d_v1 = extract_coordination(z_zr, frac_zr, cell)
        f_v3, _, d_v3 = extract_coordination_v3(z_zr, frac_zr, cell)
        self.assertEqual(d_v1["centers"][0]["cn"], 8)
        self.assertFalse(d_v1["centers"][0]["shape_valid"])
        self.assertTrue(d_v3["centers"][0]["shape_valid"])
        self.assertAlmostEqual(d_v3["centers"][0]["shape_scores"]["SAPR8"], 0.0, places=7)
        self.assertEqual(float(f_v3[0, 20]), 1.0)

        # 2. La9 with 3 chelating carboxylate C atoms (v1 CN=12 -> v2/v3 CN=9 TCTPR9)
        tctpr9 = TEMPLATE_DIRECTIONS_V3["TCTPR9"] * 2.50
        o9 = center + tctpr9
        c3 = [center + tctpr9[i] * 1.12 + np.array([0.0, 0.0, 0.85]) for i in (6, 7, 8)]
        z_la = [57] + [8] * 9 + [6] * 3
        frac_la = np.vstack([center, o9, np.asarray(c3)]) / 20.0
        _, _, d_la_v1 = extract_coordination(z_la, frac_la, cell)
        _, _, d_la_v2 = extract_coordination_v2(z_la, frac_la, cell)
        f_la_v3, q_la_v3, d_la_v3 = extract_coordination_v3(z_la, frac_la, cell)
        self.assertEqual(d_la_v1["centers"][0]["cn"], 12)
        self.assertEqual(d_la_v2["centers"][0]["cn"], 9)
        self.assertFalse(d_la_v2["centers"][0]["shape_valid"])
        self.assertEqual(d_la_v3["centers"][0]["cn"], 9)
        self.assertTrue(d_la_v3["centers"][0]["shape_valid"])
        self.assertAlmostEqual(d_la_v3["centers"][0]["shape_scores"]["TCTPR9"], 0.0, places=7)

        # 3. Standardization on CN=8 + CN=9
        scaler = CoordinationScalerV3.fit([
            (torch.from_numpy(f_v3), torch.from_numpy(np.ones_like(q_la_v3[:len(f_v3)]))),
            (torch.from_numpy(f_la_v3), torch.from_numpy(q_la_v3)),
        ])
        transformed = scaler.transform(torch.from_numpy(f_la_v3), torch.from_numpy(q_la_v3))
        self.assertTrue(torch.isfinite(transformed).all())
        self.assertEqual(float(transformed[0, 20]), 1.0)
        self.assertTrue(torch.all(transformed[0, 21:29] == 0.0))


if __name__ == "__main__":
    unittest.main()
