"""Quickstart demo comparing v1, v2, and v3 coordination descriptors on high-CN MOF motifs."""
from __future__ import annotations

import numpy as np

from coordrep_mof import (
    TEMPLATE_DIRECTIONS_V3,
    extract_coordination,
    extract_coordination_v2,
    extract_coordination_v3,
)


def run_demo() -> None:
    cell = np.diag([20.0, 20.0, 20.0])
    origin = np.array([10.0, 10.0, 10.0])

    # Case 1: ZrO8 Square Antiprism (UiO-66 SBU node)
    o8 = origin + 2.18 * TEMPLATE_DIRECTIONS_V3["SAPR8"]
    z_zr = [40] + [8] * 8
    frac_zr = np.vstack([origin, o8]) / 20.0

    # Case 2: UO7 Uranyl Pentagonal Bipyramid
    o7 = origin + np.array([1.78, 1.78, 2.36, 2.36, 2.36, 2.36, 2.36])[:, None] * TEMPLATE_DIRECTIONS_V3["PBPY7"]
    z_u = [92] + [8] * 7
    frac_u = np.vstack([origin, o7]) / 20.0

    # Case 3: LaO9 Tricapped Trigonal Prism + 3 Chelating Carboxylate Carbons
    tctpr9 = TEMPLATE_DIRECTIONS_V3["TCTPR9"] * 2.50
    o9 = origin + tctpr9
    c3 = [origin + tctpr9[k] * 1.12 + np.array([0.0, 0.0, 0.85]) for k in (6, 7, 8)]
    z_la = [57] + [8] * 9 + [6] * 3
    frac_la = np.vstack([origin, o9, np.asarray(c3)]) / 20.0

    cases = [
        ("ZrO8 (UiO-66 SBU, ideal SAPR-8)", z_zr, frac_zr),
        ("UO7 (Uranyl node, ideal PBPY-7)", z_u, frac_u),
        ("LaO9 + 3 Carboxylate-C (ideal TCTPR-9)", z_la, frac_la),
    ]

    print("=" * 88)
    print(f"{'Motif':<38} | {'v1 CN / Valid':<14} | {'v2 CN / Valid':<14} | {'v3 CN / Best Shape'}")
    print("-" * 88)
    for label, z, frac in cases:
        _, _, d1 = extract_coordination(z, frac, cell)
        _, _, d2 = extract_coordination_v2(z, frac, cell)
        _, _, d3 = extract_coordination_v3(z, frac, cell)
        c1, c2, c3_info = d1["centers"][0], d2["centers"][0], d3["centers"][0]
        scores = c3_info["shape_scores"]
        best_tpl = min(scores, key=scores.get) if scores else "None"
        best_str = f"{best_tpl} ({scores[best_tpl]:.3f})" if scores else "None"
        print(
            f"{label:<38} | "
            f"CN={c1['cn']:<2d} ({str(c1['shape_valid']):<5}) | "
            f"CN={c2['cn']:<2d} ({str(c2['shape_valid']):<5}) | "
            f"CN={c3_info['cn']:<2d} -> {best_str}"
        )
    print("=" * 88)


if __name__ == "__main__":
    run_demo()
