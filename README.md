# CoordRep-MOF-v3: Periodic Coordination Priors, Carboxylate-C Filtering, and Exact High-CN (`CN = 2..9`) Procrustes Polyhedral Descriptors for MOFs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](pyproject.toml)

**CoordRep-MOF-v3** (`periodic_covalent_radius_direction_procrustes_v3`) is a self-contained Python & PyTorch library for extracting **periodic metal-center coordination geometry descriptors** and **exact $\mathrm{SO}(3)$ unit-direction Procrustes polyhedral shape scores** across coordination numbers $\mathrm{CN} \in \{2, 3, 4, 5, 6, 7, 8, 9\}$ in Metal–Organic Frameworks (MOFs) and periodic coordination polymers.

Across the full **QMOF database** (**20,298 periodic MOF structures, 82,089 metal centers**), upgrading from `v1` ($\mathrm{CN}=2\dots 6$ only, unfiltered covalent cutoffs) and `v2` (carboxylate-C filtered, $\mathrm{CN}=2\dots 6$) to **`v3`** boosts valid polyhedral shape coverage from **81.28% (`v1`) $\to$ 83.10% (`v2`) $\to$ 97.08% (`v3`)** (+12,966 metal centers), lifting `Zr/Hf` SBU shape coverage from **0.00% to 99.56%**, `Cd` / `Zn` / `Cu` to **100.00%**, `U/Th` actinide nodes from **1.72% to 88.51%**, and `Ln` lanthanide nodes from **0.16% to 78.40%**.

---

## Key Highlights & Innovations

1. **Periodic Covalent-Radius Contact Prior (`v1` $\to$ `v3`)**
   - Uses skinless periodic neighbor enumeration (`ase.neighborlist.neighbor_list`) with Cordero (2008) covalent radii ($d \le 1.25(r_M + r_D) + 10^{-8}$ Å) to retain all qualifying periodic donor images `(site, [ix, iy, iz])` without arbitrary top-$k$ truncation or oxidation-state heuristics.
   - Isolates angular stereochemistry from radial bond-length statistics by projecting donor displacements onto unit vectors $\hat{\mathbf{u}}_i = \mathbf{r}_i / \lVert\mathbf{r}_i\rVert$ centered at the **metal atom origin** (never translating by the donor centroid).

2. **Chelating Carboxylate-Carbon Artifact Filter (`v2` & `v3`)**
   - In $\kappa^2$-chelating carboxylate MOF linkers (`-COO⁻` coordinated to $\mathrm{Co}^{2+}$, $\mathrm{Cd}^{2+}$, $\mathrm{Ln}^{3+}$, or alkaline-earth centers), the $sp^2$ carboxylate carbon ($\mathrm{C}$, $r_{\mathrm{cov}} = 0.76$ Å) frequently lies at $2.45\text{--}2.85$ Å from the metal—inside the $1.25(r_M + r_C)$ sphere—inflating true $\mathrm{CN}=6$ octahedra to $\mathrm{CN}=7,8$ and $\mathrm{CN}=8,9$ lanthanide polyhedra to $\mathrm{CN}=11\dots 13$.
   - **Rule (`D-006`)**: Any accepted carbon contact ($Z=6$) on a metal center whose Euclidean distance to an accepted oxygen or nitrogen contact ($Z \in \{7, 8\}$) of the *same* metal center satisfies $d(\mathrm{C}, \mathrm{O/N}) < 1.45$ Å is automatically filtered out as a chelating backbone carbon, eliminating **26,475 false M–C contacts** across QMOF while preserving true organometallic carbon donors ($\mathrm{M{-}CN}$, $\mathrm{M{-}CO}$, $\mathrm{M{-}C{\equiv}C}$).

3. **Exact Polyhedral Shape Templates for High Coordination Numbers ($\mathrm{CN}=7, 8, 9$ in `v3`)**
   - Extends exact $\mathrm{SO}(3)$ unit-direction Procrustes matching from $\mathrm{CN} \in \{2\dots 6\}$ to $\mathrm{CN} \in \{2\dots 9\}$ with canonical two-template polyhedral pairs for high-CN MOF secondary building units (SBUs):
     - $\mathrm{CN}=7$: **`PBPY7`** ($D_{5h}$ Pentagonal Bipyramid) vs **`OCF7`** ($C_{3v}$ Monocapped Octahedron)
     - $\mathrm{CN}=8$: **`SAPR8`** ($D_{4d}$ Archimedean Square Antiprism) vs **`CU8`** ($O_h$ Cube)
     - $\mathrm{CN}=9$: **`TCTPR9`** ($D_{3h}$ Tricapped Trigonal Prism) vs **`CSAPR9`** ($C_{4v}$ Monocapped Square Antiprism)

4. **Exact Symmetry-Quotient Reduction ($S_N / G_{\mathrm{O}(3)}$) & Closed-Form $3\times 3$ Singular Values**
   - Exhaustive permutation matching at $\mathrm{CN}=9$ requires $9! = 362{,}880$ SVD evaluations per template per center. Upstream molecular `CoordRep` worked around $\mathrm{CN} > 6$ by sorting inter-vertex distance rows and keeping only $m=40$ greedy candidate permutations—a heuristic that is **not** guaranteed to find the global Procrustes minimum under anisotropic distortion.
   - Instead of heuristics, `CoordRep-MOF-v3` exploits the exact orthogonal point-group symmetry $G_{\mathrm{O}(3)} \subset S_N$ of each achiral polyhedron, reducing the permutation search to left coset representatives $S_N / G_{\mathrm{O}(3)}$ (**zero approximation error**, up to $48\times$ fewer permutations) and evaluating the batched $3\times 3$ singular-value sums via Viète's closed-form trigonometric eigenvalue equation (~$50\times$ faster than batched `cuSOLVER` SVD on GPU).

---

## Mathematical Formulation

### 1. Metal-Centered Unit-Direction Procrustes Score

Let $U \in \mathbb{R}^{N \times 3}$ be the row matrix of unit donor direction vectors $\hat{\mathbf{u}}_i = (\mathbf{r}_i - \mathbf{r}_M)/\lVert\mathbf{r}_i - \mathbf{r}_M\rVert$ around a metal center with coordination number $N = \mathrm{CN}$, and let $V \in \mathbb{R}^{N \times 3}$ be an ideal unit-sphere polyhedral template ($\lVert U\rVert_F^2 = \lVert V\rVert_F^2 = N$). Optimizing over all vertex permutations $\Pi \in S_N$, proper rotations $R \in \mathrm{SO}(3)$, and nonnegative uniform scale $s \ge 0$:

$$
S(U, V) = 100 \left( 1 - \frac{\max(c^{\ast}, 0)^2}{N^2} \right) \in [0, 100], \qquad c^{\ast} = \max_{\Pi \in S_N} \max_{R \in \mathrm{SO}(3)} \mathrm{Tr}\left( U^{\top} \Pi V R \right)
$$

where $S(U, V) = 0$ if and only if the donor directions match template $V$ up to permutation, proper rotation, and independent radial bond lengths.

### 2. Symmetry-Quotient Reduction ($S_N / G_{\mathrm{O}(3)}$)

Let $G_{\mathrm{O}(3)}(V) = \{ g \in S_N : \exists\, Q_g \in \mathrm{O}(3) \text{ s.t. } P_g V = V Q_g \}$ be the orthogonal vertex-permutation symmetry group of template $V$. For any $g \in G_{\mathrm{O}(3)}(V)$, the $3\times 3$ cross-covariance matrix $H(\Pi) = (\Pi V)^{\top} U$ satisfies:

$$
H(\Pi P_g) = Q_g^{\top} H(\Pi)
$$

Since orthogonal left-multiplication preserves singular values, $\sigma_k(H(\Pi P_g)) = \sigma_k(H(\Pi))$ for $k=1,2,3$. Moreover, because all 6 high-CN templates (`PBPY7`, `OCF7`, `SAPR8`, `CU8`, `TCTPR9`, `CSAPR9`) are achiral (possessing improper reflections $Q_{\mathrm{ref}} \in \mathrm{O}(3)$ with $\det(Q_{\mathrm{ref}}) = -1$), every $\mathrm{O}(3)$ coset $\Pi G_{\mathrm{O}(3)}$ contains both $\det \ge 0$ and $\det \le 0$ representatives. Consequently, the maximum over the entire coset under $R \in \mathrm{SO}(3)$ is **strictly equal** to the sum of the three singular values of a single coset representative $\bar{\Pi} \in S_N / G_{\mathrm{O}(3)}$:

$$
c^{\ast} = \max_{\bar{\Pi} \in S_N / G_{\mathrm{O}(3)}(V)} \sum_{k=1}^{3} \sigma_k\left( (\bar{\Pi} V)^{\top} U \right)
$$

| CN | Template | Geometry | Point Group | Full $N!$ | $\lvert G_{\mathrm{O}(3)}\rvert$ | Exact Cosets $\lvert S_N / G_{\mathrm{O}(3)}\rvert$ | Reduction Factor |
|:---:|:---|:---|:---:|---:|---:|---:|:---:|
| **2** | `L` | Linear | $D_{\infty h}$ | 2 | — | 2 | $1\times$ |
| **3** | `TP` | Trigonal Planar | $D_{3h}$ | 6 | — | 6 | $1\times$ |
| **4** | `Td` / `SP` | Tetrahedron / Square Planar | $T_d$ / $D_{4h}$ | 24 | — | 24 | $1\times$ |
| **5** | `TBP` / `SPY` | Trigonal Bipyramid / Square Pyramid | $D_{3h}$ / $C_{4v}$ | 120 | — | 120 | $1\times$ |
| **6** | `Oh` / `TPr` | Octahedron / Trigonal Prism | $O_h$ / $D_{3h}$ | 720 | — | 720 | $1\times$ |
| **7** | `PBPY7` | Pentagonal Bipyramid | $D_{5h}$ | 5,040 | 20 | **252** | **$20\times$** |
| **7** | `OCF7` | Monocapped Octahedron | $C_{3v}$ | 5,040 | 6 | **840** | **$6\times$** |
| **8** | `SAPR8` | Archimedean Square Antiprism | $D_{4d}$ | 40,320 | 16 | **2,520** | **$16\times$** |
| **8** | `CU8` | Cube | $O_h$ | 40,320 | 48 | **840** | **$48\times$** |
| **9** | `TCTPR9` | Tricapped Trigonal Prism | $D_{3h}$ | 362,880 | 12 | **30,240** | **$12\times$** |
| **9** | `CSAPR9` | Monocapped Square Antiprism | $C_{4v}$ | 362,880 | 8 | **45,360** | **$8\times$** |

### 3. Analytical $3\times 3$ Singular-Value Sum via Viète's Trigonometric Formula

For each batched $3\times 3$ covariance matrix $H$, its singular values are $\sigma_k(H) = \sqrt{\lambda_k(M)}$, where $M = H^{\top} H$ is a $3\times 3$ symmetric positive semidefinite matrix. Setting $q = \mathrm{Tr}(M)/3$, $B = M - qI$, $p = \sqrt{\mathrm{Tr}(B^2)/6}$, and $\phi = \frac{1}{3}\arccos\left(\mathrm{clip}\left(\frac{\det B}{2 p^3}, -1, 1\right)\right)$, the three eigenvalues of $M$ are obtained in closed form without iterative SVD:

$$
\lambda_1 = q + 2p\cos(\phi), \qquad \lambda_3 = q + 2p\cos\left(\phi + \frac{2\pi}{3}\right), \qquad \lambda_2 = 3q - \lambda_1 - \lambda_3
$$

---

## QMOF Full-Database Benchmark (20,298 Structures, 82,089 Metal Centers)

All tables below are reproduced in [`benchmarks/qmof_20298_audit/`](benchmarks/qmof_20298_audit/) from a strictly label-free geometric audit across the 20,298 QMOF structures.

### 1. Coverage Progression Across Cohorts (`v1` $\to$ `v2` $\to$ `v3`)

| Cohort | Structures | Metal Centers | `v1` `shape_valid` | `v2` `shape_valid` | `v3` `shape_valid` | All-Centers-Valid Structures (`v1` $\to$ `v2` $\to$ `v3`) |
|:---|---:|---:|---:|---:|---:|:---:|
| **All QMOF** | 20,298 | 82,089 | 66,724 (**81.28%**) | 68,214 (**83.10%**) | **79,690 (97.08%)** | 15,955 $\to$ 16,408 $\to$ **19,546 (96.29%)** |
| `original_supported` ($\mathrm{CN}_{v1} \in [2,6]$) | 15,955 | 63,952 | 63,952 (100.0%) | 63,952 (100.0%) | **63,952 (100.0%)** | 15,955 $\to$ 15,955 $\to$ 15,955 |
| `newly_added` (High-CN / SBU expansion) | 4,343 | 18,137 | 2,772 (15.28%) | 4,262 (23.50%) | **15,738 (86.77%)** | 0 $\to$ 453 $\to$ **3,591 (82.68%)** |

### 2. Coverage by Metal Class (`v1` $\to$ `v2` $\to$ `v3`)

| Metal Class | Centers | Mean CN (`v1` $\to$ `v2/v3`) | `v1` `shape_valid` | `v2` `shape_valid` | `v3` `shape_valid` | Dominant High-CN Polyhedral Motif (`v3`) |
|:---|---:|:---:|---:|---:|---:|:---|
| **Zr / Hf** | 2,041 | $8.08 \to 8.01$ | 0.00% | 0.20% | **99.56%** | $\mathrm{CN}=8$: **99.85% `SAPR8`** (UiO-66/MOF-808 SBU, mean score `1.80`) |
| **Cd** | 8,821 | $5.95 \to 5.86$ | 83.79% | 89.85% | **100.00%** | $\mathrm{CN}=7$: **83.03% `PBPY7`** (mean score `3.86`) |
| **Zn** | 21,922 | $4.43 \to 4.43$ | 99.96% | 99.98% | **100.00%** | $\mathrm{CN}=4,6$ (`Td` / `Oh`) + 4 $\mathrm{CN}=7$ centers |
| **Cu** | 12,355 | $4.13 \to 4.12$ | 99.91% | 99.97% | **100.00%** | $\mathrm{CN}=4,5,6$ (`SP` / `SPY` / `Oh`) |
| **Co** | 3,404 | $5.46 \to 5.37$ | 92.83% | 98.71% | **99.82%** | 200 carboxylate-C artifacts fixed in `v2` + 38 $\mathrm{CN}=7$ in `v3` |
| **Li/Na/Mg/Ca/Be** | 4,060 | $5.82 \to 5.66$ | 77.68% | 80.34% | **99.29%** | $\mathrm{CN}=7$ (`PBPY7` 58%), $\mathrm{CN}=8$ (`SAPR8` 81%) |
| **other TM / $p$-block** | 19,934 | $4.91 \to 4.84$ | 93.33% | 95.66% | **99.19%** | +463 centers via `v2` filter, +705 via `v3` $\mathrm{CN}=7..9$ |
| **U / Th (Actinides)** | 522 | $9.56 \to 7.82$ | 1.72% | 3.26% | **88.51%** | $\mathrm{CN}=7$: **100.0% `PBPY7`** (uranyl $[\mathrm{UO}_2\mathrm{L}_5]$, mean score `0.97`) |
| **Ln (Lanthanides)** | 6,278 | $12.08 \to 9.02$ | 0.16% | 0.45% | **78.40%** | $\mathrm{CN}=8$: **99.9% `SAPR8`** (`2.74`); $\mathrm{CN}=9$: **62.7% `TCTPR9` / 37.3% `CSAPR9`** (`2.58`) |
| **K/Rb/Cs/Sr/Ba** | 2,752 | $9.74 \to 8.71$ | 4.98% | 10.17% | **71.73%** | +1,837 heavy alkaline-earth / alkali centers covered |

---

## Installation & Quickstart

### Installation

```bash
git clone https://github.com/FHRMl/CoordRep-MOF-v3.git
cd CoordRep-MOF-v3
pip install -e .
```

### Run Unit Tests & High-CN MOF Demo

```bash
python -m unittest discover -s tests -v
python examples/demo_high_cn_mof.py
```

Example output from `examples/demo_high_cn_mof.py`:

```text
========================================================================================
Motif                                  | v1 CN / Valid  | v2 CN / Valid  | v3 CN / Best Shape
----------------------------------------------------------------------------------------
ZrO8 (UiO-66 SBU, ideal SAPR-8)        | CN=8  (False)  | CN=8  (False)  | CN=8  -> SAPR8 (0.000)
UO7 (Uranyl node, ideal PBPY-7)        | CN=7  (False)  | CN=7  (False)  | CN=7  -> PBPY7 (0.000)
LaO9 + 3 Carboxylate-C (ideal TCTPR-9) | CN=12 (False)  | CN=9  (False)  | CN=9  -> TCTPR9 (0.000)
========================================================================================
```

### Python API Example

```python
import numpy as np
from coordrep_mof import extract_coordination_v3, shape_scores_v3, TEMPLATE_DIRECTIONS_V3

# 1. Direct Procrustes shape scoring on [CN, 3] donor displacement vectors (CN in 2..9)
vectors_zr8 = 2.18 * TEMPLATE_DIRECTIONS_V3["SAPR8"]
scores = shape_scores_v3(vectors_zr8)
print(scores)  # {'SAPR8': 0.0, 'CU8': 19.6562...}

# 2. Full periodic crystal extraction: returns features [N, 36], quality [N, 4], and diagnostics
cell = np.diag([20.0, 20.0, 20.0])
origin = np.array([[10.0, 10.0, 10.0]])
frac = np.vstack([origin, origin + vectors_zr8]) / 20.0
z = [40] + [8] * 8

coord_fea, quality, diag = extract_coordination_v3(z, frac, cell)
print("Center CN:", diag["centers"][0]["cn"])
print("Shape valid:", diag["centers"][0]["shape_valid"])
print("Shape scores:", diag["centers"][0]["shape_scores"])
```

---

## 36-Dimensional Coordination Feature Schema (`cgc_coord_v3`)

| Columns | Block | Description | Active Condition (`valid_coordination_channels_v3`) |
|:---:|:---:|:---|:---|
| `0:9` | **A** | Coordination number one-hot for $\mathrm{CN} \in \{2,3,4,5,6,7,8,9,\ge 10\}$ | Metal site with `coord_valid == 1` |
| `9` | **A** | Normalized scalar $\min(\mathrm{CN}, 12) / 12$ | Metal site with `coord_valid == 1` |
| `10` | **A** | Low-CN indicator $\mathbf{1}(\mathrm{CN} \le 1)$ | Metal site with `coord_valid == 1` |
| `11:19` | **B** | Donor element fraction histogram (`O, N, S, P, C, F, Cl, Other`) | Metal site with `coord_valid == 1` |
| `19` | **B** | Donor element diversity $N_{\mathrm{unique}} / \mathrm{CN}$ | Metal site with `coord_valid == 1` |
| `20` | **C** | Binary indicator `shape_valid` ($\mathbf{1}(\mathrm{CN} \in \{2\dots 9\})$) | Raw binary indicator (unscaled) |
| `21:29` | **C** | $\exp(-0.1\,S_t)$ for $\mathrm{CN}=2\dots 6$ templates (`L, TP, Td, SP, TBP, SPY, Oh, TPr`) | `shape_valid == 1` and matching $\mathrm{CN} \in \{2\dots 6\}$ |
| `29` | **C** | Best-fit polyhedral distortion $\log(1 + \min_t S_t)$ across $\mathrm{CN} \in \{2\dots 9\}$ | `shape_valid == 1` ($\mathrm{CN} \in \{2\dots 9\}$) |
| `30` | **C** | Two-template shape margin $\tanh(S_{t_1} - S_{t_0})$ across $\mathrm{CN} \in \{4\dots 9\}$ | `shape_valid == 1` and $\mathrm{CN} \in \{4\dots 9\}$ |
| `31:34` | **R** | Radial bond-length statistics: $\mu(d)$, $\sigma(d)$, $d_{\max}/d_{\min}$ | Metal site with `radial_valid == 1` ($\mathrm{CN} \ge 1$) |
| `34:36` | **D** | Second-order directional tensor invariants: $\lVert Q\rVert_F$, $\det(Q)$ where $Q = \frac{1}{\mathrm{CN}}U^{\top} U - \frac{1}{3}I_3$ | Metal site with `tensor_valid == 1` ($\mathrm{CN} \ge 1$) |

---

## References & Bibliography

If you use **CoordRep-MOF-v3** in your research, please cite the following foundational works:

1. **Continuous Shape Measures (CShM) & Classical Polyhedra ($\mathrm{CN}=2\dots 6$)**:
   - Pinsky, M.; Avnir, D. Continuous Symmetry Measures. 5. The Classical Polyhedra. *Inorg. Chem.* **1998**, *37* (21), 5575–5582. [DOI: 10.1021/ic9804925](https://doi.org/10.1021/ic9804925)
2. **Seven- and Eight-Coordinate Polyhedral Stereochemistry (`PBPY-7`, `OCF-7`, `SAPR-8`, `CU-8`)**:
   - Casanova, D.; Cirera, J.; Llunell, M.; Alemany, P.; Avnir, D.; Alvarez, S. Minimal Distortion Pathways in Polyhedral Rearrangements. *J. Am. Chem. Soc.* **2004**, *126* (6), 1755–1763. [DOI: 10.1021/ja036479n](https://doi.org/10.1021/ja036479n)
3. **Nine-Coordinate Polyhedral Stereochemistry (`TCTPR-9`, `CSAPR-9`)**:
   - Ruiz-Martínez, A.; Casanova, D.; Alvarez, S. Polyhedral Structures with an Odd Number of Vertices: Nine-Coordinate Metal Compounds. *Chem. Eur. J.* **2008**, *14* (4), 1291–1303. [DOI: 10.1002/chem.200701137](https://doi.org/10.1002/chem.200701137)
4. **Orthogonal Procrustes & $\mathrm{SO}(3)$ Optimal Rotation**:
   - Schönemann, P. H. A Generalized Solution of the Orthogonal Procrustes Problem. *Psychometrika* **1966**, *31* (1), 1–10. [DOI: 10.1007/BF02289451](https://doi.org/10.1007/BF02289451)
   - Kabsch, W. A Solution for the Best Rotation to Relate Two Sets of Vectors. *Acta Crystallogr. Sect. A* **1976**, *32* (5), 922–923. [DOI: 10.1107/S0567739476001873](https://doi.org/10.1107/S0567739476001873)
5. **Closed-Form $3\times 3$ Symmetric Eigenvalues (Viète Trigonometric Method)**:
   - Smith, O. K. Eigenvalues of a Symmetric $3 \times 3$ Matrix. *Commun. ACM* **1961**, *4* (4), 168. [DOI: 10.1145/355578.366316](https://doi.org/10.1145/355578.366316)
6. **Molecular Coordination Representation & Periodic Neighbor Priors**:
   - Luo, W.-L.; Li, C.-H. *CoordRep: A Canonical, Continuous, and Compositional Representation for Machine Learning in Coordination Chemistry*; GitHub repository, **2026**. [https://github.com/1701NJU/CoordRep](https://github.com/1701NJU/CoordRep)
   - Waroquiers, D.; George, J.; Horton, M.; Schenk, S.; Persson, K. A.; Rignanese, G.-M.; Gonze, X.; Hautier, G. ChemEnv: A Fast and Robust Coordination Environment Identification Tool. *Acta Crystallogr. Sect. B* **2020**, *76* (4), 683–695. [DOI: 10.1107/S2052520620007994](https://doi.org/10.1107/S2052520620007994)
   - Cordero, B.; Gómez, V.; Platero-Prats, A. E.; Revés, M.; Echeverría, J.; Cremades, E.; Barragán, F.; Alvarez, S. Covalent Radii Revisited. *Dalton Trans.* **2008**, (21), 2832–2838. [DOI: 10.1039/B801115J](https://doi.org/10.1039/B801115J)
   - Hjorth Larsen, A.; et al. The Atomic Simulation Environment—A Python Library for Working with Atoms. *J. Phys.: Condens. Matter* **2017**, *29* (27), 273002. [DOI: 10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)
7. **Crystal Graph Neural Networks & QMOF Benchmark Database**:
   - Xie, T.; Grossman, J. C. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. *Phys. Rev. Lett.* **2018**, *120* (14), 145301. [DOI: 10.1103/PhysRevLett.120.145301](https://doi.org/10.1103/PhysRevLett.120.145301)
   - Rosen, A. S.; Iyer, S. M.; Ray, D.; Yao, Z.; Aspuru-Guzik, A.; Gagliardi, L.; Notestein, J. M.; Snurr, R. Q. Machine Learning the Quantum-Chemical Properties of Metal–Organic Frameworks for Accelerated Materials Discovery. *Matter* **2021**, *4* (5), 1578–1597. [DOI: 10.1016/j.matt.2021.02.015](https://doi.org/10.1016/j.matt.2021.02.015)
   - Rosen, A. S.; Fung, V.; Huck, P.; O'Donnell, C. T.; Horton, M. K.; Truhlar, D. G.; Persson, K. A.; Notestein, J. M.; Snurr, R. Q. High-Throughput Predictions of Metal–Organic Framework Electronic Properties: Theoretical Challenges, Graph Neural Networks, and Data Exploration. *npj Comput. Mater.* **2022**, *8*, 112. [DOI: 10.1038/s41524-022-00796-6](https://doi.org/10.1038/s41524-022-00796-6)

### BibTeX

```bibtex
@article{pinsky1998continuous,
  author  = {Pinsky, Miquel and Avnir, David},
  title   = {Continuous Symmetry Measures. 5. The Classical Polyhedra},
  journal = {Inorganic Chemistry},
  year    = {1998},
  volume  = {37},
  number  = {21},
  pages   = {5575--5582},
  doi     = {10.1021/ic9804925}
}

@article{casanova2004minimal,
  author  = {Casanova, David and Cirera, Jordi and Llunell, Miquel and Alemany, Pere and Avnir, David and Alvarez, Santiago},
  title   = {Minimal Distortion Pathways in Polyhedral Rearrangements},
  journal = {Journal of the American Chemical Society},
  year    = {2004},
  volume  = {126},
  number  = {6},
  pages   = {1755--1763},
  doi     = {10.1021/ja036479n}
}

@article{ruiz2008polyhedral,
  author  = {Ruiz-Mart{\'i}nez, Antonio and Casanova, David and Alvarez, Santiago},
  title   = {Polyhedral Structures with an Odd Number of Vertices: Nine-Coordinate Metal Compounds},
  journal = {Chemistry -- A European Journal},
  year    = {2008},
  volume  = {14},
  number  = {4},
  pages   = {1291--1303},
  doi     = {10.1002/chem.200701137}
}

@article{cordero2008covalent,
  author  = {Cordero, Beatriz and G{\'o}mez, Ver{\'o}nica and Platero-Prats, Ana E. and Rev{\'e}s, Marc and Echeverr{\'i}a, Jorge and Cremades, Eduard and Barrag{\'a}n, Flavia and Alvarez, Santiago},
  title   = {Covalent radii revisited},
  journal = {Dalton Transactions},
  year    = {2008},
  number  = {21},
  pages   = {2832--2838},
  doi     = {10.1039/B801115J}
}

@article{waroquiers2020chemenv,
  author  = {Waroquiers, David and George, Janine and Horton, Matthew and Schenk, Stephan and Persson, Kristin A. and Rignanese, Gian-Marco and Gonze, Xavier and Hautier, Geoffroy},
  title   = {ChemEnv: a fast and robust coordination environment identification tool},
  journal = {Acta Crystallographica Section B},
  year    = {2020},
  volume  = {76},
  number  = {4},
  pages   = {683--695},
  doi     = {10.1107/S2052520620007994}
}

@misc{luo2026coordrep,
  author       = {Luo, Wen-Lin and Li, Cheng-Hui},
  title        = {CoordRep: A Canonical, Continuous, and Compositional Representation for Machine Learning in Coordination Chemistry},
  year         = {2026},
  howpublished = {\url{https://github.com/1701NJU/CoordRep}}
}

@article{rosen2021machine,
  author  = {Rosen, Andrew S. and Iyer, Shaurya M. and Ray, Debmalya and Yao, Zhenpeng and Aspuru-Guzik, Al{\'a}n and Gagliardi, Laura and Notestein, Justin M. and Snurr, Randall Q.},
  title   = {Machine Learning the Quantum-Chemical Properties of Metal--Organic Frameworks for Accelerated Materials Discovery},
  journal = {Matter},
  year    = {2021},
  volume  = {4},
  number  = {5},
  pages   = {1578--1597},
  doi     = {10.1016/j.matt.2021.02.015}
}

@article{rosen2022high,
  author  = {Rosen, Andrew S. and Fung, Victor and Huck, Patrick and O'Donnell, Cody T. and Horton, Matthew K. and Truhlar, Donald G. and Persson, Kristin A. and Notestein, Justin M. and Snurr, Randall Q.},
  title   = {High-throughput predictions of metal--organic framework electronic properties: theoretical challenges, graph neural networks, and data exploration},
  journal = {npj Computational Materials},
  year    = {2022},
  volume  = {8},
  pages   = {112},
  doi     = {10.1038/s41524-022-00796-6}
}
```
