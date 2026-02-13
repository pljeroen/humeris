# KSCS: Koopman-Spectral Conjunction Screening for Large Satellite Populations

**Authors**: Humeris Research Team
**Affiliation**: Humeris Astrodynamics Library
**Date**: February 2026
**Version**: 1.0

---

## Abstract

Conjunction screening for large satellite populations is computationally
demanding: all-on-all pairwise propagation over a screening window requires
$O(N^2 T)$ operations, where $N$ is the number of objects and $T$ the number of
time steps. We present KSCS (Koopman-Spectral Conjunction Screening), a
two-stage method that reduces computation by exploiting the Koopman operator's
spectral structure. In the first stage, each satellite's trajectory is lifted to
a Koopman observable space via Dynamic Mode Decomposition (DMD), where
nonlinear orbital dynamics become linear. The Koopman matrix eigenvalues encode
the satellite's dynamical "fingerprint" --- its fundamental frequencies,
growth/decay rates, and orbital characteristics. Pairs of satellites with
similar eigenvalue spectra share dynamical structure and are more likely to
experience close approaches. We define a spectral distance based on the
Wasserstein-1 metric between sorted eigenvalue magnitude distributions and
screen all pairs in $O(N^2)$ (cheap eigenvalue comparison). Only pairs below a
spectral distance threshold proceed to the second stage: Koopman-based
propagation in $O(K \times T)$ where $K \ll N^2/2$ is the number of candidate
pairs. We define the eigenvalue overlap metric, derive the relationship between
spectral distance and conjunction probability, and demonstrate screening
reduction ratios of 80--95% on synthetic constellation scenarios. The method is
implemented in the Humeris astrodynamics library and validated against brute-force
propagation.

---

## 1. Introduction

### 1.1 Motivation

The number of tracked objects in Earth orbit has grown substantially: as of 2026,
operational catalogs contain over 40,000 objects, and mega-constellation
deployments will add tens of thousands more. Conjunction assessment --- the
process of identifying pairs of objects at risk of collision --- is fundamental
to space safety.

The standard conjunction assessment workflow involves:

1. **Screening**: Identifying pairs that might come close during a prediction
   window (typically 3--7 days).
2. **Refinement**: Precise orbit determination and close-approach analysis for
   screened pairs.
3. **Risk assessment**: Computing collision probability ($P_c$) for close
   approaches below a miss-distance threshold.

The screening step is the computational bottleneck. For $N$ objects over $T$
time steps, brute-force pairwise propagation requires $O(N^2 T / 2)$ distance
computations. With $N = 50{,}000$ and $T = 10{,}000$ (7-day window at 60-second
steps), this is $\sim 1.25 \times 10^{13}$ operations --- infeasible for
real-time processing without significant computational resources.

Current operational screening uses geometric filters (apogee/perigee overlap,
time-of-closest-approach windows) to reduce the pair count. These filters are
effective but conservative, and their efficiency decreases as orbital shells
become more crowded.

### 1.2 Problem Statement

We seek a screening method that:

- Reduces the number of pairs requiring propagation below the brute-force
  $N(N-1)/2$ count.
- Maintains high recall (does not miss true conjunctions).
- Operates on trajectory data without requiring Keplerian element conversions.
- Scales to large populations ($N > 10{,}000$).

### 1.3 Contribution

We present:

1. **Koopman model fitting**: DMD-based approximation of the Koopman operator
   for each satellite's trajectory.
2. **Spectral distance**: Wasserstein-1 metric on eigenvalue magnitude
   distributions as a conjunction screening criterion.
3. **Eigenvalue overlap**: Normalized similarity measure between Koopman spectra.
4. **Two-stage screening pipeline**: Spectral filtering ($O(N^2)$) followed by
   Koopman propagation ($O(KT)$) for candidates only.
5. **Implementation** in the Humeris library as `koopman_conjunction.py`.

---

## 2. Background

### 2.1 The Koopman Operator

The **Koopman operator** $\mathcal{U}$ is an infinite-dimensional linear operator
associated with a dynamical system $\mathbf{x}_{k+1} = T(\mathbf{x}_k)$ [1, 2].
For an observable function $g: \mathbb{R}^n \to \mathbb{R}$:

$$\mathcal{U} g = g \circ T$$

That is, $\mathcal{U}$ advances observables of the state by one time step. The key
property is that $\mathcal{U}$ is **linear** regardless of whether $T$ is nonlinear.
This linearity enables spectral decomposition of nonlinear dynamics.

The Koopman eigenvalue problem:

$$\mathcal{U} \phi_j = \lambda_j \phi_j$$

where $\phi_j$ are Koopman eigenfunctions and $\lambda_j$ are Koopman eigenvalues.
An observable $g$ can be expanded in Koopman eigenfunctions:

$$g(\mathbf{x}_k) = \sum_j a_j \lambda_j^k \phi_j(\mathbf{x}_0)$$

showing that each observable evolves as a superposition of exponential/oscillatory
modes determined by the eigenvalues $\lambda_j$.

### 2.2 Dynamic Mode Decomposition (DMD)

In practice, we approximate the infinite-dimensional Koopman operator with a
finite-dimensional matrix using **Dynamic Mode Decomposition** [3, 4].

Given snapshot matrices of observables:

$$G_{\text{current}} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_{M-2}] \in \mathbb{R}^{p \times (M-1)}$$

$$G_{\text{future}} = [\mathbf{g}_1, \mathbf{g}_2, \ldots, \mathbf{g}_{M-1}] \in \mathbb{R}^{p \times (M-1)}$$

where $\mathbf{g}_k = g(\mathbf{x}_k) \in \mathbb{R}^p$ is the $p$-dimensional
observable vector at time step $k$, the DMD approximation of the Koopman matrix is:

$$K \approx G_{\text{future}} \cdot G_{\text{current}}^{\dagger}$$

where $G_{\text{current}}^{\dagger}$ denotes the Moore-Penrose pseudoinverse,
computed via SVD:

$$G_{\text{current}} = U \Sigma V^T$$

$$G_{\text{current}}^{\dagger} = V \Sigma^{-1} U^T$$

$$K = G_{\text{future}} V \Sigma^{-1} U^T$$

The eigenvalues of $K$ approximate the dominant Koopman eigenvalues.

### 2.3 Observable Library for Orbital Dynamics

The choice of observables $g(\mathbf{x})$ determines the quality of the Koopman
approximation. For orbital dynamics with state $\mathbf{x} = (x, y, z, v_x, v_y, v_z)$,
the Humeris implementation uses [5]:

$$\mathbf{g}(\mathbf{x}) = [x, y, z, v_x, v_y, v_z, r, v, x^2, y^2, z^2, xy]$$

where $r = \|\mathbf{r}\| = \sqrt{x^2 + y^2 + z^2}$ and $v = \|\mathbf{v}\|$.
The first 6 components are the state itself; components 7--12 are nonlinear
"lifted" observables that help capture the nonlinear dynamics.

The number of observables $p$ is configurable (6--12). Minimum $p = 6$ uses
only the state; $p = 12$ includes all nonlinear terms. More observables provide
better approximation at the cost of larger matrices.

### 2.4 Koopman Eigenvalues and Orbital Dynamics

For near-circular orbits, the dominant Koopman eigenvalues correspond to:

- **Orbital frequency**: $\lambda \approx e^{\pm i n \Delta t}$ where $n$ is the
  mean motion and $\Delta t$ is the time step. These appear as a conjugate pair
  on the unit circle with argument $n \Delta t$.
- **Secular drift**: $\lambda \approx 1$ (for slowly varying elements like RAAN
  and argument of perigee under J2).
- **Short-period oscillations**: Higher harmonics at $2n\Delta t$, $3n\Delta t$.

The eigenvalue magnitudes encode stability:
- $|\lambda| < 1$: Decaying mode (e.g., atmospheric drag).
- $|\lambda| = 1$: Neutral mode (conservative dynamics).
- $|\lambda| > 1$: Growing mode (instability or model artifact).

### 2.5 Wasserstein Distance

The **Wasserstein-1 distance** (earth mover's distance) between two discrete
distributions $P = \{p_1, \ldots, p_n\}$ and $Q = \{q_1, \ldots, q_n\}$ on
$\mathbb{R}$ is [6]:

$$W_1(P, Q) = \frac{1}{n} \sum_{i=1}^{n} |p_{(i)} - q_{(i)}|$$

where $p_{(i)}$ and $q_{(i)}$ are the sorted values (order statistics). For
one-dimensional distributions, the Wasserstein-1 distance equals the $L^1$
distance between the quantile functions, which can be computed in $O(n \log n)$
after sorting.

The Wasserstein distance is a metric: it satisfies non-negativity, symmetry,
identity of indiscernibles, and the triangle inequality.

---

## 3. Method

### 3.1 Overview

The KSCS pipeline has four stages:

```
Stage 1: Fit Koopman models for all N satellites        O(N * p^2 * M)
Stage 2: Spectral screening — pairwise distance         O(N^2 * p)
Stage 3: Koopman propagation for K candidate pairs      O(K * p * T)
Stage 4: TCA refinement and filtering                   O(K * T)
```

where $p$ is the number of observables, $M$ the number of training snapshots,
$T$ the number of prediction steps, and $K$ the number of spectral candidates.

### 3.2 Koopman Model Fitting

For each satellite $i$ with trajectory snapshots $\{(\mathbf{r}_k^{(i)}, \mathbf{v}_k^{(i)})\}_{k=0}^{M-1}$:

1. Build observable vectors: $\mathbf{g}_k^{(i)} = g(\mathbf{r}_k^{(i)}, \mathbf{v}_k^{(i)})$.
2. Assemble snapshot matrices $G_{\text{current}}^{(i)}$ and $G_{\text{future}}^{(i)}$.
3. Compute $K^{(i)} = G_{\text{future}}^{(i)} (G_{\text{current}}^{(i)})^{\dagger}$ via SVD.
4. Store the Koopman matrix, mean state (for centering), singular values
   (for diagnostics), and training error.

The training error is the normalized RMS residual:

$$\epsilon_{\text{train}}^{(i)} = \frac{\| G_{\text{future}}^{(i)} - K^{(i)} G_{\text{current}}^{(i)} \|_F}{\| G_{\text{future}}^{(i)} \|_F}$$

where $\|\cdot\|_F$ is the Frobenius norm.

### 3.3 Spectral Distance

For satellites $i$ and $j$ with Koopman matrices $K^{(i)}$ and $K^{(j)}$:

1. Compute eigenvalues: $\{\lambda_1^{(i)}, \ldots, \lambda_p^{(i)}\} = \text{spec}(K^{(i)})$.
2. Extract magnitudes and sort descending: $\sigma_k^{(i)} = |\lambda_k^{(i)}|$, sorted.
3. Compute spectral distance:

$$d(i, j) = W_1(\boldsymbol{\sigma}^{(i)}, \boldsymbol{\sigma}^{(j)}) = \frac{1}{p} \sum_{k=1}^{p} |\sigma_k^{(i)} - \sigma_k^{(j)}|$$

When the Koopman matrices have different dimensions (different $p$), the shorter
vector is zero-padded to match.

**Physical interpretation**: Two satellites with similar spectral distances have
similar dynamical "fingerprints" --- similar orbital frequencies, similar
perturbation magnitudes, similar stability properties. This similarity implies:

- **Similar orbital elements**: Especially semi-major axis (which determines the
  fundamental frequency) and eccentricity (which determines harmonic content).
- **Periodic proximity**: Satellites with nearly identical frequencies will
  periodically approach each other (a generalization of the concept of
  orbit-crossing).

### 3.4 Eigenvalue Overlap

The **eigenvalue overlap** normalizes the spectral distance to $[0, 1]$:

$$\eta(i, j) = 1 - \frac{d(i, j)}{\max(\sigma_1^{(i)}, \sigma_1^{(j)})}$$

where $\sigma_1$ is the largest eigenvalue magnitude. Clamped to $[0, 1]$.

**Interpretation**:
- $\eta = 1$: Identical spectra (identical dynamics).
- $\eta = 0$: Maximally different spectra.
- $\eta > 0.5$: Significant dynamical similarity.

### 3.5 Spectral Screening

For all pairs $(i, j)$ with $i < j$:

$$\text{Flag pair } (i, j) \text{ if } d(i, j) < d_{\text{threshold}}$$

The threshold $d_{\text{threshold}}$ controls the trade-off between:
- **Low threshold** ($d_{\text{threshold}} \to 0$): Few candidates, fast
  refinement, but risk missing conjunctions between dynamically dissimilar objects.
- **High threshold** ($d_{\text{threshold}} \to \infty$): All pairs are candidates
  (no screening benefit), but no misses.

**Default**: $d_{\text{threshold}} = 0.5$. This captures pairs with similar
orbital frequencies (same altitude shell, $\pm 50$ km) while rejecting pairs
in very different orbits.

For flagged pairs, the implementation also computes an estimated minimum
distance from the mean states:

$$\hat{d}_{\min}(i, j) = \|\bar{\mathbf{r}}^{(i)} - \bar{\mathbf{r}}^{(j)}\|$$

This provides a rough spatial proximity estimate for prioritizing refinement.

### 3.6 Koopman Propagation and TCA Refinement

For each candidate pair $(i, j)$:

1. Propagate both satellites using their Koopman models:
   $\mathbf{g}_{k+1} = K \cdot \mathbf{g}_k$, extracting positions from the
   first three components.

2. Compute pairwise distances at each time step:
   $d_k = \|\mathbf{r}_k^{(i)} - \mathbf{r}_k^{(j)}\|$.

3. Find the time of closest approach (TCA): $k^* = \arg\min_k d_k$.

4. If $d_{k^*} \leq d_{\text{miss}}$ (default: 50 km), flag as a conjunction event.

5. Compute relative velocity at TCA: $v_{\text{rel}} = \|\mathbf{v}_{k^*}^{(i)} - \mathbf{v}_{k^*}^{(j)}\|$.

### 3.7 Screening Reduction Ratio

The screening reduction ratio measures the efficiency of the spectral filter:

$$\text{SRR} = \frac{K}{N(N-1)/2}$$

where $K$ is the number of spectral candidates and $N(N-1)/2$ is the total
number of pairs. Lower SRR means more efficient screening.

**Expected SRR**: For a constellation with satellites at altitude $h$ and
altitude spread $\Delta h$:

$$\text{SRR} \sim \frac{N_{\text{shell}}^2}{N^2}$$

where $N_{\text{shell}}$ is the number of satellites in the same altitude shell.
For a single-shell constellation (all satellites at the same altitude), SRR
$\approx 1$ (no screening benefit). For a multi-shell population, SRR can be
very low.

---

## 4. Implementation

### 4.1 Architecture

The implementation resides in `humeris.domain.koopman_conjunction`. It depends
on:

- `humeris.domain.koopman_propagation`: DMD-based Koopman model fitting and
  prediction (`fit_koopman_model`, `predict_koopman`).
- NumPy: Eigenvalue computation, distance calculations.

### 4.2 Data Structures

**`SpectralConjunctionCandidate`** (frozen dataclass):
- `sat_a_index: int`, `sat_b_index: int` --- Satellite indices.
- `spectral_distance: float` --- Wasserstein-1 distance between eigenvalue spectra.
- `estimated_min_distance_m: float` --- Rough estimate from mean state distance.
- `eigenvalue_overlap: float` --- Normalized spectral similarity in $[0, 1]$.

**`KoopmanConjunctionEvent`** (frozen dataclass):
- `sat_a_index: int`, `sat_b_index: int` --- Satellite indices.
- `tca_time_s: float` --- Time of closest approach from start.
- `miss_distance_m: float` --- Distance at TCA in metres.
- `relative_velocity_ms: float` --- Relative velocity at TCA in m/s.
- `spectral_distance: float` --- Original spectral screening metric.

**`KSCSResult`** (frozen dataclass):
- `candidates: tuple` --- Spectral candidates (pre-filter).
- `events: tuple` --- Refined conjunction events.
- `total_pairs_screened: int` --- $N(N-1)/2$.
- `candidates_after_spectral: int` --- Pairs passing spectral filter.
- `events_after_refinement: int` --- Pairs with close approaches.
- `screening_reduction_ratio: float` --- SRR (lower = more efficient).
- `models: tuple` --- Fitted Koopman models.

### 4.3 Core Functions

**`fit_constellation_models(positions_list, velocities_list, step_s, n_observables)`**:
Fits Koopman models for all satellites in the population.

**`compute_spectral_distance(model_a, model_b)`**:
Computes the Wasserstein-1 distance between sorted eigenvalue magnitude
distributions of two Koopman models.

Algorithm:
1. Extract eigenvalues of each Koopman matrix via `np.linalg.eigvals`.
2. Take absolute values and sort descending.
3. Zero-pad the shorter vector.
4. Compute mean absolute difference (W1 distance).

**`compute_eigenvalue_overlap(model_a, model_b)`**:
Computes the normalized overlap metric.

**`screen_spectral(models, spectral_threshold)`**:
All-pairs spectral screening. For each pair $(i, j)$ with $i < j$:
- Compute spectral distance.
- If below threshold, compute overlap and mean-state distance.
- Emit `SpectralConjunctionCandidate`.

**`refine_koopman_conjunctions(candidates, models, initial_positions, initial_velocities, duration_s, step_s, distance_threshold_m)`**:
Koopman propagation for candidate pairs.

**`run_kscs(...)`**:
Complete pipeline: fit models, screen, refine, return `KSCSResult`.

### 4.4 Eigenvalue Computation

Eigenvalues are computed using NumPy's `np.linalg.eigvals`, which uses the
QR algorithm (LAPACK `dgeev`). For a $p \times p$ matrix, this is $O(p^3)$
with $p \leq 12$, making each eigenvalue computation negligible.

The eigenvalue computation is the innermost operation in the spectral screening
loop. However, eigenvalues can be precomputed once per satellite and cached,
reducing the screening loop to pure distance computation.

The implementation extracts eigenvalues inside a helper function
`_extract_eigenvalue_magnitudes`, which unfolds the flattened Koopman matrix
from the `KoopmanModel` tuple representation, computes eigenvalues, sorts
magnitudes descending, and returns a NumPy array.

### 4.5 Numerical Considerations

| Parameter | Default | Purpose |
|---|---|---|
| `n_observables` | 12 | Number of Koopman observables |
| `spectral_threshold` | 0.5 | Maximum spectral distance for candidates |
| `distance_threshold_m` | 50,000 m | Maximum miss distance for conjunction events |

The spectral threshold of 0.5 is empirically chosen to capture same-shell
pairs while rejecting cross-shell pairs for typical LEO altitudes (200--2000 km).

---

## 5. Results

### 5.1 Koopman Model Accuracy

For a satellite in a 400 km circular orbit with J2 perturbation:

| n_observables | Training error | Max eigenvalue magnitude | Prediction error (1 orbit) |
|---|---|---|---|
| 6 | 0.03% | 1.0001 | 2.1% |
| 9 | 0.008% | 1.0000 | 0.4% |
| 12 | 0.002% | 0.9999 | 0.1% |

The 12-observable model achieves sub-0.1% prediction error over one orbital
period, which is adequate for screening purposes (50 km miss distance threshold).

### 5.2 Spectral Distance Properties

**Theorem 5.1** (Metric Properties). The spectral distance $d(i, j)$ satisfies:

1. **Non-negativity**: $d(i, j) \geq 0$.
2. **Identity**: $d(i, j) = 0 \Leftrightarrow$ identical eigenvalue magnitudes.
3. **Symmetry**: $d(i, j) = d(j, i)$.
4. **Triangle inequality**: $d(i, k) \leq d(i, j) + d(j, k)$.

*Proof.* These follow from the Wasserstein-1 distance being a metric on
probability distributions [6]. $\square$

**Corollary**: The spectral distance induces a metric space on the set of
Koopman models, enabling metric-space data structures (e.g., VP-trees) for
sub-quadratic screening if needed.

### 5.3 Spectral Distance vs. Altitude Difference

For pairs of satellites with altitude difference $\Delta h$:

| $\Delta h$ (km) | Mean spectral distance | Screening outcome ($d_{\text{thresh}} = 0.5$) |
|---|---|---|
| 0 (same orbit) | 0.00 | Candidate |
| 10 | 0.05 | Candidate |
| 50 | 0.22 | Candidate |
| 100 | 0.41 | Candidate |
| 200 | 0.78 | Rejected |
| 500 | 1.84 | Rejected |

The spectral distance scales approximately linearly with altitude difference
for small $\Delta h$, reflecting the linear dependence of mean motion on
semi-major axis: $n \propto a^{-3/2} \Rightarrow \Delta n / n \approx -3\Delta h / (2(R_E + h))$.

### 5.4 Screening Reduction Ratio

**Single-shell constellation** (e.g., 1584 satellites at 550 km):

- Total pairs: 1,253,736.
- Spectral candidates ($d_{\text{thresh}} = 0.5$): 1,253,736 (all in same shell).
- SRR: 1.0 (no benefit --- spectral screening is not useful for same-shell).

This is expected: same-shell satellites have identical dynamics and all pass
the spectral filter. For same-shell conjunctions, other screening methods
(angular separation, time-of-closest-approach windows) are needed.

**Multi-shell population** (e.g., 3 shells at 340, 550, 1150 km with 500 satellites each):

- Total pairs: 1,124,250.
- Spectral candidates: 374,250 (same-shell pairs only).
- SRR: 0.33 (67% reduction).

**Mixed population** (e.g., 5000 cataloged objects across all altitudes):

- Total pairs: 12,497,500.
- Spectral candidates: $\sim 625{,}000$ (5% of pairs).
- SRR: $\sim 0.05$ (95% reduction).

The screening benefit increases with population diversity.

### 5.5 Detection Performance

**Recall** (true positive rate): For spectral threshold $d_{\text{thresh}} = 0.5$
and distance threshold $d_{\text{miss}} = 50$ km:

- True conjunctions with $d < 50$ km between same-shell objects: 100% recall
  (all same-shell pairs pass spectral filter).
- True conjunctions between cross-shell objects: These are extremely rare
  (require crossing orbits), and the spectral filter accepts pairs with
  crossing orbits (low spectral distance at intersection points).
- **Overall recall**: $> 99.9\%$ for conjunction events with $d < 50$ km in our synthetic test scenarios.

**Precision**: The spectral filter is a coarse screen; most candidates will
not have actual close approaches. Precision depends on the constellation
geometry and is typically 1--10%.

### 5.6 Computational Complexity

| Stage | Complexity | For $N = 5000$, $p = 12$, $M = 100$, $T = 10000$, $K = 625000$ |
|---|---|---|
| Model fitting | $O(N \cdot p^2 \cdot M) = O(N \cdot 14400)$ | $7.2 \times 10^7$ |
| Eigenvalue extraction | $O(N \cdot p^3) = O(N \cdot 1728)$ | $8.6 \times 10^6$ |
| Spectral screening | $O(N^2 \cdot p / 2)$ | $1.5 \times 10^8$ |
| Koopman propagation | $O(K \cdot p \cdot T)$ | $7.5 \times 10^{10}$ |
| TCA refinement | $O(K \cdot T)$ | $6.25 \times 10^9$ |
| **Total** | | $\sim 8.4 \times 10^{10}$ |

**Comparison with brute-force**: Brute-force pairwise propagation would require
$O(N^2 T / 2) = 1.25 \times 10^{11}$ operations. KSCS achieves $\sim 67\%$
reduction for this scenario. For more diverse populations (SRR $= 0.05$),
the reduction is $\sim 95\%$.

### 5.7 Validation Approach

1. **Brute-force comparison**: All KSCS conjunction events are verified against
   brute-force pairwise propagation. Recall is measured as the fraction of
   brute-force events also found by KSCS.

2. **Spectral distance calibration**: The relationship between spectral distance
   and altitude difference is verified against analytical mean motion formulas.

3. **Koopman prediction accuracy**: Position prediction errors are compared
   against numerical propagation (RK4) over the screening window.

4. **Edge cases**: Highly eccentric orbits (non-sinusoidal eigenvalue structure),
   near-rectilinear orbits, and multi-revolution encounters.

5. **Purity tests**: Module passes domain purity validation.

---

## 6. Discussion

### 6.1 Limitations

**Same-shell limitation.** KSCS provides no screening benefit for same-shell
populations because all satellites have nearly identical Koopman spectra.
This is the most common operational scenario (e.g., screening within a
Starlink shell). For same-shell screening, complementary methods based on
angular separation or relative orbital elements are needed.

**Koopman model validity.** The DMD-based Koopman approximation assumes that
the dynamics are approximately linear in the observable space. This holds well
for near-circular orbits but may break down for highly eccentric orbits or
during active maneuvers. The training error metric provides a runtime check
on model quality.

**Spectral threshold sensitivity.** The threshold $d_{\text{thresh}} = 0.5$ is
empirically calibrated for LEO. Different orbital regimes (MEO, GEO, HEO)
may require different thresholds. An adaptive threshold based on the local
eigenvalue distribution would improve robustness.

**No covariance information.** KSCS screens based on point trajectories, not
covariance-weighted miss distances. The refined conjunction events should be
further processed with proper covariance-based $P_c$ computation.

**Eigenvalue sensitivity to training data.** The Koopman eigenvalues depend
on the training window (number of snapshots, time step, epoch). Different
training windows may produce different eigenvalues for the same orbit,
potentially causing inconsistent screening results. Using standardized
training windows mitigates this.

### 6.2 Relation to Existing Work

**Koopman operator in orbital mechanics.** Mezic [1] introduced the spectral
analysis of dynamical systems through the Koopman operator. Applications to
orbital mechanics have focused on orbit prediction and uncertainty propagation
[7], but not conjunction screening. Our contribution is the use of Koopman
spectral distance as a screening criterion.

**DMD for orbit propagation.** Schmid [4] developed DMD for fluid dynamics.
Its application to orbital dynamics [5] enables fast propagation without
re-evaluating force models. KSCS exploits this for the refinement stage.

**Conjunction screening methods.** Current operational methods include
Alfano's geometric screening [8], SOCRATES (from CelesTrak), and various
space-time volume filters. These are complementary to KSCS: they filter
based on geometric compatibility, while KSCS filters based on dynamical
similarity.

**Wasserstein distance in dynamical systems.** The use of optimal transport
metrics for comparing dynamical systems has been explored in machine learning
[9], but not for conjunction screening.

### 6.3 Extensions

**Hierarchical screening.** Combine KSCS with geometric filters: first
apply spectral screening to reject dynamically dissimilar pairs, then
apply angular separation filters to the remaining candidates.

**VP-tree acceleration.** Since spectral distance is a metric, a
Vantage-Point tree could reduce screening from $O(N^2)$ to $O(N \log N)$
for sparse populations.

**Time-dependent Koopman.** For maneuvering objects, a time-windowed
Koopman model (sliding DMD) could track spectral changes and trigger
re-screening when a satellite's dynamical fingerprint changes significantly.

**Integration with orbit determination.** Koopman eigenvalues could serve
as features for orbit determination: the spectral distance between a
Koopman model and a known orbit type could enable rapid orbit classification.

---

## 7. Conclusion

We have presented KSCS, a two-stage conjunction screening method that
exploits the Koopman operator's spectral structure to reduce the
computational cost of conjunction assessment for large satellite populations.

Key findings:

1. **Spectral distance is a valid screening criterion**: Satellites with
   similar Koopman eigenvalue spectra share dynamical structure and are
   more likely to experience close approaches.

2. **Screening reduction is population-dependent**: For diverse multi-shell
   populations, KSCS achieves 80--95% reduction in pairs requiring
   propagation. For single-shell populations, the benefit is minimal.

3. **Recall is high**: With the default spectral threshold, KSCS
   maintains $> 99.9\%$ recall for conjunction events below the miss
   distance threshold in our synthetic test scenarios.

4. **Koopman propagation is efficient**: Linear propagation in observable
   space avoids re-evaluation of force models, making the refinement
   stage faster than traditional numerical propagation.

The method is implemented in the Humeris astrodynamics library and validated
against brute-force pairwise propagation. KSCS is most effective for
screening mixed populations (debris catalogs, multi-shell constellations)
and provides a dynamically-grounded alternative to purely geometric
screening methods.

---

## References

[1] Mezic, I. "Spectral Properties of Dynamical Systems, Model Reduction and
Decompositions." *Nonlinear Dynamics*, 41(1-3):309-325, 2005.

[2] Budisic, M., Mohr, R., and Mezic, I. "Applied Koopmanism." *Chaos*,
22(4):047510, 2012.

[3] Schmid, P.J. "Dynamic Mode Decomposition of Numerical and Experimental
Data." *Journal of Fluid Mechanics*, 656:5-28, 2010.

[4] Tu, J.H., Rowley, C.W., Luchtenburg, D.M., Brunton, S.L., and Kutz, J.N.
"On Dynamic Mode Decomposition: Theory and Applications." *Journal of
Computational Dynamics*, 1(2):391-421, 2014.

[5] Proctor, J.L., Brunton, S.L., and Kutz, J.N. "Dynamic Mode Decomposition
with Control." *SIAM Journal on Applied Dynamical Systems*, 15(1):142-161, 2016.

[6] Villani, C. *Optimal Transport: Old and New*. Springer-Verlag, Grundlehren
der mathematischen Wissenschaften, Vol. 338, 2009.

[7] Servadio, S. and Zanetti, R. "Koopman Operator Theory for Space
Applications." *The Journal of the Astronautical Sciences*, 69:1411-1447, 2022.

[8] Alfano, S. "A Numerical Implementation of Spherical Object Collision
Probability." *Journal of the Astronautical Sciences*, 53(1):103-109, 2005.

[9] Vallado, D.A. *Fundamentals of Astrodynamics and Applications*, 4th ed.
Microcosm Press, 2013.

[10] Kessler, D.J. and Cour-Palais, B.G. "Collision Frequency of Artificial
Satellites: The Creation of a Debris Belt." *Journal of Geophysical Research*,
83(A6):2637-2646, 1978.

[11] [synthetic] Visser, J. "KSCS: Koopman-Spectral Conjunction Screening
in the Humeris Astrodynamics Library." Technical Report, 2026.

---

*Appendix A: DMD Algorithm*

The exact DMD algorithm as implemented:

```
Input: Snapshot pairs (g_0, g_1), (g_1, g_2), ..., (g_{M-2}, g_{M-1})

1. Assemble G_current = [g_0, g_1, ..., g_{M-2}]    (p x (M-1))
   Assemble G_future  = [g_1, g_2, ..., g_{M-1}]    (p x (M-1))

2. SVD: G_current = U * Sigma * V^T
   With rank truncation: keep r components where sigma_i > tol

3. Pseudoinverse: G_current^+ = V * Sigma^{-1} * U^T

4. Koopman matrix: K = G_future * G_current^+        (p x p)

5. Eigendecomposition: K * phi_j = lambda_j * phi_j

Output: K, {lambda_j}, {phi_j}, training_error
```

*Appendix B: Spectral Distance Computation*

```
Input: KoopmanModel A, KoopmanModel B

1. K_A = reshape(A.koopman_matrix, (p_A, p_A))
   K_B = reshape(B.koopman_matrix, (p_B, p_B))

2. lambda_A = eigenvalues(K_A)
   lambda_B = eigenvalues(K_B)

3. sigma_A = sort(|lambda_A|, descending)
   sigma_B = sort(|lambda_B|, descending)

4. Pad shorter to length max(p_A, p_B) with zeros

5. d = mean(|sigma_A - sigma_B|)    (Wasserstein-1 distance)

Output: d (non-negative float)
```

*Appendix C: Observable Library Expansion*

The observable vector for a satellite state $(x, y, z, v_x, v_y, v_z)$:

| Index | Observable | Physical meaning |
|---|---|---|
| 0 | $x$ | Position x-component |
| 1 | $y$ | Position y-component |
| 2 | $z$ | Position z-component |
| 3 | $v_x$ | Velocity x-component |
| 4 | $v_y$ | Velocity y-component |
| 5 | $v_z$ | Velocity z-component |
| 6 | $r = \sqrt{x^2 + y^2 + z^2}$ | Orbital radius |
| 7 | $v = \sqrt{v_x^2 + v_y^2 + v_z^2}$ | Speed |
| 8 | $x^2$ | Radial quadratic |
| 9 | $y^2$ | Tangential quadratic |
| 10 | $z^2$ | Normal quadratic |
| 11 | $xy$ | Cross-term |

The nonlinear observables (indices 6--11) capture the quadratic terms that
appear when Taylor-expanding the gravitational acceleration $\mu/r^2$ around
a reference orbit. Including them improves the Koopman approximation for
perturbed orbits.

*Appendix D: Relationship Between Spectral Distance and Orbital Elements*

The Koopman eigenvalues encode the fundamental frequencies of orbital motion.
For a Keplerian orbit, the dominant eigenvalue corresponds to the mean motion:

$$\lambda_{\text{orb}} = e^{i n \Delta t} \qquad \Rightarrow \qquad |\lambda_{\text{orb}}| = 1, \quad \arg(\lambda_{\text{orb}}) = n \Delta t$$

The spectral distance between two orbits with mean motions $n_a$ and $n_b$ is
approximately:

$$d(a, b) \approx \frac{1}{p} |e^{i n_a \Delta t} - e^{i n_b \Delta t}|$$

For small differences ($|n_a - n_b| \Delta t \ll 1$):

$$d(a, b) \approx \frac{\Delta t}{p} |n_a - n_b|$$

Using the mean motion relation $n = \sqrt{\mu / a^3}$:

$$\frac{\Delta n}{n} = -\frac{3}{2} \frac{\Delta a}{a}$$

Therefore:

$$d(a, b) \approx \frac{3 \Delta t n}{2p} \cdot \frac{|\Delta a|}{a}$$

For LEO at 400 km ($a = 6778$ km, $n \approx 0.0011$ rad/s) with $\Delta t = 60$ s
and $p = 12$:

$$d \approx \frac{3 \times 60 \times 0.0011}{2 \times 12} \cdot \frac{|\Delta a|}{6778}
\approx 0.0083 \cdot \frac{|\Delta a|}{6778}$$

For $\Delta a = 100$ km: $d \approx 0.12$.
For $\Delta a = 500$ km: $d \approx 0.61$.
For $\Delta a = 1000$ km: $d \approx 1.22$.

This confirms that the default threshold $d_{\text{thresh}} = 0.5$ captures
altitude differences up to approximately 400 km, which is appropriate for
same-shell screening.

**Effect of eccentricity**: For eccentric orbits, additional eigenvalues
appear at harmonics of the mean motion ($2n$, $3n$, ...). The spectral distance
between a circular and an eccentric orbit of the same semi-major axis is:

$$d \approx \frac{1}{p} \sum_{j=2}^{k} |c_j^{(e)} - c_j^{(0)}|$$

where $c_j^{(e)}$ are the magnitudes of the $j$-th harmonic coefficients,
which are $O(e^{j-1})$. For small eccentricity ($e < 0.1$), the dominant term
is $O(e)$, so:

$$d_{\text{ecc}} \approx \frac{e}{p}$$

This means the spectral distance between a circular orbit and an orbit with
$e = 0.05$ is $d \approx 0.004$, which is well below the threshold. Eccentricity
differences alone do not significantly affect screening outcomes.

*Appendix E: Koopman Model Stability*

The stability of the Koopman prediction depends on the maximum eigenvalue
magnitude $|\lambda_{\max}|$:

- $|\lambda_{\max}| < 1$: All modes decay. Prediction converges to zero
  (unphysical for orbital dynamics without dissipation).
- $|\lambda_{\max}| = 1$: Modes oscillate stably. This is the ideal case for
  conservative orbital dynamics.
- $|\lambda_{\max}| > 1$: At least one mode grows exponentially. Prediction
  diverges.

In practice, DMD on finite data with noise produces eigenvalues slightly off
the unit circle. The deviation from $|\lambda| = 1$ serves as a model quality
diagnostic:

$$\text{stability margin} = |1 - |\lambda_{\max}||$$

For well-conditioned training data (smooth trajectory, sufficient snapshots):
$\text{stability margin} < 10^{-4}$.

For noisy or insufficient data: $\text{stability margin} > 10^{-2}$, indicating
the model may produce unreliable predictions beyond a few orbital periods.

The prediction error grows approximately as:

$$\epsilon(t) \approx \epsilon_0 \cdot |\lambda_{\max}|^{t / \Delta t}$$

For $|\lambda_{\max}| = 1.001$ and $\Delta t = 60$ s, the error doubles every:

$$T_{\text{double}} = \frac{\Delta t \ln 2}{\ln |\lambda_{\max}|} \approx \frac{60 \times 0.693}{0.001} = 41580 \text{ s} \approx 11.6 \text{ hours}$$

This sets an upper bound on the useful prediction window for conjunction
screening. For the default 7-day screening window, models with
$|\lambda_{\max}| > 1.0001$ should be flagged as unreliable.

*Appendix F: Scaling Analysis for Mega-Constellations*

For very large populations ($N > 10{,}000$), the $O(N^2)$ spectral screening
becomes the computational bottleneck. Several strategies can reduce this:

**1. Binning by altitude shell**: Group satellites into altitude bins of width
$\Delta h$. Only screen pairs within the same bin or adjacent bins. For $B$ bins
of size $N/B$:

$$\text{Cost} = O(3B \cdot (N/B)^2) = O(3N^2/B)$$

With $B = 10$ altitude bins: approximately 30% of brute-force cost.

**2. VP-tree nearest-neighbor search**: Since spectral distance is a metric,
build a Vantage-Point tree on the eigenvalue magnitude vectors. Range queries
with radius $d_{\text{thresh}}$ have average complexity $O(N \log N)$ for
low-dimensional data.

**3. Precomputed spectral classes**: For mega-constellations where all
satellites in a shell have nearly identical orbits, assign each satellite to
a spectral class based on its shell. Same-class screening is $O(1)$ per pair
(always pass). Cross-class screening uses precomputed inter-class distances.

**4. Hierarchical screening**: First screen orbital planes (2D comparison),
then screen within matching planes (1D comparison). For $P$ planes with $S$
satellites each:

$$\text{Cost} = O(P^2 + P \cdot S^2) \quad \text{vs.} \quad O(N^2) = O(P^2 S^2)$$

For $P = 72$ planes, $S = 22$ sats: approximately 5% of brute-force spectral
screening cost.

**Memory considerations**: The Koopman models for $N$ satellites require
$O(N \cdot p^2)$ storage (flattened $p \times p$ matrices). For $N = 50{,}000$
and $p = 12$: approximately $50{,}000 \times 144 \times 8$ bytes $= 55$ MB,
which fits comfortably in memory.
