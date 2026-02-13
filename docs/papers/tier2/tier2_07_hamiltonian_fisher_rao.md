# Symplectic Covariance Propagation via Hamiltonian Fisher-Rao Geometry

**Authors**: Humeris Research
**Status**: Tier 2 -- Validated Conceptually, Not Yet Implemented
**Date**: February 2026
**Library Version**: Humeris v1.22.0

---

## Abstract

Covariance propagation in orbit determination traditionally relies on the
Extended Kalman Filter (EKF), which linearizes the dynamics about the
estimated trajectory and propagates covariance via $P(t) = \Phi P_0 \Phi^T + Q$.
This approach has a structural limitation: the linearized state transition
matrix $\Phi$ does not in general satisfy the symplectic condition
$\Phi^T J \Phi = J$, violating Liouville's theorem and causing artificial
uncertainty growth unrelated to actual dynamical divergence. We propose
Hamiltonian Fisher-Rao Covariance Propagation (HFRCP), which enforces
symplecticity by propagating the state transition matrix using a symplectic
integrator and measures uncertainty through the Fisher-Rao metric on the
statistical manifold of orbital state distributions. The symplectic STM
guarantees $\det(\Phi_s) = 1$, preserving phase-space volume as required by
Hamiltonian mechanics. The Fisher-Rao metric $g_{ij}(\theta) = E[\partial_i
\log p(x|\theta) \cdot \partial_j \log p(x|\theta)]$ provides a
Riemannian geometry on the space of probability distributions, yielding
geodesic distances that respect the intrinsic curvature of the uncertainty
manifold. We derive the symplectic STM propagation equations, establish
the connection between symplectic covariance and the Fisher information
matrix, and argue that HFRCP should produce tighter uncertainty bounds than
standard EKF propagation for orbital arcs exceeding one orbital period.
The approach builds on the existing Humeris EKF (`orbit_determination.py`)
and numerical propagation infrastructure (`numerical_propagation.py`).

---

## 1. Introduction

### 1.1 Motivation

Accurate uncertainty quantification is essential for conjunction assessment,
maneuver planning, and autonomous operations. The probability of collision
$P_c$ depends critically on the covariance matrices of both objects at the
time of closest approach. Artificially inflated covariance leads to
excessive false alarm rates; artificially deflated covariance leads to
missed conjunctions. Both failure modes have operational consequences.

The Extended Kalman Filter, as implemented in the Humeris
`orbit_determination.run_ekf`, propagates covariance through linearized
dynamics. The state transition matrix (STM) $\Phi$ is computed via finite
differences (`_compute_stm`), which does not enforce the symplectic
structure inherent in Hamiltonian orbital mechanics. Over long propagation
arcs, this violation accumulates, producing covariance matrices that are
inconsistent with the underlying phase-space geometry.

### 1.2 Problem Statement

The standard EKF covariance propagation $P(t) = \Phi P_0 \Phi^T + Q$
satisfies two properties:

1. $P(t)$ is symmetric positive semi-definite (guaranteed by the Joseph
   stabilized form used in Humeris).
2. $P(t)$ grows over time (observation-free propagation always increases
   uncertainty).

However, it violates:

3. **Volume preservation.** For conservative (Hamiltonian) dynamics,
   Liouville's theorem requires that the phase-space volume
   $\det(P(t))^{1/2}$ is preserved under the dynamics alone (excluding
   process noise $Q$). The EKF STM does not guarantee
   $\det(\Phi) = 1$, so the dynamical contribution to covariance
   growth may be non-physical.

The question is: **how much of the EKF covariance growth is due to genuine
dynamical divergence versus violation of symplectic structure?**

### 1.3 Contribution

We propose HFRCP, which:

1. Propagates the STM using a symplectic integrator that guarantees
   $\Phi_s^T J \Phi_s = J$ and $\det(\Phi_s) = 1$.
2. Computes uncertainty through the Fisher-Rao metric on the statistical
   manifold, providing geodesic distances that are invariant under
   reparametrization.
3. Yields covariance bounds that are expected to be tighter than the EKF for
   conservative dynamics without process noise (see Proposition 1).
4. Reduces to the standard EKF in the short-arc limit where symplectic
   violations are negligible.

---

## 2. Background

### 2.1 Hamiltonian Orbital Mechanics

Orbital motion under a central gravitational field is a Hamiltonian system.
The Hamiltonian for the two-body problem is:

$$H(\mathbf{q}, \mathbf{p}) = \frac{|\mathbf{p}|^2}{2m} - \frac{\mu m}{|\mathbf{q}|}$$

where $\mathbf{q} = (x, y, z)$ is the position and
$\mathbf{p} = m \mathbf{v} = m(v_x, v_y, v_z)$ is the momentum.
Hamilton's equations of motion are:

$$\dot{\mathbf{q}} = \frac{\partial H}{\partial \mathbf{p}} = \frac{\mathbf{p}}{m}$$

$$\dot{\mathbf{p}} = -\frac{\partial H}{\partial \mathbf{q}} = -\mu m \frac{\mathbf{q}}{|\mathbf{q}|^3}$$

In canonical form with state vector $\mathbf{z} = (\mathbf{q}, \mathbf{p})^T$:

$$\dot{\mathbf{z}} = J \nabla H(\mathbf{z})$$

where $J$ is the symplectic matrix:

$$J = \begin{pmatrix} 0 & I_3 \\ -I_3 & 0 \end{pmatrix}$$

### 2.2 Symplectic Structure and Liouville's Theorem

The flow map $\phi_t: \mathbf{z}_0 \mapsto \mathbf{z}(t)$ of a Hamiltonian
system preserves the symplectic form: for any two tangent vectors
$\delta \mathbf{z}_1, \delta \mathbf{z}_2$:

$$\omega(\delta \mathbf{z}_1, \delta \mathbf{z}_2) = \delta \mathbf{z}_1^T J \delta \mathbf{z}_2 = \text{const}$$

The state transition matrix $\Phi = \partial \phi_t / \partial \mathbf{z}_0$
satisfies the symplectic condition:

$$\Phi^T J \Phi = J$$

**Liouville's theorem** follows immediately: since $\det(J) = 1$,

$$\det(\Phi)^2 = \det(\Phi^T J \Phi) / \det(J) = 1$$

Therefore $\det(\Phi) = \pm 1$, and by continuity from the identity,
$\det(\Phi) = 1$. Phase-space volume is preserved.

### 2.3 EKF Covariance Propagation

The Humeris EKF (`orbit_determination.run_ekf`) computes the STM via
central finite differences of the two-body propagator:

$$\Phi_{ij} \approx \frac{\phi_t(z_0 + \epsilon e_j)_i - \phi_t(z_0 - \epsilon e_j)_i}{2\epsilon}$$

This numerical STM is an approximation to the true variational equations.
The finite-difference approach does not enforce $\Phi^T J \Phi = J$, and
in general $\det(\Phi) \neq 1$. The violation grows with the propagation
time step $\Delta t$ and the perturbation magnitude $\epsilon$.

Covariance is propagated as:

$$P^- = \Phi P^+ \Phi^T + Q$$

where $P^+$ is the post-update covariance from the previous step and $Q$
is the process noise. The Joseph stabilized update form is used for the
measurement update:

$$P^+ = (I - KH) P^- (I - KH)^T + K R K^T$$

### 2.4 Information Geometry

Information geometry (Amari 2016) studies the geometry of probability
distribution families $\{p(x|\theta) : \theta \in \Theta\}$. The
Fisher information matrix (FIM) defines a Riemannian metric on the
parameter space:

$$g_{ij}(\theta) = E_\theta\left[\frac{\partial \log p(x|\theta)}{\partial \theta_i} \cdot \frac{\partial \log p(x|\theta)}{\partial \theta_j}\right]$$

For a Gaussian distribution $p(x|\mu, \Sigma) = \mathcal{N}(\mu, \Sigma)$,
the Fisher-Rao metric on the mean parameter $\mu$ is:

$$g_{ij}(\mu) = [\Sigma^{-1}]_{ij}$$

The geodesic distance between two Gaussian distributions with the same
covariance is the Mahalanobis distance. For distributions with different
covariances, the Fisher-Rao geometry is curved, and geodesic distances
differ from Euclidean distances in parameter space.

### 2.5 Symplectic Integrators

Geometric numerical integration (Hairer et al. 2006) provides methods
that preserve the symplectic structure of Hamiltonian flows. The simplest
is the Stormer-Verlet (leapfrog) method:

$$\mathbf{p}_{1/2} = \mathbf{p}_0 + \frac{h}{2} \mathbf{f}(\mathbf{q}_0)$$

$$\mathbf{q}_1 = \mathbf{q}_0 + h \frac{\mathbf{p}_{1/2}}{m}$$

$$\mathbf{p}_1 = \mathbf{p}_{1/2} + \frac{h}{2} \mathbf{f}(\mathbf{q}_1)$$

where $\mathbf{f}(\mathbf{q}) = -\nabla V(\mathbf{q})$ is the gravitational
acceleration. This method is second-order, time-reversible, and symplectic:
the resulting map satisfies $\Phi_{\text{SV}}^T J \Phi_{\text{SV}} = J$
exactly (up to machine precision).

Higher-order symplectic methods include the 4th-order Yoshida composition
and the 6th-order Kahan-Li method. The Humeris `numerical_propagation`
module currently uses RK4 and Dormand-Prince, which are not symplectic.

---

## 3. Proposed Method

### 3.1 Symplectic STM Propagation

Instead of computing the STM via finite differences, we propagate the
variational equations alongside the state using a symplectic integrator.

The variational equations for the STM are:

$$\dot{\Phi} = A(t) \Phi$$

where $A(t) = J \cdot D^2 H(\mathbf{z}(t))$ is the Hamiltonian Jacobian,
and $D^2 H$ is the Hessian of the Hamiltonian. For the two-body problem:

$$D^2 H = \begin{pmatrix} \frac{\mu}{r^3}\left(I - 3\hat{\mathbf{r}}\hat{\mathbf{r}}^T\right) & 0 \\ 0 & \frac{1}{m} I \end{pmatrix}$$

where $r = |\mathbf{q}|$ and $\hat{\mathbf{r}} = \mathbf{q}/r$.

The key insight is that if we propagate the augmented system
$(\mathbf{z}, \Phi)$ using a symplectic integrator applied to the
augmented Hamiltonian, the resulting $\Phi_s$ automatically satisfies:

$$\Phi_s^T J \Phi_s = J$$

This is because the variational equations inherit the Hamiltonian
structure of the original system (Hairer et al. 2006, Chapter VI).

### 3.2 Symplectic Covariance Propagation

Given the symplectic STM $\Phi_s$, the covariance is propagated as:

$$P_s(t) = \Phi_s(t) P_0 \Phi_s(t)^T$$

This propagation preserves the phase-space volume:

$$\det(P_s(t)) = \det(\Phi_s)^2 \det(P_0) = \det(P_0)$$

Contrast with the EKF, where $\det(P_{\text{EKF}}(t))$ may differ
from $\det(P_0)$ even without process noise.

When process noise is included (non-conservative perturbations like drag
and SRP), the propagation becomes:

$$P_s(t) = \Phi_s(t) P_0 \Phi_s(t)^T + Q_s(t)$$

where $Q_s(t)$ accounts only for genuinely non-conservative forces.
The symplectic propagation is designed so that conservative dynamics do not
artificially inflate uncertainty.

### 3.3 Fisher-Rao Metric for Orbital Distributions

We model the orbital state distribution at time $t$ as a Gaussian:

$$p(\mathbf{z}|t) = \mathcal{N}(\bar{\mathbf{z}}(t), P(t))$$

The Fisher-Rao metric on the mean parameters is:

$$g_{ij}(t) = [P(t)^{-1}]_{ij}$$

This is the precision matrix -- the inverse covariance. The geodesic
distance between two orbital state distributions
$\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$
on the Fisher-Rao manifold is:

$$d_{\text{FR}}^2 = (\mu_1 - \mu_2)^T \bar{\Sigma}^{-1} (\mu_1 - \mu_2) + \frac{1}{2} \text{tr}\left[\left(\bar{\Sigma}^{-1}(\Sigma_1 - \Sigma_2)\right)^2\right]$$

where $\bar{\Sigma} = (\Sigma_1 + \Sigma_2) / 2$ is the mean covariance.

The Fisher-Rao distance provides a principled measure of how different
two orbital estimates are, accounting for both the state difference and
the covariance difference. This is more informative than the Mahalanobis distance
alone, which ignores covariance differences.

### 3.4 Symplectic Fisher Information Propagation

The Fisher information matrix (FIM) at time $t$ is:

$$\mathcal{I}(t) = P(t)^{-1}$$

Under symplectic covariance propagation:

$$\mathcal{I}_s(t) = \Phi_s(t)^{-T} P_0^{-1} \Phi_s(t)^{-1} = \Phi_s(t)^{-T} \mathcal{I}_0 \Phi_s(t)^{-1}$$

Using the symplectic condition $\Phi_s^{-1} = -J \Phi_s^T J$:

$$\mathcal{I}_s(t) = J \Phi_s(t) J^T \mathcal{I}_0 J \Phi_s(t)^T J^T$$

This shows that the Fisher information propagation under symplectic
dynamics has a specific structure: it is a congruence transformation
by $J \Phi_s J^T$, which preserves the eigenvalue spectrum of
$\mathcal{I}$ up to the symplectic similarity.

**Theorem 1** (Symplectic Fisher Information Invariant). *For symplectic
STM $\Phi_s$, the symplectic eigenvalues of the Fisher information matrix
$\mathcal{I}(t)$ are preserved under propagation. That is, if
$\lambda_1, \ldots, \lambda_n$ are the symplectic eigenvalues of
$\mathcal{I}_0$, then $\mathcal{I}_s(t)$ has the same symplectic
eigenvalues.*

The symplectic eigenvalues are defined as the eigenvalues of
$|i J \mathcal{I}|$ and represent the fundamental uncertainty scales
in the conjugate (position-momentum) pairs.

### 3.5 HFRCP Algorithm

The complete Hamiltonian Fisher-Rao Covariance Propagation algorithm:

**Input:** Initial state $\mathbf{z}_0$, initial covariance $P_0$,
Hamiltonian $H$, propagation time $T$, step size $h$, process noise
model $Q(\cdot)$.

**Output:** Propagated state $\mathbf{z}(T)$, symplectic covariance
$P_s(T)$, Fisher-Rao metric $g(T)$, volume preservation error
$\delta_V$.

1. Initialize: $\Phi_s = I_{6 \times 6}$, $\mathbf{z} = \mathbf{z}_0$.
2. For each step $k = 0, \ldots, T/h - 1$:
   - a. Compute Hessian $D^2 H(\mathbf{z}_k)$.
   - b. Symplectic Stormer-Verlet step for $(\mathbf{z}, \Phi_s)$:
     - $\Phi_s^{(1/2)} = \Phi_s^{(k)} + \frac{h}{2} A(\mathbf{z}_k) \Phi_s^{(k)}$
     - $\mathbf{z}_{k+1/2} = \text{SV-half-step}(\mathbf{z}_k, h)$
     - $\Phi_s^{(k+1)} = \Phi_s^{(1/2)} + \frac{h}{2} A(\mathbf{z}_{k+1}) \Phi_s^{(1/2)}$
   - c. Symplecticity enforcement (Gram-Schmidt on $J$):
     - $\Phi_s \leftarrow \text{symplectify}(\Phi_s)$
   - d. Accumulate process noise: $Q_{\text{acc}} \leftarrow Q_{\text{acc}} + Q(t_k) \cdot h$
3. Compute symplectic covariance:
   $P_s(T) = \Phi_s P_0 \Phi_s^T + Q_{\text{acc}}$
4. Compute Fisher-Rao metric: $g(T) = P_s(T)^{-1}$
5. Volume preservation check: $\delta_V = |\det(\Phi_s) - 1|$
6. Return $(\mathbf{z}(T), P_s(T), g(T), \delta_V)$.

The symplectification step (3c) corrects numerical drift by projecting
$\Phi_s$ onto the symplectic group $\text{Sp}(6)$ via:

$$\Phi_s \leftarrow \Phi_s (I + \frac{1}{2}(J - \Phi_s^T J \Phi_s))$$

This Cayley-type correction preserves accuracy to $O(h^p)$ while
ensuring exact symplecticity.

### 3.6 Comparison Metric

To quantify the improvement of HFRCP over standard EKF propagation,
we define the volume ratio:

$$\rho_V(t) = \frac{\det(P_s(t))^{1/2}}{\det(P_{\text{EKF}}(t))^{1/2}}$$

For conservative dynamics without process noise, $\rho_V(t) = \det(P_0)^{1/2} / \det(P_{\text{EKF}}(t))^{1/2}$.

Since $\det(P_{\text{EKF}}) \geq \det(P_0)$ in general (volume can only
increase with EKF propagation), we have $\rho_V \leq 1$. The smaller
$\rho_V$, the more conservative (over-estimated) the EKF uncertainty is.

---

## 4. Theoretical Analysis

### 4.1 Volume Preservation Guarantee

**Theorem 2** (Exact Volume Preservation). *For the symplectic STM
$\Phi_s$ produced by a symplectic integrator applied to the Hamiltonian
variational equations:*

$$\det(\Phi_s) = 1 + O(\epsilon_{\text{mach}})$$

*where $\epsilon_{\text{mach}}$ is machine epsilon. This holds for
arbitrarily long propagation times.*

*Proof.* The symplectic integrator preserves the modified Hamiltonian
$\tilde{H} = H + O(h^p)$ exactly (Hairer et al. 2006, backward error
analysis). The flow of $\tilde{H}$ is exactly symplectic, hence
$\det(\Phi_s) = 1$ exactly for the modified system. The error is
only from floating-point arithmetic. $\square$

Contrast with the EKF: the finite-difference STM has
$\det(\Phi_{\text{FD}}) = 1 + O(\epsilon h^{-1} + h^2)$, where the
first term is finite-difference error and the second is truncation error.
For long propagation arcs, these errors accumulate.

### 4.2 Tighter Uncertainty Bounds

**Proposition 1** (Uncertainty Ordering). *For conservative dynamics
(no process noise), the symplectic covariance satisfies:*

$$P_s(t) \preceq P_{\text{EKF}}(t)$$

*in the Loewner ordering (positive semi-definite ordering), with
equality iff $\Phi_{\text{EKF}}$ is exactly symplectic.*

*Proof sketch.* Since $\det(P_s) = \det(P_0)$ and
$\det(P_{\text{EKF}}) \geq \det(P_0)$, and both are obtained from
the same initial $P_0$ by linear transformation, the volume-preserving
transformation produces a tighter ellipsoid in the determinant sense.
[CONJECTURED] The Loewner ordering $P_s \preceq P_{\text{EKF}}$ is
conjectured to follow from the AM-GM inequality on the eigenvalues of
$P_{\text{EKF}} P_s^{-1}$; a complete proof requires showing that no
individual eigenvalue of $P_s$ exceeds the corresponding eigenvalue of
$P_{\text{EKF}}$, which has not been verified for all perturbation
regimes. $\square$

### 4.3 Fisher-Rao Geodesic Properties

**Proposition 2** (Reparametrization Invariance). *The Fisher-Rao
distance between two orbital state distributions is invariant under
diffeomorphic coordinate transformations (e.g., Cartesian to Keplerian
elements).*

This is a fundamental property of the Fisher-Rao metric (Amari 2016).
It means that the uncertainty quantification is independent of the
coordinate system used for state representation -- a desirable property
that the Euclidean Mahalanobis distance does not possess.

**Proposition 3** (Cramer-Rao Bound Consistency). *The symplectic
Fisher information satisfies the Cramer-Rao inequality:*

$$P_s(t) \succeq \mathcal{I}_s(t)^{-1}$$

*with equality for Gaussian distributions. The standard EKF Fisher
information may violate this bound due to non-symplectic propagation.*

### 4.4 Error Analysis

The HFRCP covariance error has two contributions:

1. **Symplectic integrator error:** The modified Hamiltonian
   $\tilde{H} = H + O(h^p)$ introduces a bias in the STM of order
   $O(h^p)$, where $p$ is the integrator order. For Stormer-Verlet,
   $p = 2$; for Yoshida, $p = 4$.

2. **Symplectification error:** The Cayley projection introduces
   $O(\|\Phi^T J \Phi - J\|)$ error, which is $O(h^{p+1})$ for a
   $p$th-order symplectic integrator.

The total covariance error is:

$$\|P_s(T) - P_{\text{true}}(T)\| = O(T h^p)$$

compared to the EKF:

$$\|P_{\text{EKF}}(T) - P_{\text{true}}(T)\| = O(T h^2 + T \epsilon h^{-1})$$

where the second term is the finite-difference contamination.

### 4.5 Computational Cost

| Operation | EKF STM | HFRCP STM |
|-----------|---------|-----------|
| STM computation | 12 propagations (finite diff) | 1 propagation (augmented) |
| Symplecticity | Not enforced | Exact (up to $\epsilon_{\text{mach}}$) |
| Memory | $O(n^2)$ | $O(n^2)$ |
| Per-step FLOPS | $O(12 n)$ | $O(n^2)$ (variational eqs) |

The HFRCP is computationally cheaper per step (1 augmented propagation
vs. 12 perturbed propagations for 6D state), but requires implementation
of the symplectic integrator for the augmented system.

---

## 5. Proposed Validation

### 5.1 Determinant Preservation Test

Propagate a LEO orbit for 10 orbital periods with both EKF STM and
HFRCP STM (no process noise):

1. Initialize with $P_0 = \text{diag}(100^2, 100^2, 100^2, 0.1^2, 0.1^2, 0.1^2)$ (m, m/s).
2. Propagate STM using `orbit_determination._compute_stm` (EKF baseline).
3. Propagate STM using symplectic Stormer-Verlet (HFRCP).
4. Compute $\det(\Phi_{\text{EKF}})$ and $\det(\Phi_s)$ at each period.
5. Verify: $|\det(\Phi_s) - 1| < 10^{-12}$ (machine precision).
6. Plot $\det(\Phi_{\text{EKF}})$ drift over time.

### 5.2 Covariance Volume Comparison

Compare uncertainty volumes (determinant of $P$) between EKF and HFRCP
for propagation arcs of 1, 5, 10, 50 orbital periods:

1. Use the same two-body dynamics as `orbit_determination._two_body_propagate`.
2. Compute $\rho_V(t) = \det(P_s(t))^{1/2} / \det(P_{\text{EKF}}(t))^{1/2}$.
3. Plot $\rho_V$ vs. propagation time.
4. Verify that $\rho_V \leq 1$ always and $\rho_V \to 0$ for long arcs.

### 5.3 Consistency with Existing EKF

For short propagation arcs ($\Delta t < T_{\text{orbital}} / 10$), verify
that HFRCP and EKF produce identical results within tolerance:

1. Run `orbit_determination.run_ekf` with observation intervals of 60 s.
2. Replace STM computation with HFRCP symplectic STM.
3. Compare final state estimates and covariances.
4. Verify: $\|P_s - P_{\text{EKF}}\| / \|P_{\text{EKF}}\| < 0.01$ for
   short arcs.

### 5.4 Fisher-Rao Distance Validation

Compute the Fisher-Rao distance between EKF-propagated and
HFRCP-propagated distributions:

1. Propagate both filters for 10 orbital periods.
2. Compute $d_{\text{FR}}$ between the two Gaussian state distributions.
3. Compare with the Bhattacharyya distance (which HFRCP should improve).

### 5.5 Perturbation Sensitivity

Test with perturbed dynamics (J2, drag) using `numerical_propagation`:

1. Add J2 perturbation (conservative -- should preserve symplecticity).
2. Add drag (non-conservative -- process noise $Q$ should capture this).
3. Verify that HFRCP with J2-only matches volume preservation.
4. Verify that HFRCP with drag produces $\det(P_s) > \det(P_0)$ (as expected).

---

## 6. Discussion

### 6.1 Limitations

**Gaussian assumption.** HFRCP assumes Gaussian state distributions.
For highly nonlinear dynamics (e.g., close approaches, low-perigee orbits),
the distribution becomes non-Gaussian, and the Fisher-Rao metric on
Gaussian families is no longer appropriate. The existing Humeris particle
filter (`orbit_determination.run_particle_filter`) handles non-Gaussian
distributions but does not provide the Fisher-Rao metric.

**Non-conservative perturbations.** The symplectic structure is exact
only for conservative (Hamiltonian) dynamics. Atmospheric drag, solar
radiation pressure, and maneuver forces break Hamiltonian structure. The
process noise $Q$ must absorb all non-conservative effects, which requires
careful calibration.

**Augmented system stiffness.** Propagating the variational equations
alongside the state doubles the system dimension from 6 to 42 (state +
36 STM elements). For stiff dynamics (e.g., close to Earth), the augmented
system may require very small step sizes.

**Implementation complexity.** Implementing a symplectic integrator for the
augmented variational system is substantially more complex than the current
finite-difference STM approach. The Stormer-Verlet method is straightforward
for separable Hamiltonians (kinetic + potential), but non-separable
perturbations (e.g., velocity-dependent drag) require splitting techniques.

### 6.2 Open Questions

1. **Optimal process noise model.** Given the symplectic propagation,
   how should $Q$ be calibrated to account for unmodeled non-conservative
   forces without over-inflating uncertainty?

2. **Connection to D-optimal scheduling.** The existing
   `orbit_determination.compute_optimal_observation_schedule` uses the
   Fisher information matrix. Can the symplectic FIM from HFRCP improve
   the observation scheduling?

3. **Extension to non-Gaussian distributions.** The Fisher-Rao metric
   extends to exponential family distributions. Can HFRCP be generalized
   to propagate higher-order cumulants (skewness, kurtosis) while
   preserving symplectic structure?

4. **Integration with Bayesian model selection.** The existing
   `orbit_determination.select_force_model` uses BIC for force model
   ranking. Can the HFRCP Fisher information provide a better Bayesian
   evidence computation?

### 6.3 Prerequisites for Implementation

Required new components:

1. **Symplectic integrator:** Stormer-Verlet for the two-body problem,
   with Yoshida composition for higher order. Must handle the augmented
   $(\mathbf{z}, \Phi)$ system.
2. **Hessian computation:** $D^2 H$ for two-body plus perturbations
   (J2, J3). Can reuse existing gravitational acceleration derivatives
   from `numerical_propagation`.
3. **Symplectification routine:** Cayley projection
   $\Phi \mapsto \Phi(I + \frac{1}{2}(J - \Phi^T J \Phi))$.
4. **Fisher-Rao distance computation:** For Gaussian distributions,
   requires covariance inversion and matrix logarithm.

Domain purity is maintained: all operations use linear algebra
(available in `linalg.py` and NumPy) and standard mathematical functions.

---

## 7. Conclusion

We have proposed Hamiltonian Fisher-Rao Covariance Propagation (HFRCP),
which combines symplectic state transition matrix propagation with
Fisher-Rao information geometry for orbital uncertainty quantification.
By enforcing $\Phi_s^T J \Phi_s = J$ through symplectic integration,
HFRCP guarantees phase-space volume preservation ($\det(\Phi_s) = 1$),
avoiding the artificial uncertainty growth associated with the standard EKF
finite-difference STM. The Fisher-Rao metric provides a coordinate-invariant
measure of distributional distance that respects the Riemannian geometry
of the uncertainty manifold.

The approach is expected to yield tighter uncertainty bounds for conservative
dynamics, with the improvement growing with propagation arc length. For
short observation intervals typical of ground-station tracking, HFRCP
reduces to the standard EKF; the benefit emerges primarily for autonomous
operations with long observation gaps.

The framework builds on the existing Humeris EKF infrastructure in
`orbit_determination.py` and the numerical propagation capabilities in
`numerical_propagation.py`, requiring primarily the addition of a
symplectic integrator for the augmented variational system.

---

## References

1. Amari, S. (2016). *Information Geometry and Its Applications*. Springer.

2. Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations* (2nd ed.). Springer.

3. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.

4. Broucke, R. A. (2003). Solution of the elliptic rendezvous problem with the time as independent variable. *Journal of Guidance, Control, and Dynamics*, 26(4), 615--621.

5. de Melo, C. F., & Winter, O. C. (2006). Alternative paths for insertion of probes into high inclination orbits around Jupiter. *Advances in Space Research*, 38(11), 2398--2405.

6. Gurfil, P. (2007). Relative motion between elliptic orbits: Generalized boundedness conditions and optimal formationkeeping. *Journal of Guidance, Control, and Dynamics*, 28(4), 761--767.

7. Leok, M., & Zhang, J. (2011). Discrete Hamiltonian variational integrators. *IMA Journal of Numerical Analysis*, 31(4), 1497--1532.

8. Park, R. S., & Scheeres, D. J. (2006). Nonlinear mapping of Gaussian statistics: Theory and applications to spacecraft trajectory design. *Journal of Guidance, Control, and Dynamics*, 29(6), 1367--1375.

9. Rao, C. R. (1945). Information and the accuracy attainable in the estimation of statistical parameters. *Bulletin of the Calcutta Mathematical Society*, 37, 81--91.

10. Selig, J. M. (2005). *Geometric Fundamentals of Robotics*. Springer.
