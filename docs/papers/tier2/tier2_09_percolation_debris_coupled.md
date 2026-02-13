# Coupled Phase Transitions in Debris-ISL Network Systems: A Percolation Analysis

**Authors**: Humeris Research
**Status**: Tier 2 -- Validated Conceptually, Not Yet Implemented
**Date**: February 2026
**Library Version**: Humeris v1.22.0

---

## Abstract

The growth of orbital debris and the maintenance of inter-satellite link
(ISL) constellation networks are not independent phenomena. As debris
spatial density $\rho$ increases, ISL links fail due to widened conjunction
avoidance exclusion zones and physical damage, reducing network connectivity.
This coupling creates a dynamical system with two interacting order
parameters: the debris density $\rho(t)$ and the ISL bond occupation
probability $p(\rho)$. We analyze this coupled system through the lens of
percolation theory. The ISL network undergoes a percolation phase transition
at a critical bond probability $p_c$, below which the giant connected
component vanishes and the network fragments. The debris population evolves
through SIR-like dynamics (or logistic growth). The coupling
$p(\rho) = \exp(-\rho \sigma v_{\text{rel}} \tau_{\text{link}})$ creates a
feedback loop: debris growth reduces connectivity, which may impair
coordinated collision avoidance, which accelerates debris growth. We derive
the critical debris density $\rho_c$ at which ISL network percolation fails,
and examine conditions under which this threshold can be lower than the Kessler cascade
threshold $\rho_K$, and analyze the coupled dynamics near criticality. The
Fiedler value $\lambda_2(L)$ of the graph Laplacian serves as the
connectivity order parameter, approaching zero as $p \to p_c^+$ and
exhibiting critical scaling $\lambda_2 \sim (p - p_c)^\nu$. The giant
component fraction follows $G(p) \sim (p - p_c)^\beta$ with mean-field
exponent $\beta \approx 1$. This analysis suggests the possibility of an "information
Kessler syndrome" where the constellation loses its communication backbone
before the debris density reaches the classical cascade threshold.

---

## 1. Introduction

### 1.1 Motivation

The Kessler syndrome threshold is traditionally defined as the debris
spatial density at which collisional fragment production exceeds drag
removal: the onset of a self-sustaining cascade. This threshold depends
on collision cross-sections, relative velocities, fragment multipliers,
and atmospheric drag lifetimes. The existing Humeris modules
(`cascade_analysis`, `kessler_heatmap`) compute this threshold through
SIR epidemic dynamics and the cascade multiplication factor $k_{\text{eff}}$.

However, modern mega-constellations depend on ISL networks for their
operational functionality. These networks are vulnerable to a different
but related hazard: as debris density increases, conjunction events
become more frequent, forcing satellites to widen their collision
avoidance maneuvering zones. ISL links that pass through high-density
debris regions may be interrupted by avoidance maneuvers, reducing the
effective network connectivity. Even before a physical Kessler cascade
begins, the ISL network may fragment, creating an "information Kessler
syndrome" where the constellation loses coordinated operation capability.

### 1.2 Problem Statement

Two thresholds exist in the debris-constellation system:

1. **Kessler cascade threshold** $\rho_K$: Debris density at which
   collisional cascade becomes self-sustaining ($k_{\text{eff}} > 1$
   or $R_0 > 1$ in the SIR model).

2. **Network percolation threshold** $\rho_c$: Debris density at which
   the ISL network fragments (giant component vanishes).

The relationship between these thresholds is unknown. If $\rho_c < \rho_K$,
then the constellation loses its communication backbone before the
classical cascade begins -- a failure mode not captured
by the existing mean-field analysis.

### 1.3 Contribution

We analyze the coupled debris-ISL system and derive that, under stated assumptions:

1. The bond occupation probability $p(\rho)$ decreases exponentially
   with debris density under a Poisson conjunction avoidance model.
2. The network percolation threshold defines a critical debris density
   $\rho_c$ that may be lower than $\rho_K$.
3. Near criticality, the Fiedler value scales as
   $\lambda_2 \sim (p - p_c)^\nu$, providing an observable
   early-warning indicator.
4. The coupling between debris growth and network degradation creates
   a positive feedback loop that can accelerate both transitions.

---

## 2. Background

### 2.1 Percolation Theory

Percolation theory (Stauffer & Aharony 1994) studies the emergence of
long-range connectivity in random media. In bond percolation on a graph
$G = (V, E)$, each edge is independently "open" with probability $p$
and "closed" with probability $1-p$. Key quantities:

**Giant component fraction** $G(p)$: The fraction of nodes in the
largest connected component.

**Percolation threshold** $p_c$: The critical probability at which
$G(p)$ transitions from zero (no giant component) to positive:

$$G(p) \sim \begin{cases} 0 & \text{if } p < p_c \\ (p - p_c)^\beta & \text{if } p > p_c \end{cases}$$

For mean-field (random graph) models, $\beta = 1$. For 2D lattice
percolation, $\beta = 5/36 \approx 0.139$.

**Correlation length** $\xi(p)$: The typical size of connected clusters
below the threshold:

$$\xi(p) \sim |p - p_c|^{-\nu}$$

For mean-field, $\nu = 1/2$; for 2D lattice, $\nu = 4/3$.

### 2.2 Percolation Thresholds for Standard Graphs

| Graph type | Bond $p_c$ | Site $p_c$ |
|-----------|-----------|-----------|
| Erdos-Renyi $(N, p)$ | $1/(N-1)$ | $1/(N-1)$ |
| Random $d$-regular | $1/(d-1)$ | varies |
| Square lattice | $1/2$ | $0.5927$ |
| Triangular lattice | $2 \sin(\pi/18) \approx 0.347$ | $1/2$ |
| Bethe lattice (degree $z$) | $1/(z-1)$ | $1/(z-1)$ |

For ISL networks, the most relevant models are random geometric graphs
(satellites within communication range) and random regular graphs
(fixed ISL degree per satellite).

### 2.3 Kessler Cascade Models in Humeris

The Humeris library provides multiple cascade models:

- **SIR model** (`cascade_analysis.compute_cascade_sir`): Mean-field
  epidemic dynamics with $R_0 = N_f \beta S_0 / \gamma$.
- **Birth-death chain** (`cascade_analysis.compute_debris_birth_death`):
  Stochastic population model with geometric stationary distribution.
- **Lotka-Volterra** (`cascade_analysis.compute_lotka_volterra_debris`):
  Multi-species interaction model (rocket bodies, mission debris, fragments).
- **Spectral $k_{\text{eff}}$** (`kessler_heatmap.compute_spectral_kessler`):
  Perron-Frobenius eigenvalue of the debris migration matrix.

### 2.4 ISL Network Analysis in Humeris

The graph-theoretic infrastructure includes:

- **Fiedler value** $\lambda_2(L)$: Algebraic connectivity
  (`graph_analysis.compute_topology_resilience`).
- **Ising phase transition**: Critical eclipse fraction for network
  coherence (`graph_analysis.compute_isl_phase_transition`).
- **Cheeger bottleneck**: Isoperimetric constant
  (`graph_analysis.compute_cheeger_bottleneck`).

The Ising model analysis is particularly relevant: it identifies a phase
transition in ISL network coherence driven by eclipse fraction. The
debris-driven percolation transition we propose is an analogous phenomenon
driven by debris density rather than eclipse.

---

## 3. Proposed Method

### 3.1 Bond Occupation Probability

Each ISL link $(i, j)$ operates when:

1. The satellites $i$ and $j$ are within communication range.
2. No debris conjunction avoidance maneuver interrupts the link.
3. The link is not physically destroyed by debris impact.

Condition (2) dominates for current debris levels. The probability that
a link operates during a time window $\tau_{\text{link}}$ (typical ISL
communication session duration) is:

$$p(\rho) = \exp(-\rho \cdot \sigma_{\text{avoid}} \cdot v_{\text{rel}} \cdot \tau_{\text{link}})$$

where:
- $\rho$ is the local debris spatial density ($\text{km}^{-3}$).
- $\sigma_{\text{avoid}}$ is the effective conjunction avoidance
  cross-section ($\text{km}^2$). This is much larger than the physical
  collision cross-section because avoidance maneuvers have a wider
  exclusion zone (typically 1--10 km miss distance threshold).
- $v_{\text{rel}}$ is the mean relative velocity (km/s).
- $\tau_{\text{link}}$ is the ISL communication session duration (s).

The exponential form arises from the Poisson process assumption:
conjunction events arrive as a Poisson process with rate
$\lambda = \rho \sigma_{\text{avoid}} v_{\text{rel}}$, and the
link survives if zero events occur in $[0, \tau_{\text{link}}]$.

### 3.2 Critical Debris Density

Setting $p(\rho_c) = p_c$ (the percolation threshold of the ISL graph):

$$\rho_c = -\frac{\ln p_c}{\sigma_{\text{avoid}} \cdot v_{\text{rel}} \cdot \tau_{\text{link}}}$$

For an ISL network with mean degree $z = 4$ (typical for Walker
constellations with 4 ISL per satellite), $p_c \approx 1/(z-1) = 1/3$
for the mean-field approximation:

$$\rho_c = \frac{\ln 3}{\sigma_{\text{avoid}} \cdot v_{\text{rel}} \cdot \tau_{\text{link}}}$$

### 3.3 Numerical Estimates

Using representative parameters for LEO ISL operations:

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Avoidance cross-section | $\sigma_{\text{avoid}}$ | $10^{-3}$ | $\text{km}^2$ |
| Relative velocity | $v_{\text{rel}}$ | 10 | km/s |
| Link session duration | $\tau_{\text{link}}$ | 600 | s |
| Kessler threshold | $\rho_K$ | $10^{-7}$ | $\text{km}^{-3}$ |

The critical debris density for ISL percolation:

$$\rho_c = \frac{\ln 3}{10^{-3} \times 10 \times 600} \approx \frac{1.099}{6} \approx 0.183 \text{ km}^{-3}$$

This is far above the Kessler threshold, suggesting that for these
parameters, the Kessler cascade occurs first. However, the avoidance
cross-section $\sigma_{\text{avoid}}$ is highly scenario-dependent.
For conservative avoidance policies (e.g., 10 km miss distance for
automated avoidance), $\sigma_{\text{avoid}}$ can be much larger:

With $\sigma_{\text{avoid}} = 0.3 \text{ km}^2$ (10 km miss distance circle):

$$\rho_c = \frac{1.099}{0.3 \times 10 \times 600} \approx 6.1 \times 10^{-4} \text{ km}^{-3}$$

Still above $\rho_K = 10^{-7}$. The network percolation threshold
exceeds the Kessler threshold for these parameters.

The regimes where $\rho_c < \rho_K$ arise when:

$$\frac{\sigma_{\text{avoid}}}{\sigma_{\text{collision}}} > \frac{N_f}{z - 1} \cdot \frac{\tau_{\text{link}}}{\tau_{\text{drag}}}$$

That is, when the avoidance cross-section is sufficiently larger than
the collision cross-section relative to the fragment multiplier and
drag lifetime.

### 3.4 Coupled Dynamics

The coupled debris-ISL system evolves as:

**Debris evolution** (logistic or SIR):

$$\frac{d\rho}{dt} = f(\rho) = \alpha \rho (1 - \rho / \rho_{\text{cap}}) - \gamma \rho + S$$

where $\alpha$ is the collision-driven growth rate,
$\rho_{\text{cap}}$ is the carrying capacity (limited by shell volume),
$\gamma$ is the drag removal rate, and $S$ is the external source
(launch debris, breakup events).

**ISL bond probability** (instantaneous coupling):

$$p(t) = \exp(-\rho(t) \cdot \sigma_{\text{avoid}} \cdot v_{\text{rel}} \cdot \tau_{\text{link}})$$

**Bond probability dynamics** (differentiating):

$$\frac{dp}{dt} = -p \cdot \sigma_{\text{avoid}} \cdot v_{\text{rel}} \cdot \tau_{\text{link}} \cdot \frac{d\rho}{dt}$$

**Giant component fraction:**

$$G(p) = \begin{cases} 0 & \text{if } p \leq p_c \\ c_1 (p - p_c)^\beta & \text{if } p > p_c \end{cases}$$

**Fiedler value:**

$$\lambda_2(p) \sim c_2 (p - p_c)^\nu$$

### 3.5 Feedback Loop

The coupling can create a positive feedback loop:

1. Debris density $\rho$ increases (natural growth or event).
2. ISL bond probability $p(\rho)$ decreases.
3. Network connectivity degrades ($\lambda_2$ decreases, $G$ decreases).
4. [SPECULATIVE] Coordinated collision avoidance becomes less effective (fragmented
   network cannot share tracking data or coordinate maneuvers).
5. [SPECULATIVE] Collision rate increases (uncoordinated avoidance is less efficient).
6. Debris density increases further (back to step 1).

This feedback loop can be modeled by making the collision rate
$\alpha$ depend on network connectivity:

$$\alpha(G) = \alpha_0 \cdot (1 + \kappa (1 - G))$$

where $\kappa$ is the coordination degradation factor. When the network
is fully connected ($G = 1$), $\alpha = \alpha_0$ (baseline collision
rate). When the network is fragmented ($G = 0$), $\alpha = \alpha_0 (1 + \kappa)$
(elevated collision rate due to uncoordinated operations).

### 3.6 Phase Diagram

The coupled system has a two-dimensional phase space $(\rho, p)$ with
three critical curves:

1. **Debris nullcline:** $d\rho/dt = 0$, defining equilibrium debris density.
2. **Percolation threshold:** $p = p_c$, separating connected from
   fragmented network.
3. **Kessler threshold:** $\rho = \rho_K$, separating subcritical from
   supercritical cascade.

The phase diagram has four regimes:

| Regime | Debris | Network | State |
|--------|--------|---------|-------|
| I | $\rho < \rho_K, \; p > p_c$ | Subcritical, connected | Nominal operations |
| II | $\rho < \rho_K, \; p < p_c$ | Subcritical, fragmented | Information Kessler |
| III | $\rho > \rho_K, \; p > p_c$ | Supercritical, connected | Classical Kessler |
| IV | $\rho > \rho_K, \; p < p_c$ | Supercritical, fragmented | Catastrophic |

The key question is whether the system trajectory passes through
Regime II (information Kessler) before reaching Regime III (classical
Kessler). This depends on the relative magnitudes of $\rho_c$ and $\rho_K$.

### 3.7 Integration Algorithm

The coupled system is integrated as follows:

1. Initialize: $\rho(0)$, constellation ISL adjacency $A_0$, parameters.
2. For each time step $\Delta t$:
   - a. Compute $p(\rho(t))$ from the bond occupation formula.
   - b. Generate random graph: keep each edge of $A_0$ with probability $p$.
   - c. Compute Fiedler value $\lambda_2$ and giant component $G$.
   - d. Compute collision rate $\alpha(G)$ with coordination feedback.
   - e. Euler step: $\rho(t + \Delta t) = \rho(t) + f(\rho, \alpha(G)) \Delta t$.
   - f. Record $(\rho, p, \lambda_2, G)$.
3. Identify transitions: $\lambda_2 \to 0$ (percolation) and
   $\rho \to \rho_K$ (Kessler).

The stochastic element (random edge retention) can be replaced by
the expected values for a deterministic mean-field analysis:

$$G(p) = 1 - \sum_k P(k) \left(1 - \frac{G(p) \cdot p}{1 - (1-p) G(p) k / \langle k \rangle}\right)^k$$

This self-consistency equation for $G(p)$ on a random graph with
degree distribution $P(k)$ is solved iteratively.

---

## 4. Theoretical Analysis

### 4.1 Critical Exponents

Near the percolation threshold $p \to p_c^+$, the system exhibits
universal scaling behavior:

**Giant component:** $G(p) \sim (p - p_c)^\beta$

**Correlation length:** $\xi(p) \sim |p - p_c|^{-\nu}$

**Average cluster size:** $\langle s \rangle \sim |p - p_c|^{-\gamma_{\text{perc}}}$

**Fiedler value:** $\lambda_2(p) \sim (p - p_c)^{\nu_F}$

For random graphs (mean-field universality class): $\beta = 1$, $\nu = 1/2$,
$\gamma_{\text{perc}} = 1$. For lattice percolation (if the ISL graph has
spatial structure): $\beta = 5/36$, $\nu = 4/3$, $\gamma_{\text{perc}} = 43/18$.

The Fiedler exponent $\nu_F$ is related to but distinct from the
correlation length exponent. For random graphs, $\nu_F = 1$ (linear
approach to zero).

### 4.2 Coupled Critical Behavior

**Theorem 1** (Coupled Threshold Ordering). *For the coupled
debris-ISL system with logistic debris growth and exponential bond
coupling, the network percolation threshold $\rho_c$ satisfies:*

$$\rho_c = \frac{-\ln p_c}{\sigma_{\text{avoid}} v_{\text{rel}} \tau_{\text{link}}}$$

*and the Kessler threshold satisfies:*

$$\rho_K = \frac{\gamma}{N_f \sigma_{\text{collision}} v_{\text{rel}}}$$

*The information Kessler syndrome precedes the classical cascade iff:*

$$\frac{\sigma_{\text{avoid}} \tau_{\text{link}}}{\sigma_{\text{collision}} / \gamma} > \frac{N_f}{-\ln p_c}$$

### 4.3 Stability Analysis

The equilibrium of the coupled system satisfies:

$$f(\rho^*, \alpha(G(p(\rho^*)))) = 0$$

Linearizing around the equilibrium:

$$\frac{d\delta\rho}{dt} = \left[\frac{\partial f}{\partial \rho} + \frac{\partial f}{\partial \alpha} \cdot \frac{d\alpha}{dG} \cdot \frac{dG}{dp} \cdot \frac{dp}{d\rho}\right] \delta\rho$$

The term in brackets is the effective growth rate with network feedback.
The system is stable when this is negative, unstable when positive.

The feedback contribution
$\frac{\partial f}{\partial \alpha} \cdot \frac{d\alpha}{dG} \cdot \frac{dG}{dp} \cdot \frac{dp}{d\rho}$
is positive under this model (debris growth $\to$ connectivity loss $\to$ more collisions),
making the equilibrium less stable and potentially triggering instability
at debris densities below the uncoupled threshold.

**Proposition 1** (Feedback-Enhanced Threshold). *The coupled system's
effective cascade threshold $\rho_K^{\text{eff}}$ satisfies:*

$$\rho_K^{\text{eff}} \leq \rho_K$$

*with equality when the coordination feedback $\kappa = 0$.*

### 4.4 Time Scale Analysis

The coupled system has two intrinsic time scales:

1. **Debris time scale:** $\tau_{\text{debris}} = 1/|\alpha - \gamma|$
   (years to decades).
2. **Network time scale:** $\tau_{\text{network}} = 1/(\sigma_{\text{avoid}} v_{\text{rel}} \rho |\dot{\rho}/\rho|)$
   (seconds to hours for individual link failures; months for
   mean-field connectivity evolution).

When $\tau_{\text{network}} \ll \tau_{\text{debris}}$ (typical), the
network responds quasi-statically to debris evolution: $p(t) \approx p(\rho(t))$
with no transient dynamics. The coupled system reduces to a single ODE:

$$\frac{d\rho}{dt} = f(\rho, \alpha(G(p(\rho))))$$

This is the regime where the analysis of Section 3.6 applies directly.

### 4.5 Computational Complexity

| Component | Complexity |
|-----------|-----------|
| Debris ODE step | $O(1)$ |
| Bond probability | $O(M)$ (per edge) |
| Fiedler value | $O(N^2)$ (Jacobi) or $O(M)$ (Lanczos) |
| Giant component | $O(N + M)$ (BFS) |
| Total per step | $O(N^2)$ or $O(M)$ with Lanczos |
| Total for $T$ steps | $O(T \cdot N^2)$ |

---

## 5. Proposed Validation

### 5.1 Bond Probability Calibration

Validate the exponential bond occupation probability against the
Kessler heatmap:

1. Generate a debris population using `kessler_heatmap.compute_kessler_heatmap`.
2. For each ISL link, compute the local debris density from the
   heatmap cell containing the link midpoint.
3. Compute $p(\rho)$ from the exponential model.
4. Compare with Monte Carlo simulation of conjunction events using
   the Poisson model from `kessler_heatmap.compute_conjunction_intensity`.

### 5.2 Percolation Threshold Verification

Verify percolation thresholds for known graph topologies:

1. Construct ISL adjacency matrices for Walker constellations.
2. Randomly remove edges with probability $1-p$ for a range of $p$.
3. Compute Fiedler value and giant component fraction.
4. Identify $p_c$ as the point where $\lambda_2 \to 0$.
5. Compare with theoretical predictions: $p_c \approx 1/(z-1)$ for
   random graphs with mean degree $z$.

### 5.3 Fiedler Value Scaling

Verify the critical scaling of the Fiedler value near $p_c$:

1. For each $p$ value near $p_c$, generate 100 random edge-deleted graphs.
2. Compute $\lambda_2$ for each using
   `graph_analysis.compute_topology_resilience`.
3. Plot $\log \lambda_2$ vs. $\log(p - p_c)$.
4. Verify linear scaling with slope $\nu_F$.

### 5.4 Coupled Dynamics Simulation

Run the full coupled system and compare with uncoupled models:

1. Initialize with a Walker constellation and debris density
   $\rho_0 = 10^{-9} \text{ km}^{-3}$ (current LEO background).
2. Evolve debris using `cascade_analysis.compute_cascade_sir`
   (uncoupled baseline).
3. Evolve the coupled system with ISL feedback.
4. Compare trajectories and identify whether the coupled system
   reaches fragmentation ($\lambda_2 = 0$) before or after $\rho_K$.

### 5.5 Comparison with Ising Phase Transition

The Ising phase transition analysis in
`graph_analysis.compute_isl_phase_transition` identifies a critical
eclipse fraction for network coherence. Compare:

1. Ising critical eclipse fraction $f_c^{\text{eclipse}}$.
2. Debris-driven percolation critical density $\rho_c$.
3. Verify that both describe the same underlying connectivity loss
   phenomenon through different physical mechanisms.

---

## 6. Discussion

### 6.1 Limitations

**Mean-field approximation.** The bond percolation analysis assumes
statistically independent edge failures. In reality, debris density
varies spatially, creating correlated edge failures: ISL links in the
same altitude band experience similar debris densities. This spatial
correlation can shift $p_c$ from the mean-field prediction.

**Quasi-static assumption.** The analysis assumes the network responds
instantaneously to debris density changes. In reality, ISL link failures
have transient dynamics (conjunction avoidance maneuvers are temporary;
the link resumes after the conjunction passes). The time-averaged bond
probability may differ from the instantaneous probability.

**Linear coordination feedback.** The feedback model
$\alpha(G) = \alpha_0 (1 + \kappa(1-G))$ is a first-order approximation.
The actual relationship between network connectivity and collision
avoidance effectiveness is complex, depending on tracking data sharing,
maneuver coordination protocols, and ground segment architecture.

**Single debris density.** The model uses a single scalar $\rho$ for
the debris density, whereas the actual density varies strongly with
altitude and inclination (as captured by the Humeris Kessler heatmap).
A spatially resolved model would assign different bond probabilities
to ISL links at different altitudes.

### 6.2 Open Questions

1. **Spatial percolation on orbital shells.** ISL networks have spatial
   structure (random geometric graph, not Erdos-Renyi). How does this
   affect $p_c$ and the critical exponents?

2. **Adaptive avoidance.** If the constellation can dynamically adjust
   its avoidance cross-section $\sigma_{\text{avoid}}$ based on debris
   density, can it delay the percolation transition? This creates an
   optimal control problem.

3. **Multi-shell coupling.** Debris migrates between altitude bands via
   drag (as modeled by `kessler_heatmap.compute_spectral_kessler`).
   How does inter-shell debris migration affect the ISL percolation of
   constellations at different altitudes?

4. **Relationship to cascade multiplication factor.** The Perron-Frobenius
   eigenvalue from `compute_spectral_kessler` is the true $k_{\text{eff}}$
   for the debris cascade. Is there an analogous spectral quantity for
   the ISL percolation transition?

5. **Renormalization group analysis.** The existing
   `kessler_heatmap.compute_renormalization_group` computes RG flow for
   debris density. Can this be extended to a joint RG for the coupled
   $(\rho, p)$ system?

### 6.3 Prerequisites for Implementation

The following existing Humeris modules would be composed:

- `cascade_analysis`: SIR debris dynamics (debris subsystem).
- `graph_analysis`: Fiedler value, giant component detection,
  Ising phase transition (network subsystem).
- `kessler_heatmap`: Spatial density, conjunction intensity,
  Perron-Frobenius cascade eigenvalue.
- `inter_satellite_links`: ISL topology and distances.

New implementation required:

1. **Bond probability calculator:** $p(\rho)$ from debris density
   and avoidance parameters.
2. **Random bond deletion:** Stochastic edge removal for Monte Carlo
   percolation analysis.
3. **Giant component computation:** BFS-based connected component
   analysis (or leverage existing Fiedler value: $\lambda_2 > 0$
   iff connected).
4. **Coupled ODE integrator:** Forward Euler for the
   $(\rho, p, \lambda_2, G)$ system.
5. **Phase diagram plotter:** Identify and visualize the four regimes.

Estimated complexity: **Medium**. The main challenge is coupling the
debris density evolution (which operates on year time scales) with the
ISL bond probability (which fluctuates on orbital time scales). The
quasi-static approximation simplifies this to a single ODE, but
validating the approximation requires both time scales to be resolved.

---

## 7. Conclusion

We have analyzed the coupled phase transitions in the debris-ISL network
system, showing that debris density growth and ISL connectivity loss form
a dynamical system with interacting order parameters. The bond occupation
probability $p(\rho) = \exp(-\rho \sigma_{\text{avoid}} v_{\text{rel}} \tau_{\text{link}})$
couples debris evolution to network percolation. The critical debris density
$\rho_c$ for ISL network fragmentation provides a new threshold that is
distinct from (and potentially lower than) the classical Kessler cascade
threshold $\rho_K$.

The Fiedler value $\lambda_2(L)$ serves as the observable order parameter
for the network percolation transition, exhibiting critical scaling
$\lambda_2 \sim (p - p_c)^\nu$ near the threshold. The existing Humeris
infrastructure for Fiedler value computation, Ising phase transitions, and
SIR cascade dynamics provides the building blocks for implementing this
coupled analysis.

The central observation is that constellation vulnerability has two dimensions
-- physical (debris collision) and informational (network connectivity) --
and the informational vulnerability may be the binding constraint for
operational sustainability.

---

## References

1. Albert, R., Jeong, H., & Barabasi, A.-L. (2000). Error and attack tolerance of complex networks. *Nature*, 406(6794), 378--382.

2. Callaway, D. S., Newman, M. E. J., Strogatz, S. H., & Watts, D. J. (2000). Network robustness and fragility: Percolation on random graphs. *Physical Review Letters*, 85(25), 5468--5471.

3. Kessler, D. J. (1991). Collisional cascading: The limits of population growth in low Earth orbit. *Advances in Space Research*, 11(12), 63--66.

4. Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.

5. Pastor-Satorras, R., & Vespignani, A. (2001). Epidemic spreading in scale-free networks. *Physical Review Letters*, 86(14), 3200--3203.

6. Stauffer, D., & Aharony, A. (1994). *Introduction to Percolation Theory* (2nd ed.). Taylor & Francis.

7. Bollobas, B. (2001). *Random Graphs* (2nd ed.). Cambridge University Press.

8. Grimmett, G. R. (1999). *Percolation* (2nd ed.). Springer.

9. Cohen, R., Erez, K., ben-Avraham, D., & Havlin, S. (2001). Breakdown of the Internet under intentional attack. *Physical Review Letters*, 86(16), 3682--3685.
