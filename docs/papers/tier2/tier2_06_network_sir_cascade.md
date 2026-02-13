# Network-Aware SIR Epidemic Model for Debris Cascade Propagation on ISL Graphs

**Authors**: Humeris Research
**Status**: Tier 2 -- Validated Conceptually, Not Yet Implemented
**Date**: February 2026
**Library Version**: Humeris v1.22.0

---

## Abstract

Debris cascade modeling in low Earth orbit has traditionally employed
mean-field SIR (Susceptible-Infected-Recovered) epidemic dynamics, where
spatial structure is abstracted into a single well-mixed shell volume. This
approximation discards information that may be significant: in operational constellations with
inter-satellite links (ISLs), the network topology creates preferential
channels for cascade propagation. When a satellite is destroyed by debris
impact, the resulting fragment cloud threatens geometrically proximate
satellites -- precisely those connected by ISL links. We propose a
Network-Aware SIR (NA-SIR) model that evolves epidemic dynamics directly
on the ISL adjacency graph. The per-link infection rate
$\beta_{ij} = \sigma v_{\text{rel}} V_{\text{shell}}^{-1} \exp(-d_{ij} / L_{\text{frag}})$
depends on ISL link distance $d_{ij}$ and a fragment dispersion length scale
$L_{\text{frag}}$. The network basic reproduction number
$R_0^{\text{net}} = \lambda_{\max}(A) \langle \beta \rangle / \langle \gamma \rangle$
is governed by the largest eigenvalue of the adjacency matrix, linking cascade
criticality to graph spectral properties. We derive conditions under which
the Fiedler value $\lambda_2(L)$ of the graph Laplacian drops below a
fragmentation threshold during cascade evolution, causing the ISL network to
partition. This framework is intended to enable vulnerability assessment that accounts for
constellation-specific topology rather than relying on shell-averaged density
estimates.

---

## 1. Introduction

### 1.1 Motivation

The Kessler syndrome -- a self-sustaining collisional cascade in orbital
debris -- has been studied primarily through mean-field models that treat
the orbital environment as a spatially homogeneous gas of debris particles
and intact satellites (Kessler & Cour-Palais 1978, Kessler 1991). The SIR
epidemic analogy was formalized in this mean-field context: intact satellites
are *susceptible*, debris fragments are *infected*, and deorbited fragments
are *recovered* (Rossi et al. 1998). The mean-field SIR model, as
implemented in the Humeris `cascade_analysis` module, provides useful
aggregate metrics -- the basic reproduction number $R_0$, time to peak debris,
and equilibrium population -- but necessarily averages over the spatial
structure of the constellation.

In practice, modern mega-constellations (Starlink, OneWeb, Kuiper) are not
randomly distributed shells of independent satellites. They form structured
networks connected by inter-satellite laser links (ISLs), where the network
topology is a first-order design variable affecting both communication
performance and cascade vulnerability. When a satellite at position $i$ is
destroyed, the resulting fragment cloud does not threaten all other satellites
equally. The probability of secondary impact decreases with distance from the
initial event, following the ballistic dispersion of the fragment cloud. This
creates a spatial correlation: satellites close to the destroyed node --
precisely those connected by ISL links -- face elevated risk.

### 1.2 Problem Statement

The gap between mean-field SIR cascade models and the network reality of
operational constellations creates two specific deficiencies:

1. **Incorrect criticality thresholds.** Mean-field models predict cascade
   onset at a critical debris density $\rho_c$ that depends only on shell
   volume and collision parameters. Under this model, the threshold also depends on the
   network's spectral properties -- highly connected constellations may
   cascade at lower debris densities than mean-field predictions suggest.

2. **No topology-dependent vulnerability.** Mean-field models cannot
   distinguish between network configurations. A constellation with ISL
   links forming a regular lattice has different cascade vulnerability than
   one with small-world or scale-free topology, even if both occupy the same
   orbital shell at identical spatial density.

### 1.3 Contribution

We propose the Network-Aware SIR (NA-SIR) model that:

- Formulates SIR dynamics on the ISL adjacency graph with
  distance-dependent contact rates.
- Derives a network-specific $R_0^{\text{net}}$ from the spectral radius
  of the weighted adjacency matrix.
- Establishes a connection between cascade progression and Fiedler value
  degradation, predicting the moment of ISL network fragmentation.
- Accounts for dynamic graph evolution: as nodes are destroyed, the
  adjacency matrix changes, creating a coupled epidemic-network system.

---

## 2. Background

### 2.1 Mean-Field SIR for Debris Cascades

The classical SIR model for debris cascade dynamics partitions the
population into three compartments:

- $S(t)$: Susceptible -- intact operational satellites.
- $I(t)$: Infected -- active debris fragments capable of causing collisions.
- $R(t)$: Recovered -- fragments that have deorbited or become inactive.

The mean-field equations (as implemented in `cascade_analysis.compute_cascade_sir`) are:

$$\frac{dS}{dt} = \Lambda - \beta S I$$

$$\frac{dI}{dt} = N_{\text{frag}} \beta S I - \gamma I$$

$$\frac{dR}{dt} = \gamma I$$

where $\Lambda$ is the launch replenishment rate,
$\beta = \sigma v_{\text{rel}} / V_{\text{shell}}$ is the volumetric
collision rate, $N_{\text{frag}}$ is the average fragment count per
collision, and $\gamma = 1 / \tau_{\text{drag}}$ is the drag removal rate.

The basic reproduction number is:

$$R_0 = \frac{N_{\text{frag}} \beta S_0}{\gamma}$$

When $R_0 > 1$, the debris population grows exponentially in the initial
phase; when $R_0 < 1$, debris is removed faster than it is produced.

### 2.2 Epidemic Models on Networks

The extension of SIR dynamics to structured networks was pioneered by
Pastor-Satorras and Vespignani (2001), who showed that on scale-free
networks, the epidemic threshold vanishes: $R_0^c \to 0$ as $N \to \infty$
for networks with power-law degree distributions $P(k) \sim k^{-\gamma}$
with $2 < \gamma \leq 3$. This result has notable implications: in
heterogeneous networks, even weakly infectious diseases can become endemic.

For a general network with adjacency matrix $A$, the heterogeneous mean-field
(HMF) approximation gives the epidemic threshold (Wang et al. 2003):

$$\frac{\langle \beta \rangle}{\langle \gamma \rangle} > \frac{1}{\lambda_{\max}(A)}$$

where $\lambda_{\max}(A)$ is the spectral radius (largest eigenvalue) of
the adjacency matrix. This result is tight for uncorrelated networks and
provides a lower bound for general graphs (Chakrabarti et al. 2008).

### 2.3 ISL Network Topology in Humeris

The Humeris library provides comprehensive graph-theoretic analysis of ISL
networks through the `graph_analysis` module. Key capabilities include:

- **Algebraic connectivity:** The Fiedler value $\lambda_2(L)$ of the
  graph Laplacian measures bottleneck connectivity (`compute_topology_resilience`).
- **Hodge Laplacian analysis:** First Betti number $\beta_1$ quantifies
  independent routing cycles (`compute_hodge_topology`).
- **Fragmentation timeline:** Time series of $\lambda_2(t)$ with
  eclipse-degraded weights (`compute_fragmentation_timeline`).
- **Percolation analysis:** Ising model phase transitions for ISL networks
  (`compute_isl_phase_transition`).

The mean-field SIR model exists in `cascade_analysis.compute_cascade_sir`,
along with the birth-death chain (`compute_debris_birth_death`) and
multi-species Lotka-Volterra model (`compute_lotka_volterra_debris`). None
of these models incorporate network structure.

### 2.4 Fragment Cloud Dispersion

After a hypervelocity collision at orbital altitude $h$, the resulting
fragment cloud disperses over time. The initial velocity perturbation
follows a distribution determined by the collision geometry and the
NASA Standard Breakup Model (Johnson et al. 2001). For our purposes,
we model the fragment cloud as an expanding sphere with characteristic
dispersion length:

$$L_{\text{frag}}(t) = \Delta v_{\text{rms}} \cdot t$$

where $\Delta v_{\text{rms}}$ is the root-mean-square velocity perturbation
of the fragments. For a typical LEO collision, $\Delta v_{\text{rms}} \approx$
100--300 m/s, giving $L_{\text{frag}} \approx$ 100--1000 km over one orbital
period. This length scale determines how rapidly the elevated collision risk
transitions from a localized threat to a shell-wide average.

---

## 3. Proposed Method

### 3.1 Network SIR Formulation

Consider a constellation of $N$ satellites connected by an ISL network
described by adjacency matrix $A \in \{0, 1\}^{N \times N}$. We assign
to each satellite $i$ a state variable $s_i(t) \in \{S, I, R\}$:

- $S_i(t)$: Probability that satellite $i$ is intact at time $t$.
- $I_i(t)$: Probability that satellite $i$ has been destroyed and its
  fragment cloud is active.
- $R_i(t)$: Probability that the fragments from satellite $i$ have
  deorbited.

The continuous-time network SIR equations are:

$$\frac{dS_i}{dt} = -S_i \sum_{j=1}^N A_{ij} \beta_{ij} I_j$$

$$\frac{dI_i}{dt} = S_i \sum_{j=1}^N A_{ij} \beta_{ij} I_j - \gamma_i I_i$$

$$\frac{dR_i}{dt} = \gamma_i I_i$$

where $\beta_{ij}$ is the per-link infection rate from node $j$ to node $i$,
and $\gamma_i$ is the recovery rate for fragments originating from node $i$.

### 3.2 Distance-Dependent Contact Rate

The distinguishing element of this formulation is the distance-dependent contact rate. When satellite $j$
is destroyed, its fragment cloud preferentially threatens nearby satellites.
We model this using the ISL link distance $d_{ij}$ as a proxy for spatial
proximity:

$$\beta_{ij} = \frac{\sigma v_{\text{rel}}}{V_{\text{shell}}} \exp\left(-\frac{d_{ij}}{L_{\text{frag}}}\right)$$

where:
- $\sigma$ is the collision cross-section ($\text{km}^2$).
- $v_{\text{rel}}$ is the mean relative velocity (m/s).
- $V_{\text{shell}}$ is the shell volume ($\text{km}^3$).
- $d_{ij}$ is the ISL link distance between satellites $i$ and $j$ (km).
- $L_{\text{frag}}$ is the fragment cloud dispersion length scale (km).

The exponential decay captures the physical fact that fragment density
decreases with distance from the collision event. When $L_{\text{frag}} \to \infty$,
we recover the mean-field model (all $\beta_{ij}$ equal); when
$L_{\text{frag}} \to 0$, only directly connected neighbors are threatened.

### 3.3 Weighted Adjacency and Effective Infection Matrix

Define the weighted infection matrix:

$$W_{ij} = A_{ij} \cdot \beta_{ij}$$

The network basic reproduction number is:

$$R_0^{\text{net}} = \frac{\lambda_{\max}(W)}{\langle \gamma \rangle}$$

This follows from the linearized stability analysis of the disease-free
equilibrium (DFE) $(S_i = 1, I_i = 0, R_i = 0)$ for all $i$. Near the
DFE, the infection dynamics are governed by:

$$\frac{dI_i}{dt} \approx \sum_j W_{ij} I_j - \gamma_i I_i$$

In matrix form: $\dot{\mathbf{I}} = (W - \Gamma) \mathbf{I}$, where
$\Gamma = \text{diag}(\gamma_1, \ldots, \gamma_N)$. The DFE is unstable
(cascade occurs) when $\lambda_{\max}(W - \Gamma) > 0$, i.e., when:

$$\lambda_{\max}(W) > \gamma_{\min}$$

For homogeneous recovery rates $\gamma_i = \gamma$:

$$R_0^{\text{net}} = \frac{\lambda_{\max}(W)}{\gamma} > 1$$

### 3.4 Dynamic Graph Evolution

A critical feature of debris cascades is that the network changes as nodes
are destroyed. When satellite $j$ transitions from $S$ to $I$, row $j$ and
column $j$ of the adjacency matrix are effectively removed (or zeroed out,
since the ISL links to a destroyed satellite no longer function). This gives
rise to a coupled system:

$$A(t) = A_0 \odot \mathbf{s}(t) \mathbf{s}(t)^T$$

where $\mathbf{s}(t) \in \{0, 1\}^N$ is the survival indicator vector
($s_i = 1$ if $S_i > 0.5$, 0 otherwise) and $\odot$ denotes
element-wise product. The spectral properties of $A(t)$ change as nodes
are removed, potentially triggering cascading disconnections.

### 3.5 Fiedler-Linked Fragmentation Criterion

The Fiedler value $\lambda_2(L(t))$ of the graph Laplacian
$L(t) = D(t) - A(t)$ (where $D$ is the degree matrix) measures the
algebraic connectivity of the surviving ISL network. We define the
fragmentation criterion:

$$\lambda_2(L(t)) < \epsilon_{\text{frag}}$$

When this condition is met, the ISL network has effectively fragmented into
disconnected components. This has operational consequences: communication
routing fails, distributed computation halts, and coordinated collision
avoidance becomes impossible.

The cascade-induced Fiedler evolution can be approximated by:

$$\frac{d\lambda_2}{dt} \approx -\lambda_2 \cdot \frac{d}{dt}\left[\sum_i I_i\right] \cdot c_{\text{Fiedler}}$$

where $c_{\text{Fiedler}}$ depends on the eigenvector structure of the
Laplacian and the location of destroyed nodes. Nodes with large Fiedler
vector components are the critical bottlenecks; their destruction causes
disproportionate connectivity loss.

### 3.6 Algorithm Description

The NA-SIR cascade algorithm proceeds as follows:

**Input:** Adjacency matrix $A_0$, ISL distances $d_{ij}$, model
parameters $(\sigma, v_{\text{rel}}, V_{\text{shell}}, L_{\text{frag}},
\gamma, N_{\text{frag}})$, initial infection set $\mathcal{I}_0$.

**Output:** Time series $(S_i(t), I_i(t), R_i(t))$ for all nodes,
Fiedler evolution $\lambda_2(t)$, fragmentation time $t_{\text{frag}}$.

1. Initialize: $S_i(0) = 1$ for $i \notin \mathcal{I}_0$;
   $I_i(0) = 1$ for $i \in \mathcal{I}_0$; $R_i(0) = 0$ for all $i$.
2. Compute weighted infection matrix $W_{ij} = A_{ij} \cdot \beta_{ij}$.
3. For each time step $t_k = k \Delta t$:
   - a. Update adjacency: $A(t_k) = A_0 \odot \mathbf{s}(t_k) \mathbf{s}(t_k)^T$.
   - b. Recompute $W(t_k) = A(t_k) \odot B$ where $B_{ij} = \beta_{ij}$.
   - c. Forward Euler step for each node $i$:
     - $S_i \leftarrow S_i - S_i \sum_j W_{ij}(t_k) I_j \cdot \Delta t$
     - $I_i \leftarrow I_i + (S_i \sum_j W_{ij}(t_k) I_j - \gamma_i I_i) \cdot \Delta t$
     - $R_i \leftarrow R_i + \gamma_i I_i \cdot \Delta t$
   - d. Clamp: $S_i, I_i, R_i \geq 0$; normalize $S_i + I_i + R_i = 1$.
   - e. Compute graph Laplacian $L(t_k)$ and Fiedler value $\lambda_2(t_k)$.
   - f. If $\lambda_2(t_k) < \epsilon_{\text{frag}}$: record fragmentation.
4. Compute $R_0^{\text{net}} = \lambda_{\max}(W_0) / \gamma$.
5. Return results.

### 3.7 Fragment Multiplication

When a collision destroys satellite $j$, it generates $N_{\text{frag}}$
fragments. In the network SIR model, this is captured by the factor
$N_{\text{frag}}$ in $R_0^{\text{net}}$. Specifically, the infection
matrix should be:

$$W_{ij}^{\text{eff}} = N_{\text{frag}} \cdot A_{ij} \cdot \frac{\sigma v_{\text{rel}}}{V_{\text{shell}}} \cdot \exp\left(-\frac{d_{ij}}{L_{\text{frag}}}\right)$$

This accounts for the amplification: each collision event produces multiple
fragments, each of which independently threatens neighboring satellites.

---

## 4. Theoretical Analysis

### 4.1 Epidemic Threshold

**Theorem 1** (Network Cascade Threshold). *The disease-free equilibrium
of the NA-SIR model is locally asymptotically stable if and only if:*

$$\lambda_{\max}(W^{\text{eff}}) < \gamma_{\min}$$

*Equivalently, the cascade is subcritical when:*

$$R_0^{\text{net}} = \frac{\lambda_{\max}(W^{\text{eff}})}{\gamma_{\min}} < 1$$

*Proof sketch.* The Jacobian of the infection subsystem at the DFE is
$J = W^{\text{eff}} - \Gamma$. By the Perron-Frobenius theorem,
$W^{\text{eff}}$ (non-negative and irreducible for a connected ISL graph)
has a dominant real eigenvalue $\lambda_{\max}(W^{\text{eff}})$. The DFE
stability requires all eigenvalues of $J$ to have negative real parts, which
holds iff $\lambda_{\max}(W^{\text{eff}}) < \gamma_{\min}$.

**Corollary 1.** *The network $R_0^{\text{net}}$ is bounded:*

$$R_0^{\text{MF}} \leq R_0^{\text{net}} \leq R_0^{\text{MF}} \cdot \frac{\lambda_{\max}(A)}{d_{\text{avg}}}$$

*where $d_{\text{avg}}$ is the average degree and $R_0^{\text{MF}}$ is
the mean-field reproduction number.*

This bound shows that for regular graphs (all degrees equal),
$R_0^{\text{net}} = R_0^{\text{MF}}$, while for heterogeneous graphs
with high spectral radius relative to mean degree, the network
$R_0$ can be substantially larger.

### 4.2 Percolation Connection

The NA-SIR model connects to bond percolation theory. Define the
transmissibility of link $(i, j)$:

$$T_{ij} = 1 - \exp\left(-\frac{\beta_{ij}}{\gamma}\right)$$

This is the probability that infection transmits from $j$ to $i$ before $j$
recovers. The cascade percolates (infects a macroscopic fraction of the
network) when the link occupation probability exceeds the bond percolation
threshold:

$$\langle T \rangle > p_c(G)$$

where $p_c(G)$ is the bond percolation threshold for the graph topology $G$.
For random $d$-regular graphs, $p_c \approx 1/(d-1)$; for Erdos-Renyi
graphs with mean degree $z$, $p_c \approx 1/(z-1)$ (Newman 2010).

### 4.3 Fragmentation Dynamics

As the cascade progresses and nodes are removed, the Fiedler value
$\lambda_2$ decreases. We analyze this through interlacing inequalities.

**Lemma 1** (Monotone Fiedler Decrease). *For a connected graph $G$ and
a vertex $v$, the Fiedler value of the subgraph $G - v$ satisfies:*

$$\lambda_2(G - v) \leq \lambda_2(G) + \frac{\|L_v\|^2}{\lambda_2(G)}$$

*where $L_v$ is the row of the Laplacian corresponding to vertex $v$.
In particular, removing high-degree nodes decreases $\lambda_2$ more.*

The cascade preferentially threatens high-degree nodes (they have more ISL
links, hence more exposure to fragments). This creates a positive feedback
loop: the most topologically critical nodes are destroyed first, causing
disproportionate connectivity loss.

**Definition 1** (Fragmentation Time). *The fragmentation time is:*

$$t_{\text{frag}} = \inf\{t : \lambda_2(L(t)) < \epsilon_{\text{frag}}\}$$

We estimate $t_{\text{frag}}$ from the cascade growth rate and the initial
Fiedler value:

$$t_{\text{frag}} \approx \frac{\lambda_2(L(0))}{\lambda_{\max}(W^{\text{eff}}) - \gamma} \cdot c_{\text{topology}}$$

where $c_{\text{topology}}$ depends on the correlation between node degree
and Fiedler vector component.

### 4.4 Computational Complexity

For a network with $N$ nodes and $M$ ISL links:

- **Per-step computation:** The forward Euler step requires $O(M)$ for
  the infection sums (sparse adjacency). The adjacency update is $O(N)$
  (mark destroyed nodes). Total per-step: $O(M + N)$.
- **Fiedler computation:** Each Laplacian eigendecomposition costs $O(N^3)$
  with dense methods or $O(M \cdot k)$ with Lanczos iteration for the
  $k$ smallest eigenvalues. Since we only need $\lambda_2$, Lanczos
  with $k = 3$ is efficient.
- **Total complexity:** For $T$ time steps: $O(T(M + N))$ for the SIR
  dynamics plus $O(T \cdot M)$ for Fiedler tracking.
  Total: $O(T \cdot (M + N))$.
- **Spectral radius computation:** Computing $\lambda_{\max}(W)$ via
  power iteration is $O(M \cdot k_{\text{iter}})$, done once at
  initialization.

For a Walker constellation with $N = 1000$ satellites and $M \approx 4N$
ISL links, the per-step cost is $O(4000)$, well within real-time
computation budgets.

### 4.5 Convergence of the Forward Euler Scheme

The forward Euler discretization introduces numerical error. Stability
requires:

$$\Delta t < \frac{1}{\max_i \sum_j W_{ij}^{\text{eff}} + \gamma_{\max}}$$

This is the CFL-analogous condition for the network SIR system. The
Humeris `cascade_analysis` module already implements adaptive sub-stepping
for the mean-field SIR model when the growth metric exceeds 1.0; the same
approach applies here with the network-specific growth metric
$\max_i \sum_j W_{ij}^{\text{eff}}$.

### 4.6 Relationship to Existing Humeris Models

The NA-SIR model generalizes the existing mean-field SIR in `cascade_analysis`:

| Property | Mean-field SIR | NA-SIR |
|----------|---------------|--------|
| Spatial structure | None (well-mixed) | ISL adjacency graph |
| Contact rate | Uniform $\beta$ | Distance-dependent $\beta_{ij}$ |
| $R_0$ | $N_{\text{frag}} \beta S_0 / \gamma$ | $\lambda_{\max}(W^{\text{eff}}) / \gamma$ |
| Graph evolution | N/A | Dynamic adjacency |
| Fragmentation | Not tracked | Fiedler-based criterion |
| Complexity | $O(T)$ | $O(T \cdot (M + N))$ |

---

## 5. Proposed Validation

### 5.1 Validation Against Mean-Field Limit

As $L_{\text{frag}} \to \infty$, all $\beta_{ij} \to \beta$ (constant),
and the NA-SIR should converge to the mean-field SIR. Validation test:

1. Generate a Walker constellation with $N = 100$ satellites.
2. Compute ISL topology using `inter_satellite_links.compute_isl_topology`.
3. Run `cascade_analysis.compute_cascade_sir` with mean-field parameters.
4. Run NA-SIR with $L_{\text{frag}} = 10^6$ km (effectively infinite).
5. Verify that total $S(t)$, $I(t)$, $R(t)$ curves match within 5%.

### 5.2 Spectral Radius Validation

Use `graph_analysis.compute_topology_resilience` to compute the Fiedler
value and verify the spectral properties of the adjacency matrix:

1. Construct adjacency matrices for known graph topologies (ring, complete,
   star, random regular).
2. Compute $\lambda_{\max}(A)$ analytically (known for these graphs).
3. Verify that $R_0^{\text{net}}$ computed from $\lambda_{\max}(W)$
   matches the analytical prediction.

### 5.3 Fragmentation Tracking

Use `graph_analysis.compute_fragmentation_timeline` to validate the
Fiedler evolution:

1. Simulate a cascade with sequential node removal.
2. At each step, compute $\lambda_2(L)$ using both the NA-SIR tracker
   and the existing `compute_topology_resilience` function.
3. Verify consistency.

### 5.4 Topology Sensitivity Analysis

Compare cascade outcomes across different ISL topologies:

1. **Regular lattice** (each satellite linked to $k$ nearest neighbors).
2. **Random geometric graph** (ISL links for satellites within range $r$).
3. **Small-world** (lattice with random rewiring, Watts-Strogatz model).
4. **Scale-free** (preferential attachment, though unusual for ISLs).

For each topology, compute $R_0^{\text{net}}$ and $t_{\text{frag}}$,
demonstrating that topology matters.

### 5.5 Comparison with Birth-Death and Lotka-Volterra Models

Cross-validate the aggregate behavior (total infected fraction over time)
of NA-SIR against the existing stochastic models in `cascade_analysis`:

1. `compute_debris_birth_death` (stochastic birth-death chain).
2. `compute_lotka_volterra_debris` (multi-species competitive dynamics).
3. Compare the supercritical/subcritical classification.

---

## 6. Discussion

### 6.1 Limitations

**Static fragment dispersion model.** The exponential decay
$\exp(-d_{ij}/L_{\text{frag}})$ is a first-order approximation. In reality,
fragment cloud evolution depends on orbital mechanics, differential
precession, and atmospheric drag. A more accurate model would propagate
fragment trajectories using `numerical_propagation` and compute true
encounter probabilities.

**Deterministic continuous approximation.** The continuous-time SIR
equations approximate what is fundamentally a stochastic discrete process.
For small constellations ($N < 50$), stochastic effects dominate and the
continuous approximation may be poor. A stochastic network SIR (Gillespie
algorithm on graphs) would be more appropriate for small $N$.

**ISL distance as proximity proxy.** ISL link distance is a reasonable
but imperfect proxy for geometric proximity. Two satellites may be
geometrically close but not connected by an ISL (e.g., in different
orbital planes). The model should be augmented with a geometric distance
matrix $D$ independent of the ISL topology.

**Computational cost of dynamic eigendecomposition.** Computing $\lambda_2(L(t))$
at every time step is the computational bottleneck for large constellations.
Incremental eigenvalue updates after node removal (rank-1 downdates) could
reduce this cost.

### 6.2 Open Questions

1. **Optimal constellation design for cascade resistance.** Given the
   NA-SIR framework, what ISL topology minimizes $R_0^{\text{net}}$ while
   maintaining communication performance (high Fiedler value)?

2. **Vaccination analogy.** In epidemic terms, "vaccinating" a satellite
   means hardening it against debris impact (e.g., debris shields) or
   making it non-fragmenting (e.g., design for demise). Which nodes
   should be "vaccinated" to maximally reduce $R_0^{\text{net}}$? The
   answer involves the eigenvector centrality of the infection matrix.

3. **Coupling with Kessler heatmap.** The `kessler_heatmap` module
   provides spatial density by altitude and inclination band. Can the
   NA-SIR per-link $\beta_{ij}$ be informed by the local cell density
   from the Kessler heatmap?

4. **Time-dependent $L_{\text{frag}}$.** The fragment dispersion length
   scale grows over time as the cloud expands. How does a time-varying
   $L_{\text{frag}}(t)$ affect the cascade dynamics?

### 6.3 Prerequisites for Implementation

The following existing Humeris modules would be composed:

- `cascade_analysis`: Mean-field SIR model (baseline comparison).
- `graph_analysis`: Adjacency construction, Fiedler computation, Hodge
  topology, ISL phase transitions.
- `inter_satellite_links`: ISL topology and link distances.
- `kessler_heatmap`: Spatial density and collision velocity fields.
- `linalg`: Eigendecomposition (`mat_eigenvalues_symmetric`), DFT.

New implementation required:
- Network SIR integrator (forward Euler on graph).
- Dynamic adjacency update (node removal).
- Spectral radius computation for weighted infection matrix.
- Fragment dispersion kernel computation.

Estimated complexity: **Medium**. The core algorithm is a straightforward
extension of the existing SIR integrator in `cascade_analysis`, but the
coupling between SIR state evolution and graph adjacency updates (node
removal changes $A$, which changes $W$, which changes infection dynamics)
introduces implementation complexity. The dynamic eigendecomposition for
Fiedler tracking adds further cost.

---

## 7. Conclusion

We have proposed the Network-Aware SIR (NA-SIR) model for debris cascade
propagation on ISL constellation graphs. By formulating epidemic dynamics
directly on the network adjacency matrix with distance-dependent contact
rates, the model captures the spatial structure that mean-field approaches
discard. The network basic reproduction number $R_0^{\text{net}}$,
governed by the spectral radius of the weighted infection matrix, provides
a topology-specific criticality criterion. The Fiedler-linked fragmentation
criterion connects cascade progression to ISL network connectivity loss,
providing a basis for estimating when the constellation's communication
backbone fractures.

The framework composes naturally with existing Humeris capabilities:
graph-theoretic analysis from `graph_analysis`, spectral topology from
`spectral_topology`, and cascade models from `cascade_analysis`. The
primary implementation challenge lies in coupling the SIR state evolution
with dynamic graph updates, requiring careful numerical treatment of a
system where the governing equations change their structure as the
simulation progresses.

---

## References

1. Chakrabarti, D., Wang, Y., Wang, C., Leskovec, J., & Faloutsos, C. (2008). Epidemic thresholds in real networks. *ACM Transactions on Information and System Security*, 10(4), 1--26.

2. Johnson, N. L., Krisko, P. H., Liou, J.-C., & Anz-Meador, P. D. (2001). NASA's new breakup model of EVOLVE 4.0. *Advances in Space Research*, 28(9), 1377--1384.

3. Kessler, D. J., & Cour-Palais, B. G. (1978). Collision frequency of artificial satellites: The creation of a debris belt. *Journal of Geophysical Research*, 83(A6), 2637--2646.

4. Kessler, D. J. (1991). Collisional cascading: The limits of population growth in low Earth orbit. *Advances in Space Research*, 11(12), 63--66.

5. Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.

6. Pastor-Satorras, R., & Vespignani, A. (2001). Epidemic spreading in scale-free networks. *Physical Review Letters*, 86(14), 3200--3203.

7. Rossi, A., Cordelli, A., Farinella, P., & Anselmo, L. (1998). Collisional evolution of the Earth's orbital debris cloud. *Journal of Geophysical Research*, 99(E11), 23195--23210.

8. Stauffer, D., & Aharony, A. (1994). *Introduction to Percolation Theory* (2nd ed.). Taylor & Francis.

9. Wang, Y., Chakrabarti, D., Wang, C., & Faloutsos, C. (2003). Epidemic spreading in real networks: An eigenvalue viewpoint. *Proceedings of the 22nd IEEE Symposium on Reliable Distributed Systems*, 25--34.
