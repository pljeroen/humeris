# BIJE: Bayesian Intent Joint Estimation for Space Situational Awareness

**Authors**: Humeris Research
**Status**: Tier 2 -- Validated Conceptually, Not Yet Implemented
**Date**: February 2026
**Library Version**: Humeris v1.22.0

---

## Abstract

Current space situational awareness (SSA) systems detect maneuvers and
estimate orbital states as separate tasks. Maneuver detection algorithms
(CUSUM, EWMA, chi-squared) identify *when* a maneuver occurs; orbit
determination (EKF, particle filter) estimates *where* the satellite is.
Neither answers the operationally critical question: *what does the operator
intend?* We propose Bayesian Intent Joint Estimation (BIJE), a unified
framework that simultaneously estimates the orbital state and the operator's
intent by coupling an Extended Kalman Filter (EKF) for state estimation with
a Hidden Markov Model (HMM) for intent classification. The intent state
$I_t \in \{\text{SK}, \text{DO}, \text{EV}, \text{UN}\}$ (station-keeping,
deorbit, evasive maneuver, uncontrolled) evolves according to a
physics-informed transition matrix $P(I_{t+1} | I_t)$. The EKF residuals
serve as HMM observations, with emission distributions
$P(\mathbf{r}_t | I_t)$ calibrated to the residual signatures of each
intent mode. The forward algorithm computes the intent posterior
$P(I_t | \mathbf{r}_{1:t})$ in $O(K^2 T)$ time, where $K = 4$ is the
number of intent states and $T$ is the number of observations. The
predictive collision probability integrates over intent:
$P_c = \sum_I P(\text{collision} | I) P(I | \mathbf{r}_{1:t})$, providing
a risk assessment that accounts for estimated operator intent. BIJE builds on
the existing Humeris EKF (`orbit_determination.run_ekf`) and maneuver
detection suite (`maneuver_detection`) while adding the HMM layer for
intent estimation.

---

## 1. Introduction

### 1.1 Motivation

The space environment around Earth hosts over 30,000 tracked objects,
with an increasing fraction being actively controlled satellites in
mega-constellations. Conjunction assessment -- predicting potential
collisions and deciding whether to maneuver -- is the central challenge
of space traffic management.

Current conjunction assessment workflows compute the probability of
collision $P_c$ assuming both objects follow their predicted trajectories.
When a maneuver is detected (via CUSUM, EWMA, or chi-squared residual
analysis), the predicted trajectory is updated, and $P_c$ is recomputed.
However, the maneuver detection only identifies that a deviation from
the predicted orbit has occurred; it does not classify the *purpose*
of the maneuver.

The purpose matters enormously for operational decisions:

- **Station-keeping (SK):** Routine orbit maintenance. The satellite
  will return to its nominal orbit. Future trajectory is predictable.
- **Deorbit (DO):** End-of-life disposal. The satellite is leaving the
  operational altitude. Collision risk decreases over time.
- **Evasive maneuver (EV):** The operator is responding to a conjunction.
  The satellite will move away from the predicted closest approach point.
  Collision risk should decrease, but the new trajectory may create
  secondary conjunctions.
- **Uncontrolled (UN):** The satellite is not responding to commands
  (attitude failure, propulsion failure). Future trajectory is determined
  only by orbital mechanics. This is the highest-risk state.

### 1.2 Problem Statement

Existing Humeris maneuver detection methods
(`maneuver_detection.detect_maneuvers_cusum`,
`detect_maneuvers_chi_squared`, `detect_maneuvers_ewma`,
`wald_sequential_test`) detect the *occurrence* of maneuvers but not
their *intent*. The orbit determination EKF (`orbit_determination.run_ekf`)
estimates the *state* but not the *mode*. There is no unified framework
that jointly estimates state and intent.

The question is: **given a sequence of tracking observations, can we
jointly estimate the orbital state AND the operator's intent in a
principled Bayesian framework?**

### 1.3 Contribution

We propose BIJE, which:

1. Models operator intent as a discrete Hidden Markov Model with
   states $\{SK, DO, EV, UN\}$ and physics-informed transitions.
2. Uses EKF residuals as HMM observations, with intent-specific
   emission distributions.
3. Computes the intent posterior via the forward algorithm.
4. Provides intent-conditioned trajectory predictions and collision
   probabilities.
5. Integrates with the existing EKF and maneuver detection infrastructure.

---

## 2. Background

### 2.1 Hidden Markov Models

A Hidden Markov Model (Rabiner 1989) is defined by:

- **State space** $\mathcal{I} = \{I_1, \ldots, I_K\}$: Discrete hidden states.
- **Transition matrix** $\mathbf{A}$: $A_{ij} = P(I_{t+1} = j | I_t = i)$.
- **Emission distributions** $B_k(\mathbf{o})$:
  $B_k(\mathbf{o}) = P(\mathbf{o}_t | I_t = k)$.
- **Initial distribution** $\boldsymbol{\pi}$:
  $\pi_k = P(I_0 = k)$.

The forward algorithm computes the filtered posterior:

$$\alpha_t(k) = P(\mathbf{o}_{1:t}, I_t = k) = B_k(\mathbf{o}_t) \sum_{j=1}^K \alpha_{t-1}(j) A_{jk}$$

The intent posterior is obtained by normalization:

$$P(I_t = k | \mathbf{o}_{1:t}) = \frac{\alpha_t(k)}{\sum_{k'} \alpha_t(k')}$$

The forward algorithm has complexity $O(K^2 T)$ per observation sequence.

### 2.2 EKF for Orbit Determination

The Humeris EKF (`orbit_determination.run_ekf`) provides:

- State estimate $\hat{\mathbf{x}}_t = (x, y, z, v_x, v_y, v_z)_t$.
- Covariance $P_t$ (6x6).
- Post-fit residual $r_t = |\mathbf{z}_t - H \hat{\mathbf{x}}_t|$.

The residual $r_t$ is the magnitude of the innovation vector. Under
the null hypothesis (no maneuver, correct dynamics model), the normalized
residual $r_t / \sigma_t$ follows a chi distribution with 3 degrees of
freedom (position-only observations).

### 2.3 Maneuver Detection in Humeris

The Humeris maneuver detection suite provides four detectors:

1. **CUSUM** (`detect_maneuvers_cusum`): Two-sided cumulative sum with
   Hawkins-Olwell reset. Optimal for detecting abrupt level shifts.
   Parameterized by threshold $h$ and drift $k$.

2. **Chi-squared** (`detect_maneuvers_chi_squared`): Windowed variance
   ratio test with $\text{DOF} = w - 1$.

3. **EWMA** (`detect_maneuvers_ewma`): Exponentially weighted moving
   average. Optimal for small sustained shifts (low-thrust maneuvers).
   Parameterized by smoothing $\lambda$ and control limit $L$.

4. **SPRT** (`wald_sequential_test`): Sequential Probability Ratio
   Test with exact Wald bounds. Minimax optimal for simple hypotheses.

Each detector outputs `ManeuverEvent` objects with detection time,
CUSUM/test value, and residual magnitude. None provide intent
classification.

### 2.4 Multiple Model Estimation

The Interacting Multiple Model (IMM) estimator (Bar-Shalom et al. 2001)
is the standard approach for tracking targets with unknown dynamics mode.
IMM runs multiple Kalman filters in parallel, one per mode, and combines
their outputs weighted by the mode posterior. BIJE can be viewed as an
IMM variant specialized for the orbital mechanics domain, with the key
difference that the emission model is based on residual signatures
rather than dynamics models.

---

## 3. Proposed Method

### 3.1 Intent State Space

We define four intent states:

$$\mathcal{I} = \{SK, DO, EV, UN\}$$

| State | Meaning | Typical $\Delta v$ | Duration | Residual Signature |
|-------|---------|-------------------|----------|-------------------|
| SK | Station-keeping | 0.5--5 m/s | 1--10 min | Moderate, periodic |
| DO | Deorbit | 10--100 m/s | 1--60 min | Large, sustained decrease in $a$ |
| EV | Evasive maneuver | 0.1--1 m/s | 1--5 min | Small, impulsive, directional |
| UN | Uncontrolled | 0 | Indefinite | Growing, aperiodic |

### 3.2 Transition Matrix

The transition matrix $A_{ij} = P(I_{t+1} = j | I_t = i)$ encodes
physics-informed priors about how operator intent changes between
observation epochs. For an observation interval $\Delta t$:

$$\mathbf{A}(\Delta t) = \begin{pmatrix} 1 - a_{SK} & p_{SK \to DO} & p_{SK \to EV} & p_{SK \to UN} \\ p_{DO \to SK} & 1 - a_{DO} & p_{DO \to EV} & p_{DO \to UN} \\ p_{EV \to SK} & p_{EV \to DO} & 1 - a_{EV} & p_{EV \to UN} \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Key design choices:

1. **UN is absorbing:** An uncontrolled satellite remains uncontrolled.
   This is a conservative assumption; in reality, control may be
   recovered, but this is rare and should be modeled explicitly when
   evidence supports it.

2. **SK is the most common state:** The self-transition probability
   $1 - a_{SK}$ is high (e.g., 0.99 per epoch).

3. **EV is transient:** Evasive maneuvers are brief. The transition
   $EV \to SK$ is high (e.g., 0.5 per epoch).

4. **DO is semi-absorbing:** Once deorbit begins, the satellite rarely
   returns to station-keeping. $1 - a_{DO}$ is high (e.g., 0.95).

5. **Time-dependent transitions:** Transition probabilities scale with
   the observation interval $\Delta t$ via:

   $$p_{ij}(\Delta t) = p_{ij}^{(0)} \cdot \frac{\Delta t}{\Delta t_{\text{ref}}}$$

   where $\Delta t_{\text{ref}}$ is the reference interval (e.g., 1 hour).

### 3.3 Emission Model

The emission distribution $P(\mathbf{r}_t | I_t = k)$ models the
probability of observing residual vector $\mathbf{r}_t$ given intent
$k$. We parameterize this as a mixture of components:

**Station-keeping (SK):**

$$P(\mathbf{r}_t | SK) = \mathcal{N}(\mathbf{r}_t | 0, \sigma_{SK}^2 I_3) + \text{periodic component}$$

Residuals are small, near-zero, possibly with periodic structure from
the orbital dynamics.

**Deorbit (DO):**

$$P(\mathbf{r}_t | DO) = \mathcal{N}(\mathbf{r}_t | \mu_{DO}, \sigma_{DO}^2 I_3)$$

Residuals have a large, sustained radial component ($\mu_{DO}$ is
dominated by the along-track deceleration residual).

**Evasive maneuver (EV):**

$$P(\mathbf{r}_t | EV) = \mathcal{N}(\mathbf{r}_t | \mu_{EV}, \sigma_{EV}^2 I_3) \cdot \mathbb{1}[\text{impulsive}]$$

Residuals show a sharp impulse in a specific direction (typically
cross-track or along-track, depending on the conjunction geometry).

**Uncontrolled (UN):**

$$P(\mathbf{r}_t | UN) = \mathcal{N}(\mathbf{r}_t | 0, \sigma_{UN}^2(t) I_3)$$

where $\sigma_{UN}^2(t)$ grows over time (no station-keeping corrections,
so model errors accumulate). This is the key signature: growing residual
variance is the hallmark of an uncontrolled object.

For practical computation, we use the scalar residual magnitude
$r_t = |\mathbf{r}_t|$ (chi-distributed) rather than the vector
$\mathbf{r}_t$:

$$P(r_t | I_t = k) = \frac{r_t^2}{\sigma_k^2} \exp\left(-\frac{r_t^2}{2\sigma_k^2}\right) \cdot \frac{1}{\sigma_k}$$

This is the Rayleigh distribution (chi distribution with 3 DOF
marginalized over direction).

### 3.4 Forward Algorithm for Intent Estimation

At each observation epoch $t$:

1. **Prediction step:** Propagate the intent distribution forward:

   $$\bar{\alpha}_t(k) = \sum_{j=1}^K \alpha_{t-1}(j) \cdot A_{jk}(\Delta t)$$

2. **Update step:** Incorporate the residual observation:

   $$\alpha_t(k) = P(r_t | I_t = k) \cdot \bar{\alpha}_t(k)$$

3. **Normalization:**

   $$P(I_t = k | r_{1:t}) = \frac{\alpha_t(k)}{\sum_{k'} \alpha_t(k')}$$

The initialization is:

$$\alpha_0(k) = \pi_k \cdot P(r_0 | I_0 = k)$$

where $\boldsymbol{\pi}$ is the prior intent distribution (e.g.,
$\pi_{SK} = 0.9$, $\pi_{UN} = 0.05$, $\pi_{DO} = 0.03$, $\pi_{EV} = 0.02$).

### 3.5 Predictive Collision Probability

The intent-conditioned collision probability is:

$$P_c = \sum_{k \in \mathcal{I}} P(\text{collision} | I = k) \cdot P(I_t = k | r_{1:t})$$

where:

- $P(\text{collision} | SK)$: Computed using the nominal trajectory
  and covariance from the EKF (standard conjunction assessment).
- $P(\text{collision} | DO)$: Reduced, since the satellite is
  decreasing altitude (moving away from the conjunction geometry).
- $P(\text{collision} | EV)$: Significantly reduced, since the operator
  is actively avoiding the conjunction.
- $P(\text{collision} | UN)$: Elevated, since no avoidance will occur.

The conditional collision probabilities can be computed using the
existing Humeris `conjunction` module with mode-specific trajectory
predictions.

### 3.6 Most Likely Intent Sequence

The Viterbi algorithm finds the most likely intent sequence
$\hat{I}_{1:T} = \arg\max_{I_{1:T}} P(I_{1:T} | r_{1:T})$:

1. Initialize: $\delta_0(k) = \pi_k \cdot P(r_0 | k)$,
   $\psi_0(k) = 0$.
2. Recursion:
   $\delta_t(k) = P(r_t | k) \cdot \max_j [\delta_{t-1}(j) A_{jk}]$,
   $\psi_t(k) = \arg\max_j [\delta_{t-1}(j) A_{jk}]$.
3. Backtrack: $\hat{I}_T = \arg\max_k \delta_T(k)$,
   $\hat{I}_t = \psi_{t+1}(\hat{I}_{t+1})$.

The Viterbi algorithm has the same $O(K^2 T)$ complexity as the
forward algorithm.

### 3.7 Integration with Existing Detectors

BIJE complements rather than replaces the existing maneuver detectors:

1. **CUSUM** provides the detection time $t_{\text{detect}}$ with
   controlled false alarm rate (ARL$_0$).
2. **BIJE** provides the intent classification at $t_{\text{detect}}$
   and subsequent times.
3. **Fusion:** At detection time, the CUSUM alarm triggers a BIJE
   intent estimation pass. The BIJE posterior at the detection time
   provides immediate intent classification with quantified uncertainty.

The sequential nature of both algorithms makes them naturally compatible:
CUSUM operates on the same residual sequence as the BIJE forward algorithm.

### 3.8 Adaptive Emission Parameters

The emission distribution parameters ($\sigma_{SK}$, $\sigma_{DO}$,
$\sigma_{EV}$, $\sigma_{UN}$) should adapt to the specific satellite:

1. **Calibration phase:** Use the first $N_{\text{cal}}$ observations
   (assumed SK) to estimate the baseline residual variance
   $\sigma_{SK}^2$. This is analogous to the baseline estimation in
   `maneuver_detection.detect_maneuvers_cusum`.

2. **Relative scaling:** Set $\sigma_{DO} = c_{DO} \sigma_{SK}$,
   $\sigma_{EV} = c_{EV} \sigma_{SK}$, etc., where the scaling
   factors $c_k$ are physics-informed constants.

3. **Online update:** As more observations arrive, update the emission
   parameters using the expected sufficient statistics under the
   current intent posterior (EM-like update).

---

## 4. Theoretical Analysis

### 4.1 Identifiability

**Theorem 1** (Intent Identifiability). *The intent states are
identifiable from residual observations if and only if the emission
distributions are distinct: for any $k \neq k'$, there exists
$r$ such that $P(r | k) \neq P(r | k')$.*

For our emission model, the states are identifiable when:

$$\sigma_{SK} < \sigma_{EV} < \sigma_{DO} < \sigma_{UN}$$

or more generally, when the emission distributions have different
means or variances. The uncontrolled state is identifiable by its
growing variance; the others are distinguished by residual magnitude.

### 4.2 Detection Delay

**Proposition 1** (Expected Detection Delay). *The expected number of
observations to reach intent posterior $P(I_t = k | r_{1:t}) > \theta$
starting from prior $P(I_0 = k) = \pi_k$ is bounded by:*

$$E[T_{\text{detect}}] \leq \frac{\log(\theta / \pi_k)}{D_{\text{KL}}(P(r|k) \| P(r|k'))}$$

*where $D_{\text{KL}}$ is the Kullback-Leibler divergence and $k'$
is the most confusable alternative intent.*

This result follows from the HMM filtering convergence rate (Douc et al.
2004). The detection delay is inversely proportional to the KL divergence
between emission distributions: more distinct signatures allow faster
detection.

### 4.3 Comparison with CUSUM and EWMA

The relationship between BIJE and the existing detectors:

| Property | CUSUM | EWMA | BIJE |
|----------|-------|------|------|
| Detects *when* | Yes | Yes | Yes |
| Classifies *what* | No | No | Yes |
| Optimal for abrupt shifts | Yes (Moustakides) | No | Depends on $\mathbf{A}$ |
| Optimal for gradual shifts | No | Yes (Roberts) | Depends on $\mathbf{A}$ |
| False alarm control | ARL$_0$ | Control limits | Prior-dependent |
| Computational cost | $O(T)$ | $O(T)$ | $O(K^2 T) = O(16T)$ |

BIJE is approximately $16\times$ more expensive than CUSUM per
observation, which is negligible for the typical observation rates
in SSA (seconds to minutes between observations).

### 4.4 Bayes Factor for Intent Discrimination

The Bayes factor between intent $k$ and $k'$ after $T$ observations:

$$B_{kk'} = \frac{P(r_{1:T} | I \text{ includes } k)}{P(r_{1:T} | I \text{ includes } k')} = \frac{\alpha_T(k)}{\alpha_T(k')} \cdot \frac{\pi_{k'}}{\pi_k}$$

A Bayes factor $B_{kk'} > 10$ provides "strong evidence" for intent
$k$ over $k'$ (Jeffreys' scale). The number of observations needed for
strong evidence is:

$$T_{\text{strong}} \approx \frac{\ln 10 + \ln(\pi_{k'} / \pi_k)}{D_{\text{KL}}(P(r|k) \| P(r|k'))}$$

### 4.5 Convergence Properties

**Proposition 2** (Posterior Convergence). *Under the true intent $I^*$,
the posterior probability $P(I_t = I^* | r_{1:t}) \to 1$ as
$t \to \infty$, provided:*

1. *The transition matrix $\mathbf{A}$ is ergodic (except for absorbing
   states which are handled separately).*
2. *The emission distributions are distinct (identifiability).*
3. *The true intent does not change faster than the observation rate.*

*The convergence rate is geometric with exponent determined by the
second-largest eigenvalue of the observation-weighted transition matrix.*

### 4.6 Intent-Informed Covariance Scaling

The BIJE intent posterior can be used to adjust the EKF process noise:

$$Q_t = \sum_{k \in \mathcal{I}} P(I_t = k | r_{1:t}) \cdot Q_k$$

where $Q_k$ is the intent-specific process noise:

- $Q_{SK}$: Small process noise (predictable dynamics).
- $Q_{DO}$: Moderate process noise (known thrust direction, uncertain magnitude).
- $Q_{EV}$: Large process noise (uncertain maneuver timing and magnitude).
- $Q_{UN}$: Growing process noise (unmodeled perturbations accumulate).

This adaptive process noise is expected to improve the EKF state estimate by matching
the dynamics uncertainty to the estimated intent.

---

## 5. Proposed Validation

### 5.1 Simulated Maneuver Scenarios

Generate synthetic observation sequences for each intent mode and
verify correct classification:

1. **SK scenario:** Propagate a LEO orbit with periodic station-keeping
   burns (1 m/s every 30 days). Generate observations with noise.
   Verify: $P(SK) > 0.9$ between burns, transient $P(EV)$ spike during burns.

2. **DO scenario:** Propagate with a sustained retrograde burn
   (50 m/s). Verify: $P(DO) > 0.9$ within 3 observation epochs.

3. **EV scenario:** Insert a 0.5 m/s cross-track impulse at a
   specific time. Verify: $P(EV) > 0.5$ within 2 epochs of the
   impulse, returning to $P(SK) > 0.9$ within 5 epochs.

4. **UN scenario:** Propagate without any corrections for 30 days.
   Verify: $P(UN)$ monotonically increases as residuals grow.

### 5.2 Consistency with Existing Detectors

Verify that BIJE detections are consistent with the existing suite:

1. Run `maneuver_detection.detect_maneuvers_cusum` on a test sequence.
2. Run BIJE on the same sequence.
3. Verify: Every CUSUM detection corresponds to a BIJE intent transition
   (change in most-probable intent).
4. Verify: BIJE provides additional information (intent classification)
   not available from CUSUM alone.

### 5.3 EKF Integration

Replace the constant process noise in `orbit_determination.run_ekf` with
BIJE-adaptive process noise:

1. Run standard EKF on a maneuver scenario.
2. Run BIJE-enhanced EKF with intent-adaptive $Q_t$.
3. Compare RMS residuals and covariance consistency.
4. Verify: BIJE-enhanced EKF has lower RMS residual during maneuver
   recovery (faster convergence to the new trajectory).

### 5.4 Collision Probability Comparison

Compare standard $P_c$ with intent-conditioned $P_c$:

1. Generate a conjunction scenario where satellite A is maneuvering
   (intent: EV).
2. Compute $P_c$ assuming ballistic trajectory (standard approach).
3. Compute $P_c$ using BIJE intent-conditioned prediction.
4. Verify: BIJE $P_c$ is lower when $P(EV) > 0.5$ (the operator is
   avoiding), and higher when $P(UN) > 0.5$ (no avoidance expected).

### 5.5 SPRT Cross-Validation

The SPRT (`wald_sequential_test`) provides optimal detection with
exact false alarm bounds. Compare BIJE with SPRT:

1. Run both on the same observation sequence.
2. SPRT detects binary (maneuver / no maneuver).
3. BIJE provides 4-way classification.
4. Verify: BIJE binary (maneuver = max of {DO, EV}) matches SPRT
   detection timing.

---

## 6. Discussion

### 6.1 Limitations

**Fixed intent state space.** The four-state model
$\{SK, DO, EV, UN\}$ is a simplification. In reality, there are
sub-categories: station-keeping includes East-West and North-South
maneuvers; evasive maneuvers include conjunction avoidance and
collision avoidance (different urgency levels). The state space can
be expanded at the cost of more transition parameters and weaker
emission separation.

**Known transition probabilities.** The transition matrix $\mathbf{A}$
must be specified a priori or learned from training data. The physics
constraints (UN absorbing, EV transient) provide structure, but the
numerical values ($p_{SK \to DO}$, etc.) are operator-dependent.
Different satellite operators have different maneuver cadences.

**Gaussian emission model.** The emission distributions are modeled
as Gaussian (or Rayleigh for the magnitude). For complex maneuver
profiles (multi-burn sequences, low-thrust spirals), the actual
residual distribution may be multimodal or heavy-tailed. The
emission model should be validated against real tracking data.

**No direct intent observation.** BIJE infers intent from residual
signatures, which are indirect observations. Two different intents
may produce similar residual signatures (e.g., a deorbit burn and
an orbit raise both produce along-track acceleration). Additional
observables (e.g., orbital element changes, area-to-mass ratio
changes) could improve discrimination.

**Sequential processing.** The forward algorithm processes
observations sequentially. For batch reprocessing (e.g., after a
new observation contradicts the current intent estimate), the
forward-backward algorithm should be used to smooth the entire
sequence. This doubles the computational cost but provides better
estimates.

### 6.2 Open Questions

1. **Learning transition probabilities from data.** Given historical
   tracking data with labeled intent (from operator announcements or
   post-hoc analysis), can $\mathbf{A}$ and the emission parameters
   be learned via the Baum-Welch (EM) algorithm?

2. **Multi-object BIJE.** In a conjunction scenario, both objects
   have intents. The joint intent $(I_A, I_B)$ has $4 \times 4 = 16$
   states. Can the BIJE framework handle multi-object intent estimation
   efficiently?

3. **Connection to maneuver detection ARL.** The CUSUM ARL$_0$
   controls the false alarm rate. What is the analogous quantity for
   BIJE? The prior distribution $\boldsymbol{\pi}$ acts as an implicit
   false alarm control, but the relationship is not as clean as the
   ARL framework.

4. **Real-time intent updating.** In an operational SSA system, BIJE
   would run continuously on all tracked objects. For 30,000 objects
   with observations every 10 seconds, the total computational load
   is $30000 \times 16 \times 0.1 = 48000$ operations per second --
   computationally feasible.

5. **Integration with BIJE and Koopman conjunction screening.** The
   existing `koopman_conjunction` module screens conjunctions by
   spectral similarity. Can the BIJE intent posterior inform the
   Koopman screening (e.g., higher screening priority for objects
   with $P(UN) > 0.5$)?

### 6.3 Prerequisites for Implementation

The following existing Humeris modules would be composed:

- `orbit_determination`: EKF for state estimation and residual
  generation (`run_ekf`, residual time series).
- `maneuver_detection`: CUSUM, EWMA, chi-squared, SPRT detectors
  (for baseline comparison and alarm fusion).
- `conjunction`: Collision probability computation (for
  intent-conditioned $P_c$).
- `numerical_propagation`: Intent-specific trajectory propagation
  (SK with periodic burns, DO with deorbit thrust, EV with avoidance
  maneuver, UN with ballistic propagation).

New implementation required:

1. **HMM forward algorithm:** $O(K^2)$ per observation step.
   Straightforward implementation using NumPy matrix operations.
2. **Viterbi algorithm:** Same complexity as forward, plus backtracking.
3. **Emission distribution evaluator:** Rayleigh or chi-distribution
   PDF computation.
4. **Transition matrix constructor:** Physics-informed
   $\mathbf{A}(\Delta t)$ with configurable parameters.
5. **Intent-adaptive process noise:** Maps intent posterior to
   process noise matrix $Q_t$.

Estimated complexity: **High**. The HMM implementation itself is
straightforward ($O(K^2 T)$ with $K = 4$), but the careful design of
transition probabilities based on orbital dynamics knowledge requires
domain expertise. The emission model calibration against real tracking
data is the primary challenge. The integration with the EKF (adaptive
process noise based on intent posterior) requires modifications to the
existing EKF processing loop.

---

## 7. Conclusion

We have proposed Bayesian Intent Joint Estimation (BIJE), a unified
framework for simultaneously estimating orbital state and operator intent
in space situational awareness. By coupling an EKF for state estimation
with a Hidden Markov Model for intent classification, BIJE addresses the
operationally relevant question "what does the operator intend?" rather
than just "when did a maneuver occur?"

The four-state intent model $\{SK, DO, EV, UN\}$ with physics-informed
transitions models the primary operational modes. The EKF residuals
serve as natural HMM observations, with intent-specific emission
distributions calibrated to the residual signatures of each mode. The
forward algorithm provides the intent posterior in $O(16T)$ time, a
negligible overhead compared to the EKF state estimation.

The predictive collision probability $P_c = \sum_I P(\text{collision}|I)
P(I|r_{1:t})$ integrates over intent uncertainty, providing a risk
assessment that accounts for estimated operator intent. When the intent posterior
strongly favors "evasive" ($P(EV) > 0.9$), the effective $P_c$ is reduced;
when it favors "uncontrolled" ($P(UN) > 0.9$), $P_c$ is elevated. This
intent-informed risk assessment represents a different approach from the
current practice of recomputing $P_c$ on a single assumed trajectory.

The framework builds on the existing Humeris infrastructure:
EKF from `orbit_determination`, maneuver detectors from
`maneuver_detection`, conjunction assessment from `conjunction`, and
trajectory propagation from `numerical_propagation`.

---

## References

1. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software*. Wiley.

2. Blom, H. A. P., & Bar-Shalom, Y. (1988). The interacting multiple model algorithm for systems with Markovian switching coefficients. *IEEE Transactions on Automatic Control*, 33(8), 780--783.

3. Douc, R., Moulines, E., & Ryden, T. (2004). Asymptotic properties of the maximum likelihood estimator in autoregressive models with Markov regime. *Annals of Statistics*, 32(5), 2254--2304.

4. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257--286.

5. Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. *Annals of Mathematical Statistics*, 41(1), 164--171.

6. Moustakides, G. V. (1986). Optimal stopping times for detecting changes in distributions. *Annals of Statistics*, 14(4), 1379--1387.

7. Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100--115.

8. Viterbi, A. J. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. *IEEE Transactions on Information Theory*, 13(2), 260--269.

9. Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*, 16(2), 117--186.

10. Kelecy, T. M., & Jah, M. K. (2015). Detection and orbit determination of a satellite executing low-thrust maneuvers. *Acta Astronautica*, 66(5--6), 798--809.
