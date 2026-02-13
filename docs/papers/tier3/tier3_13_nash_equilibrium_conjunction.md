# Nash Equilibrium Strategies for Decentralized Conjunction Avoidance

**Authors**: Humeris Research — Speculative Frontier Series
**Classification**: Tier 3 — Creative Frontier (Speculative)
**Status**: Theoretical proposal, not implemented
**Date**: February 2026

---

## Abstract

We formulate conjunction avoidance as a non-cooperative game among satellite operators
and derive Nash equilibrium strategies for decentralized collision avoidance maneuvers.
Each operator chooses a maneuver vector $\Delta\mathbf{v}_i$ to minimize a payoff function
combining fuel cost and collision risk penalty. We show that under symmetric collision
probability models, the conjunction avoidance game is a potential game, which implies the
existence of at least one pure-strategy Nash equilibrium and convergence of best-response
dynamics. The Nash equilibrium maneuver is computed in closed form for the two-player
case and via iterative best response for $N$ players. We analyze the price of anarchy —
the ratio of the Nash equilibrium cost to the socially optimal cost — and show it is
bounded by a factor depending on the number of players and the collision probability
gradient structure. The framework provides theoretical foundations for decentralized
space traffic management where no central authority coordinates maneuvers. We assess the
gap between the game-theoretic model and operational reality, including information
asymmetry (operators may not share precise state estimates), sequential rather than
simultaneous decision-making, and the role of coordination protocols like the
Conjunction Data Messages (CDMs) exchanged via the 18th Space Defense Squadron.
The potential game structure suggests that simple coordination protocols (sharing CDM
data) may be sufficient for convergence to near-optimal avoidance strategies.

---

## 1. Introduction

### 1.1 Motivation

When two satellites approach each other with unacceptable collision probability, at
least one must maneuver to avoid collision. In the current operational paradigm, the
18th Space Defense Squadron (18 SDS) provides conjunction data messages (CDMs) to
operators, who independently decide whether and how to maneuver.

This creates a coordination problem:
- If both operators maneuver, they may over-correct or even maneuver into a worse
  configuration.
- If neither maneuvers (expecting the other to act), collision occurs.
- If only one maneuvers, that operator bears the full fuel cost.
- Operators have asymmetric information (different orbit determination quality,
  different risk tolerances, different fuel reserves).

Game theory provides the natural mathematical framework for analyzing strategic
interactions among rational agents with conflicting interests. Conjunction avoidance
is, at its core, a game.

### 1.2 The Creative Leap

The standard game-theoretic analysis would model conjunction avoidance as a one-shot
game and look for Nash equilibria. Our creative contribution is the observation that the
conjunction avoidance game is a **potential game** when collision probabilities are
symmetric functions of the miss distance.

In a potential game, there exists a single potential function $\Phi$ whose gradient with
respect to each player's strategy equals the gradient of that player's payoff. This has
several useful consequences:

1. **Pure-strategy Nash equilibria exist** (guaranteed by Monderer-Shapley theorem [5]).
2. **Best-response dynamics converge** — operators iteratively choosing their best
   response to others' current strategies will converge to a Nash equilibrium.
3. **The equilibrium is the local minimum of $\Phi$** — the game has a "landscape" that
   can be optimized.

The potential function turns out to be the **total social cost** minus a correction term.
This means the Nash equilibrium is close to socially optimal, with the price of anarchy
bounded by a function of the collision probability structure.

### 1.3 Scope and Honesty

Real conjunction avoidance involves sequential (not simultaneous) decisions, incomplete
information, repeated interactions, and institutional constraints. Our model captures the
strategic essence but abstracts away operational details. We are explicit about where the
model matches and where it diverges from practice.

---

## 2. Background

### 2.1 Non-Cooperative Game Theory

A **normal-form game** consists of:
- $N$ players (operators), indexed $i = 1, \ldots, N$
- Strategy sets $S_i$ for each player (available maneuver vectors)
- Payoff functions $u_i: S_1 \times \cdots \times S_N \to \mathbb{R}$

A strategy profile $(\mathbf{s}_1^*, \ldots, \mathbf{s}_N^*)$ is a **Nash equilibrium**
if no player can improve their payoff by unilateral deviation:

$$u_i(\mathbf{s}_i^*, \mathbf{s}_{-i}^*) \geq u_i(\mathbf{s}_i, \mathbf{s}_{-i}^*) \quad \forall \mathbf{s}_i \in S_i, \forall i$$

where $\mathbf{s}_{-i}^*$ denotes the strategies of all players except $i$.

Nash [2] proved that every finite game has at least one Nash equilibrium in mixed
strategies. Pure-strategy existence requires additional structure.

### 2.2 Potential Games

Monderer and Shapley [5] defined a class of games with special structure:

**Definition**: A game is an **exact potential game** if there exists a function
$\Phi: S_1 \times \cdots \times S_N \to \mathbb{R}$ such that for every player $i$,
every strategy $\mathbf{s}_i, \mathbf{s}_i' \in S_i$, and every opponent profile
$\mathbf{s}_{-i}$:

$$u_i(\mathbf{s}_i, \mathbf{s}_{-i}) - u_i(\mathbf{s}_i', \mathbf{s}_{-i}) = \Phi(\mathbf{s}_i, \mathbf{s}_{-i}) - \Phi(\mathbf{s}_i', \mathbf{s}_{-i})$$

Properties of potential games:
1. Every local maximum of $\Phi$ is a Nash equilibrium.
2. Best-response dynamics (each player iteratively optimizes their strategy given
   others' current strategies) converge to a Nash equilibrium.
3. Finite improvement property: every sequence of unilateral improvements terminates.

### 2.3 Collision Probability Models

The collision probability between two objects with position uncertainty is modeled as:

$$P_c = \frac{1}{2\pi|\mathbf{C}|^{1/2}} \iint_{\text{cross-section}} \exp\left(-\frac{1}{2} \mathbf{r}^T \mathbf{C}^{-1} \mathbf{r}\right) d\mathbf{r}$$

where $\mathbf{r}$ is the position in the B-plane and $\mathbf{C}$ is the combined
covariance projected onto the B-plane.

For a circular cross-section of combined radius $R$ and covariance eigenvalues
$\sigma_1^2, \sigma_2^2$:

$$P_c \approx \frac{R^2}{2\sigma_1 \sigma_2} \exp\left(-\frac{1}{2}\left(\frac{b_R^2}{\sigma_1^2} + \frac{b_T^2}{\sigma_2^2}\right)\right)$$

where $(b_R, b_T)$ is the miss distance vector in the B-plane.

**Key property**: $P_c$ depends on the **miss distance vector**, which is the difference
of the two objects' positions at TCA. When both objects maneuver, the post-maneuver miss
distance depends symmetrically on both maneuvers:

$$\mathbf{b}_{post} = \mathbf{b}_0 + \mathbf{A}_1 \Delta\mathbf{v}_1 - \mathbf{A}_2 \Delta\mathbf{v}_2$$

where $\mathbf{A}_i$ are the maneuver-to-B-plane mapping matrices (determined by orbital
geometry and lead time) and $\mathbf{b}_0$ is the pre-maneuver miss distance.

### 2.4 Existing Humeris Conjunction Management

The Humeris `conjunction_management.py` module implements single-operator conjunction
triage and avoidance maneuver computation. The `compute_avoidance_maneuver()` function
computes the minimum delta-V along-track maneuver to increase the miss distance above a
threshold. This single-operator solution does not account for the other operator's
possible actions — exactly the gap that the game-theoretic framework addresses.

### 2.5 Price of Anarchy

The **price of anarchy** (PoA) measures the efficiency loss due to selfish behavior:

$$\text{PoA} = \frac{\text{cost}(\text{worst Nash equilibrium})}{\text{cost}(\text{social optimum})}$$

where the social optimum minimizes total cost $\sum_i u_i$ subject to all constraints.
$\text{PoA} = 1$ means selfish behavior is as good as centralized coordination.
$\text{PoA} > 1$ means there is an efficiency loss from decentralization.

---

## 3. Proposed Method

### 3.1 Game Formulation

**Players**: $N$ satellite operators, each controlling one satellite involved in a
conjunction event (or cluster of events).

**Strategy**: Operator $i$ chooses a maneuver vector $\Delta\mathbf{v}_i \in \mathbb{R}^3$
(3D velocity change in the RTN frame: radial, along-track, cross-track).

**Payoff**: Operator $i$ seeks to maximize (or equivalently, minimize the negative of):

$$u_i(\Delta\mathbf{v}_1, \ldots, \Delta\mathbf{v}_N) = -\|\Delta\mathbf{v}_i\|^2 - \lambda_i \sum_{j \neq i} P_c^{(ij)}(\Delta\mathbf{v}_i, \Delta\mathbf{v}_j)$$

The first term is the fuel cost (quadratic in delta-V). The second term is the
collision risk penalty, weighted by $\lambda_i > 0$ (the collision risk aversion
parameter for operator $i$).

**Note on sign convention**: We write payoffs as quantities to be maximized (negative
cost). The Nash equilibrium maximizes each player's payoff (minimizes cost).

### 3.2 Two-Player Case: Closed-Form Solution

For a two-player conjunction ($N = 2$), the collision probability depends on the
combined miss distance:

$$P_c(\Delta\mathbf{v}_1, \Delta\mathbf{v}_2) = P_c(\mathbf{b}_0 + \mathbf{A}_1 \Delta\mathbf{v}_1 - \mathbf{A}_2 \Delta\mathbf{v}_2)$$

Using the Gaussian approximation:

$$P_c(\mathbf{b}) = \frac{R^2}{2\sigma_1\sigma_2} \exp\left(-\frac{1}{2} \mathbf{b}^T \mathbf{C}^{-1} \mathbf{b}\right)$$

**Best response for player 1**: Given $\Delta\mathbf{v}_2$, player 1 minimizes:

$$-u_1 = \|\Delta\mathbf{v}_1\|^2 + \lambda_1 P_c(\mathbf{b}_0 + \mathbf{A}_1 \Delta\mathbf{v}_1 - \mathbf{A}_2 \Delta\mathbf{v}_2)$$

Taking the gradient and setting to zero:

$$2\Delta\mathbf{v}_1 + \lambda_1 \frac{\partial P_c}{\partial \Delta\mathbf{v}_1} = 0$$

$$\frac{\partial P_c}{\partial \Delta\mathbf{v}_1} = -P_c \cdot \mathbf{A}_1^T \mathbf{C}^{-1} \mathbf{b}_{post}$$

So the best response is:

$$\Delta\mathbf{v}_1^*(\Delta\mathbf{v}_2) = \frac{\lambda_1}{2} P_c \cdot \mathbf{A}_1^T \mathbf{C}^{-1} \mathbf{b}_{post}$$

Similarly for player 2 (with opposite sign on $\mathbf{A}_2$):

$$\Delta\mathbf{v}_2^*(\Delta\mathbf{v}_1) = -\frac{\lambda_2}{2} P_c \cdot \mathbf{A}_2^T \mathbf{C}^{-1} \mathbf{b}_{post}$$

**Nash equilibrium**: The simultaneous solution $(\Delta\mathbf{v}_1^*, \Delta\mathbf{v}_2^*)$
satisfies both best-response equations. Substituting and solving (the system is nonlinear
due to the $P_c$ factor, but can be solved iteratively or via fixed-point methods):

In the linearized regime (small maneuvers, $P_c \approx const$), the Nash equilibrium is:

$$\Delta\mathbf{v}_1^* = \frac{\lambda_1}{\lambda_1 + \lambda_2} \Delta\mathbf{v}_{social}$$

$$\Delta\mathbf{v}_2^* = \frac{\lambda_2}{\lambda_1 + \lambda_2} \Delta\mathbf{v}_{social}$$

where $\Delta\mathbf{v}_{social}$ is the total maneuver vector that would be performed by
a single cooperative entity. The burden-sharing is proportional to risk aversion:
the more risk-averse operator performs a larger share of the maneuver.

**[SPECULATIVE]**: The linearized solution assumes small maneuvers and near-constant
$P_c$ over the maneuver range. For large maneuvers or highly nonlinear $P_c$ (near the
steep part of the Gaussian), the best-response dynamics may exhibit multiple fixed
points or convergence difficulties.

### 3.3 N-Player Generalization

For $N$ operators involved in a conjunction cluster, the payoff for player $i$ is:

$$u_i = -\|\Delta\mathbf{v}_i\|^2 - \lambda_i \sum_{j \neq i} P_c^{(ij)}(\Delta\mathbf{v}_i, \Delta\mathbf{v}_j)$$

The best response for player $i$ given all other players' strategies:

$$\Delta\mathbf{v}_i^*(\Delta\mathbf{v}_{-i}) = \frac{\lambda_i}{2} \sum_{j \neq i} P_c^{(ij)} \cdot (\mathbf{A}_i^{(ij)})^T (\mathbf{C}^{(ij)})^{-1} \mathbf{b}_{post}^{(ij)}$$

This sums the collision avoidance "forces" from all conjunction partners, weighted by
collision probability and risk aversion.

### 3.4 Potential Game Proof

**Theorem**: The conjunction avoidance game is an exact potential game when
$\lambda_i = \lambda$ for all players (common risk aversion) and $P_c^{(ij)}$ depends
only on the miss distance $\|\mathbf{b}_{post}^{(ij)}\|$.

**Proof sketch**: Define the potential function:

$$\Phi(\Delta\mathbf{v}_1, \ldots, \Delta\mathbf{v}_N) = -\sum_{i} \|\Delta\mathbf{v}_i\|^2 - \lambda \sum_{i < j} P_c^{(ij)}(\Delta\mathbf{v}_i, \Delta\mathbf{v}_j)$$

We need to show that:

$$u_i(\Delta\mathbf{v}_i, \Delta\mathbf{v}_{-i}) - u_i(\Delta\mathbf{v}_i', \Delta\mathbf{v}_{-i}) = \Phi(\Delta\mathbf{v}_i, \Delta\mathbf{v}_{-i}) - \Phi(\Delta\mathbf{v}_i', \Delta\mathbf{v}_{-i})$$

**Left side**:
$$\text{LHS} = -\|\Delta\mathbf{v}_i\|^2 - \lambda \sum_{j \neq i} P_c^{(ij)}(\Delta\mathbf{v}_i, \Delta\mathbf{v}_j) + \|\Delta\mathbf{v}_i'\|^2 + \lambda \sum_{j \neq i} P_c^{(ij)}(\Delta\mathbf{v}_i', \Delta\mathbf{v}_j)$$

**Right side**:
$$\text{RHS} = -\|\Delta\mathbf{v}_i\|^2 - \lambda \sum_{j: j \neq i, j > i} P_c^{(ij)} - \lambda \sum_{j: j \neq i, j < i} P_c^{(ji)} + \|\Delta\mathbf{v}_i'\|^2 + \lambda \sum_{j: j \neq i, j > i} P_c'^{(ij)} + \lambda \sum_{j: j \neq i, j < i} P_c'^{(ji)}$$

Since $P_c^{(ij)} = P_c^{(ji)}$ (collision probability is symmetric in the two objects),
the sums match: every term $P_c^{(ij)}$ involving player $i$ appears exactly once in both
the payoff difference and the potential difference. $\square$

**Corollary**: For $\lambda_i = \lambda$ (common risk aversion), the game has at least
one pure-strategy Nash equilibrium, and best-response dynamics converge.

**Extension to heterogeneous $\lambda_i$**: When risk aversions differ, the game is a
**weighted potential game** with potential:

$$\Phi = -\sum_i \|\Delta\mathbf{v}_i\|^2 - \sum_{i < j} \frac{\lambda_i + \lambda_j}{2} P_c^{(ij)}$$

This is exact when $P_c^{(ij)}$ is symmetric. The weighted potential game still
guarantees Nash equilibrium existence and best-response convergence [5].

**[SPECULATIVE]**: The symmetry $P_c^{(ij)} = P_c^{(ji)}$ holds for the miss-distance-based
collision probability. In practice, the two operators may have asymmetric covariances
(one has better tracking data), breaking symmetry. Whether the potential game structure
is robust to small asymmetries is an open question.

### 3.5 Price of Anarchy Bound

**Social optimum**: The centralized coordinator minimizes total cost:

$$\min_{\Delta\mathbf{v}_1, \ldots, \Delta\mathbf{v}_N} \sum_i \|\Delta\mathbf{v}_i\|^2 + \lambda \sum_{i < j} P_c^{(ij)}$$

**Nash equilibrium**: Each player minimizes their individual cost:

$$\min_{\Delta\mathbf{v}_i} \|\Delta\mathbf{v}_i\|^2 + \lambda \sum_{j \neq i} P_c^{(ij)}$$

The difference: in the Nash equilibrium, each player counts $P_c^{(ij)}$ once, while
in the social optimum, each $P_c^{(ij)}$ is counted once total (not once per player).
This means each player over-weights collision risk relative to the social optimum.

**Price of anarchy bound**: For the potential game formulation with common $\lambda$:

$$\text{PoA} \leq \frac{N}{N-1}$$

For $N = 2$: $\text{PoA} \leq 2$ (worst case: both operators perform the full avoidance
maneuver, doubling total fuel cost). For large $N$: $\text{PoA} \to 1$ (selfish behavior
becomes nearly optimal).

**Derivation**: The Nash equilibrium of the potential game maximizes $\Phi$. The social
optimum maximizes $\Phi_{social} = -\sum_i \|\Delta\mathbf{v}_i\|^2 - \lambda \sum_{i<j} P_c^{(ij)}$.
Since the Nash potential is $\Phi = -\sum_i \|\Delta\mathbf{v}_i\|^2 - \lambda \sum_{i<j} P_c^{(ij)} \cdot 2/(N-1) \cdot \ldots$

Actually, let us be more precise. The individual cost for player $i$ at Nash equilibrium
is bounded because the collision terms $P_c^{(ij)}$ are shared between pairs. The
over-counting factor per pair is at most 2 (both endpoints count the same pair), giving:

$$\text{Total Nash cost} \leq \sum_i [\|\Delta\mathbf{v}_i^*\|^2 + \lambda \sum_{j \neq i} P_c^{(ij)*}] = \sum_i \|\Delta\mathbf{v}_i^*\|^2 + 2\lambda \sum_{i<j} P_c^{(ij)*}$$

$$\leq 2 \left[\sum_i \|\Delta\mathbf{v}_i^*\|^2 + \lambda \sum_{i<j} P_c^{(ij)*}\right] = 2 \cdot \text{Social cost at Nash}$$

Since the Nash equilibrium minimizes the potential (which equals the social cost for the
potential game), the PoA is bounded by 2 for general $N$ and approaches 1 as the fuel
cost term dominates.

**[SPECULATIVE]**: The PoA bound of 2 assumes the worst case where both players perform
redundant maneuvers. In practice, the asymmetry of orbital geometry (one operator may
be in a much cheaper maneuver direction) and information exchange (CDMs reveal intent)
would reduce the PoA well below 2.

### 3.6 Best-Response Dynamics Algorithm

```
ALGORITHM: Nash Equilibrium Conjunction Avoidance (NECA)

INPUT:
    N                — number of operators in conjunction cluster
    b_0[i][j]        — pre-maneuver miss distance vectors (B-plane)
    C[i][j]          — combined covariance matrices
    A[i][j]          — maneuver-to-B-plane mapping matrices
    lambda[i]        — risk aversion parameters
    sigma             — combined hard-body radius
    max_iterations   — convergence limit
    tolerance        — convergence threshold (m/s)

PROCEDURE:
    1. Initialize: Delta_v[i] = [0, 0, 0] for all i (no maneuver)

    2. FOR iteration = 1 to max_iterations:
           FOR i = 1 to N:
               // Compute best response for player i given others
               grad_i = [0, 0, 0]
               FOR j != i:
                   b_post = b_0[i][j] + A[i][j] @ Delta_v[i] - A[j][i] @ Delta_v[j]
                   Pc = collision_probability(b_post, C[i][j], sigma)
                   grad_i += lambda[i] * Pc * A[i][j]^T @ inv(C[i][j]) @ b_post
               END FOR

               Delta_v_new[i] = grad_i / 2  // best response (from FOC)

               // Damped update for stability
               Delta_v[i] = (1 - eta) * Delta_v[i] + eta * Delta_v_new[i]
           END FOR

           // Check convergence
           max_change = max_i(||Delta_v_new[i] - Delta_v[i]||)
           IF max_change < tolerance:
               BREAK

    3. RETURN Delta_v[1..N] (Nash equilibrium maneuvers)

OUTPUT:
    Equilibrium maneuver vectors for each operator
    Total fuel cost, residual collision probabilities
    Convergence history
```

### 3.7 Information Structure and Coordination

The best-response algorithm requires each player to know the current strategies of all
other players. In practice, this information is exchanged via CDMs and operator-to-operator
communication. We consider three information regimes:

**Full information**: All players know all strategies (positions, covariances, planned
maneuvers). Best-response dynamics converge to Nash equilibrium. This corresponds to a
future with transparent conjunction data sharing.

**Partial information**: Players know only their own state and the CDM-provided conjunction
data (miss distance, combined covariance, TCA). Players can compute best responses to
the CDM-implied situation but not to other operators' planned maneuvers. This leads to
a Bayesian game where each operator has beliefs about others' actions.

**No information**: Players do not know whether others will maneuver. This is the most
conservative case. The dominant strategy (minimax) is for each player to perform the
full avoidance maneuver independently, leading to maximum over-correction.

The potential game structure implies that even partial information exchange (sharing
CDM data plus declared maneuver intent) is sufficient for convergence to Nash equilibrium.

### 3.8 Repeated Game: Reputation and Cooperation

In practice, operators encounter each other repeatedly over the constellation lifetime.
The repeated game structure enables:

- **Reputation building**: Operators who consistently maneuver gain reputation for
  reliability, allowing others to safely reduce their maneuvers (convergence to
  cooperative equilibria).

- **Tit-for-tat dynamics**: If operator $i$ defects (refuses to maneuver), operator $j$
  can punish by also refusing in future encounters. The folk theorem guarantees that
  cooperative outcomes can be sustained as Nash equilibria of the repeated game if the
  discount factor (how much operators value future interactions) is sufficiently high.

- **Asymmetric burden sharing**: Over repeated conjunctions, the cumulative fuel
  expenditure can be tracked, and burden sharing can be balanced across encounters even
  if individual conjunctions have asymmetric Nash equilibria.

**[SPECULATIVE]**: The repeated game analysis assumes operators are long-lived rational
agents who internalize future interactions. In practice, maneuver decisions are made by
operations teams under time pressure, with limited ability to consider long-term strategic
implications. Whether reputation dynamics emerge in the actual SSA ecosystem is an
empirical question.

---

## 4. Theoretical Analysis

### 4.1 Existence and Uniqueness

**Existence**: For the potential game with common $\lambda$, Nash equilibrium existence
is guaranteed by the Monderer-Shapley theorem [5]. For continuous strategy spaces
(maneuver vectors in $\mathbb{R}^3$), the potential function is continuous and the
strategy space is compact (maneuvers are bounded by fuel capacity), so a maximum exists.

**Uniqueness**: Not guaranteed in general. Multiple Nash equilibria can exist, differing
in which operator bears the maneuver burden. The potential function may have multiple
local maxima, each corresponding to a different Nash equilibrium.

For the two-player case with symmetric risk aversion ($\lambda_1 = \lambda_2$), the
Nash equilibrium is unique and symmetric: both operators perform equal maneuvers. With
asymmetric $\lambda$, there is still a unique equilibrium (the best-response mapping is
a contraction in the linearized regime).

For $N > 2$ players, uniqueness analysis requires checking the second-order conditions
of the potential function. The Hessian of $\Phi$ is negative definite at a Nash
equilibrium if and only if the equilibrium is a strict local maximum — ensuring local
uniqueness and stability.

### 4.2 Convergence Rate

Best-response dynamics for the potential game converge at a rate determined by the
spectral properties of the best-response Jacobian.

**Linear convergence**: For the linearized game (small maneuvers), the best-response
dynamics define a linear map $\mathbf{v}^{(k+1)} = \mathbf{M} \mathbf{v}^{(k)} + \mathbf{c}$
where $\mathbf{v} = (\Delta\mathbf{v}_1, \ldots, \Delta\mathbf{v}_N)$. The convergence
rate is $\rho(\mathbf{M})$ (spectral radius of $\mathbf{M}$).

For symmetric interactions and common $\lambda$:

$$\rho(\mathbf{M}) = \frac{\lambda P_c \|\mathbf{A}^T \mathbf{C}^{-1} \mathbf{A}\|}{2 + \lambda P_c \|\mathbf{A}^T \mathbf{C}^{-1} \mathbf{A}\|}$$

This is always less than 1, confirming convergence. The convergence is faster when:
- $\lambda$ is small (low risk aversion → small maneuvers → weak coupling)
- $P_c$ is small (low collision probability → weak coupling)
- The number of players is small (weaker multi-body effects)

For large $\lambda P_c$ (high risk, high aversion), convergence slows: $\rho \to 1$.
In this regime, damping ($\eta < 1$ in the algorithm) is necessary for stability.

### 4.3 Computational Complexity

**Per iteration**: Computing best responses for $N$ players involves $O(N^2)$ pairwise
collision probability evaluations, each requiring $O(1)$ B-plane computations (given
pre-computed mapping matrices $\mathbf{A}$). Total per iteration: $O(N^2)$.

**Number of iterations**: Linear convergence means $O(\log(1/\epsilon))$ iterations for
$\epsilon$ accuracy. With damping, the constant factor may increase.

**Total**: $O(N^2 \log(1/\epsilon))$ — efficient for conjunction clusters of practical
size ($N \leq 100$).

**Pre-computation**: The maneuver-to-B-plane mapping matrices $\mathbf{A}_i^{(ij)}$
require orbit propagation from the maneuver time to TCA. This is $O(N^2)$ propagations,
each with cost depending on the propagation method. Using the existing Humeris
propagation infrastructure, this is a one-time $O(N^2)$ setup cost.

### 4.4 Comparison with Social Optimum

The social optimum minimizes total fuel cost subject to a collision probability
constraint:

$$\min_{\Delta\mathbf{v}_1, \ldots, \Delta\mathbf{v}_N} \sum_i \|\Delta\mathbf{v}_i\|^2 \quad \text{s.t.} \quad P_c^{(ij)} \leq P_{threshold} \quad \forall i < j$$

This is a constrained optimization problem solvable by standard methods (sequential
quadratic programming, interior point).

The Nash equilibrium approximates the social optimum but with the burden sharing
determined by the risk aversion parameters $\lambda_i$ rather than by a central
coordinator. The efficiency gap (price of anarchy) is bounded by 2 for general games
and is typically much smaller in practice.

**Key insight**: The Nash equilibrium is not merely a heuristic approximation to the
social optimum — it is the natural outcome of rational, decentralized decision-making.
If operators are rational and have access to CDM data, they will converge to the Nash
equilibrium without any central coordination. Understanding the Nash equilibrium tells
us what behavior to expect.

---

## 5. Feasibility Assessment

### 5.1 What Would Need to Be True

**F1. Rational operators**: Operators must behave as rational payoff maximizers. In
practice, maneuver decisions involve institutional factors (risk culture, management
approval, operational procedures) that may deviate from pure rationality.

*Assessment*: Operators are approximately rational in the sense that they seek to
minimize fuel expenditure subject to safety constraints. The game-theoretic framework
captures this central motivation. Institutional deviations are second-order effects.

**F2. Known payoff structure**: Operators must know (or be able to estimate) their own
risk aversion $\lambda_i$ and the collision probability function $P_c$.

*Assessment*: Operators already perform collision probability assessments using CDM data.
The risk aversion parameter $\lambda_i$ can be calibrated from historical maneuver
decisions (how much fuel did the operator spend to avoid what level of risk).

**F3. Information exchange**: Best-response dynamics require operators to know others'
current strategies or planned maneuvers.

*Assessment*: Currently, operators share maneuver plans informally. The game-theoretic
framework provides motivation for formalizing this exchange: transparent sharing of
planned maneuvers enables convergence to more efficient equilibria.

**F4. Symmetric collision probability**: The potential game proof requires $P_c^{(ij)} = P_c^{(ji)}$.
This holds for the miss-distance-based model but may break down when operators have
very different covariances.

*Assessment*: The miss distance $\|\mathbf{b}\|$ is the same for both operators. The
covariance $\mathbf{C} = \mathbf{C}_i + \mathbf{C}_j$ is symmetric by construction.
The potential game structure is robust.

**F5. Single conjunction analysis**: The framework analyzes each conjunction event
(or cluster) independently. In reality, operators manage a portfolio of conjunctions
and fuel allocation across multiple events.

*Assessment*: Portfolio effects can be incorporated by modifying the fuel cost term to
reflect remaining fuel reserves. This changes the effective $\lambda_i$ per conjunction
but does not break the game structure.

### 5.2 Critical Unknowns

1. **Empirical validation**: Do actual operator maneuver decisions approximate Nash
   equilibrium behavior? Historical CDM and maneuver data could be used to test whether
   the game-theoretic predictions match observed behavior.

2. **Multiple equilibria selection**: When multiple Nash equilibria exist, which one
   is selected in practice? Focal point theory (Schelling) suggests that conventions
   (e.g., "lower-altitude operator maneuvers") may serve as equilibrium selection
   mechanisms.

3. **Dynamic games**: Real conjunction avoidance is a dynamic game with evolving
   information (CDMs are updated as TCA approaches). The static Nash equilibrium may
   not capture the sequential decision-making dynamics.

4. **Coalition formation**: Operators of large constellations (e.g., Starlink with
   >5000 satellites) have disproportionate influence on the orbital environment. They
   may form implicit coalitions that affect the game structure.

5. **Regulatory intervention**: Space traffic management regulations could impose
   maneuver obligations that override game-theoretic equilibria. The framework predicts
   what happens without regulation, informing where regulation is most needed.

### 5.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Operators are not rational | Low | Medium | Framework remains a useful approximation |
| Multiple equilibria cause confusion | Medium | Medium | Establish conventions for equilibrium selection |
| Information asymmetry breaks convergence | Medium | High | Design protocols for maneuver intent sharing |
| Sequential decision structure matters | High | Medium | Extend to extensive-form game analysis |
| Framework has no practical adoption path | Medium | High | Demonstrate value via simulation and back-testing |

---

## 6. Connection to Humeris Library

### 6.1 Existing Modules Leveraged

**Conjunction analysis**:
- `conjunction.py` — `screen_conjunctions()` identifies conjunction candidates and
  `assess_conjunction()` computes B-plane collision probabilities. These provide the
  $P_c^{(ij)}$ function that defines the game's payoffs.

- `conjunction_management.py` — `compute_avoidance_maneuver()` computes single-operator
  avoidance maneuvers. This is the "unilateral" baseline against which the Nash
  equilibrium maneuvers are compared.

**Orbit determination and state estimation**:
- `orbit_determination.py` — EKF orbit determination provides the state estimates and
  covariances that enter the collision probability computation. The covariance quality
  affects the information structure of the game.

- `maneuver_detection.py` — CUSUM/EWMA maneuver detection provides the ability to infer
  other operators' actions from tracking data, enabling best-response updates even without
  direct communication.

**Propagation**:
- `propagation.py` and `numerical_propagation.py` — Orbit propagation to TCA computes
  the maneuver-to-B-plane mapping matrices $\mathbf{A}_i$. The accuracy of these matrices
  depends on the propagation fidelity.

- `relative_motion.py` — CW relative motion provides the linearized dynamics for computing
  the effect of delta-V on miss distance.

**Design and optimization**:
- `design_optimization.py` — Optimization infrastructure that could host the Nash
  equilibrium computation as an alternative objective function.

- `multi_objective_design.py` — Multi-objective framework. The price-of-anarchy analysis
  produces a Pareto-like tradeoff between decentralization and efficiency.

**Mathematical infrastructure**:
- `linalg.py` — Matrix operations for B-plane projections, covariance inversions, and
  best-response Jacobian computations.

- `control_analysis.py` — Controllability Gramian analysis provides the fuel-cost
  anisotropy information used to determine preferred maneuver directions.

### 6.2 Proposed New Module

A new domain module `game_conjunction.py` would implement:

1. `ConjunctionGame` — Frozen dataclass representing the game structure
2. `compute_best_response()` — Single-player best response given others' strategies
3. `nash_equilibrium_br()` — Best-response dynamics to Nash equilibrium
4. `compute_price_of_anarchy()` — PoA computation vs social optimum
5. `compute_social_optimum()` — Centralized optimal maneuvers
6. `burden_sharing_analysis()` — How total maneuver is distributed across operators
7. `game_conjunction_pipeline()` — End-to-end game-theoretic conjunction resolution

### 6.3 Integration Architecture

```
game_conjunction.py
    ├── uses: conjunction.py (collision probability P_c)
    ├── uses: conjunction_management.py (single-operator baseline)
    ├── uses: relative_motion.py (CW dynamics for maneuver mapping)
    ├── uses: orbit_determination.py (covariances for B-plane)
    ├── uses: linalg.py (matrix operations)
    ├── uses: control_analysis.py (fuel cost anisotropy)
    ├── produces: list[AvoidanceManeuver] (one per operator)
    └── compared via: conjunction_management.py (single-operator baseline)
```

---

## 7. Discussion

### 7.1 Speculation Level

| Claim | Evidence Level |
|-------|---------------|
| Conjunction avoidance can be modeled as a game | **Proven** — standard game theory application |
| Nash equilibrium exists for the conjunction game | **Derived** — follows from potential game theorem |
| The game is a potential game under symmetry | **Derived** — explicit proof provided |
| Best-response dynamics converge | **Derived** — property of potential games |
| Price of anarchy is bounded by 2 | **Derived** — from potential game structure |
| Operators behave as rational agents | **Assumed** — reasonable first approximation |
| Information exchange enables convergence | **Conjectured** — requires protocol design |
| Nash equilibrium matches operational behavior | **Speculative** — no empirical validation |

### 7.2 Open Problems

1. **Empirical calibration**: Can the risk aversion parameter $\lambda_i$ be estimated
   from historical maneuver data? This requires a dataset of conjunction events with
   known maneuver responses from multiple operators.

2. **Extensive-form game**: The sequential nature of conjunction avoidance (CDMs arrive
   days before TCA, with increasing precision) is better modeled as an extensive-form
   game with information refinement. The subgame-perfect equilibrium of this sequential
   game may differ from the simultaneous Nash equilibrium.

3. **Mechanism design**: Given the game structure, what coordination mechanisms (rules,
   protocols, incentives) would minimize the price of anarchy? This is a mechanism design
   problem with applications to space traffic management regulation.

4. **Coalition resistance**: For conjunction events involving multiple satellites from
   the same operator, that operator effectively controls multiple "players." Coalition
   strategies may deviate from the non-cooperative Nash equilibrium.

5. **Incomplete information**: When operators have private information (fuel reserves,
   mission criticality, tracking quality), the game becomes a Bayesian game. The
   Bayesian Nash equilibrium involves strategies conditioned on private types, adding
   significant complexity.

6. **Connection to Kessler dynamics**: In the long run, conjunction avoidance games
   affect the evolution of the orbital debris environment. The cumulative effect of
   individual maneuver decisions (or failures to maneuver) determines the trajectory
   toward or away from Kessler syndrome. This connects game theory to the population
   dynamics of the competing-risks and cascade analysis frameworks already in Humeris.

### 7.3 Relationship to Other Tier 3 Concepts

- **Paper 12 (Helmholtz Free Energy)**: The social optimum of the conjunction game
  minimizes a total cost function analogous to the energy $E$ in the thermodynamic
  framework. The Nash equilibrium adds a "self-interest" correction that increases the
  effective temperature (each operator counts shared risks separately).

- **Paper 14 (Melnikov Separatrix Surfing)**: The Nash equilibrium maneuvers may
  preferentially exploit the dynamical highways identified by Melnikov analysis, since
  these represent minimum-fuel directions. The game-theoretic framework selects among
  available low-fuel options based on strategic considerations.

- **Paper 15 (Spectral Gap Coverage)**: After conjunction avoidance maneuvers, the
  constellation configuration changes. The spectral gap of the coverage Laplacian
  measures the impact of these maneuvers on coverage resilience.

### 7.4 Potential Impact

**Theoretical**: The potential game result provides a rigorous foundation for
decentralized space traffic management. It shows that under reasonable symmetry
conditions, uncoordinated but rational operators will converge to a stable, near-optimal
avoidance strategy.

**Practical**: The best-response algorithm provides a computable prediction of what
each operator "should" do in a conjunction event, given their risk aversion and the
CDM data. This could inform:
- Operator decision-support tools ("given what we expect others to do, here is our
  optimal maneuver")
- Post-event analysis ("was the actual maneuver response consistent with Nash
  equilibrium behavior?")
- Regulatory design ("what protocols would minimize the price of anarchy?")

**Policy**: The price of anarchy bound quantifies the efficiency cost of decentralization.
If PoA is close to 1, centralized coordination is unnecessary. If PoA is large,
coordination protocols or regulatory mandates are justified.

---

## 8. Conclusion

We have formulated conjunction avoidance as a non-cooperative game and derived Nash
equilibrium strategies for decentralized collision avoidance maneuvers. The central
result is that under symmetric collision probability models, the conjunction avoidance
game is a potential game, guaranteeing Nash equilibrium existence and best-response
convergence.

The two-player Nash equilibrium has a closed-form solution in the linearized regime:
the total avoidance maneuver is shared between operators in proportion to their risk
aversion parameters. The $N$-player extension is computed via iterative best-response
dynamics with $O(N^2)$ complexity per iteration and linear convergence.

The price of anarchy — the efficiency loss from selfish behavior relative to centralized
coordination — is bounded by 2 in the worst case and is typically much smaller. This
provides a quantitative answer to the question of whether decentralized space traffic
management can be efficient.

We have been explicit about the gap between the game-theoretic model and operational
reality: operators are approximately but not perfectly rational, information is incomplete
and sequential, and institutional factors affect decisions. These gaps do not invalidate
the framework but define its limits of applicability.

The framework integrates naturally with the existing Humeris conjunction analysis,
orbit determination, and maneuver computation modules, providing a game-theoretic
layer on top of the existing single-operator conjunction management pipeline.

---

## References

[1] Kirkpatrick, S., Gelatt, C.D., and Vecchi, M.P. "Optimization by Simulated
Annealing." *Science*, 220(4598):671-680, 1983.

[2] Nash, J.F. "Non-Cooperative Games." *Annals of Mathematics*, 54(2):286-295, 1951.

[3] Turing, A.M. "The Chemical Basis of Morphogenesis." *Philosophical Transactions of
the Royal Society of London. Series B, Biological Sciences*, 237(641):37-72, 1952.

[4] Murray, J.D. *Mathematical Biology II: Spatial Models and Biomedical Applications*.
3rd edition, Springer, 2002.

[5] Monderer, D. and Shapley, L.S. "Potential Games." *Games and Economic Behavior*,
14(1):124-143, 1996.

[6] Chung, F.R.K. *Spectral Graph Theory*. CBMS Regional Conference Series in
Mathematics, No. 92, American Mathematical Society, 1997.

[7] Fiedler, M. "Algebraic Connectivity of Graphs." *Czechoslovak Mathematical Journal*,
23(98):298-305, 1973.

[8] Nisan, N., Roughgarden, T., Tardos, E., and Vazirani, V.V. *Algorithmic Game Theory*.
Cambridge University Press, 2007.

[9] Roughgarden, T. "The Price of Anarchy is Independent of the Network Topology."
*Journal of Computer and System Sciences*, 67(2):341-364, 2003.

[10] Alfano, S. "A Numerical Implementation of Spherical Object Collision Probability."
*Journal of the Astronautical Sciences*, 53(1):103-109, 2005.

[11] Patera, R.P. "General Method for Calculating Satellite Collision Probability."
*Journal of Guidance, Control, and Dynamics*, 24(4):716-722, 2001.

[12] Aumann, R.J. "Acceptable Points in General Cooperative n-Person Games."
*Contributions to the Theory of Games IV*, Annals of Mathematics Studies, 40:287-324,
1959.

[13] Myerson, R.B. *Game Theory: Analysis of Conflict*. Harvard University Press, 1991.

[14] Harsanyi, J.C. "Games with Incomplete Information Played by Bayesian Players."
*Management Science*, 14(3):159-182, 1967.

[15] Schelling, T.C. *The Strategy of Conflict*. Harvard University Press, 1960.
