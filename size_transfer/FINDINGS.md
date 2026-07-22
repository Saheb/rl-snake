# Size generalization: a size-agnostic Snake ConvNet transfers, then cliffs

Fix: replaced ConvDuelingQNet's board-size-locked `flatten→Linear` head with a fully
size-agnostic head (conv trunk → global avg-pool ‖ max-pool → dueling FC). Trained on 6×6,
evaluated **zero-shot** on larger boards with identical weights. Script: `scripts/size_transfer.py`.

Training (hard periodic target updates fixed the earlier Q-drift collapse): greedy score climbed
stably to **17.77** on 6×6 (~50% board fill — strong).

## Zero-shot transfer curve

| board | greedy (food) | score/cells |
|---|---|---|
| 6×6 (trained) | 18.5 | 0.514 |
| 8×8 | 22.5 | 0.352 |
| 10×10 | 21.0 | 0.210 |
| 12×12 | 5.96 | 0.041 |
| 16×16 | 1.6 | 0.006 |

## The anomaly

Transfer **improves** up to 8–10×8–10 (eats MORE food than at training size), then **falls off a
cliff** between 10×10 and 12×12 (21 → 6 → 1.6). Sharp, localized failure at a characteristic
scale — not diffuse decay. The local navigation policy is genuinely size-agnostic for modest
scale-ups; something breaks abruptly around 12×12.

Nuances: raw-score rise at 8×8 is partly headroom (6×6 is near its 50%-fill planning ceiling);
10×10 is high-variance (±9.3); normalized score decays monotonically throughout; the cliff itself
is unambiguous.

## Lead hypothesis (testable): global-pooling dilution

GAP+GMP squash the board into a 256-vector. On small boards the single-cell head/food signals
dominate the pool enough to steer; as the board grows they are **diluted** among hundreds of
cells (avg-pool → ~0; max-pool keeps "food exists" but loses *where*). Past a threshold the pooled
representation can't encode *direction to food* → navigation collapses. Predicts a cliff, and
predicts its location.

Competing mechanisms to rule in/out: 5×5 receptive field; value miscalibration with horizon;
out-of-range input statistics (board fill, snake length).

## Mechanism (resolved) — decodable but not actionable at novel range

Two hypotheses fell to cheap probes:
- **Pooling dilution: REFUTED.** food-direction stays decodable 0.97→0.90 and danger stays perfect
  even on 16×16 where the agent is dead. Information is not lost.
- **Activation scale-shift: REFUTED.** |GAP| shifts only 2.14→1.93 (~10%), |hidden| ~stable.

Then we looked at *how it dies* (greedy play, 60 eps/board):

| board | self-death% | timeout% | moves-toward-food% | score |
|---|---|---|---|---|
| 6 | 88 | 8 | 86 | 18 |
| 8 | 90 | 10 | 85 | 23 |
| 10 | 57 | 43 | 74 | 22 |
| 12 | 7 | 93 | 55 | 7.4 |
| 16 | 0 | 100 | 51 | 2.0 |

On 6–8 the agent navigates to food (86%) and dies by self-trapping like a strong player. On 12–16
it **times out wandering at ~chance (51–55% toward food)** — food-blind in ACTION despite
food-direction being decodable at 0.90.

**Cause: the training distribution confounded food-direction with food-proximity.** On 6×6 food is
always ≤~5 cells, so the policy learned to seek food via a signal entangled with closeness; it
never learned to act on pure direction to *distant* food. The conv trunk computes direction at any
range (probe decodes it), but the frozen policy head only learned direction→action in the
short-range regime — so at novel range it wanders.

Three ideas in one minimal result: (1) size-agnostic *architecture* ≠ scale-general *behavior*;
(2) **decodable ≠ used** (probe says 0.90, policy acts at chance); (3) length generalization,
mechanistically — a spurious train-time confound (direction≈proximity) that breaks OOD.

## Causal confirmation — CONFIRMED

Trained the SAME size-agnostic architecture on 10×10 (where food can be distant, decoupling
direction from proximity) and re-ran zero-shot transfer:

| board | 6×6-trained (score/cells) | 10×10-trained (score/cells) |
|---|---|---|
| 6 | 0.514 | 0.477 |
| 8 | 0.352 | 0.376 |
| 10 | 0.210 | 0.271 |
| 12 | **0.041 (cliff)** | **0.219 (no cliff)** |
| 16 | 0.006 | 0.060 |

The 12×12 cliff **vanishes**: 10×10-trained scores 31.6 on 12×12 (its best board) vs the
6×6-agent's 5.96 (5×); 16×16 goes 1.6→15.4 (~10×). Downward transfer to 6×6 is unharmed (17.2).

**The cliff didn't disappear — it MOVED**, from ~11 (6×6-trained) to ~14+ (10×10-trained), with
16×16 beginning to fail again. **The cliff location tracks the maximum food-distance seen in
training.** Extend the training range → extend the transfer range. This is the confound mechanism
confirmed with a knob.

## Interpolation vs extrapolation (hardened: fixed length cap=5, fixed budget=400, rate metrics)

Re-evaluated both agents on a fine board grid with self-collision and timeout removed as
confounds. toward-food% (coverage in parens):

| board | 6×6-trained | 10×10-trained |
|---|---|---|
| 6 | 96 (36) | 97 (36) |
| 8 | 94 (64) | 98 (62) |
| 10 | 70 (78) | 99 (94) |
| 11 | 56 (58) | 98 (111) |
| 12 | 55 (59) | 93 (121) |
| 14 | 52 (47) | 66 (93) |
| 16 | 52 (49) | 57 (60) |

- **Interpolation (test ≤ training range) is flawless** — 10×10 agent == 6×6 agent on small boards.
- **The decline curve shifts RIGHT by the training gap** (6→10): same sigmoid shape, translated ~4
  boards. The extrapolation boundary tracks the training food-distance distribution.
- **Graceful margin ~1.2–1.3× training size**, then sigmoid decay to chance (52–57% = chance).
- **Coverage peaks at the competence boundary** (10×10 agent: 121 cells at board 12), then collapses
  as navigation fails — a clean boundary detector, and a two-phase failure (roam → local cycling).
- Caveat: eval is greedy (ε=0), so the low-coverage "stuck" phase is partly deterministic cycling;
  the failure is *exploitation* of visible food (Snake is fully observed), not exploration — no ε
  knob fixes it, only widening the training food-distance distribution.

## ISOLATION: food-distance vs board-size — food-distance hypothesis REFUTED

Decoupled the two variables: fixed board at 10×10 for both agents, varied only training food
distance. near-only agent (food ≤4 during training) vs full-range agent (all distances). Both
evaluated on the SAME 10×10 board with food at controlled distances (capped length, greedy).

| food dist | near-only toward-food% (avg/max score) | full-range toward-food% (avg/max) |
|---|---|---|
| 2 | 95 (150/168) | 94 (142/165) |
| 4 | 96 (90/95) | 97 (92/96) |
| 8 | 84 (34/48) | 98 (48/51) |
| 12 | 90 (27/33) | 100 (34/35) |
| 14 | 90 (23/29) | 100 (29/31) |

**The near-only agent navigates to food 14 cells away at 90% — it does NOT collapse.** Training on
only-near food does not prevent far-food navigation. So **food-distance extrapolation is easy**, and
the earlier "food-distance confound" explanation is **refuted** (demoted to a weak secondary effect:
near-only is modestly worse at far food, 84–90% vs 98–100%, but nowhere near chance).

**Corrected mechanism.** The near-only agent trained on a 10×10 *board* (just never saw far food)
and handles far food fine. The 6×6 agent failed on 12×12. The distinguishing variable is therefore
**board size seen in training, not food distance.** The agent extrapolates across novel food-
distances but NOT across novel board-sizes. Food-direction is computed *relative* to the head
(translation-invariant → transfers across distance); *board size* changes global/absolute structure
(wall positions, empty-space extent, what pooling aggregates over) → that's what fails to extrapolate.

The board-size causal result stands (train-size sets transfer-size); the MECHANISM is characterized
below (isolated: global average pooling floods the representation with the empty-cell background as the
board grows, diluting the food-steering signal): why board-size extrapolation fails when food-distance
extrapolation succeeds.

Confirming test — near-only-10×10 vs full-range-10×10 on a BOARD sweep (normal food, capped
length, greedy). toward-food% | avg score | max score:

| board | near-only (d≤4) tf% / avg / max | full-range tf% / avg / max |
|---|---|---|
| 10 | 94 / 50 / 60 | 98 / 56 / 64 |
| 11 | 85 / 37 / 51 | 98 / 52 / 59 |
| 12 | 75 / 25 / 47 | 89 / 39 / 54 |
| 14 | 61 / 10 / 31 | 66 / 13 / 36 |
| 16 | 58 / 6 / 21 | 59 / 7 / 22 |

**max score reveals: the big-board failure is a RELIABILITY failure, not a capability floor.** At
board 14, avg score ≈ 10–13 (near chance) but max ≈ 31–36 — in the best episodes both agents DO
navigate a 14×14 board competently (~5× their average). The capability is intermittently present;
it just isn't deployed reliably (works when food falls favorably, wanders otherwise). Soft,
high-variance boundary; consistent with decodable-but-not-reliably-used.

**Final mechanism — board-size primary, food-distance secondary:**
- Both 10×10-trained agents hit the SAME hard wall at 14–16 regardless of food-distance training →
  **board-size-in-training sets the transfer ceiling** (~1.3–1.4× training board; 6→~8, 10→~13).
  This is board-size / spatial-extent extrapolation, and it's the primary failure.
- Food-distance diversity is a **secondary modulator**: full-range holds better in the marginal zone
  (boards 11–12: 98/92 vs 89/70) but does NOT raise the ceiling.

Reconciles everything: near-only handled far food ON 10×10 (board interpolation, distance extrapolation
= easy) but fails on bigger BOARDS (board extrapolation = hard). The original single-variable
"food-distance confound" was wrong as the primary story (isolation caught it); food-distance is a
genuine secondary effect. WHY board-size extrapolation fails — not info loss (food-dir decodable at
0.90) nor gross activation-scale shift (|GAP| shifts ~10%) — is isolated in the next section (global
average pooling floods with the empty-cell background as the board grows; spatial exclusion recovers it).

## MECHANISM — ISOLATED: global-average-pool floods with the empty-cell background as the board grows

Two zero-shot probes on the 6×6 agent resolve *why* board-size extrapolation fails. Arch reference:
`size_transfer/arch_diagram.svg` (conv trunk → GAP‖GMP → fixed dueling head).

**A — frozen vs refit reader** (`scripts/size_frozen_probe.py`). The "still decodable at 0.90" number
came from a probe REFIT on each board's own states. The policy head is the *same kind* of fixed
linear reader (the advantage stream reads the pooled hidden), but frozen at 6×6. So we ran one reader
in both roles on food-direction:

| board | frozen-AUC (6×6-fit) | refit-AUC | policy toward-food |
|---|---|---|---|
| 6 | 0.990 | 0.986 | 84.5% |
| 10 | 0.957 | 0.973 | 73.1% |
| 12 | 0.922 | 0.964 | 54.2% |
| 16 | 0.853 | 0.953 | 50.9% (chance) |

The frozen–refit gap widens with size (0.00→0.10): representation shift is real and does penalize a
fixed reader. **But the frozen reader only sags to 0.85 — it still reads direction well, while the
policy is at chance.** A fixed linear reader of the same hidden does NOT fail the way the policy does,
so the collapse is *not* "the shifted representation is unreadable." AUC is rank-based → it survives
because dilution scales magnitude, not ordering. That points the finger at *magnitude*, not info.

**B — advantage-margin dilution** (`scripts/size_margin.py`). Decompose the advantage vector A(4)
(== Q up to constants that cancel in differences) into a food-steering margin (best toward-food Q −
best away-from-food Q) and a survival margin (best safe Q − best fatal Q), per board:

| board | food margin | danger margin | food/spread | toward-argmax |
|---|---|---|---|---|
| 6 | 0.956 | 6.08 | 0.239 | 97.0% |
| 10 | 0.590 | 6.04 | 0.194 | 89.4% |
| 12 | 0.420 | 6.05 | 0.161 | 80.6% |
| 16 | **0.192** | **5.99** | 0.117 | 64.7% |

**Food-steering margin collapses 5× (0.96→0.19); danger margin is invariant (flat to <2%).**

**What is established.** The survival margin is invariant while the food margin collapses 5× with
board size — a *locally-invariant* signal (danger at one head cell) keeps its scale, a *relational*
signal (head-vs-food geometry) loses its scale, and the fixed head has baked in their 6×6 ratio.
That much is measured.

**Ruling out the naive fixes (two area-normalization ablations).** First guess: global avg-pool (÷ H·W)
dilutes the spatially-aggregated food signal as area grows, while max-pool (area-invariant) preserves
the head-local survival signal. But "the food margin shrinks" doesn't isolate the operator, so we
tested it. (1) *No-retrain* (`scripts/size_rescale_ablation.py`): uniformly rescaling the GAP branch by
area/36 (the sum-pool equivalent, on frozen features) *worsened* transfer (greedy 19.5→2.0 at board
10). (2) *Retrain* (`scripts/size_pool_ablation.py`): MaskedAvgPoolQNet — avg branch divides by
ACTIVE-cell count instead of H·W (zero new params) — trains to the same 6×6 score (18.4) but its food
margin **still collapses ~5× (1.07→0.22)**, toward-argmax unchanged. So **changing the pooling
*denominator* is not the fix** (count-norm gives only a modest incidental transfer gain, greedy
6.5→10.3 at board 12, with no food-steering improvement). This *looked* like it exonerated pooling —
but the canonical-scene test below shows the operator is guilty; the denominator was just the wrong knob.

**Mechanism ISOLATED (`scripts/size_canonical_scene.py`).** Hand-build ONE identical local scene (head
+ short snake + food a fixed d=3 cells to the right) and embed it in boards 6→20 — fixed food distance,
fixed local geometry, only the surrounding board area changes.

| board | corner fm (full pool) | corner fm (local-box pool) | center fm (full) | center fm (local-box) |
|---|---|---|---|---|
| 8 | 0.397 | 0.477 | 0.312 | 0.377 |
| 10 | 0.312 | 0.429 | 0.073 | 0.160 |
| 14 | 0.265 | 0.429 | 0.041 | 0.141 |
| 20 | 0.238 | **0.429** | 0.014 | **0.141** |

- **Board area ALONE collapses the food margin** (corner 0.61→0.24; center, all walls receding, 0.31→
  0.01) with identical local geometry and fixed food distance. Not food-distance, not local geometry,
  not state distribution — pure spatial extent. This is the isolation the earlier probes couldn't give.
- **The locus is the avg-pool branch flooding with the empty-cell background constant.** As area grows,
  avg-pool → the pure-empty activation (`avg_dev` 0.84→0.14, ~1/area) while max-pool is invariant
  (`max_dev` flat at 6.85). The growing empty region's constant activation dominates the average and
  washes out the local food signal.
- **The fix is spatial exclusion, not renormalization.** Pooling the avg branch over only the occupied
  bounding box (excluding empty cells) **flattens the food margin** (constant 0.429 corner / 0.141
  center for all boards ≥10–14). Both anchors agree → it is the empty area, not wall proximity.

**Reconciliation.** All four experiments are consistent: the food signal *does* dilute through avg-pool
(canonical scene), but you cannot fix it by reweighting the denominator — uniform rescale also scales
the dominant empty constant, and count-norm divides by active-cell count while empty cells carry
positive post-ReLU bias so they *count as active*. Only removing the empty region from the average
(spatial exclusion / masking / attention) restores the signal. So: **global average pooling floods the
pooled representation with the background constant as the board grows, diluting the local food-steering
signal; survival is spared because it rides on the area-invariant max-pool.** The *local/global
scale-competition* framing holds throughout (danger margin invariant in every condition); the canonical
scene upgrades it from "consistent with" to an isolated, positively-identified mechanism. Deployable
fix (untested): a learned area-invariant aggregator that suppresses empty cells — attention/softmax
pooling or a background-subtracted (masked) average — should extend the transfer range; local-box
pooling is the hand-specified oracle showing the ceiling is recoverable.

## FIX TEST — attention pooling: mechanism fixed, transfer NOT (a second barrier surfaces)

Trained `AttnPoolQNet` (`scripts/size_attn_pool.py`) on 6×6 — avg branch → 4-head attention pool
(softmax over cells, area-invariant, can zero active-but-empty cells); max branch + dueling head
unchanged. Two readouts disagree, and the disagreement is the result:

| board | canonical fm (fixed near food d=3) base→attn | real-play fm base→attn | transfer score/cells base→attn |
|---|---|---|---|
| 8  | 0.397 → 0.655 | 0.739 → 0.713 | 0.353 → 0.298 |
| 12 | 0.283 → 0.981 | 0.420 → 0.394 | 0.063 → 0.069 |
| 16 | 0.253 → **1.002** | 0.192 → 0.184 | 0.006 → 0.002 |

- **On the isolated mechanism the fix WORKS.** With a fixed *near* scene and only board area growing,
  the baseline's food margin collapses 2.4× while attention **holds/strengthens** (0.59→1.00).
  Background flooding is eliminated — softmax zeros the active-but-empty cells that defeated count-norm.
  This confirms the diagnosis with a learned, deployable operator.
- **On real-world transfer it does NOT help** — same cliff, dead by 16, real-play food margin still
  collapses ~5×. Same model, two regimes: near food (canonical) holds, far food (real play) collapses.
- **A second barrier is now binding: long-range direction.** On big boards food is far; the ~5×5
  receptive field with no positional encoding cannot compute direction to *distant* food, and pooling
  (however clean) aggregates translation-invariant features that don't carry absolute position. Fixing
  barrier #1 (background flooding) cleanly exposes barrier #2 (position/direction at range).

**Consequence for a 64×64 / 100×100 zero-shot agent:** the area-invariant pool is *necessary but not
sufficient*. It must be combined with a scale-invariant way to encode head→food direction — e.g.
normalized coordinate channels (position preserved, sign of displacement scale-invariant) feeding the
attention pool — and trained on a multi-size curriculum so the policy sees varied extents. Attention
pool alone is validated as the fix for the mechanism we isolated; it is not the whole agent.

### Iteration — coordinate channels (6×6, no curriculum): still not enough

Added normalized coord channels [row/(H−1), col/(W−1)] to the attention-pool agent
(`scripts/size_coord_attn.py`), trained on 6×6 only (one variable, no curriculum). Coords supply
per-cell position (in [0,1] at ANY size), so attention selecting the food cell can *in principle*
recover direction at range.

Result: a small marginal-zone gain over attention-only (boards 12–16: toward-food 60%/53% vs 53%/51%,
score/cells 0.061/0.040 vs 0.028/0.008), but **real-play transfer still collapses to chance by board
~24** (toward-food 50%, greedy 0 at 48/64/100). **Coord channels alone are not sufficient.** (A
far-food canonical probe looked positive for both models but is *confounded by border-proximity* —
far food was placed near the right edge, solvable by a "food-near-border" cue, not true direction; the
tell is that the *no-coord* attention model also scored 0.55–0.65 on it. Discounted; the robust signal
is real-play toward-food%/score.)

Interpretation: coords provide the *information* for scale-invariant direction, but single-size training
does not force the policy to *use* it size-invariantly — it only ever saw 6×6-scale coordinate and
displacement patterns, so it learned a 6×6-specific mapping.

### Iteration — egocentric relative encoding (6×6, no curriculum): direction fixed, exposes a barrier stack

Replaced absolute coords with head-RELATIVE squashed offsets `tanh((i−head_i)/κ)`, κ=3
(`scripts/size_ego_attn.py`); attention pool kept; trained 6×6 only. Scale-invariance verified — "food
3 right" encodes to 0.762 on 6×6, 16×16, AND 100×100 identically (board size never enters).

A clean dissociation:
- **Direction IS fixed at the representation level.** Unconfounded far-food probe (mid-board,
  axis-aligned): strong positive margin, argmax_toward=True at every size incl. 100×100 (0.785); the
  absolute-coord agent is ~0.
- **But real-play transfer still fails** — toward-food 50%, greedy 0 on boards ≥32. ε-greedy does not
  help (ε 0.05–0.25); seeding an initial body segment does not help (max 2 food on 16, 0 on 64/100).

Tracing the failure exposed a STACK of barriers behind #1/#2, all symptoms of a single-size-trained
policy being OOD on big-board trajectories:
- **Greedy limit cycles.** The agent navigates from distance, overshoots, then traps in a 2-cycle at
  moderate range (Q goes flat, argmax flips by hundredths — right/left/right/left). ε doesn't break it
  (no net directional drift on the visited states).
- **Length-1 orientation degeneracy.** With no body (no orientation) AND diagonal food, argmax moves
  *away* (margin −0.7 to −1.2 at all sizes). On big boards the snake never eats → stays length-1 → the
  trap is self-perpetuating. (Same-row food is survivable at length-1; diagonal is not.)
- **Diagonal saturation at extreme scale.** Even *with* a body, the diagonal-food margin decays with
  board size (0.58 at 16 → −0.01 at 100): both tanh offset channels saturate to ~1 for large diagonal
  offsets, losing which-axis-is-closer. Axis-aligned far food holds; diagonal fades.

**Conclusion.** Input-encoding fixes (attention, ego) are each necessary and each removes a specific
barrier — but they cannot fix a POLICY/VALUE function on a trajectory distribution it never trained on.
Single-size 6×6 training yields a value function that is OOD on big-board play in several compounding
ways (cycling, orientation degeneracy, diagonal saturation), none removable by eval-time tricks (ε,
body-seeding). Reaching 64/100 requires the policy to TRAIN on larger/varied boards.

### REALITY CHECK — the probe-validated fixes made real-play transfer WORSE, not better

Fine-grained real-play sweep (toward-food% and avg score, all trained on 6×6, greedy):

| board | baseline avg-pool | attention pool | ego + attention |
|---|---|---|---|
| 6  | 87% (13.6) | 89% (13.9) | 90% (15.0) |
| 8  | **87% (17.4)** | 82% (15.1) | 66% (7.9) |
| 10 | **75% (15.0)** | 70% (12.0) | 53% (1.6) |
| 11 | 61% (7.1) | 61% (6.5) | 52% (1.1) |
| 12 | 56% (4.2) | 55% (3.8) | 51% (0.8) |
| 14 | 55% (4.2) | 53% (2.4) | 50% (0.1) |
| 16 | 51% (1.1) | 51% (1.1) | 50% (0.2) |
| 25 | 50% | 50% | 50% |

**On the real objective (zero-shot navigation), the plain baseline is the BEST of the three; each
architectural "fix" degraded it, and ego degraded it most.** The knee is board ~10–11 for all three.
Every fix improved its target *probe* (canonical food margin, far-food direction) while doing nothing
good — or actively harming — actual navigation.

This is the session's central lesson turned on our own fixes: **improving a mechanistic probe ≠
improving behavior** (same shape as the original "decodable ≠ used"). Background flooding is a real
mechanism, but it was never the *behavioral* bottleneck, so removing it (and adding 6×6-overfittable
structure) moved the probe, not the policy — and the extra structure made OOD transfer worse. The
genuine bottleneck is shared by all three and untouched by any of them: a policy/value fit only to 6×6
is OOD on bigger boards. The simplest model (baseline) transfers least-badly because it has the least
6×6-specific structure to overfit. Corrective takeaway: validate scale fixes on the *behavioral*
metric (real-play toward-food%/score across a fine board grid), not on probe margins; and the only
lever shown to matter for behavior is training the policy across sizes.

### ROOT CAUSE — it's a state-distribution shift, not the architecture (`scripts/size_state_stats.py`)

Stop tweaking architecture; characterize the actual STATES. Rolling out a FIXED scripted competent
policy (greedy-to-food + collision avoidance — identical behavior at every size, so any change is pure
board-size) gives:

| statistic | 6 | 10 | 16 | 25 |
|---|---|---|---|---|
| food dist (mean) | 3.2 | 4.9 | 7.2 | 11.0 |
| % food beyond 5×5 RF | 44% | 67% | 79% | 87% |
| wall dist (mean) | 0.8 | 1.7 | 3.1 | 5.3 |
| **% far from ALL walls (>2)** | **0%** | 27% | 60% | 79% |
| steps per food | 4.7 | 7.9 | 11.8 | 17.5 |
| **% states OUTSIDE 6×6 support** | **0%** | **31%** | **71%** | **88%** |

The decisive fact: on 6×6 it is **geometrically impossible** to be >2 cells from every wall, yet that
"open space" regime is **60% of 16×16 play and 79% of 25×25 play.** Aggregate, **71% of 16×16 and 88%
of 25×25 competent play is OUTSIDE the 6×6 training support** (food farther than any 6×6 food, or
farther from walls than any 6×6 cell). The 6×6 policy was never in those states — not rarely, *never*.

**This is the real conclusion of the whole investigation, and it demotes the architecture story to
secondary.** The pooling/direction mechanisms are real, but board-size extrapolation fails primarily
because the state distribution shifts so far that most big-board play is out-of-support for a
6×6-trained policy — architecture-invariant. It reconciles everything: the fixes addressed
*representation* of states, not the missing *policy coverage* of a state region that never existed at
6×6 (mid-board, far walls, far food, sparse reward); the simplest model transferred least-badly but
still failed (71% out-of-support at 16); probe≠play because the probe was one in-support point while
play is mostly out-of-support; the length-1 trap exists because the snake grows within ~4.7 steps on
6×6 so mid-board-short-snake states don't occur there. The only lever that can fix behavior is training
the policy on states with big-board statistics (bigger boards / a size range) — coverage, not
architecture. "Size-agnostic architecture ≠ size-general behavior" resolves, ultimately, to
"single-size training ≠ coverage of multi-size state distributions."

### INTERPOLATION test + ego redemption (`scripts/size_interp_test.py`)

Trained ego on 10×10 (best greedy 27.5 in-distribution) and compared to the existing baseline avg-pool
trained on 10×10, fine grid:

| board | baseline-10 tf% (score) | ego-10 tf% (score) | region |
|---|---|---|---|
| 6 | 86% (16.6) | 85% (15.8) | interp |
| 8 | 88% (22.2) | 88% (22.7) | interp |
| 10 | 87% (26.5) | 89% (27.4) | interp |
| 12 | 84% (31.1) | **92% (30.4)** | extrap |
| 14 | 66% (18.3) | **85% (34.5)** | extrap |
| 16 | 64% (19.3) | **75% (26.7)** | extrap |

- **Interpolation (≤ trained size) is flawless for BOTH and identical** — the attention/egocentric
  machinery does NOT hurt in-distribution learning; ego ≈ baseline at 6–10. The extrapolation table's
  "ego is worse" was purely about *extrapolation from too-small a training board*.
- **Ego REDEEMED: trained on 10×10, ego extrapolates BETTER than baseline** (board 14: 85% vs 66%),
  the reverse of the 6×6-trained result. The egocentric encoding is a genuine asset for extrapolation
  *once trained on a board big enough to learn its scale-invariant use* — on 6×6 the coords saturate
  and the support is too small; on 10×10 there's enough scale variety.
- **Corroborates coverage:** the failure point moves outward with training board size (6×6-trained
  cracks ~board 11; 10×10-trained holds through 12, softens at 14–16). Bigger training board → more of
  the far-from-walls/far-food regime in-support → transfer extends.
- **Pushed to the full grid (6→100), a single training size still never reaches 100.** 10×10 avg-pool:
  92%@6, 59%@16, chance (≈50%) by 32. 10×10 egocentric: 88%@16, 68%@32, 53%@64, chance by 100 — the
  egocentric representation carries a *single* training size further (to ~32) but coverage is what
  removes the cliff entirely (curriculum reaches 98% at 100).

**Resolved recipe for a cross-size agent:** egocentric arch + train across sizes. Interpolation is
free; extrapolation range tracks training coverage; and with adequate training scale the egocentric
direction encoding generalizes *beyond* the trained range better than the plain baseline. The
architecture work is not useless — it is the multiplier that, combined with coverage, extends transfer;
it was just never a substitute for coverage.

### PAYOFF — a single agent that solves 6→100 zero-shot (`scripts/size_curriculum.py`)

Egocentric arch (attention pool + head-relative squashed coords) trained on a curriculum of sizes
{6,10,14,18,22} (per-size replay buffers so mixed sizes don't share a ragged batch), evaluated zero-shot:

| board | toward-food% | avg food | region |
|---|---|---|---|
| 6 | 88.5% | 14.7 | train |
| 14 | 93.0% | 31.3 | train |
| 22 | 93.9% | 42.5 | train |
| **32** | 95.6% | 46.1 | **extrap** |
| **48** | 97.4% | 58.0 | **extrap** |
| **64** | 98.0% | 66.5 | **extrap** |
| **100** | 97.6% | 86.5 | **extrap** |

**A single agent, trained only up to 22×22, plays 100×100 zero-shot at 97.6% toward-food (~86 food/game)
— a 4.5× extrapolation with NO degradation (it actually improves with size).** Baseline single-size
agents were at chance (50%) by board 16. Goal met.

Why it extrapolates so far, composing exactly as predicted:
- The **egocentric encoding** makes the agent's input size-invariant (local neighborhood + head-relative
  food direction) — a 100×100 board yields the same representation near the head as a 22×22 one.
- **Coverage 6–22** puts the big-board state statistics (far-from-walls, far food, sparse reward)
  in-support, and those statistics *saturate* beyond ~20, so 100 looks in-distribution.
- Neither ingredient worked alone (proven the hard way); together, extrapolation is essentially free.

Why score *rises* with board size (not mysterious — larger boards remove constraints;
`scripts/size_collision_rate.py`):

| | 6 | 22 | 100 |
|---|---|---|---|
| collisions / 1000 steps | 2.01 | 1.51 | **0.09** (~22× lower) |
| % steps on a wall edge | 54% | 7% | **1%** |
| wall deaths (% of deaths) | 50% | 0% | 0% |
| eating rate (food/1000 steps) | 187 | 62 | **14** (slower!) |
| survival (steps/episode) | 71 | 661 | **6501** (~90×) |
| avg / MAX score | 13 / 17 | 41 / 62 | 90 / 109 |

The agent does NOT navigate better on big boards — it eats *slower* per step (food is farther). Score
rises purely because the environment stops killing it: wall interactions collapse (54%→1% of steps),
wall deaths vanish, collision rate drops ~22×, so survival explodes ~90×. On 6×6 the walls are always
adjacent (half of all deaths); on 100×100 the agent lives in the open space it learned to handle and
rarely touches a wall. Constraints removed → longer life → higher total score. This is the flip side of
the core finding: the "far-from-walls, open-space" regime that was *out-of-support* (and fatal) for a
single-size agent is exactly the regime the curriculum agent learned — and it's an *easier* regime.

**Final synthesis of the whole investigation.** Board-size generalization failed not for the
architectural reasons one would guess (pooling, receptive field — real but secondary), but because
single-size training leaves ~71–88% of big-board states out-of-support. Fixing it needs BOTH a
size-invariant representation (so big boards look like training) AND training coverage of the
multi-size state distribution (so the policy has seen that regime). "Size-agnostic architecture ≠
size-general behavior" — but *size-invariant representation + coverage of the state distribution =
size-general behavior*, and it extrapolates far past the training range for free.

The measured part already **retro-explains the death-mode table** above: on 12–16 the agent dies 0%
self-collision / 100% timeout = survival margin preserved (never walks into death) + food margin gone
(never reaches food). And it **reconciles A with B**: the shrinkage is in *magnitude* not *ordering*,
so a rank-based refit probe still reads direction (0.85) while the magnitude-dependent argmax fails.
Decodable ≠ used, made concrete — the food signal survives in *rank* but loses the *margin* argmax
needs.

**The general phenomenon (broader than pooling).** Strip the CNN specifics and the pattern is:
*global-extrapolation fails when globally-aggregated features compete with locally-invariant features
inside a fixed decision rule.* Local signals (danger at the head) tend to have
environment-size-independent scale; globally-aggregated signals (relational food geometry) tend to
have scale that depends on environment size. A policy trained at one size implicitly fixes a *ratio*
between the two; change the size and the ratio drifts, so the policy can fail even though every
relevant feature is still encoded (A: direction decodable at 0.85). Average pooling is one concrete
way this arises — the same failure mode should appear in any architecture where feature types scale
differently with problem size: GNNs, transformers with global/CLS pooling, set encoders. That
framing, not "avg-pool is bad," is the transferable lesson.

## Bottom line

A "size-agnostic" ConvNet Snake agent fails to scale to bigger boards — but not for any architectural
reason you'd guess, and not because information is lost. The conv trunk is genuinely
translation-invariant and food-direction stays decodable at every size (A). It fails because a
**globally-aggregated signal (food-steering) and a locally-invariant signal (survival) scale
differently with board size, and the fixed head has baked in their 6×6 ratio** (B): the survival
margin holds while the food margin dilutes 5×, so past a threshold the policy degenerates to "avoid
death and wander." The cause is isolated: **global average pooling floods the pooled representation
with the empty-cell background constant as the board grows** — a fixed local scene loses its food
margin under full-board pooling (board area alone, food distance held fixed) but keeps it under
local-box pooling that excludes the empty region. It is *not* fixable by reweighting the pool's
denominator (uniform rescale and count-normalization both fail — empty cells carry positive activation
and dominate either way); it needs spatial exclusion of the background (masking / attention). Survival
is spared because it rides on the area-invariant max-pool. Training on a bigger board shifts the ratio
the head expects (the earlier train-size→transfer-size result); food-distance is a secondary modulator.
The transferable lesson is
general: **global extrapolation fails when globally-aggregated features compete with locally-invariant
features inside a fixed decision rule — size-agnostic architecture ≠ scale-general behavior;
decodable ≠ used.**
