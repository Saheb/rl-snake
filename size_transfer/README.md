# Size Generalization: does a Snake agent trained on a small board work on a bigger one?

The intuition: if an agent has learned to play Snake on a 6×6 board, it *should* also handle 10×10 or
16×16 — the game is the same, only bigger. It doesn't. A "size-agnostic" ConvNet (fully convolutional
+ global pooling, so it *runs* on any board) trained on 6×6 collapses to random-walk behaviour by
board ~16.

This directory is a full investigation into **why**, ending in a fix.

## The result

A single agent, trained only on boards **6–22**, plays **100×100 zero-shot** at 97.6% toward-food
(~86 food/game). Every single-size-trained agent was at chance (50%) by board 16.

| board | 6 | 16 | 32 | 64 | **100** |
|---|---|---|---|---|---|
| single-size (6×6) agent, toward-food% | 87% | 51% | 50% | 50% | 50% |
| **curriculum agent (trained 6–22), toward-food%** | 88% | — | 96% | 98% | **98%** |

## The finding, in one line

Board-size generalization fails **not for the architectural reasons you'd guess** (pooling, receptive
field — real but secondary) but because single-size training leaves **71–88% of big-board states
out-of-support** (e.g. "far from all walls" is geometrically impossible on 6×6 yet is 60–79% of
big-board play). The fix needs **both** a size-invariant *representation* (so big boards *look* like
training) **and** training *coverage* of the multi-size state distribution (so the policy has *seen*
that regime). Neither alone works; together, extrapolation is essentially free.

Full narrative with every table and dead-end: **[FINDINGS.md](FINDINGS.md)**. Architecture of the final
agent: [arch_final.svg](arch_final.svg) (the size-agnostic baseline it evolved from: [arch_diagram.svg](arch_diagram.svg)).

## Try it yourself

The trained agent ships here: **`curriculum_ego_best.pth`** (824 KB, ~209K params). Play it on any
board size (6→100) — pick a size, watch it go:

```bash
uv run marimo edit notebooks/size_generalization.py   # from the repo root
```

## Reproduce

Interpreter is `.venv/bin/python` from the repo root (torch 2.8, MPS/CPU). Each `size_*.py` runs
standalone (`--skip-train` / `--eval-only` where noted). Trained agent: `curriculum_ego_best.pth`.

```bash
.venv/bin/python scripts/size_curriculum.py            # train the cross-size agent (6..22) + eval 6..100
.venv/bin/python scripts/size_collision_rate.py        # why score rises with board size
```

## The scripts, in narrative order

**Setup & baseline**
| script | what it shows |
|---|---|
| `size_transfer.py` | Size-agnostic net (conv → global avg+max pool → dueling head). Trains on one board, measures zero-shot transfer. Transfers a little, then cliffs. |

**Mechanism (probes) — the collapse is real but secondary**
| script | what it shows |
|---|---|
| `size_frozen_probe.py` | Frozen vs refit linear reader of the pooled hidden → *decodable ≠ used*: info survives, a fixed reader can't use it. |
| `size_margin.py` | Decomposes the advantage vector → **food-steering margin collapses 5×, danger margin invariant**. |
| `size_diagnostic.py` | Per-size decodability of food-direction vs danger. |
| `size_canonical_scene.py` | Fixed local scene, only board area grows → isolates **global-avg-pool flooding with the empty-cell background**; a local-box pool is the oracle fix. |

**Ablations — what does *not* fix it**
| script | what it shows |
|---|---|
| `size_rescale_ablation.py` | No-retrain: rescale the avg-pool branch by area — makes transfer *worse*. |
| `size_pool_ablation.py` | Retrain with count-normalized pooling — food margin *still* collapses. Pooling denominator is not the fix. |

**Encoding fixes — necessary, not sufficient**
| script | what it shows |
|---|---|
| `size_attn_pool.py` | Attention pooling (background-suppressing). Fixes the *probe*, not real-play behaviour. |
| `size_coord_attn.py` | Absolute normalized coordinate channels. Not enough (6×6-scale patterns don't transfer). |
| `size_ego_attn.py` | **Egocentric** head-relative squashed coords → scale-invariant direction (verified). Fixes direction; exposes a stack of policy-OOD failures. |

**Root cause & solution**
| script | what it shows |
|---|---|
| `size_state_stats.py` | The actual state distribution across sizes → **71–88% of big-board play is outside 6×6 support**. Architecture was never the bottleneck. |
| `size_interp_test.py` | Interpolation (≤ trained size) is flawless for all archs; trained on 10×10 the ego arch extrapolates *better* than baseline. |
| `size_curriculum.py` | **The solution.** Egocentric arch + curriculum over sizes 6–22 (per-size replay buffers) → 100×100 zero-shot. |
| `size_collision_rate.py` | Why score *rises* with board size: larger boards remove constraints (wall interactions 54%→1%, collisions ~22× lower, survival ~90× longer). |

Exploratory / superseded (food-distance confound, later demoted to a secondary effect):
`size_food_distance.py`, `size_interp_extrap.py`.

## Key figures

`transfer_curve.png` · `frozen_probe.png` · `margin_dilution.png` · `canonical_scene.png` ·
`pool_ablation.png` · `attn_pool.png` · `ego_attn.png` · `coord_attn.png`
