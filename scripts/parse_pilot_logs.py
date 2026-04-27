"""Parse pilot logs into structured numbers for notebooks/curiosity.py.

Outputs a Python literal that can be pasted into the notebook's placeholder dicts.
Also computes coverage curves (for the chart cell).

Usage:
    .venv/bin/python scripts/parse_pilot_logs.py
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "pilot_logs"

GAME_RE = re.compile(
    r"Game\s+(\d+)\s*\|\s*Score:\s*(-?\d+)\s*\|\s*Record:\s*\d+\s*\|\s*Mean Score:\s*(-?\d+\.\d+)"
)
COVERAGE_RE = re.compile(
    r"\[coverage\]\s*Game\s+(\d+)\s*\|\s*unique_states_cumulative:\s*(\d+)/(\d+)"
)
FINAL_MEAN_RE = re.compile(r"Final Mean Score:\s*(-?\d+\.\d+)")


def parse_log(path: Path):
    """Return dict with per-game series and final metrics."""
    games = []
    scores = []
    mean_scores = []
    coverage = []  # list[(game, count, max)]
    final_mean = None
    completed = False
    with path.open() as f:
        for line in f:
            m = GAME_RE.search(line)
            if m:
                games.append(int(m.group(1)))
                scores.append(int(m.group(2)))
                mean_scores.append(float(m.group(3)))
                continue
            m = COVERAGE_RE.search(line)
            if m:
                coverage.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
                continue
            m = FINAL_MEAN_RE.search(line)
            if m:
                final_mean = float(m.group(1))
            if "Training Complete" in line:
                completed = True
    return {
        "games": games,
        "mean_scores": mean_scores,
        "coverage": coverage,
        "final_mean": final_mean,
        "completed": completed,
        "n_checkpoints": len(games),
    }


def aggregate(group_paths: list[Path]):
    """Aggregate over seeds: returns final mean (mean + std), avg coverage curve."""
    finals = []
    coverage_aggregate = defaultdict(list)  # game -> [count, ...]
    coverage_max = None
    for p in group_paths:
        d = parse_log(p)
        if d["final_mean"] is not None:
            finals.append(d["final_mean"])
        for g, c, mx in d["coverage"]:
            coverage_aggregate[g].append(c)
            coverage_max = mx
    out = {
        "n_seeds": len(finals),
        "final_mean": statistics.mean(finals) if finals else None,
        "final_std": statistics.stdev(finals) if len(finals) > 1 else 0.0,
        "all_seeds": finals,
    }
    if coverage_aggregate:
        gs = sorted(coverage_aggregate.keys())
        out["coverage"] = [(g, statistics.mean(coverage_aggregate[g])) for g in gs]
        out["coverage_max"] = coverage_max
    return out


def fmt(v):
    if v is None:
        return "None"
    return f"{v:.2f}"


def main():
    # Group by (algo, board, reward, has_icm)
    groups = {
        # 8x8 DQN
        ("DQN", "8x8", "dense", False): list(LOG_DIR.glob("sparse_S1_dense_dqn_8x8_*.log")),
        ("DQN", "8x8", "dense", True): list(LOG_DIR.glob("sparse_S2_dense_icm_8x8_*.log")),
        ("DQN", "8x8", "sparse", False): list(LOG_DIR.glob("sparse_S3_sparse_dqn_8x8_*.log")),
        ("DQN", "8x8", "sparse", True): list(LOG_DIR.glob("sparse_S4_sparse_icm_8x8_*.log")),
        ("DQN", "8x8", "pure_sparse", False): list(LOG_DIR.glob("puresparse_S3_pure_sparse_dqn_8x8_*.log")),
        ("DQN", "8x8", "pure_sparse", True): list(LOG_DIR.glob("puresparse_S4_pure_sparse_icm_8x8_*.log")),
        # 10x10 DQN
        ("DQN", "10x10", "dense", False): list(LOG_DIR.glob("puresparse_S1_dense_dqn_10x10_*.log")),
        ("DQN", "10x10", "dense", True): list(LOG_DIR.glob("puresparse_S2_dense_icm_10x10_*.log")),
        ("DQN", "10x10", "sparse", False): list(LOG_DIR.glob("sparse10_S5_sparse_dqn_10x10_*.log")),
        ("DQN", "10x10", "sparse", True): list(LOG_DIR.glob("sparse10_S6_sparse_icm_10x10_*.log")),
        ("DQN", "10x10", "pure_sparse", False): list(LOG_DIR.glob("puresparse_S3_pure_sparse_dqn_10x10_*.log")),
        ("DQN", "10x10", "pure_sparse", True): list(LOG_DIR.glob("puresparse_S4_pure_sparse_icm_10x10_*.log")),
        # 10x10 PPO
        ("PPO", "10x10", "dense", False): list(LOG_DIR.glob("ppo_dense_baseline_10x10_*.log")),
        ("PPO", "10x10", "dense", True): list(LOG_DIR.glob("ppo_dense_icm_10x10_*.log")),
        ("PPO", "10x10", "pure_sparse", False): list(LOG_DIR.glob("ppo_puresparse_baseline_10x10_*.log")),
        ("PPO", "10x10", "pure_sparse", True): list(LOG_DIR.glob("ppo_puresparse_icm_10x10_*.log")),
    }

    # Filter only completed runs (a still-running run with partial data shouldn't pollute means).
    print("# === SCOREBOARD (final mean ± std across seeds) ===")
    print(f"# {'condition':<40} {'n':>3}  {'mean':>6}  {'std':>5}  {'seeds'}")
    print("# " + "-" * 90)
    results = {}
    for key, paths in groups.items():
        completed = [p for p in paths if parse_log(p)["completed"]]
        agg = aggregate(completed) if completed else {"n_seeds": 0, "final_mean": None, "final_std": 0, "all_seeds": [], "coverage": None}
        results[key] = agg
        algo, board, reward, has_icm = key
        condname = f"{algo} {board} {reward} {'+ICM' if has_icm else ''}"
        seeds_str = ", ".join(f"{s:.2f}" for s in agg["all_seeds"]) if agg["all_seeds"] else "-"
        print(f"# {condname:<40} {agg['n_seeds']:>3}  {fmt(agg['final_mean']):>6}  {agg['final_std']:>5.2f}  [{seeds_str}]")

    # Emit Python-literal for sparsity_scoreboard placeholder
    print("\n# === Python literal for notebook scoreboard ===\n")
    sb = {"8x8": {}, "10x10": {}}
    for (algo, board, reward, has_icm), agg in results.items():
        if algo != "DQN":
            continue
        if board not in sb:
            continue
        sb[board].setdefault(reward, {})[("DQN+ICM" if has_icm else "DQN")] = agg["final_mean"]
    print("sparsity_scoreboard =", json.dumps(sb, indent=4, default=lambda x: round(x, 3) if isinstance(x, float) else x))

    # PPO scoreboard
    print("\n# === PPO scoreboard ===\n")
    ppo_sb = {}
    for (algo, board, reward, has_icm), agg in results.items():
        if algo != "PPO":
            continue
        ppo_sb.setdefault(reward, {})[("PPO+ICM" if has_icm else "PPO")] = {
            "mean": agg["final_mean"],
            "std": agg["final_std"],
            "n": agg["n_seeds"],
        }
    print("ppo_scoreboard =", json.dumps(ppo_sb, indent=4, default=lambda x: round(x, 3) if isinstance(x, float) else x))

    # Learning curves for the headline 2x2 chart (10x10 only).
    # For each condition, aggregate the `Mean Score` checkpoint series across seeds.
    print("\n# === Learning curves (10x10) for chart ===\n")
    lc = {}
    for (algo, board, reward, has_icm), paths in groups.items():
        if board != "10x10":
            continue
        if reward not in ("dense", "pure_sparse"):
            continue
        completed = [p for p in paths if parse_log(p)["completed"]]
        if not completed:
            continue
        # Aggregate per-game: collect mean_scores across seeds at matching game checkpoints
        by_game = defaultdict(list)
        for p in completed:
            d = parse_log(p)
            for g, ms in zip(d["games"], d["mean_scores"]):
                by_game[g].append(ms)
        gs = sorted(by_game.keys())
        means = [statistics.mean(by_game[g]) for g in gs]
        stds = [statistics.stdev(by_game[g]) if len(by_game[g]) > 1 else 0.0 for g in gs]
        key = f"{algo}_{reward}_{'icm' if has_icm else 'baseline'}"
        lc[key] = {"games": gs, "mean": [round(m, 3) for m in means], "std": [round(s, 3) for s in stds], "n": len(completed)}
    print("learning_curves =", json.dumps(lc, indent=4))
    out_path = Path(__file__).resolve().parent.parent / "assets" / "learning_curves.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(lc, indent=2))
    print(f"# Wrote {out_path}")

    # Coverage curves
    print("\n# === Coverage curves (game, mean_count) ===\n")
    cc = {"8x8": {}, "10x10": {}}
    for (algo, board, reward, has_icm), agg in results.items():
        if algo != "DQN" or "coverage" not in agg or agg["coverage"] is None:
            continue
        label = f"{reward} / {'DQN+ICM' if has_icm else 'DQN'}"
        cc[board][label] = [(g, round(c, 1)) for g, c in agg["coverage"]]
    # Compact print
    for board, curves in cc.items():
        print(f"# {board} coverage:")
        for label, pts in curves.items():
            n = len(pts)
            last = pts[-1] if pts else None
            print(f"#   {label:<35} n_pts={n:>3}  last={last}")


if __name__ == "__main__":
    main()
