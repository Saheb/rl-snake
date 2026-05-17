"""
Parse ConvNet DQN training logs and print the data-update commands for convnet.py.

Usage:
    uv run python scripts/parse_convnet_logs.py

Reads from pilot_logs/ConvDQN_*.log and FeatureDQN_*.log (16x16 runs).
Prints the three data blocks to paste into notebooks/convnet.py.
"""
import re
import sys
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path(__file__).parent.parent / "pilot_logs"

# v1: "Mean Score: 0.84"  v2: "Mean: 0.84"
PATTERN = re.compile(
    r"Game\s+(\d+).*?Mean(?:\s+Score)?:\s*([\d.]+)"
)


def parse_log(path: Path) -> dict[int, float]:
    """Return {game: mean_score} for every 100-game checkpoint."""
    result = {}
    for line in path.read_text().splitlines():
        m = PATTERN.search(line)
        if m:
            result[int(m.group(1))] = float(m.group(2))
    return result


def gather(prefix: str, board: int, v2: bool = False) -> list[dict[int, float]]:
    """Collect seed results; v2=True looks for {prefix}_v2_... logs."""
    seeds = []
    tag = f"{prefix}_v2" if v2 else prefix
    for seed in (0, 1, 2):
        p = LOG_DIR / f"{tag}_{board}x{board}_dense_seed{seed}.log"
        if p.exists():
            d = parse_log(p)
            if d:
                seeds.append(d)
    return seeds


def to_arrays(seed_dicts: list[dict]) -> tuple[list, list, list]:
    if not seed_dicts:
        return [], [], []
    all_games = sorted(set().union(*[d.keys() for d in seed_dicts]))
    means, stds = [], []
    import statistics
    for g in all_games:
        vals = [d[g] for d in seed_dicts if g in d]
        if vals:
            means.append(round(sum(vals) / len(vals), 3))
            stds.append(round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 3))
    return all_games, means, stds


def report(label: str, games, means, stds, n):
    print(f"\n# {label} (n={n})")
    if not games:
        print("  ⏳ No data yet")
        return
    print(f"  games={games}")
    print(f"  mean={means}")
    print(f"  std ={stds}")
    print(f"  Final mean score: {means[-1]:.3f}")


if __name__ == "__main__":
    # ConvNet v1 10×10
    c10 = gather("ConvDQN", 10)
    g, m, s = to_arrays(c10)
    report("ConvNet DQN v1 10×10 dense", g, m, s, len(c10))

    # Feature DQN 16×16
    f16 = gather("FeatureDQN", 16)
    g, m, s = to_arrays(f16)
    report("Feature DQN 16×16 dense", g, m, s, len(f16))

    # ConvNet v1 16×16
    c16 = gather("ConvDQN", 16)
    g, m, s = to_arrays(c16)
    report("ConvNet DQN v1 16×16 dense", g, m, s, len(c16))

    # ConvNet v2 10×10
    c10v2 = gather("ConvDQN", 10, v2=True)
    g, m, s = to_arrays(c10v2)
    report("ConvNet DQN v2 10×10 dense", g, m, s, len(c10v2))

    # ConvNet v2 16×16
    c16v2 = gather("ConvDQN", 16, v2=True)
    g, m, s = to_arrays(c16v2)
    report("ConvNet DQN v2 16×16 dense", g, m, s, len(c16v2))

    # ConvNet v3 10×10 (sweep-tuned, 10k games)
    c10v3 = gather("ConvDQN", 10, v2=False)
    # override: v3 uses "ConvDQN_v3_" prefix
    c10v3 = []
    for seed in (0, 1, 2):
        p = LOG_DIR / f"ConvDQN_v3_10x10_dense_seed{seed}.log"
        if p.exists():
            d = parse_log(p)
            if d: c10v3.append(d)
    g, m, s = to_arrays(c10v3)
    report("ConvNet DQN v3 10×10 dense", g, m, s, len(c10v3))

    # ConvNet v3 16×16
    c16v3 = []
    for seed in (0, 1, 2):
        p = LOG_DIR / f"ConvDQN_v3_16x16_dense_seed{seed}.log"
        if p.exists():
            d = parse_log(p)
            if d: c16v3.append(d)
    g, m, s = to_arrays(c16v3)
    report("ConvNet DQN v3 16×16 dense", g, m, s, len(c16v3))

    # Feature DQN v2 10×10 (10k games — matched to ConvNet v3)
    f10v2 = []
    for seed in (0, 1, 2):
        p = LOG_DIR / f"FeatureDQN_v2_10x10_dense_seed{seed}.log"
        if p.exists():
            d = parse_log(p)
            if d: f10v2.append(d)
    g, m, s = to_arrays(f10v2)
    report("Feature DQN 10×10 dense (10k games)", g, m, s, len(f10v2))

    # Feature DQN v2 16×16 (16k games — matched to ConvNet v3)
    f16v2 = []
    for seed in (0, 1, 2):
        p = LOG_DIR / f"FeatureDQN_v2_16x16_dense_seed{seed}.log"
        if p.exists():
            d = parse_log(p)
            if d: f16v2.append(d)
    g, m, s = to_arrays(f16v2)
    report("Feature DQN 16×16 dense (16k games)", g, m, s, len(f16v2))
