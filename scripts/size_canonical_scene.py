"""
Canonical-scene invariance: does board size ALONE (fixed local geometry, fixed food distance) kill the
food signal — and if so, WHERE?

We hand-build ONE identical local scene — head + short snake + food a fixed d=3 cells to the right —
and embed it in boards 6..20. Food distance is fixed, local geometry is fixed; the ONLY thing that
changes is the surrounding board area. Two anchorings separate wall-proximity from pure area:
  - corner : head at (2,2), near walls fixed, far walls recede
  - center : head at (B/2,B/2), all walls recede symmetrically

Readouts per board:
  food_margin_full   : Q[toward] - Q[away] with the normal full-board avg+max pool
  food_margin_localbox: same, but the AVG branch pools only over the occupied bounding box (+2 margin),
                        i.e. the growing empty expanse is EXCLUDED from the average (max stays full)
  avg_dev / max_dev  : ||pool - empty_constant||, how far each branch is from the pure-empty activation

Logic:
  - full food_margin FLAT across board  -> board size per se doesn't matter (reopens food-distance/state).
  - full food_margin COLLAPSES          -> board area ALONE kills the food signal (isolation achieved).
      then: localbox FLAT while full collapses -> the empty expanse entering the AVG pool is the cause
            (a spatial-exclusion fix works even though the count-norm renormalization did not).
      or:   localbox ALSO collapses      -> not the pool; the collapse is upstream / in the head.
"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import SizeAgnosticQNet, DEVICE

OUT = "/Users/saheb/home/rl-snake/size_transfer"
CKPT = f"{OUT}/sizeagnostic_6x6_best.pth"
SIZES = [6, 8, 10, 12, 14, 16, 20]
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # 0 up, 1 right, 2 down, 3 left
D = 3                                          # fixed food distance (to the right)


def make_scene(B, anchor):
    s = np.zeros((3, B, B), np.float32)
    hr, hc = (2, 2) if anchor == "corner" else (B // 2, B // 2)
    s[0, hr, hc - 2] = 1.0; s[0, hr, hc - 1] = 1.0     # body (excl head), moving right
    s[1, hr, hc] = 1.0                                  # head
    s[2, hr, hc + D] = 1.0                              # food, D cells straight ahead (right)
    occ = [(hr, hc - 2), (hr, hc - 1), (hr, hc), (hr, hc + D)]
    return s, (hr, hc), (hr, hc + D), occ


def _head(model, z):
    a = model.advantage_stream(z); v = model.value_stream(z)
    return (v + (a - a.mean(dim=1, keepdim=True))).detach().cpu().numpy().ravel()


@torch.no_grad()
def q_full(model, s):
    h = model.conv(torch.tensor(s[None], device=DEVICE))
    return _head(model, torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)), h


@torch.no_grad()
def q_localbox(model, s, occ, margin=2):
    B = s.shape[1]
    h = model.conv(torch.tensor(s[None], device=DEVICE))
    rs = [p[0] for p in occ]; cs = [p[1] for p in occ]
    r0, r1 = max(0, min(rs) - margin), min(B, max(rs) + margin + 1)
    c0, c1 = max(0, min(cs) - margin), min(B, max(cs) + margin + 1)
    box = h[:, :, r0:r1, c0:c1]
    return _head(model, torch.cat([box.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1))


def food_margin(q, head, food):
    d0 = abs(head[0] - food[0]) + abs(head[1] - food[1])
    toward = np.array([abs(head[0]+dx-food[0]) + abs(head[1]+dy-food[1]) < d0 for dx, dy in DELTAS])
    return float(q[toward].max() - q[~toward].max()), bool(toward[int(q.argmax())])


@torch.no_grad()
def empty_constant(model, B):
    h = model.conv(torch.zeros(1, 3, B, B, device=DEVICE))
    return h[0, :, B // 2, B // 2].cpu().numpy()   # activation of a pure-empty interior cell


def main():
    model = SizeAgnosticQNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, anchor in zip(axes, ["corner", "center"]):
        print(f"\n== anchor={anchor}, fixed local scene (food d={D} right), varying board ==")
        print(f"{'board':>5} | {'fm_full':>8} {'arg_full':>8} | {'fm_localbox':>11} {'arg_lbox':>8} "
              f"| {'avg_dev':>8} {'max_dev':>8}")
        ff, fl, sizes_used = [], [], []
        for B in SIZES:
            hc = 2 if anchor == "corner" else B // 2
            if hc + D >= B or hc - 2 < 0:      # scene must fit on the board
                continue
            sizes_used.append(B)
            s, head, food, occ = make_scene(B, anchor)
            qf, h = q_full(model, s)
            ql = q_localbox(model, s, occ)
            mf, af = food_margin(qf, head, food)
            ml, al = food_margin(ql, head, food)
            c = empty_constant(model, B)
            avg_dev = float(np.linalg.norm(h.mean(dim=(2, 3)).cpu().numpy().ravel() - c))
            max_dev = float(np.linalg.norm(h.amax(dim=(2, 3)).cpu().numpy().ravel() - c))
            ff.append(mf); fl.append(ml)
            print(f"{B:5d} | {mf:8.3f} {str(af):>8} | {ml:11.3f} {str(al):>8} "
                  f"| {avg_dev:8.2f} {max_dev:8.2f}", flush=True)
        ax.plot(sizes_used, ff, "s-", color="#dc2626", label="full-board avg-pool")
        ax.plot(sizes_used, fl, "o-", color="#16a34a", label="local-box avg-pool")
        ax.axhline(0, ls=":", color="#9ca3af", lw=1); ax.axvline(6, ls=":", color="gray", alpha=0.5)
        ax.set_xlabel("board size (fixed local scene)"); ax.set_ylabel("food margin (Q units)")
        ax.set_title(f"anchor = {anchor}"); ax.legend()
    fig.suptitle("Canonical scene: identical local geometry + fixed food distance, only board area grows")
    fig.tight_layout(); fig.savefig(f"{OUT}/canonical_scene.png", dpi=130)
    print(f"\nWrote {OUT}/canonical_scene.png")


if __name__ == "__main__":
    main()
