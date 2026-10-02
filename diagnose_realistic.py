"""
diagnose_realistic.py
=====================
Why does the MLP drop from 0.43 (no path loss) to 0.11 (realistic)?

Two candidate explanations, and they have very different implications:
  (a) BUG -- the 56 dB dynamic range is destroying the weak users' signal, or
             the noise is being applied at a fixed absolute level and so wipes
             out exactly the far users.
  (b) PHYSICS -- distant users have lower SNR and a wider angular spread, so
             they genuinely carry less information. If so the low number is the
             honest answer and the fix is better input handling, not a bug fix.

The test: stratify accuracy by distance. Under (a) accuracy would be flat-ish
and uniformly bad. Under (b) accuracy should fall off monotonically with
distance, and near users should still be well predicted.
"""

import numpy as np
import torch
import torch.nn as nn

from train import build, CFG
from train_realistic import load_variant, train_one

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    tr, va, te, Gte, blocked, moving = load_variant("realistic")
    d = np.load("data/beam_pred_realistic.npz")
    P, M, B = int(CFG["n_probes"]), int(CFG["n_ant"]), int(CFG["n_beams"])

    # Rebuild the test indices identically to load_variant
    n = len(d["X"])
    n_val, n_test = int(0.15 * n), int(0.15 * n)
    idx = np.random.default_rng(0).permutation(n)
    te_idx = idx[n_val:n_val + n_test]
    d_te = d["d_ue"][te_idx]

    model = train_one("mlp", tr, M, P, 40)
    model.eval()
    with torch.no_grad():
        pred = model(te[0]).argmax(1)

    y = te[1].cpu().numpy()
    pred = pred.cpu().numpy()
    G = Gte
    rel = G[np.arange(len(y)), pred] / G.mean(axis=1)

    print(f"{'distance bin':>16s} {'users':>7s} {'top1':>8s} {'rel gain dB':>12s}")
    print("-" * 48)
    edges = [0, 50, 100, 200, 300, 400, 600]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_te >= lo) & (d_te < hi)
        if m.sum() < 10:
            continue
        acc = (pred[m] == y[m]).mean()
        gain = 10 * np.log10(rel[m].mean())
        print(f"{f'{lo}-{hi} m':>16s} {m.sum():>7d} {acc:>8.4f} {gain:>12.2f}")

    print("\nInterpretation:")
    near = (d_te < 100)
    far = (d_te >= 300)
    print(f"  near (<100 m)  top1 {(pred[near] == y[near]).mean():.4f}  "
          f"gain {10 * np.log10(rel[near].mean()):.2f} dB")
    print(f"  far  (>=300 m) top1 {(pred[far] == y[far]).mean():.4f}  "
          f"gain {10 * np.log10(rel[far].mean()):.2f} dB")

    # Per-user SNR check: is the far-user signal genuinely buried in noise?
    X = d["X"][te_idx]
    pw = np.mean(X[:, 0, :, 0] ** 2 + X[:, 0, :, 1] ** 2, axis=1)
    for name, m in [("near", near), ("far", far)]:
        print(f"  {name:5s} mean received power {10 * np.log10(pw[m].mean() + 1e-20):.1f} dB")

    print("\n--- control: same model on no_pathloss ---")
    tr2, va2, te2, G2, _, _ = load_variant("no_pathloss")
    d2 = np.load("data/beam_pred_no_pathloss.npz")
    d_te2 = d2["d_ue"][te_idx]
    m2 = train_one("mlp", tr2, M, P, 40)
    m2.eval()
    with torch.no_grad():
        p2 = m2(te2[0]).argmax(1).cpu().numpy()
    y2 = te2[1].cpu().numpy()
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_te2 >= lo) & (d_te2 < hi)
        if m.sum() < 10:
            continue
        print(f"{f'{lo}-{hi} m':>16s} {m.sum():>7d} {(p2[m] == y2[m]).mean():>8.4f}")


if __name__ == "__main__":
    main()
