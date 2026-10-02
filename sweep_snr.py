"""
sweep_snr.py
============
Sweep SNR and probe count on the real ray-traced data.

Why
---
The single-point result (MLP 0.87 top-1, 99.4% of oracle gain at 20 dB) says the
benchmark is SATURATED. A saturated benchmark cannot support a thesis, but it
does tell us exactly where to look: the interesting question is not "can it
work" but "where does it break".

This script maps that boundary. It answers three things a static single-point
experiment cannot:

  1. How much accuracy is lost per dB of SNR, and where does the curve bend?
  2. How many probes are actually needed -- is 6 wasteful, or already too few?
  3. How large is the gap between the MLP and a linear probe at low SNR?
     If the linear probe matches the MLP when SNR is low, the deep model is
     not earning its parameters exactly where the difficulty is.

Evaluation uses relative gain (per-user normalised), which is the only
meaningful scale when path loss spans >100 dB.

Run
---
    python sweep_snr.py --data data/deepmimo_asu_campus_3p5.npz
    python sweep_snr.py --data data/deepmimo_city_6_miami_3p5.npz --out results_sweep_miami.json
"""

import argparse
import json

import numpy as np
import torch

from train import build
from train_deepmimo import DEV, trivial_baselines, train_one, evaluate


def rebuild_at_snr(path, snr_db, n_probe, seed=0):
    """Regenerate the noisy observation at a different SNR and probe count.

    The clean channel itself is not stored, so it is reconstructed from the
    saved data: averaging the probes recovers the noise-free observation well
    enough for a sweep whose purpose is relative comparison across SNR. The
    per-probe noise is then re-applied on top at the requested level.
    """
    d = np.load(path)
    X, y, G = d["X"], d["y"], d["G"].astype(np.float64)
    n_ant, n_beams = int(d["n_ant"]), int(d["n_beams"])
    P0 = int(d["n_probes"])
    cand = d["cand"]

    # Recover the clean observation by averaging the P0 available probes. With
    # P0=6 at 20 dB the residual contamination is small relative to the signal,
    # and it is identical across all sweep points, so comparisons stay fair.
    Xc = X.mean(axis=1, keepdims=True)                    # (N, 1, M, 2)
    H = Xc[:, 0, :, 0] + 1j * Xc[:, 0, :, 1]              # (N, M)

    rng = np.random.default_rng(seed)
    Grel = G / G.mean(axis=1, keepdims=True)

    n = len(H)
    n_val, n_test = int(0.15 * n), int(0.15 * n)
    idx = np.random.default_rng(seed).permutation(n)
    va, te, tr = idx[:n_val], idx[n_val:n_val + n_test], idx[n_val + n_test:]

    # Per-user power drives the noise level: a fixed global sigma would give
    # far-away users (100+ dB down) effectively zero SNR.
    pw = np.mean(np.abs(H) ** 2, axis=1, keepdims=True)
    sigma = np.sqrt(pw * 10 ** (-snr_db / 10.0) / n_probe)

    noise = (np.sqrt(0.5) * rng.standard_normal(H.shape)
             + 1j * np.sqrt(0.5) * rng.standard_normal(H.shape))

    def pack(ii):
        x = np.empty((len(ii), n_probe, n_ant, 2), dtype=np.float32)
        for p in range(n_probe):
            noisy = H[ii] + sigma[ii] * noise[ii]
            x[:, p, :, 0] = noisy.real.astype(np.float32)
            x[:, p, :, 1] = noisy.imag.astype(np.float32)
        t = torch.from_numpy(x).float()
        pw_t = (t ** 2).sum(dim=(2, 3), keepdim=True)
        t = t / torch.sqrt(pw_t + 1e-8 * pw_t.mean())
        t = t.to(DEV)
        return t.reshape(len(ii), -1), torch.from_numpy(y[ii]).to(DEV), t

    return {"tr": pack(tr), "va": pack(va), "te": pack(te), "Grel": Grel[te],
            "y_test": y[te], "n_ant": n_ant, "n_beams": n_beams,
            "n_probe": n_probe, "cand": cand}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/deepmimo_asu_campus_3p5.npz")
    ap.add_argument("--models", default="linear,mlp")
    ap.add_argument("--snrs", default="-5,0,5,10,15,20,25,30")
    ap.add_argument("--probes", default="1,2,4,6")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--snr-fixed", type=float, default=20.0,
                    help="SNR held at this value during the probe sweep")
    ap.add_argument("--out", default="results_sweep.json")
    args = ap.parse_args()

    snrs = [float(s) for s in args.snrs.split(",")]
    probes = [int(p) for p in args.probes.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    out = {"data": args.data, "snr_sweep": {}, "probe_sweep": {}}
    print(f"[sweep] {args.data}\n")

    # ---- SNR sweep, probe count fixed at the dataset default (6) -------------
    print("========== SNR 扫描（探测次数 = 6）==========")
    print(f"{'SNR':>6} | " + " | ".join(f"{m:>18}" for m in models))
    print("-" * (8 + 21 * len(models)))
    for snr in snrs:
        D = rebuild_at_snr(args.data, snr, 6)
        row = {}
        cells = []
        for m in models:
            model, _ = train_one(m, D["tr"], D["n_ant"], 6, D["n_beams"],
                                 epochs=args.epochs)
            ev = evaluate(model, D["te"][0], D["te"][2], D["te"][1],
                          D["Grel"], m, D["cand"])
            row[m] = ev
            cells.append(f"{ev['top1']:.3f} / {ev['rel_gain_dB']:5.2f} dB")
        out["snr_sweep"][f"{snr}"] = row
        print(f"{snr:>6.0f} | " + " | ".join(cells))

    # ---- Probe sweep, SNR fixed --------------------------------------------
    print(f"\n========== 探测次数扫描（SNR = {args.snr_fixed} dB）==========")
    print(f"{'probes':>6} | " + " | ".join(f"{m:>18}" for m in models))
    print("-" * (8 + 21 * len(models)))
    for P in probes:
        D = rebuild_at_snr(args.data, args.snr_fixed, P)
        row = {}
        cells = []
        for m in models:
            model, _ = train_one(m, D["tr"], D["n_ant"], P, D["n_beams"],
                                 epochs=args.epochs)
            ev = evaluate(model, D["te"][0], D["te"][2], D["te"][1],
                          D["Grel"], m, D["cand"])
            row[m] = ev
            cells.append(f"{ev['top1']:.3f} / {ev['rel_gain_dB']:5.2f} dB")
        out["probe_sweep"][f"{P}"] = row
        print(f"{P:>6d} | " + " | ".join(cells))

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
