"""
beam-prediction-baseline
========================
A minimal, fully self-contained baseline for:
    "Deep Learning for Beam Management at 5G/6G Wireless Systems"
    -- sub-task: ML-based beam prediction for mmWave MIMO

Why this file exists
--------------------
This is the minimum experiment needed to make the project proposal credible.
It is intentionally NOT a full reproduction of any paper. It is a working,
runnable sanity check that answers three questions:

    1. Is the task well-posed?        (can a network predict the best beam?)
    2. Is the task non-trivial?       (is it harder than random guessing?)
    3. Do I have the engineering base to build the real thing later?

Physical setup (no hardware required, pure simulation)
------------------------------------------------------
    - Uniform linear array (ULA) of M antennas, mmWave carrier
    - Line-of-sight + multipath channel, generated from a ray-tracing-lite
      model with a handful of geometric paths (path length, angle, gain)
    - Base station sweeps a DFT codebook of B candidate beams
    - For each user, the "best" beam is argmax over the codebook of the
      received power gain  |w_b^H h|^2
    - A neural net receives ONLY the sampled channel vector and must
      predict the best beam index

The key point for the proposal: the network never sees the position,
the path angles, or the ground-truth best beam during training. It only
sees a noisy observation of the channel -- exactly the information a real
UE would have. This is what makes the task realistic rather than trivial.

Run
---
    python sim_channel.py     # generate and cache the dataset
    python train.py           # train / evaluate the models
    python train.py --model mlp   # or the linear-probe baseline
"""

import argparse
import json
import os
import time

import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

CFG = {
    # Array / carrier
    "n_ant": 32,             # ULA elements (even)
    "fc_ghz": 28.0,          # mmWave carrier, GHz
    "n_paths": 5,            # geometric paths (1 LoS + 4 reflected)
    "n_users": 6000,         # simulated users
    "snr_db": 20.0,          # observation SNR

    # Codebook
    "n_beams": 64,           # DFT codebook size (B)

    # Observation (what the network gets to see)
    "n_probes": 6,           # full channel uses in the observation
    "obs_noise": True,

    # Train / val / test
    "frac_val": 0.15,
    "frac_test": 0.15,
    "seed": 20261002,
}


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def ula_steering_vector(n_ant, sin_theta):
    """Normalized DFT steering vector for a ULA, given sin(theta) in [-1, 1].

    Uses the standard progressive phase shift convention. Normalised by
    sqrt(n_ant) so that |w^H h|^2 is comparable across angles.
    """
    m = np.arange(n_ant)
    return np.exp(-2j * np.pi * m * float(sin_theta)) / np.sqrt(n_ant)


def dft_codebook(n_ant, n_beams):
    """B = n_beams uniform DFT beams covering the visible angular range."""
    return np.stack([ula_steering_vector(n_ant, s) for s in np.linspace(-1, 1, n_beams)], axis=1)


def array_response(n_ant, paths_angles, paths_gains, paths_phases):
    """Sum of path contributions into the receive-side channel vector.

    Parameters
    ----------
    paths_angles : (P,) sin(theta) of each path, in [-1, 1]
    paths_gains  : (P,) complex amplitude (includes path loss)
    paths_phases : (P,) initial phase

    Returns
    -------
    h : (M,) complex128
    """
    m = np.arange(n_ant)
    h = np.zeros(n_ant, dtype=complex)
    for s, g, p in zip(paths_angles, paths_gains, paths_phases):
        h += g * np.exp(-2j * np.pi * m * s + 1j * p)
    return h


# ----------------------------------------------------------------------------
# Channel simulation
# ----------------------------------------------------------------------------

def simulate_users(cfg, rng):
    """Generate multi-user mmWave channel realisations.

    Returns
    -------
    H : (U, M) complex128
        True channel vector for each user (from the M full channel uses).
    Users are distributed in angle, with a user-dependent angular spread:
    nearby / well-served users have a tight LoS, poorly-served users a wide
    spread. This heterogeneity is what makes beam prediction non-trivial.
    """
    U, M, P = cfg["n_users"], cfg["n_ant"], cfg["n_paths"]

    # User centre angles, uniform across the visible sector
    centre = rng.uniform(-0.95, 0.95, size=U)

    # Per-user angular spread: log-uniform from 0.005 (well-conditioned)
    # to 0.45 (genuinely hard)
    spread = np.exp(rng.uniform(np.log(0.005), np.log(0.45), size=U))

    # LoS power and the power split between LoS and reflected paths
    los_power = rng.uniform(0.4, 1.0, size=U) ** 2
    refl_ratio = (1.0 - los_power) / (P - 1)
    path_power = np.concatenate([los_power[:, None], refl_ratio[:, None].repeat(P - 1, 1)], axis=1)

    # Random phases
    path_phase = rng.uniform(0, 2 * np.pi, size=(U, P))

    H = np.zeros((U, M), dtype=complex)
    for u in range(U):
        # LoS path sits at the user centre; reflections scatter around it
        offsets = rng.normal(0.0, 1.0, size=P - 1) * spread[u]
        angles = np.clip(centre[u] + np.concatenate([[0.0], offsets]), -1.0, 1.0)
        gains = np.sqrt(path_power[u]) * np.exp(1j * 0)  # real amplitude
        H[u] = array_response(M, angles, gains, path_phase[u])

    return H


def beam_gains(H, W):
    """Received power for every codebook beam.

    G[u, b] = |w_b^H h_u|^2

    Parameters
    ----------
    H : (U, M) complex
    W : (M, B) complex, the DFT codebook

    Returns
    -------
    G : (U, B) float64
    """
    return np.abs(H.conj() @ W) ** 2


def make_dataset(cfg, rng):
    """Build the full supervised dataset.

    For every user:
      - ground-truth best beam   : argmax_b |w_b^H h|^2
      - observed channel        : P full channel uses at random OFDM symbols,
                                  i.i.d. AWGN at the configured SNR

    The observation is what a real UE could plausibly obtain: it knows the
    channel on a few pilots, and must infer which beam to point at.
    """
    W = dft_codebook(cfg["n_ant"], cfg["n_beams"])

    H = simulate_users(cfg, rng)
    G = beam_gains(H, W)
    y = np.argmax(G, axis=1)                       # (U,) best beam index

    # Observation: a UE measuring pilots obtains the COMPLEX channel, not just
    # its magnitude. This matters: for a ULA the angle-of-arrival information
    # lives in the *phase* progression across the array (a linear ramp whose
    # slope is -2*pi*sin(theta)). The magnitude |h| of a single path is
    # constant across the array and carries almost no angular information.
    # Feeding |h| only would make the task artificially impossible.
    #
    # So each probe yields a full complex channel estimate, and we let the
    # network see both real and imaginary parts.
    P = cfg["n_probes"]
    noise_pow = 10 ** (-cfg["snr_db"] / 10.0)
    sig_pow = float(np.mean(np.abs(H) ** 2))
    sigma = np.sqrt(sig_pow * noise_pow / P)

    X = np.empty((cfg["n_users"], P, cfg["n_ant"], 2), dtype=np.float32)
    for p in range(P):
        noisy = H + sigma * (
            np.sqrt(0.5) * rng.standard_normal(H.shape) + 1j * np.sqrt(0.5) * rng.standard_normal(H.shape)
        )
        X[:, p, :, 0] = noisy.real.astype(np.float32)
        X[:, p, :, 1] = noisy.imag.astype(np.float32)

    return X, y.astype(np.int64), G, W


# ----------------------------------------------------------------------------
# CLI / cache
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--users", type=int, default=CFG["n_users"])
    args = ap.parse_args()

    cfg = dict(CFG)
    cfg["n_users"] = args.users

    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    X, y, G, W = make_dataset(cfg, rng)
    print(f"[sim] {cfg['n_users']} users, {cfg['n_ant']} ants, {cfg['n_beams']} beams "
          f"({time.time() - t0:.1f}s)")

    # Sanity check: how non-trivial is the label distribution?
    counts = np.bincount(y, minlength=cfg["n_beams"])
    print(f"[sim] best-beam label spread: {counts.min()}..{counts.max()} "
          f"(uniform would be {len(y) / cfg['n_beams']:.0f})")

    # How much would an oracle gain vs the average beam?
    best_gain = G.max(axis=1).mean()
    avg_gain = G.mean()
    print(f"[sim] mean best-beam gain {best_gain:.3f} vs mean random beam {avg_gain:.3f} "
          f"({10 * np.log10(best_gain / avg_gain):.1f} dB available)")

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, "beam_pred.npz"),
        X=X, y=y, G=G.astype(np.float32),
        n_ant=cfg["n_ant"], n_beams=cfg["n_beams"], n_probes=cfg["n_probes"],
    )
    print(f"[sim] saved to {args.out}/beam_pred.npz  ({X.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
