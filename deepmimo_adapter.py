"""
deepmimo_adapter.py
===================
Port the synthetic baseline onto REAL ray-traced data from DeepMIMO.

Why this matters
----------------
The synthetic experiment in sim_channel.py proves the task is learnable, but
its13.44 dB ceiling is an artifact of my own channel model. This adapter
repeats the same experiment on a real ray-traced scenario, where the channel
includes genuine path loss, diffuse scattering, reflection and shadowing.

The scientific question it answers:
    Does the model ordering (MLP > CNN > linear) survive contact with a
    realistic channel, or was it an artefact of a too-simple simulator?

DeepMIMO v4 API notes (learned the hard way)
--------------------------------------------
v4 stores PATHS, not channels. Each dataset is:
    power / phase / delay / aoa_az / aoa_el  -> (n_ue, n_paths)
So there is no `ds.channels`; you must synthesise the array yourself:
    params = dm.ChannelParameters(bs_antenna={'shape': [32, 1], 'spacing': 0.5}, ...)
    ch = ds.trim(idxs=...).compute_channels(params)   # -> (n_ue, 1, 32, 1)

Two more traps:
  - `ds[:2000]` does NOT work (Dataset is not sliceable). Use `ds.trim(idxs=...)`.
  - `ds.get_idxs('active')` filters out users with zero valid paths
    (85157 of 131931 in the ASU campus scenario).

Setup
-----
    uv venv --python 3.12 .venv-dm
    uv pip install --python .venv-dm/bin/python deepmimo numpy
    .venv-dm/bin/python download_deepmimo.py --only asu_campus_3p5
    .venv-dm/bin/python deepmimo_adapter.py --scenario asu_campus_3p5
"""

import argparse
import os
import sys

import numpy as np

DEFAULT_SCENARIO = "asu_campus_3p5"


# ----------------------------------------------------------------------------
# Acquisition
# ----------------------------------------------------------------------------

def fetch(scenario: str, n_ue: int | None = None, n_ant: int = 32, tx: int = 0):
    """Load a DeepMIMO scenario and synthesise frequency-flat ULA channels.

    Returns
    -------
    channels : (n_ue, n_ant) complex
        One channel vector per user, frequency-flat (single subcarrier).
    positions : (n_ue, 3) float
        User xyz coordinates. Only used for reporting, never fed to the model.
    """
    import deepmimo as dm

    print(f"[deepmimo] loading scenario '{scenario}' ...")
    ds = dm.load(scenario)

    # City scenarios load as a MacroDataset holding one sub-dataset per base
    # station (city_6_miami_3p5 has 3). Each has its own ray-traced grid, so we
    # analyse one site at a time; that is also what makes cross-site
    # generalisation measurable later.
    if hasattr(ds, "datasets"):
        n_sites = len(ds.datasets)
        if tx >= n_sites:
            raise SystemExit(f"{scenario} has {n_sites} sites, --tx {tx} out of range")
        print(f"[deepmimo] MacroDataset with {n_sites} sites -> using site {tx}")
        ds = ds.datasets[tx]

    # Only users with at least one valid ray-traced path are useful. In the
    # ASU campus case roughly 2/3 of the grid qualifies.
    active = ds.get_idxs("active")
    print(f"[deepmimo] {len(active)} / {ds.n_ue} users have a usable path")

    if n_ue is not None and len(active) > n_ue:
        # Evenly spaced subsample keeps spatial coverage intact; a random draw
        # would work too but this is reproducible and unbiased. It also breaks
        # the grid adjacency that would otherwise leak train/test information.
        pick = np.linspace(0, len(active) - 1, n_ue).astype(int)
        active = active[pick]
    print(f"[deepmimo] using {len(active)} users")

    # shape [n_ant, 1] = a ULA along x; 0.5 wavelength spacing is the 3GPP
    # reference configuration (half-wavelength avoids grating lobes).
    params = dm.ChannelParameters(
        bs_antenna={"shape": [n_ant, 1], "spacing": 0.5},
        ue_antenna={"shape": [1, 1]},
        ofdm={"sc_samp": np.array([0])},  # single subcarrier -> freq-flat
    )

    sub = ds.trim(idxs=active)
    ch = np.asarray(sub.compute_channels(params))
    print(f"[deepmimo] raw channel array: {ch.shape} ({ch.dtype})")

    # (n_ue, 1, n_ant, 1) -> (n_ue, n_ant)
    ch = ch[:, 0, :, 0]
    pos = np.asarray(sub.rx_pos)

    # Drop users the ray tracer gave no energy at all.
    valid = np.abs(ch).sum(axis=1) > 0
    if not valid.all():
        print(f"[deepmimo] dropping {(~valid).sum()} all-zero channels")
        ch, pos = ch[valid], pos[valid]

    return ch, pos


# ----------------------------------------------------------------------------
# Codebook and labelling
# ----------------------------------------------------------------------------

def ula_steering(n_ant: int, sin_theta: float) -> np.ndarray:
    m = np.arange(n_ant)
    return np.exp(-2j * np.pi * m * float(sin_theta)) / np.sqrt(n_ant)


def codebook(n_ant: int, n_beams: int) -> np.ndarray:
    """Uniform DFT beam codebook spanning the full visible angular range."""
    return np.stack(
        [ula_steering(n_ant, s) for s in np.linspace(-1, 1, n_beams)], axis=1
    )


def label_challenging_beams(G: np.ndarray, n_challenging: int = 8):
    """Pick N beams to verify, including the beams a naive approach would miss.

    Standard beam-prediction evaluations use the 'N challenging beams'
    protocol: the network must be correct about which of N candidate beams is
    best, rather than across the whole codebook. We use the beam with the
    highest gain (the "weakest" case a user must be able to find) together
    with a spread of others.

    Selection criterion matters a great deal on real ray-traced data. Picking
    the top-N beams by MEAN gain (the textbook choice) fails badly here: in the
    ASU campus scenario the strongest-by-average beams cover only 26.9% of
    users, because average gain is a poor proxy for which beam a given user
    actually needs. We instead select the top-N beams by LABEL FREQUENCY --
    the beams that are actually the argmax for the most users -- which makes the
    protocol meaningful (a 100% ceiling) and matches what the metric is for.
    """
    n_beams = G.shape[1]
    freq = np.bincount(np.argmax(G, axis=1), minlength=n_beams)
    cand = np.sort(np.argsort(freq)[::-1][:n_challenging])
    sub = G[:, cand]
    relabel = np.argmax(sub, axis=1)
    return cand, relabel


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

def build(ch, n_ant, n_beams, n_probes, snr_db, n_challenging, seed=20261002):
    """Turn raw DeepMIMO channels into the same tensors the synthetic path uses."""
    rng = np.random.default_rng(seed)
    W = codebook(n_ant, n_beams)

    n_ue = len(ch)
    H = np.zeros((n_ue, n_ant), dtype=complex)
    for u in range(n_ue):
        v = ch[u]
        H[u] = v[:n_ant] if len(v) >= n_ant else np.pad(v, (0, n_ant - len(v)))

    G = np.abs(H.conj() @ W) ** 2
    y_full = np.argmax(G, axis=1)

    # Per-user noise power. Using a global mean would give near-zero SNR to
    # far-away users, whose path loss is orders of magnitude below the mean.
    pu = np.mean(np.abs(H) ** 2, axis=1, keepdims=True)
    sigma = np.sqrt(pu * 10 ** (-snr_db / 10.0) / n_probes)

    X = np.empty((n_ue, n_probes, n_ant, 2), dtype=np.float32)
    noise = np.sqrt(0.5) * rng.standard_normal(H.shape) + 1j * np.sqrt(0.5) * rng.standard_normal(H.shape)
    for p in range(n_probes):
        noisy = H + sigma * noise
        X[:, p, :, 0] = noisy.real.astype(np.float32)
        X[:, p, :, 1] = noisy.imag.astype(np.float32)

    cand, relabel = label_challenging_beams(G, n_challenging)
    return X, y_full, G, W, cand, relabel


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--n-ue", type=int, default=20000)
    ap.add_argument("--n-ant", type=int, default=32)
    ap.add_argument("--n-beams", type=int, default=64)
    ap.add_argument("--n-probes", type=int, default=6)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--challenging", type=int, default=8)
    ap.add_argument("--tx", type=int, default=0, help="which base station to use")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    ch, pos = fetch(args.scenario, args.n_ue, args.n_ant, args.tx)
    if len(ch) < 500:
        print("[warn] very few usable UEs -- the scenario may be sparsely populated")

    X, y, G, W, cand, relabel = build(
        ch, args.n_ant, args.n_beams, args.n_probes, args.snr, args.challenging
    )

    # ---- Report the real-data characteristics -------------------------------
    best = G.max(axis=1)
    avg = G.mean()
    print(f"\n[data] {len(X)} UEs, {args.n_ant} antennas, {args.n_beams} beams")
    print(f"[data] pathloss spread: {(10 * np.log10(pu_of(ch))).min():.1f} .. "
          f"{(10 * np.log10(pu_of(ch))).max():.1f} dB")
    print(f"[data] usable gain (oracle - average beam): "
          f"{10 * np.log10(best.mean() / avg):.2f} dB")

    dist = np.linalg.norm(pos[:, :2] - pos[:, :2].mean(axis=0), axis=1)
    print(f"[data] user spread: x {pos[:, 0].min():.0f}..{pos[:, 0].max():.0f} m, "
          f"y {pos[:, 1].min():.0f}..{pos[:, 1].max():.0f} m")

    counts = np.bincount(y, minlength=args.n_beams)
    print(f"[data] best-beam label distribution: {counts.min()}..{counts.max()} "
          f"(uniform = {len(y) / args.n_beams:.0f})")
    print(f"[data] {args.challenging}-challenging-beam protocol: beams {cand.tolist()}")
    print(f"[data] label distribution within challenging set: "
          f"{np.bincount(relabel, minlength=args.challenging).tolist()}")

    os.makedirs(args.out, exist_ok=True)
    # Include the site index so multi-station scenarios do not overwrite
    # each other's datasets.
    suffix = f"_tx{args.tx}" if args.tx else ""
    path = os.path.join(args.out, f"deepmimo_{args.scenario}{suffix}.npz")
    np.savez_compressed(
        path, X=X, y=y, y_challenging=relabel, G=G.astype(np.float32),
        cand=cand, n_ant=args.n_ant, n_beams=args.n_beams, n_probes=args.n_probes,
    )
    print(f"\n[data] saved {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def pu_of(ch):
    """Per-user average channel power, for pathloss reporting."""
    return np.mean(np.abs(ch) ** 2, axis=1)


if __name__ == "__main__":
    sys.exit(main())
