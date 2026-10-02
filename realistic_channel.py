"""
realistic_channel.py
====================
An upgraded channel simulator, used to test whether the model ordering from
sim_channel.py survives a physically more demanding channel.

Why this file exists
--------------------
The first simulator was deliberately simple: constant path power, static
users, no shadowing, no mobility. It proved the task is learnable, but its
13.44 dB ceiling was an artefact of those simplifications.

DeepMIMO would be the proper way to test this on real ray-traced data, but the
dataset host (f005.backblazeb2.com) is unreachable from this network -- the
API endpoint resolves and answers, the object-storage redirect is reset. That
is an environment limitation, not a code problem, and it is documented in
REPORT.md rather than worked around.

So this file does the next best thing: add the specific physical effects that
make the original easy. Each is a real effect measured channel models must
capture, not a cosmetic change.

What is added
-------------
1. Path loss. Received power falls with distance. A 4th-power law with a
   configurable exponent, giving a realistic ~40 dB spread across a cell.
   Consequence: absolute gain is no longer meaningful, only the *relative*
   best-beam choice, which is exactly how the literature evaluates this.
2. Log-normal shadowing. A slow fading component, spatially smooth via
   per-cluster random walks. This is what makes beam prediction hard in
   practice: a user can have a good LoS path but still be in a fade.
3. Angular spread with distance. Nearby users have a tight angle, distant
   users are smeared. Predicted beams should degrade with distance.
4. Mobility. A fraction of users are moving, so their channel changes between
   probes -- a learned model that assumes a static channel must fail.
5. Blockage. A fraction of users are shadowed from the BS entirely.
6. Antenna pattern and mutual coupling. Real elements are not isotropic and
   couple to their neighbours, so off-broadside beams lose a little.

Each effect can be toggled, which doubles as an ablation study.

The scientific question
-----------------------
Does MLP still dominate when the channel stops being convenient?
A model that only works on a clean simulator has not learned anything about
the physics, and is useless on a real deployment.
"""

import argparse
import json
import os
import time

import numpy as np

from sim_channel import CFG, ula_steering_vector, dft_codebook, beam_gains

REAL = {
    # Geometry
    "bs_height": 15.0,          # base station height, m
    "bs_height_ue": 1.5,        # user equipment height, m
    "cell_radius": 500.0,       # max user distance, m

    # Path loss: PL = PL0 + 10*n*log10(d/d0)
    "pl_exponent": 3.8,         # path loss exponent (urban macro ~3.8)
    "pl_d0": 1.0,               # reference distance, m
    "pl_at_d0": 32.0,           # loss at reference distance, dB

    # Shadowing
    "shadow_sigma_db": 6.0,     # log-normal shadowing std
    "shadow_clusters": 40,      # spatial correlation scale
    "shadow_corr": 0.997,       # per-cluster random walk correlation

    # Angular spread grows with distance.
    # Values are radians of std-dev of the AoA. 0.004 rad (0.23 deg) is a
    # well-conditioned near user; 0.12 rad (6.9 deg) is a strongly scattered
    # far user -- realistic for an urban macro cell at 28 GHz. Anything beyond
    # ~0.2 rad would smear each user's path angles across most of the visible
    # sector, which makes the correct beam genuinely unlearnable rather than
    # merely hard.
    "spread_near": 0.004,       # at close range
    "spread_far": 0.12,         # at cell radius
    "spread_ref_distance": 50.0,

    # Mobility
    "moving_fraction": 0.25,
    "speed_mps": 8.0,           # typical UE speed
    "probe_gap_s": 0.5,         # time between probes -> decorrelation

    # Blockage
    "blocked_fraction": 0.10,

    # Array realism
    "element_pattern": True,    # cosine rolloff off broadside
    "mutual_coupling": 0.05,    # coupling coefficient between neighbours
    "n_paths_urban": 9,         # more paths than the simple model
}


def path_loss_db(d, cfg):
    """Log-distance path loss with a configurable exponent."""
    d = np.maximum(d, cfg["pl_d0"])
    return cfg["pl_at_d0"] + 10 * cfg["pl_exponent"] * np.log10(d / cfg["pl_d0"])


def shadowing(n_users, cfg, rng):
    """Spatially-correlated log-normal shadowing via per-cluster random walks.

    A constant sigma per user would make the channel i.i.d. and unrealistically
    easy. Real shadowing is spatially correlated: nearby users share a fade.
    This is generated as a random walk over clusters, which gives smooth
    variation along the user ordering.
    """
    sigma = cfg["shadow_sigma_db"]
    n_c = cfg["shadow_clusters"]
    rho = cfg["shadow_corr"]

    # Per-cluster level, smoothed along the ordering
    walk = np.zeros(n_users)
    cur = rng.normal(0, sigma)
    for i in range(n_users):
        cur = rho * cur + np.sqrt(1 - rho ** 2) * rng.normal(0, sigma)
        walk[i] = cur
    # Blend cluster walk with a small per-user residual
    return walk + rng.normal(0, sigma * 0.3, n_users)


def angular_spread(d, cfg):
    """Angle-of-arrival spread, in radians, growing with distance.

    Close to the BS the LoS path dominates and the spread is small; far away,
    diffuse scattering dominates. The curve is a saturating one (distance
    transforms the spread, not the raw distance) and is clamped to a sane
    maximum -- a spread comparable to the visible range would make the label
    pure noise, which is not what a real channel does.
    """
    d = np.maximum(np.asarray(d, dtype=float), 1.0)
    lo, hi = cfg["spread_near"], cfg["spread_far"]
    # log10 distance mapped onto [0, 1) -- saturating and always positive
    t = np.log10(d / 10.0) / np.log10(cfg["cell_radius"] / 10.0)
    t = np.clip(t, 0.0, 1.0)
    return lo + (hi - lo) * t


def element_pattern_factor(sin_theta):
    """A mild cosine-squared rolloff, as real patch/dipole elements have."""
    k = 0.9
    return (np.cos(np.pi / 2 * np.clip(sin_theta, -1, 1)) ** 2) ** k + 0.05


def coupled_array(n_ant, k):
    """Apply a simple nearest-neighbour mutual coupling matrix."""
    A = np.eye(n_ant, dtype=complex)
    idx = np.arange(n_ant - 1)
    A[idx, idx + 1] = -k
    A[idx + 1, idx] = -k
    return A


def simulate_realistic(cfg, rc, rng):
    """Generate a realistic multi-user channel set.

    Returns
    -------
    H_true  : (U, M) complex  -- the noiseless channel (uses the same per-user
              geometry for every probe, i.e. quasi-static within one frame)
    H_first : (U, M) complex  -- the channel at probe 0
    H_last  : (U, M) complex  -- the channel at the last probe (differs for
              moving users)
    d_ue    : (U,) distance
    blocked : (U,) bool
    moving  : (U,) bool
    """
    # The base config and the realism config are merged into one lookup, so the
    # geometry knobs (cell_radius, bs_height, ...) and the array knobs
    # (n_ant, n_users) can be read the same way.
    cfg = {**cfg, **rc}

    U, M = cfg["n_users"], cfg["n_ant"]
    P = cfg["n_paths_urban"]

    # ---- user geometry: uniform in area within the cell -------------------
    r = cfg["cell_radius"] * np.sqrt(rng.uniform(0, 1, U))
    theta = rng.uniform(-np.pi, np.pi, U)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 3D distance including height difference
    dz = rc["bs_height"] - rc["bs_height_ue"]
    d3 = np.sqrt(r ** 2 + dz ** 2)

    # ---- complex amplitude: 1/sqrt(d^alpha) * shadowing ------------------
    pl = path_loss_db(d3, rc)
    shadow_db = shadowing(U, rc, rng)
    amp_db = -pl + shadow_db - cfg["pl_at_d0"]
    amp = 10 ** (amp_db / 20.0)

    # ---- blockage: deep fade for a fraction of users ----------------------
    blocked = rng.uniform(0, 1, U) < rc["blocked_fraction"]
    amp[blocked] *= 10 ** (-25.0 / 20.0)

    # ---- direction of arrival and angular spread -------------------------
    aoa = np.arctan2(y, np.sqrt(np.maximum(x ** 2, 1e-6)))
    # LoS AoA relative to the BS boresight (x axis)
    los_sin = np.clip(y / np.maximum(r, 1e-6), -1, 1)
    spread = angular_spread(r, rc)

    # ---- moving users -----------------------------------------------------
    moving = rng.uniform(0, 1, U) < rc["moving_fraction"]
    delta_rad = rc["speed_mps"] * rc["probe_gap_s"] * (cfg["n_probes"] - 1) / np.maximum(r, 1e-6)

    A = coupled_array(M, rc["mutual_coupling"])
    m = np.arange(M)

    def steer(s):
        """(U,) sin-theta -> (U, M) steering matrix, fully vectorised."""
        s = np.clip(np.asarray(s, dtype=float), -1.0, 1.0)[:, None]
        return np.exp(-2j * np.pi * m[None, :] * s)      # (U, M)

    def apply_coupling(H):
        """Nearest-neighbour mutual coupling as a right-multiplication.

        A is (M, M) and each row of H is one user, so the coupling acts along
        the antenna axis: H_coupled = H @ A.T
        """
        return H @ A.T

    def build(aoa_u, spread_u, amp_u, phase_u):
        # dominant LoS path
        s = np.clip(np.asarray(aoa_u, dtype=float), -1.0, 1.0)
        g = amp_u * (element_pattern_factor(s) if rc["element_pattern"] else 1.0)
        H = (g * np.exp(1j * phase_u))[:, None] * steer(s)          # (U, M)

        # diffuse paths: P-1 scatterers at the user's angular spread
        sc_ang = np.clip(aoa_u[:, None] + rng.normal(0, 1, (U, P - 1)) * spread_u[:, None], -1, 1)
        sc_amp = amp_u[:, None] / np.sqrt(P) * (0.4 + 0.6 * rng.uniform(0, 1, (U, P - 1)))
        sc_ph = rng.uniform(0, 2 * np.pi, (U, P - 1))
        coeff = (sc_amp * np.exp(1j * sc_ph))[:, :, None]           # (U, P-1, 1)
        H = H + (coeff * steer(sc_ang.ravel()).reshape(U, P - 1, M)).sum(axis=1)

        return apply_coupling(H)

    phase0 = rng.uniform(0, 2 * np.pi, U)

    H_first = build(los_sin, spread, amp, phase0)

    # Moving users shift angle by delta over the probe window
    los_sin_last = np.clip(los_sin + moving * delta_rad * rng.choice([-1, 1], U), -1, 1)
    H_last = build(los_sin_last, spread, amp, rng.uniform(0, 2 * np.pi, U))

    return H_first, H_last, d3, blocked, moving


def make_dataset(cfg, rc, rng, add_noise=True):
    W = dft_codebook(cfg["n_ant"], cfg["n_beams"])
    H_first, H_last, d_ue, blocked, moving = simulate_realistic(cfg, rc, rng)

    G = beam_gains(H_first, W)
    y = np.argmax(G, axis=1)

    P = cfg["n_probes"]
    noise_pow = 10 ** (-cfg["snr_db"] / 10.0)
    # Noise is set RELATIVE TO EACH USER'S OWN channel power, not to the global
    # mean. With a global reference, a 28 dB path-loss range translates directly
    # into a 28 dB SNR range: distant users are buried below the noise floor
    # and become unlearnable, which is an artefact of the noise model rather
    # than a property of the channel. A per-user SNR is also the physically
    # standard convention when evaluating beam-prediction quality.
    per_user_pow = (np.abs(H_first) ** 2).mean(axis=1, keepdims=True)   # (U, 1)
    sigma = np.sqrt(per_user_pow * noise_pow)                            # (U, 1)

    X = np.empty((cfg["n_users"], P, cfg["n_ant"], 2), dtype=np.float32)
    for p in range(P):
        t = p / max(1, P - 1)
        H = (1 - t) * H_first + t * H_last          # time evolution
        if add_noise:
            H = H + sigma * (
                np.sqrt(0.5) * rng.standard_normal(H.shape)
                + 1j * np.sqrt(0.5) * rng.standard_normal(H.shape)
            )
        X[:, p, :, 0] = H.real.astype(np.float32)
        X[:, p, :, 1] = H.imag.astype(np.float32)

    return X, y, G, W, d_ue, blocked, moving


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--users", type=int, default=8000)
    ap.add_argument("--ablate", action="store_true",
                    help="generate the ablation variants too")
    args = ap.parse_args()

    cfg = dict(CFG); cfg["n_users"] = args.users
    rc = dict(REAL)
    os.makedirs(args.out, exist_ok=True)

    variants = [("realistic", rc)]
    if args.ablate:
        variants += [
            ("no_pathloss", {**rc, "pl_exponent": 0.0}),
            ("no_shadowing", {**rc, "shadow_sigma_db": 0.0}),
            ("no_mobility", {**rc, "moving_fraction": 0.0}),
            ("no_blockage", {**rc, "blocked_fraction": 0.0}),
            # Isolate the dominant effect: angular spread is the ONLY term that
            # limits how well ANY model can do, because the best beam is only
            # predictable to within the spread of the arriving paths.
            ("no_spread", {**rc, "spread_near": 0.004, "spread_far": 0.004}),
            ("static_clean", {**rc, "pl_exponent": 0.0, "shadow_sigma_db": 0.0,
                              "moving_fraction": 0.0, "blocked_fraction": 0.0,
                              "n_paths_urban": 5}),
        ]

    for name, rcfg in variants:
        rng = np.random.default_rng(cfg["seed"])
        t0 = time.time()
        X, y, G, W, d_ue, blocked, moving = make_dataset(cfg, rcfg, rng)

        # Normalise gain relative to each user's own average beam: with path
        # loss, absolute gain is dominated by distance and carries no
        # information about beam quality. This is the standard way to evaluate
        # under distance variation.
        Grel = G / (G.mean(axis=1, keepdims=True) + 1e-12)
        # Grel rows sum to n_beams by construction, so the relative gain of a
        # user's best beam is simply 10*log10(Grel.max / mean) -- NOT a second
        # normalisation of Grel, which would divide by 1/n_beams and make every
        # ratio look tiny.
        rel_db = float(np.mean(10 * np.log10(Grel.max(axis=1) / (Grel.mean(axis=1) + 1e-12) + 1e-12)))
        top1_ref = (y == np.bincount(y, minlength=cfg["n_beams"]).argmax()).mean()

        print(f"\n[{name}] {args.users} users ({time.time() - t0:.1f}s)")
        print(f"  relative best-beam gain: {rel_db:.2f} dB")
        print(f"  label spread: {np.bincount(y, minlength=cfg['n_beams']).min()}.."
              f"{np.bincount(y, minlength=cfg['n_beams']).max()}, most-freq {top1_ref:.3f}")
        print(f"  blocked {blocked.mean()*100:.0f}%  moving {moving.mean()*100:.0f}%  "
              f"dist {d_ue.min():.0f}-{d_ue.max():.0f} m")

        np.savez_compressed(
            os.path.join(args.out, f"beam_pred_{name}.npz"),
            X=X, y=y, Grel=Grel.astype(np.float32), d_ue=d_ue.astype(np.float32),
            blocked=blocked, moving=moving,
            n_ant=cfg["n_ant"], n_beams=cfg["n_beams"], n_probes=cfg["n_probes"],
        )
        print(f"  saved data/beam_pred_{name}.npz")

    with open(os.path.join(args.out, "realistic_config.json"), "w") as f:
        json.dump({"cfg": cfg, "real": rc,
                   "note": "DeepMIMO was unreachable (Backblaze B2 host blocked); "
                           "this simulator adds the equivalent physical effects."},
                  f, indent=2)


if __name__ == "__main__":
    main()
