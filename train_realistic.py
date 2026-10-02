"""
train_realistic.py
==================
Train the baselines on the physically-realistic channel and run the ablation.

The question
------------
sim_channel.py (static, no path loss, no shadowing, no mobility) gave:
    MLP 0.806 top-1 / 13.31 dB,  CNN 0.538 / 12.73 dB

realistic_channel.py adds path loss (3.8 exponent), spatially-correlated
log-normal shadowing, distance-dependent angular spread, a quarter of users
moving, a tenth blocked, plus element patterns and mutual coupling.

If the model ranking survives, the synthetic result was not an artefact of a
convenient simulator. If it collapses, the task is much harder in practice and
that is worth knowing before committing to it.

Evaluation
----------
Absolute gain is meaningless once path loss varies by 40 dB across the cell, so
every gain here is RELATIVE to each user's own average-beam gain. That is the
standard way this problem is reported.

Run
---
    python train_realistic.py --variant realistic
    python train_realistic.py --ablation
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from train import build, CFG

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VARIANTS = ["realistic", "no_pathloss", "no_shadowing", "no_mobility",
            "no_blockage", "no_spread", "static_clean"]


def load_variant(name, data_dir="data", seed=0, per_user_norm=True):
    """Load a variant and normalise the observation.

    per_user_norm matters. With path loss and shadowing the received power
    spans ~56 dB across the cell, so a single global scale factor shrinks the
    weakest users to near-zero amplitude: the network cannot see them at all
    and falls back on predicting the majority class, landing BELOW the trivial
    "always guess the commonest beam" baseline.

    Normalising each user's observation by its own power removes the irrelevant
    absolute level and keeps only the channel *shape* -- which is the entire
    information content needed to choose a beam. This mirrors the evaluation,
    where gain is always reported relative to the user's own average beam.
    """
    d = np.load(f"{data_dir}/beam_pred_{name}.npz")
    X, y, Grel = d["X"], d["y"], d["Grel"]
    n = len(X)
    n_val = int(0.15 * n)
    n_test = int(0.15 * n)
    idx = np.random.default_rng(seed).permutation(n)
    va, te, tr = idx[:n_val], idx[n_val:n_val + n_test], idx[n_val + n_test:]

    def pack(ii):
        x = torch.from_numpy(X[ii]).float()
        if per_user_norm:
            # (B, P, M, 2) -> power per (B, P) -> divide, keeping shape
            pw = (x ** 2).sum(dim=(2, 3), keepdim=True)      # (B, P, 1, 1)
            # A FIXED absolute epsilon is wrong here: with path loss the whole
            # dataset sits around 1e-13 W, so an epsilon of 1e-12 exceeds the
            # signal and the divisor degenerates to a constant, blinding the
            # model. Scale the floor relative to the batch instead.
            eps = 1e-8 * pw.mean()
            x = x / torch.sqrt(pw + eps)
        else:
            x = x / (x.std() + 1e-8)
        return x.reshape(len(ii), -1).to(DEV), torch.from_numpy(y[ii]).to(DEV), x.to(DEV)

    return pack(tr), pack(va), pack(te), Grel[te], d.get("blocked"), d.get("moving")


@torch.no_grad()
def evaluate(model, xflat, x4, y, Grel, name, P, M):
    """Relative gain: 10*log10(G[pred] / mean_b G). Mean over users."""
    model.eval()
    logits = model(x4 if name == "cnn" else xflat)
    top3 = torch.topk(logits, 3, dim=1).indices
    t1 = (top3[:, 0] == y).float().mean().item()
    t3 = (top3 == y[:, None]).any(1).float().mean().item()

    G = torch.from_numpy(Grel).to(DEV)
    gmean = G.mean(dim=1, keepdim=True)
    gp = (G.gather(1, top3[:, :1]) / gmean).mean().item()
    gb = (G.gather(1, y[:, None]) / gmean).mean().item()
    return {
        "top1": t1, "top3": t3,
        "rel_gain_dB": float(10 * np.log10(gp)),
        "oracle_dB": float(10 * np.log10(gb)),
    }


def train_one(name, tr, n_ant, P, epochs=30, lr=3e-3, wd=1e-4):
    xtr, ytr, xtr4 = tr
    model = build(name, xtr.shape[1], n_ant, P, int(CFG["n_beams"])).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=DEV)
        for i in range(0, len(perm) - 255, 256):
            b = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            out = model(xtr4[b]) if name == "cnn" else model(xtr[b])
            l = lf(out, ytr[b])
            l.backward()
            opt.step()
    return model


def run_variant(variant, data_dir, epochs):
    tr, va, te, Gte, blocked, moving = load_variant(variant, data_dir)
    P, M, B = int(CFG["n_probes"]), int(CFG["n_ant"]), int(CFG["n_beams"])
    out = {}
    maj = torch.bincount(torch.as_tensor(tr[1].cpu().numpy()), minlength=B).argmax().item()
    out["chance_top1"] = float((te[1] == maj).float().mean().item())
    for name in ["linear", "mlp", "cnn"]:
        m = train_one(name, tr, M, P, epochs)
        out[name] = evaluate(m, te[0], te[2], te[1], Gte, name, P, M)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="realistic")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results_realistic.json")
    args = ap.parse_args()

    variants = VARIANTS if args.ablation else [args.variant]
    all_res = {}

    for v in variants:
        print(f"\n{'=' * 58}\n=== {v} ===\n{'=' * 58}")
        r = run_variant(v, "data", args.epochs)
        all_res[v] = r
        print(f"chance {r['chance_top1']:.4f}")
        for name in ["linear", "mlp", "cnn"]:
            x = r[name]
            print(f"  {name:7s} top1 {x['top1']:.4f}  top3 {x['top3']:.4f}  "
                  f"rel_gain {x['rel_gain_dB']:.2f} dB  (oracle {x['oracle_dB']:.2f})")

    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\n[saved] {args.out}")

    if args.ablation:
        print("\n--- MLP top-1 across physical effects (8000 users) ---")
        for v in VARIANTS:
            if v in all_res:
                print(f"  {v:14s} {all_res[v]['mlp']['top1']:.4f}   "
                      f"gain {all_res[v]['mlp']['rel_gain_dB']:.2f} dB")


if __name__ == "__main__":
    main()
