"""
train.py
========
Train and evaluate neural baselines for ML-based beam prediction.

Models
------
  linear : per-antenna linear probe. The "can a trivial model do this?" check.
  mlp    : 2-layer MLP. The workhorse baseline.
  cnn    : 1D-CNN over the antenna axis. The literature's usual choice.

Metrics
-------
  top1   : accuracy of predicting the single best beam.
  top3   : accuracy of the true beam being in the network's top 3.
  gain_dB: realised beamforming gain relative to the average codebook
           beam. This is the metric that actually matters in a paper --
           top-1 alone is a proxy, because a 1-beam error may still cost
           very little gain.

Baselines for reference
-----------------------
  chance  : always predict the most frequent beam in the training set.
  oracle  : always predict the true best beam (upper bound, gain = peak).

Run
---
    python train.py                    # trains all three
    python train.py --model mlp --epochs 40
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim_channel import CFG

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load(path="data/beam_pred.npz"):
    d = np.load(path)
    X, y, G = d["X"], d["y"], d["G"]
    n = len(X)
    # Deterministic split -- no leakage, and reproducible across models.
    n_val = int(CFG["frac_val"] * n)
    n_test = int(CFG["frac_test"] * n)
    idx = np.random.default_rng(0).permutation(n)
    va, te, tr = idx[:n_val], idx[n_val:n_val + n_test], idx[n_val + n_test:]

    def pack(ii):
        x = torch.from_numpy(X[ii]).float()            # (B, P, M, 2)
        x = x / (x.std() + 1e-8)
        flat = x.reshape(len(ii), -1)                  # (B, P*M*2) for linear/mlp
        return flat.to(DEV), torch.from_numpy(y[ii]).to(DEV), x.to(DEV)

    return pack(tr), pack(va), pack(te), G[te], G[va], y[te]


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Minimal sanity baseline: one linear map from observation to beams."""

    def __init__(self, d_in, n_beams):
        super().__init__()
        self.fc = nn.Linear(d_in, n_beams)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, d_in, n_beams, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_beams),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Treats the antenna axis as a 1-D signal -- the natural inductive bias
    for a ULA, since neighbouring elements are physically coupled.

    Input is (B, P, M, 2); we merge the P probes and the real/imag pair into
    the channel axis so Conv1d slides along the *antenna* dimension.
    """

    def __init__(self, n_ant, n_probe, n_beams, width=64):
        super().__init__()
        self.n_ant, self.n_probe = n_ant, n_probe
        in_ch = n_probe * 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.1), nn.Linear(width * n_ant, n_beams)
        )

    def forward(self, x):
        b, p, m, two = x.shape
        x = x.permute(0, 1, 3, 2).reshape(b, p * two, m)   # (B, P*2, M)
        h = self.conv(x)                                   # (B, width, M)
        return self.head(h)


def build(name, d_in, n_ant, n_probe, n_beams):
    if name == "linear":
        return LinearProbe(d_in, n_beams)
    if name == "mlp":
        return MLP(d_in, n_beams)
    if name == "cnn":
        return CNN1D(n_ant, n_probe, n_beams)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, xflat, y, G, name, n_probe, n_ant):
    """Top-1, top-3, and realised gain in dB.

    Realised gain is computed by looking up, for the *predicted* beam, the
    true gain column of G. A top-1 error that lands on an adjacent beam costs
    only ~0.1 dB, which is why top-1 and gain_dB can disagree sharply.
    """
    model.eval()
    logits = model(xflat.view(len(xflat), n_probe, n_ant, 2) if name == "cnn" else xflat)
    top3 = torch.topk(logits, 3, dim=1).indices
    top1 = top3[:, 0]

    t1 = (top1 == y).float().mean().item()
    t3 = (top3 == y[:, None]).any(dim=1).float().mean().item()

    Gt = torch.from_numpy(G).to(DEV)
    g_pred = Gt.gather(1, top1[:, None]).mean().item()
    g_best = Gt.gather(1, y[:, None]).mean().item()
    g_avg = Gt.mean().item()
    gain = 10 * np.log10(g_pred / g_avg)

    return {"top1": t1, "top3": t3, "gain_dB": float(gain),
            "oracle_dB": float(10 * np.log10(g_best / g_avg))}


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------

def run(name, tr, va, te, Gte, epochs, lr, wd):
    xtr, ytr, xtr4 = tr
    xva, yva, _ = va
    d_in = xtr.shape[1]
    n_ant, n_probe = int(CFG["n_ant"]), int(CFG["n_probes"])
    n_beams = int(CFG["n_beams"])

    model = build(name, d_in, n_ant, n_probe, n_beams).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=max(1, len(xtr) // 256)
    )
    lossf = nn.CrossEntropyLoss()

    def fwd(model_, xflat, x4):
        return model_(x4 if name == "cnn" else xflat)

    print(f"\n=== {name} ===  {n_par:,} params | device {DEV}")
    best_va, best_state, t0 = -1.0, None, time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=DEV)
        tot, nb = 0.0, 0
        for i in range(0, len(perm) - 255, 256):
            b = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            loss = lossf(fwd(model, xtr[b], xtr4[b]), ytr[b])
            loss.backward()
            opt.step()
            try:
                sched.step()
            except ValueError:
                pass
            tot += loss.item(); nb += 1

        model.eval()
        with torch.no_grad():
            vl = lossf(fwd(model, xva, xva.view(len(xva), n_probe, n_ant, 2)), yva).item()
            vacc = (fwd(model, xva, xva.view(len(xva), n_probe, n_ant, 2)).argmax(1) == yva).float().mean().item()
        if vacc > best_va:
            best_va = vacc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % max(1, epochs // 6) == 0 or ep == epochs:
            print(f"  ep {ep:3d}/{epochs}  train {tot / nb:.4f}  val {vl:.4f}  val_acc {vacc:.4f}")

    model.load_state_dict(best_state)
    res = evaluate(model, te[0], te[1], Gte, name, n_probe, n_ant)
    res["params"] = n_par
    res["train_s"] = round(time.time() - t0, 1)
    print(f"  TEST  top1 {res['top1']:.4f}  top3 {res['top3']:.4f}  "
          f"gain {res['gain_dB']:.2f} dB  (oracle {res['oracle_dB']:.2f} dB)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", "linear", "mlp", "cnn"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--data", default="data/beam_pred.npz")
    args = ap.parse_args()

    tr, va, te, Gte, Gva, yva = load(args.data)
    print(f"train {len(tr[0])}  val {len(va[0])}  test {len(te[0])}  "
          f"| {tr[0].shape[1]} features -> {CFG['n_beams']} beams")

    # Reference baselines
    maj = torch.bincount(tr[1], minlength=int(CFG["n_beams"])).argmax().item()
    chance = (tr[1] == maj).float().mean().item()
    print(f"\n[baseline] most-frequent beam = {maj}  ->  test top1 {chance:.4f}")
    print(f"[baseline] oracle gain = {10 * np.log10(Gte.max(1).mean() / Gte.mean()):.2f} dB")

    names = ["linear", "mlp", "cnn"] if args.model == "all" else [args.model]
    results = {n: run(n, tr, va, te, Gte, args.epochs, args.lr, args.wd) for n in names}
    results["chance_top1"] = chance

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results.json")


if __name__ == "__main__":
    main()
