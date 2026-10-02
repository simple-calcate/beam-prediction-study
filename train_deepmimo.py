"""
train_deepmimo.py
=================
Train the beam-prediction baselines on REAL ray-traced DeepMIMO data.

Why this run is different from train.py and train_realistic.py
-------------------------------------------------------------
The synthetic runs gave MLP ~0.81 top-1. The realistic simulator dropped it to
~0.54, and the ablation showed the cause was angular spread: when a user's
energy is smeared over many directions, the "best beam" label stops being a
well-defined function of the observation, so no model can recover it.

Real ray-tracing data behaves DIFFERENTLY, and this script measures that
honestly:

1. Label distribution is extremely concentrated. The strongest single beam
   covers 12.1% of users and the top 8 cover 43.3%, versus 1/64 = 1.6% under
   uniform. A model can therefore score well by predicting common beams, so a
   "majority class" baseline is reported alongside every model. Without it the
   numbers are meaningless.

2. Absolute gain is meaningless. Received power spans 112 dB across the
   campus (path loss -209..-97 dB), so all gains are reported RELATIVE to each
   user's own average-beam gain -- the standard convention in this literature.

3. Per-user normalisation is mandatory. The observation is divided by its own
   power, discarding the absolute level (which says nothing about which beam is
   best) and keeping the channel shape. A fixed epsilon is wrong here: the
   dataset sits around 1e-13 W, so epsilon=1e-12 would exceed the signal and
   collapse the divisor to a constant, blinding the model.

Evaluation protocols
--------------------
  full codebook : predict the best of all 64 beams (the standard task)
  challenging-8 : predict the best of 8 candidate beams, the protocol used by
                  most beam-prediction papers, where the task is easier and
                  the metric more comparable to published numbers

Run
---
    .venv-dm/bin/python deepmimo_adapter.py --scenario asu_campus_3p5
    python train_deepmimo.py --data data/deepmimo_asu_campus_3p5.npz
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from train import build

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_npz(path, seed=0, per_user_norm=True):
    """Load the DeepMIMO tensors and split into train/val/test.

    Returns flat inputs for the linear/MLP probes, the 4D tensor for the CNN,
    and the relative-gain matrix used for evaluation.
    """
    d = np.load(path)
    X, y, G = d["X"], d["y"], d["G"].astype(np.float64)
    n_ant, n_beams = int(d["n_ant"]), int(d["n_beams"])
    n_probe = int(d["n_probes"])

    # Relative gain: divide each user by their own mean beam gain. This removes
    # path loss entirely so 112 dB of received-power range does not swamp the
    # quantity we actually care about.
    Grel = G / G.mean(axis=1, keepdims=True)

    n = len(X)
    n_val = int(0.15 * n)
    n_test = int(0.15 * n)
    idx = np.random.default_rng(seed).permutation(n)
    va, te, tr = idx[:n_val], idx[n_val:n_val + n_test], idx[n_val + n_test:]

    def pack(ii):
        x = torch.from_numpy(X[ii]).float()
        if per_user_norm:
            # (B, P, M, 2) -> per-(user, probe) power -> normalise
            pw = (x ** 2).sum(dim=(2, 3), keepdim=True)          # (B, P, 1, 1)
            # Floor scaled to the batch: a fixed absolute epsilon is larger
            # than the entire signal in this dataset and would blind the model.
            eps = 1e-8 * pw.mean()
            x = x / torch.sqrt(pw + eps)
        else:
            x = x / (x.std() + 1e-12)
        x = x.to(DEV)
        return x.reshape(len(ii), -1), torch.from_numpy(y[ii]).to(DEV), x

    return {
        "tr": pack(tr), "va": pack(va), "te": pack(te),
        "Grel": Grel[te], "y_test": y[te],
        "n_ant": n_ant, "n_beams": n_beams, "n_probe": n_probe,
    }


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, flat, x4, y, Grel, name, cand=None):
    """Top-k accuracy and relative realised gain in dB.

    Rel gain = 10*log10( G[pred] / mean_b G[b] ), averaged over users. The
    absolute value is meaningless when path loss varies by >100 dB.
    """
    model.eval()
    logits = model(x4 if name == "cnn" else flat)
    top3 = torch.topk(logits, 3, dim=1).indices
    top1 = top3[:, 0]

    t1 = (top1 == y).float().mean().item()
    t3 = (top3 == y[:, None]).any(dim=1).float().mean().item()

    G = torch.from_numpy(Grel).to(DEV).float()
    gmean = G.mean(dim=1, keepdim=True)
    g_pred = (G.gather(1, top1[:, None]) / gmean).mean().item()
    g_best = (G.gather(1, y[:, None]) / gmean).mean().item()

    out = {
        "top1": round(t1, 4),
        "top3": round(t3, 4),
        "rel_gain_dB": round(float(10 * np.log10(g_pred)), 2),
        "oracle_dB": round(float(10 * np.log10(g_best)), 2),
    }

    if cand is not None:
        # 8-challenging-beam protocol: mask out everything outside the
        # candidate set, then re-rank among the 8 survivors.
        mask = torch.full_like(logits, float("-inf"), device=DEV)
        mask[:, torch.as_tensor(np.asarray(cand), device=DEV)] = 0.0
        cl = (logits + mask).topk(8, dim=1).indices
        out["challenging8_top1"] = round((cl[:, 0] == y).float().mean().item(), 4)
        out["challenging8_top3"] = round(
            (cl == y[:, None]).any(dim=1).float().mean().item(), 4)
    return out


def trivial_baselines(y_test, Grel, cand=None):
    """Reference points. Without these, a model number means nothing.

    'majority' always predicts the single most common beam. On real data that
    is NOT a strawman: it alone reaches 12.1% top-1.
    """
    n_beams = Grel.shape[1]
    counts = np.bincount(y_test, minlength=n_beams)
    maj = int(counts.argmax())
    G = torch.from_numpy(Grel).to(DEV).float()
    gmean = G.mean(dim=1, keepdim=True)
    g_maj = (G[:, maj] / gmean).mean().item()

    out = {
        "random": {"top1": round(1 / n_beams, 4), "rel_gain_dB": 0.0},
        "majority": {"top1": round(float((y_test == maj).mean()), 4),
                     "rel_gain_dB": round(float(10 * np.log10(g_maj)), 2),
                     "beam": maj},
    }
    if cand is not None:
        c = np.asarray(cand)
        rel = np.searchsorted(c, y_test)
        rel = np.clip(rel, 0, len(c) - 1)
        out["majority"]["challenging8_top1"] = round(
            float((y_test == c[int(np.bincount(rel, minlength=len(c)).argmax())]).mean()), 4)
    return out


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_one(name, tr, n_ant, n_probe, n_beams, epochs=30, lr=3e-3, wd=1e-4,
              label_smoothing=0.0):
    flat, y, x4 = tr
    model = build(name, flat.shape[1], n_ant, n_probe, n_beams).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lf = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    nparam = sum(p.numel() for p in model.parameters())
    loss_sum, correct, total = 0.0, 0, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(flat), device=DEV)
        for i in range(0, len(perm) - 255, 256):
            b = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            out = model(x4[b]) if name == "cnn" else model(flat[b])
            loss = lf(out, y[b])
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(b)
            correct += (out.argmax(1) == y[b]).sum().item()
            total += len(b)
    return model, {"params": nparam,
                   "final_train_loss": round(loss_sum / total, 4),
                   "final_train_acc": round(correct / total, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/deepmimo_asu_campus_3p5.npz")
    ap.add_argument("--models", default="linear,mlp,cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--out", default="results_deepmimo.json")
    args = ap.parse_args()

    D = load_npz(args.data)
    n_ant, n_beams, n_probe = D["n_ant"], D["n_beams"], D["n_probe"]
    te_flat, te_y, te_x4 = D["te"]
    cand = np.load(args.data)["cand"]

    print(f"[data] {args.data}")
    print(f"[data] test users: {len(te_y)}, {n_ant} antennas, "
          f"{n_beams} beams, {n_probe} probes")
    orc = 10 * np.log10(D["Grel"].max(1).mean() / D["Grel"].mean())
    print(f"[data] oracle relative gain: {orc:.2f} dB")

    results = {"data": args.data, "n_ant": n_ant, "n_beams": n_beams,
               "n_probe": n_probe, "oracle_dB": round(float(orc), 2), "models": {}}

    print("\n=== 参考基线 ===")
    tb = trivial_baselines(D["y_test"], D["Grel"], cand)
    for k, v in tb.items():
        print(f"  {k:9s} top1={v['top1']:.4f}  gain={v['rel_gain_dB']:6.2f} dB")
    results["baselines"] = tb

    for name in args.models.split(","):
        name = name.strip()
        print(f"\n=== {name} (epochs={args.epochs}) ===")
        model, info = train_one(name, D["tr"], n_ant, n_probe, n_beams,
                                epochs=args.epochs, lr=args.lr,
                                label_smoothing=args.label_smoothing)
        ev = evaluate(model, te_flat, te_x4, te_y, D["Grel"], name, cand)
        ev.update(info)
        print(f"  params={info['params']:,}  train_acc={info['final_train_acc']:.4f}")
        print(f"  top1={ev['top1']:.4f}  top3={ev['top3']:.4f}  "
              f"gain={ev['rel_gain_dB']:.2f} dB (oracle {ev['oracle_dB']:.2f})")
        print(f"  challenging-8: top1={ev['challenging8_top1']:.4f}  "
              f"top3={ev['challenging8_top3']:.4f}")
        results["models"][name] = ev

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
