"""
cnn_ablation.py
===============
Diagnostic experiment: WHY does a 1-D CNN underperform the MLP here?

Hypothesis
----------
For a ULA the angle-of-arrival information is encoded as a *global* linear
phase ramp across all M elements. Estimating it requires integrating over the
whole array. A CNN with two k=5 layers has a receptive field of only
1 + 4 + 4 = 9 < 32, so it physically cannot see the full ramp.

Prediction
----------
Widening the first kernel to 32 (= full array) should close most of the gap.

This file is a diagnostic, not part of the main deliverable. It exists because
"the CNN lost to the MLP" is only interesting if you can explain WHY.
"""

import torch
import torch.nn as nn

import train as T

tr, va, te, Gte, _, _ = T.load()
(xtr, ytr, _), (xva, yva, _), (xte, yte, _) = tr, va, te
P, M, B = T.CFG["n_probes"], T.CFG["n_ant"], T.CFG["n_beams"]


class CNNWide(nn.Module):
    """First kernel spans the full array -> global receptive field in one layer."""

    def __init__(self, w=128, k=M):
        super().__init__()
        # Conv1d with kernel M and no padding collapses the antenna axis to
        # length 1, i.e. a genuinely global receptive field. The first layer
        # alone sees the whole array, which is exactly the hypothesis under
        # test. The second conv operates on that single position.
        self.net = nn.Sequential(
            nn.Conv1d(P * 2, w, k), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w, w, 1), nn.BatchNorm1d(w), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(w, B))

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, P, M, 2).permute(0, 1, 3, 2).reshape(b, P * 2, M)
        return self.head(self.net(x))


def main():
    m = CNNWide().to(T.DEV)
    print(f"CNNWide (kernel={M}, full array)  params: {sum(p.numel() for p in m.parameters()):,}")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss()
    best, best_state = -1.0, None

    for ep in range(1, 41):
        m.train()
        perm = torch.randperm(len(xtr), device=T.DEV)
        for i in range(0, len(perm) - 255, 256):
            b = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            lf(m(xtr[b].view(-1, P * M * 2)), ytr[b]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            a = (m(xva.view(-1, P * M * 2)).argmax(1) == yva).float().mean().item()
        if a > best:
            best = a
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if ep % 10 == 0:
            print(f"  ep {ep:3d}  val_acc {a:.4f}")

    m.load_state_dict(best_state)
    r = T.evaluate(m, xte, yte, Gte, "cnn", P, M)
    print(f"\nTEST  top1 {r['top1']:.4f}  top3 {r['top3']:.4f}  "
          f"gain {r['gain_dB']:.2f} dB  (oracle {r['oracle_dB']:.2f} dB)")
    print("\nCompare against train.py output:")
    print("  linear  top1 0.2517  gain 10.40 dB")
    print("  mlp     top1 0.7933  gain 13.30 dB")
    print("  cnn(k=5)     top1 0.5525  gain 12.75 dB   <- narrow receptive field")
    print(f"  cnn(k={M})      top1 {r['top1']:.4f}  gain {r['gain_dB']:.2f} dB   <- full array")


if __name__ == "__main__":
    main()
