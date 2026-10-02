# Beam Prediction for 5G/6G — Preliminary Study

Deep learning-based beam prediction for millimetre-wave MIMO, evaluated on
ray-traced channel data.

This repository contains my preliminary work toward the graduation project
topic *"Deep Learning for Beam Management at 5G/6G Wireless Systems"*, carried
out while I am based in Hangzhou. I am still very much at the beginning of the
wireless-communication side of this, and the point of uploading it is to show
where I have got to and what I think the interesting question actually is.

---

## The task

In millimetre-wave systems the channel is highly directional, so a beam must be
pointed at the user before transmission. The standard approach sweeps a codebook
and measures which beam is strongest. That sweep costs time and power, and gets
worse as the codebook grows.

The question here is whether a learned model can predict the right beam from a
small number of channel measurements, so most of the sweep can be skipped.

**Setup** — 32-element uniform linear array (ULA), 64-beam DFT codebook,
6 noisy channel probes as input, never seeing user position or path angles.
Metrics are top-1 / top-3 accuracy and *relative* gain, where each user's gain
is normalised by their own average-beam gain (absolute gain is meaningless when
path loss varies by over 100 dB across a cell).

## The finding that shaped my proposal

The first thing I built was a channel simulator. It suggested that **angular
spread** was the dominant performance limit — smearing a user's energy across
many directions makes the "best beam" label ambiguous, and an ablation gave
0.54 top-1.

I did not trust my own simulator, so I obtained real ray-traced data
(DeepMIMO, Remcom InSite, 3.5 GHz) and ran the identical experiment:

| Environment | Model | Top-1 | Relative gain | % of oracle |
|---|---|---|---|---|
| My own channel simulator | MLP | 0.543 | 11.18 dB | 82% |
| Real ray-traced (ASU campus) | MLP | **0.866** | **13.52 dB** | **99.4%** |
| Real ray-traced (Miami city) | MLP | **0.899** | **13.83 dB** | **99.6%** |

The real data **overturned** the conclusion. In an actual deployment the energy
concentrates into a few dominant directions far more than my model assumed, so
the target is sharply defined and a small network recovers essentially all of
the achievable gain. A linear probe on the same data reaches only 0.19, which
confirms the task is genuinely being solved rather than being trivially easy.

**So the straightforward version of this project is already solved.** That is
the useful result, because it points at what is not.

## The question I want to work on

I noticed a defect in my own data while checking the above: all six probes
observe an *identical* channel, differing only in noise. That is physically
wrong. In a real deployment successive probes are separated in time and the user
has moved — which is exactly why 3GPP uses multiple probes in the first place.

So the question I would like to pursue is:

> **When the user moves between probes, how many channel measurements are
> actually needed to select the right beam — and what is the boundary at which
> the prediction breaks down?**

The static case is saturated, so the research content lives in the moving case.
This also connects back to beam failure detection and recovery in the 3GPP
procedure, which my current work does not touch at all.

## What I have and have not done

**Done** — full pipeline from raw ray-traced data to trained baselines; two
independent scenarios; an ablation; an SNR sweep; verification that the train/test
split has no spatial leakage; and a written record of the bugs I hit and fixed.

**Not done** — mobility between probes (the defect above); beam failure
handling; overhead accounting; multiple seeds and error bars; comparison against
published numbers on the same scenarios. Section 6 of `REPORT.md` lists these
in full, including the ones that make my current results look better than they
should.

## Reproducing

Requires PyTorch. The DeepMIMO path additionally needs Python ≤ 3.12.

```bash
# Synthetic baseline and the simulator ablation
python train.py --epochs 30
python train_realistic.py --ablation --epochs 40

# Real ray-traced data
uv venv --python 3.12 .venv-dm
uv pip install --python .venv-dm/bin/python deepmimo numpy
.venv-dm/bin/python download_deepmimo.py --only asu_campus_3p5
.venv-dm/bin/python download_deepmimo.py --only city_6_miami_3p5
.venv-dm/bin/python deepmimo_adapter.py --scenario asu_campus_3p5 --n-ue 20000
.venv-dm/bin/python deepmimo_adapter.py --scenario city_6_miami_3p5 --n-ue 20000 --tx 0
python train_deepmimo.py --data data/deepmimo_asu_campus_3p5.npz --epochs 40
python sweep_snr.py --data data/deepmimo_asu_campus_3p5.npz
```

DeepMIMO downloads are one-shot tokens with a daily quota, so the two download
commands may need to be run on separate occasions.

## Files

| File | Purpose |
|---|---|
| `REPORT.md` | Full write-up: method, results, ablation, two methodology bugs, limitations |
| `CONCEPTS.md` | The beam-management concepts I had to learn first, from first principles |
| `sim_channel.py`, `train.py` | Synthetic channel and the first baselines |
| `cnn_ablation.py` | Receptive-field diagnostic |
| `realistic_channel.py`, `train_realistic.py` | Path loss, shadowing, spread, mobility, blockage + ablation |
| `diagnose_realistic.py` | Distance-stratified accuracy |
| `download_deepmimo.py`, `probe_deepmimo.py` | Scenario download and availability probing |
| `deepmimo_adapter.py` | Ray-traced data → tensors |
| `train_deepmimo.py` | Training and the N-challenging-beams protocol |
| `sweep_snr.py` | SNR and probe-count sweeps |
