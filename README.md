# Beam Prediction for 5G/6G — Preliminary Study

Learned beam prediction for 5G/6G beam management, evaluated on ray-traced
channel data at both Sub-6 and millimetre-wave frequencies.

This repository contains my preliminary work toward the graduation project
topic *"Deep Learning for Beam Management at 5G/6G Wireless Systems"*, carried
out while I am based in Hangzhou. I am still very much at the beginning of the
wireless-communication side of this, and the point of uploading it is to show
where I have got to and what I think the interesting question actually is.

---

## The task

In directional systems a beam must be pointed at the user before transmission.
The standard approach sweeps a codebook and measures which beam is strongest.
That sweep costs time and power, and gets worse as the codebook grows.

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
(DeepMIMO, Remcom InSite) and ran the identical experiment across three
scenarios — a campus and a city, at two different frequencies:

| Scenario | Frequency | Model | Top-1 | Relative gain | % of oracle |
|---|---|---|---|---|---|
| My own channel simulator | — | MLP | 0.543 | 11.18 dB | 82% |
| Real ray-traced (ASU campus) | 3.5 GHz | MLP | 0.866 | 13.52 dB | 99.4% |
| Real ray-traced (Miami city) | 3.5 GHz | MLP | 0.899 | 13.83 dB | 99.6% |
| Real ray-traced (Miami city) | **28 GHz** | MLP | **0.903** | **13.82 dB** | **99.6%** |

The real data **overturned** the conclusion. In an actual deployment the energy
concentrates into a few dominant directions far more than my model assumed, so
the target is sharply defined and a small network recovers essentially all of
the achievable gain. A linear probe on the same data reaches only 0.19, which
confirms the task is genuinely being solved rather than being trivially easy.

**So the straightforward version of this project is already solved.** That is
the useful result, because it points at what is not.

## Going to millimetre-wave did not change this — and that was surprising

My first instinct was that the task was easy because 3.5 GHz is Sub-6, where
beam management matters less, and that millimetre-wave (where beams are narrow
and the sweep is genuinely expensive) would be much harder. So I downloaded the
28 GHz version of the same Miami scenario and reran everything.

The physics changed substantially — path loss went from −200…−81 dB to
−227…−100 dB. The task did not change at all:

| | 3.5 GHz | 28 GHz |
|---|---|---|
| Path loss range | −200 … −81 dB | −227 … −100 dB |
| MLP top-1 | 0.899 | 0.903 |
| Strongest beam covers | 30.1% of users | 30.7% of users |
| Median gap between best and 2nd-best beam | 1.38 dB | 1.38 dB |

That last row is the explanation. Beamwidth for a ULA is approximately `2/N`
radians — set by the **antenna count alone**, independent of frequency. Raising
the frequency shortens the wavelength and shrinks the physical aperture, but it
also makes the channel more directional, and the two effects cancel out.

So difficulty is not driven by frequency. It is driven by the ratio of codebook
size to array size — which is what actually sets how many beams are close
enough together to be confused. Under this framing the saturated result is
expected rather than surprising, and it tells me what would actually make the
problem harder: **more beams, or more antennas**, not a different band.

## The question I want to work on

Two things my experiments have not touched, both of which the static
single-snapshot setup hides:

**1. The user never moves.** All six probes observe an *identical* channel,
differing only in noise (the difference is exactly 0.0 for every user). That is
physically wrong. In a real deployment successive probes are separated in time
and the user has moved — which is exactly why 3GPP uses multiple probes.

**2. The array is unrealistically small.** 32 antennas with a 64-beam codebook
is a benign ratio. Real mmWave deployments use 256-element arrays, where the
sweep is far more expensive and the neighbouring-beam confusion is much worse.

So the question I would like to pursue is:

> **Under a realistic array size and a moving user, how many channel
> measurements are actually needed to select the right beam — and what is the
> boundary at which the prediction breaks down?**

Both parts are needed: without them the answer is a trivial "one probe, static,
done", which is what I have now. This also connects to beam failure detection
and recovery in the 3GPP procedure, which my current work does not touch at all.

## What I have and have not done

**Done** — full pipeline from raw ray-traced data to trained baselines; three
scenarios spanning two frequencies; an ablation; SNR and probe-count sweeps; a
controlled frequency comparison; verification that the train/test split has no
spatial leakage; and a written record of the bugs I hit and fixed.

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
.venv-dm/bin/python download_deepmimo.py --only city_6_miami_28
.venv-dm/bin/python deepmimo_adapter.py --scenario asu_campus_3p5 --n-ue 20000
.venv-dm/bin/python deepmimo_adapter.py --scenario city_6_miami_3p5 --n-ue 20000 --tx 0
.venv-dm/bin/python deepmimo_adapter.py --scenario city_6_miami_28 --n-ue 20000 --tx 0
python train_deepmimo.py --data data/deepmimo_asu_campus_3p5.npz --epochs 40
python train_deepmimo.py --data data/deepmimo_city_6_miami_28.npz --epochs 40
python sweep_snr.py --data data/deepmimo_asu_campus_3p5.npz
```

DeepMIMO downloads are one-shot tokens with a daily quota, so the three download
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
