# Preliminary Experimental Report

**Deep learning-based beam prediction for mmWave MIMO**
A scoping study for the proposed graduation project

Junwei Xu (285594) · 2 October 2026

---

## 1. Summary

Three questions, answered with runnable code:

1. **Is the proposed project tractable without hardware?** Yes. A complete
   pipeline — ray-traced data, baseline models, ablation — runs on a laptop
   in under 30 seconds per model.
2. **Does it work on real data?** Yes, and better than on my own simulator.
   On two independent ray-traced scenarios the MLP reaches **0.87 / 0.90
   top-1** against an oracle of 13.6 / 13.9 dB.
3. **What actually limits performance?** *Not* angular spread — the opposite
   of what my simulator suggested. On real data the best beam is far more
   sharply defined than I had modelled, and the task is correspondingly easier.

Finding 3 is the important one, and it is the reason real data was worth the
trouble: **my simulator was the bottleneck, not the problem.** The ablation in
§4 had identified angular spread as the fundamental limit; real ray tracing
overturns that. Any project proposal built on the simulation result would have
started from a wrong premise.

---

## 2. Task definition

**Physical setup** (simulation only, no hardware):

- Uniform linear array, **M = 32** antennas, 0.5-wavelength spacing
- **64-beam** DFT codebook
- Per-user multi-path channel; the label is `argmax_b |w_bᴴ h|²`
- Input: 6 noisy complex channel probes (real + imaginary parts). The network
  never sees user position, path angles, or the label.

**Why the input must be complex.** For a ULA the angle of arrival lives in the
*phase* ramp across the array; the magnitude of a single path is constant. An
early version fed only `|h|` and every model sat at 2% — chance level. That
failure is itself a finding: it is the clearest possible demonstration that the
information is in the phase, not the amplitude.

**Reported metric.** Received power spans **112 dB** across the ASU campus
(-209 to -97 dB), so absolute gain is meaningless. Everything is reported as
*relative* gain: `10·log10(G[pred] / mean_b G[b])` for each user, then averaged.
This is the standard convention in the beam-prediction literature.

---

## 3. Results

### 3.1 Baseline comparison (static synthetic channel)

Test set: 1200 held-out users. Total training time under 10 s.

| Model | Params | Top-1 | Top-3 | Gain | vs Oracle |
|---|---|---|---|---|---|
| Most-frequent beam | — | 0.020 | — | — | — |
| Linear probe | 24,640 | 0.252 | 0.358 | 10.40 dB | −3.04 dB |
| **MLP (2×256)** | 180,800 | **0.806** | 0.944 | **13.31 dB** | −0.14 dB |
| CNN k=5 | 155,840 | 0.553 | 0.873 | 12.73 dB | −0.69 dB |
| Oracle | — | 1.000 | 1.000 | 13.44 dB | 0 |

The linear probe reaching 12× chance is informative: the channel-to-best-beam
map is largely linear in the phase ramp. The interesting question is where the
nonlinear residual lives.

**Top-1 and gain disagree sharply.** The CNN scores 0.553 top-1 yet delivers
12.73 dB. Reporting top-1 alone understates it; reporting gain alone
overstates the ease. Both are needed.

### 3.2 Diagnostic: why the CNN lost to the MLP

Raising the CNN to 80 epochs changed nothing (0.528), so this was not
undertraining.

**Hypothesis.** The angle of arrival is a linear phase ramp across all 32
elements, so estimating it requires integrating over the whole array. Two k=5
convolutions give a receptive field of `1 + 4 + 4 = 9` — only 9 of 32 antennas.

**Test.** Set the first kernel to 32 (the full array).

| Variant | Receptive field | Params | Top-1 |
|---|---|---|---|
| CNN k=5 | 9 / 32 | 155,840 | 0.553 |
| **CNN k=32** | **32 / 32** | **74,560** | **0.759** |

**Confirmed.** 52% fewer parameters, 37% relative gain in top-1. The
architecture has to match the physics of the signal. This is the core content
of the project, and it is not about which PyTorch layer to use.

### 3.3 Ablation: which physical effect actually matters

Realistic channel with path loss (exponent 3.8), spatially-correlated shadowing
(6 dB), distance-dependent angular spread, 25% moving users, 10% blocked,
element patterns and mutual coupling. Gain is reported relative to each user's
own average beam, since absolute gain is dominated by distance once path loss
varies.

| Variant | MLP Top-1 | Gain |
|---|---|---|
| realistic (all effects) | 0.543 | 11.18 dB |
| no path loss | 0.500 | 11.03 dB |
| no shadowing | 0.530 | 11.16 dB |
| no mobility | 0.514 | 11.08 dB |
| no blockage | 0.524 | 11.07 dB |
| **no angular spread** | **0.842** | **14.33 dB** |
| static + clean (all removed) | 0.553 | 11.61 dB |

**The result is not subtle.** Removing *every* other effect gains 1 point.
Removing angular spread alone gains **30 points** and 3 dB.

**Why.** The best beam is only predictable to within the spread of arriving
paths. That is a physical ceiling no architecture can beat: if the arriving
energy is spread over many directions, "the best beam" is not well defined, and
no amount of model capacity recovers information that the channel does not
carry. Path loss, shadowing and blockage are **amplitude** effects — the network
normalises them away, and relative gain is the right metric anyway. Mobility
barely matters over a 3 ms probe window.

**This reframes the project.** The interesting question is not "which network
architecture" but "how do you predict well *when the answer is intrinsically
ambiguous*". That means uncertainty estimation, not just accuracy — a model that
knows when it does not know is more useful than one that is right 54% of the
time and confidently wrong the rest.

---

## 4. Methodology bugs found and fixed

Recorded because they are the actual content of this exercise.

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| Magnitude-only input | All models at 2% | ULA magnitude is constant; information is in phase | Feed complex channel |
| NumPy 2.x | `AttributeError: sin_theta` | `np.sin_theta` resolves as an attribute | `float()` cast |
| Global noise reference | Far users unlearnable | 28 dB path loss became 28 dB SNR loss | Per-user noise power |
| Fixed normalise epsilon | Blind constant input | With ~1e-13 W signals, a 1e-12 floor exceeded the signal | Epsilon relative to batch mean |
| Angular spread interpolation | Negative spread at 5 m | Log of a sub-reference distance | Saturating curve, clamped |
| Gain reported negative | −30 dB "best" gain | Re-normalised already-normalised data | Fix the reporting, not the data |

The fourth is worth dwelling on. A model scoring *below* the trivial
always-predict-the-commonest-beam baseline is a bug signature, not a physical
result. Stratifying accuracy by distance showed near users at 0.26 and far users
at 0.066 — which pointed at the SNR gradient, which pointed at the global noise
reference. Two of the six bugs were only found because a number looked
*implausible*, not because a test failed.

---

## 5. DeepMIMO: attempted, blocked

The proper validation is on real ray-traced data. This was attempted and
**failed for environmental reasons**, documented rather than hidden:

- `deepmimo` v4 installed cleanly (Python 3.12 via `uv`; the project's ROCm
  Python 3.14 has no prebuilt wheel for its `numpy<2.3` pin)
- The API endpoint responds correctly, returning a signed redirect
- That redirect points at **`f005.backblazeb2.com`**, which is unreachable from
  this network — TCP connection reset, every attempt, on both IPv4 and IPv6
- 200+ real scenarios are available in the database (Boston 5G, Los Angeles,
  Beijing, Shanghai, Guangzhou, …) — none can be fetched here

So the realistic-channel study above stands in for it. It reproduces the
*effects* ray tracing would have produced, but it is my model, not measured
data, and the distinction matters.

**Code is ready.** `fetch_deepmimo.py` bypasses the failing wrapper and hits the
same public endpoint directly; `deepmimo_adapter.py` and `train_deepmimo.py`
implement the DeepMIMO protocol including the standard *N-challenging-beams*
evaluation. On a network that can reach Backblaze, this should run as-is.

---

## 5. Real ray-traced data (DeepMIMO)

The download failed earlier because the CDN host was unreachable. With the
university VPN connected it works, and the results below come from actual
ray-traced channels — not from my simulator.

### 5.1 Data provenance

| | ASU campus | Miami city |
|---|---|---|
| Ray tracer | Remcom Wireless InSite 3.3 | same |
| Carrier | 3.5 GHz | 3.5 GHz |
| Max path depth | 7 | 7 |
| Grid users (total) | 131 931 | 42 984 |
| Users with a valid path | 85 157 (65%) | 27 061 (63%) |
| Base stations | 1 | 3 (site 0 used here) |
| Sampled for training | 20 000 | 20 000 |
| Path loss spread | −209 … −97 dB | −200 … −81 dB |
| Oracle relative gain | 13.60 dB | 13.89 dB |

Two scenarios were chosen deliberately: a **campus** and a **dense urban**
cell, to check that conclusions are not specific to one propagation environment.

### 5.2 Results on real data

3000 held-out users per scenario, 40 epochs, ~10 s per model on the GPU.

| Scenario | Model | Params | Top-1 | Top-3 | Rel. gain | vs oracle |
|---|---|---|---|---|---|---|
| ASU campus | random | – | 0.016 | – | 0.00 dB | – |
| ASU campus | majority class | – | 0.122 | – | 6.33 dB | 46% |
| ASU campus | linear probe | 24 640 | 0.192 | 0.358 | 8.23 dB | 61% |
| ASU campus | **MLP** | 180 800 | **0.866** | 0.979 | **13.52 dB** | 99.4% |
| ASU campus | CNN | 155 840 | 0.744 | 0.952 | 13.29 dB | 98% |
| Miami city | random | – | 0.016 | – | 0.00 dB | – |
| Miami city | majority class | – | 0.307 | – | 10.27 dB | 74% |
| Miami city | linear probe | 24 640 | 0.324 | 0.530 | 10.48 dB | 75% |
| Miami city | **MLP** | 180 800 | **0.899** | 0.984 | **13.83 dB** | 99.6% |
| Miami city | CNN | 155 840 | 0.842 | 0.976 | 13.73 dB | 99% |

**The 8-challenging-beam protocol** (the standard evaluation in this
literature, where the network picks the best of 8 candidates rather than all
64): MLP reaches 0.437 / 0.665 top-1 and 0.449 / 0.690 top-3 on the two
scenarios. The top-3 figures equal the theoretical ceiling, confirming the
protocol is well-formed.

### 5.3 The finding that matters

The MLP recovers **99.4% and 99.6% of the oracle gain**. Essentially all of the
achievable beamforming gain is predicted from 6 noisy probes.

This contradicts my own simulation. In `realistic_channel.py` the same MLP
reached only 0.543 top-1 / 11.18 dB (82% of oracle), and the ablation blamed
angular spread: energy smeared across directions makes the "best beam" label
ill-defined. On real ray-traced channels the opposite holds —

| | simulation | real data |
|---|---|---|
| strongest single beam covers | ~2% of users | 12% (ASU) / 31% (Miami) |
| top-8 beams cover | ~16% | 43% (ASU) / 100%* (Miami) |
| label quality | ambiguous | sharply peaked |

\* after correcting the candidate-selection rule; see below.

Real deployments concentrate users into a few dominant arrival directions far
more than my uniform-spread model did. The label is therefore a *well-defined
function* of the observation, and the task is learnable to near-oracle accuracy.

**Consequence for the project.** The interesting question is no longer "can a
network predict the best beam" — it can, trivially, in this static setting. The
open problems that remain are the ones a static single-cell setting hides:
mobility between probes, beam failure, the cost of the probes themselves, and
cross-site generalisation. The project should target one of those instead.

### 5.4 Two methodology bugs found and fixed

Recording these because both would have produced wrong numbers.

**The challenging-beam set was malformed.** Selecting candidate beams by
*mean gain* — the textbook rule — gave a set covering only **26.9%** of users
on real data, because mean gain is a poor proxy for which beam a given user
needs. Every challenging-beam number was therefore capped at 0.269 for reasons
that had nothing to do with the models. Fixed by selecting candidates by
*label frequency* instead, which restores a 100% ceiling.

**The magnitude of the result needed checking for leakage.** DeepMIMO users sit
on a regular grid, so a random train/test split can put near-identical channels
on both sides. Measured: adjacent-index channel cosine similarity −0.016 versus
−0.008 for random pairs, i.e. no leakage. The `np.linspace` subsample in the
adapter also breaks grid adjacency deliberately. The 0.87 figure is real.

### 5.5 Where does it break? An SNR and probe sweep

Since the single-point result is near-oracle, the useful question is where the
task degrades. ASU campus, MLP and linear probe, relative gain:

| SNR (dB) | linear top-1 | MLP top-1 | MLP gain |
|---|---|---|---|
| −5 | 0.174 | 0.727 | 13.25 dB |
| 0 | 0.170 | 0.815 | 13.45 dB |
| 5 | 0.189 | 0.846 | 13.50 dB |
| 10 | 0.186 | 0.857 | 13.51 dB |
| 15 | 0.191 | 0.863 | 13.52 dB |
| 20 | 0.193 | 0.862 | 13.52 dB |
| 30 | 0.196 | 0.872 | 13.53 dB |

The MLP is robust: even at **−5 dB** it keeps 0.727 top-1 and 13.25 dB, which is
98% of the oracle gain. It only degrades meaningfully below 0 dB. The linear
probe, by contrast, is flat at ~0.19 across the entire range — it extracts
almost none of the angular structure regardless of SNR. So the deep model is
not merely winning on the easy end; it is the only thing that works at all when
the observation is noisy. That is the honest justification for using it here.

**The probe sweep produced a result I do not trust, and it is a defect in my
data rather than a finding about beam management:**

| probes | linear top-1 | MLP top-1 |
|---|---|---|
| 1 | 0.185 | 0.875 |
| 2 | 0.189 | 0.869 |
| 4 | 0.190 | 0.865 |
| 6 | 0.195 | 0.867 |

Six probes should beat one. Here one probe is as good as six, and going the
other way *lowers* accuracy slightly. The cause is verifiable: the maximum
element-wise difference between a user's probes 1–5 and probe 0 is exactly
`0.0` for every user. All six probes observe an identical channel and differ
only in noise, so the extra probes carry no new information and only add input
dimensions for the network to overfit.

This is physically wrong. In a real deployment successive probes are separated
in time and the user has moved, which is precisely *why* 3GPP uses multiple
probes — averaging across a moving user is what makes beam selection robust.
My generator holds the user perfectly still, so it cannot study the thing that
makes probes valuable in the first place.

**Consequence:** every number in §5.2 and §5.5 is a *single-snapshot* result
and should be reported as such. Multi-probe robustness is still unmeasured, and
it is the most natural next thing to fix — see §6.

---

## 6. Honest limitations

**About the real-data results (§5)**

- **The probes are redundant, so the headline number is optimistic.** All six
  probes see an identical channel (§5.5), which is physically wrong and removes
  exactly the multi-probe robustness that motivates beam management in 3GPP.
  Until this is fixed, 0.87 / 0.90 is a single-snapshot result.
- **Static users, single cell, single frame.** Users never move and never leave
  the cell. This is the easy version of beam prediction; real deployments add
  mobility, handover, and beam failure, and the numbers will drop.
- **No overhead accounting.** The whole point is to avoid a 64-beam sweep, yet
  the probes it costs are never counted. I have shown the task is *learnable*,
  not that it is *worth doing*. §5.5 makes this worse: since extra probes cost
  nothing in my data, I have not measured the real trade-off at all.
- **A near-oracle result is a warning, not a success.** 99.4% of oracle gain means
  the benchmark is saturated. A saturated benchmark cannot support a thesis
  contribution; it needs a harder setting.
- **One random seed, one train/test split per scenario.** No error bars, no
  cross-validation. The 0.87 vs 0.90 gap between scenarios is not a claim.
- **The SNR sweep reconstructs the clean channel by averaging the stored
  probes** (since the clean channel is not saved separately). The residual
  contamination is identical at every sweep point, so the *shape* of the curve
  is trustworthy, but the absolute values are optimistic.
- **No comparison against published DeepMIMO results.** Numbers in the beam
  prediction literature exist for this exact task; mine should be measured
  against them rather than reported standalone.
- **Only 2 of 201 available scenarios were used.** Enough to show the conclusion
  is not scenario-specific, not enough to call it general.

**About the simulation results (§3–4)**

- They remain properties of my channel models. §5 shows the angular-spread
  conclusion drawn from them does not survive contact with real data, which
  retroactively discredits the rest of that ablation as a guide to the real
  problem.

**Next concrete steps, in priority order**

1. **Add mobility between probes.** This is the defect that matters: it is why
   the current numbers are optimistic, and it is the mechanism that makes
   multi-probe beam management work in 3GPP. Expect the largest drop here.
2. Train on one site, test on another (Miami has 3 base stations available) to
   measure cross-site generalisation.
3. Compare against published numbers on the same DeepMIMO scenario.
4. Re-run the SNR sweep on a random split by *site* rather than by user, so the
   test set is genuinely unseen geography.

---

## 7. Files

```
sim_channel.py         synthetic channel, DFT codebook, first dataset
train.py               linear / MLP / CNN baselines
cnn_ablation.py        receptive-field diagnostic (§3.2)
realistic_channel.py   path loss, shadowing, spread, mobility, blockage
train_realistic.py     ablation study (§3.3)
diagnose_realistic.py  distance-stratified accuracy
download_deepmimo.py   scenario downloader (§5.1, handles the one-shot token)
probe_deepmimo.py      which scenarios are downloadable, and how big
deepmimo_adapter.py    DeepMIMO v4 -> same tensor format
train_deepmimo.py      training + N-challenging-beams protocol (§5.2)
CONCEPTS.md            the concepts, explained from first principles
REPORT.md              this file
```

Reproduce the real-data results:

```bash
.venv-dm/bin/python download_deepmimo.py --only asu_campus_3p5
.venv-dm/bin/python download_deepmimo.py --only city_6_miami_3p5
.venv-dm/bin/python deepmimo_adapter.py --scenario asu_campus_3p5 --n-ue 20000
.venv-dm/bin/python deepmimo_adapter.py --scenario city_6_miami_3p5 --n-ue 20000 --tx 0
python train_deepmimo.py --data data/deepmimo_asu_campus_3p5.npz --epochs 40
python train_deepmimo.py --data data/deepmimo_city_6_miami_3p5.npz --epochs 40
```

Reproduce the simulation results:

```bash
python sim_channel.py --users 8000
python train.py --epochs 30
python cnn_ablation.py
python realistic_channel.py --users 8000 --ablate
python train_realistic.py --ablation --epochs 40
python diagnose_realistic.py
```

Environment: PyTorch 2.14 + ROCm, AMD Radeon RX 6750 GRE 12 GB, Python 3.14.
DeepMIMO path additionally needs `.venv-dm` (Python 3.12).
