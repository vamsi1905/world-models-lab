# world-models-lab

World models from first principles, on free compute. Weeks 0–2 of a six-week plan: first the evidence that sequence models build internal worlds, then the smallest complete world model, typed out component by component.

| Week | Notebook | What it shows | Colab time |
| --- | --- | --- | --- |
| 0 | `00_othello_probe` | A move-predicting transformer holds a readable, editable board | ~75 min (T4) |
| 1 | `01_vae` | V: 64×64 frames compressed to 32 numbers | ~45 min |
| 1 | `02_mdnrnn` | M: next-latent prediction, and where the dream blurs | ~25 min (T4) |
| 2 | `03_dream_controller` | C: an 867-parameter driver evolved inside M's imagination | ~45 min (T4) |

## Getting started

1. Create an empty **public** GitHub repo called `world-models-lab` and push this folder to it:
   ```bash
   cd world-models-lab
   git init && git add . && git commit -m "Week 0-2 scaffold"
   git branch -M main
   git remote add origin https://github.com/vamsi1905/world-models-lab.git
   git push -u origin main
   ```
2. In each notebook's first cell, set `REPO_URL` to your repo. (Search-and-replace `YOUR_GITHUB_USERNAME` across `notebooks/` once and commit.)
3. Open a notebook in Colab: `https://colab.research.google.com/github/vamsi1905/world-models-lab/blob/main/notebooks/00_othello_probe.ipynb`
4. Runtime → Change runtime type → **T4 GPU**. Run all.

Data and checkpoints go to `MyDrive/world-models-lab/`, so a disconnect costs minutes, not the evening. Every expensive cell checks for its output first and skips if it exists.

Locally: `pip install swig && pip install -e .`, then run the notebooks from `notebooks/`. `WM_SMOKE=1` shrinks every size so a notebook runs end to end on a laptop CPU in a few minutes; use it to check the plumbing, not to judge results.

## Layout

```
wmlab/othello/game.py        Othello engine + fast bitboard random-game generator
wmlab/othello/model.py       Minimal GPT with residual-stream read and edit hooks
wmlab/othello/probe.py       Linear probes, interventions, board plotting
wmlab/carracing/rollouts.py  Random-walk rollouts, 64x64 preprocessing
wmlab/carracing/vae.py       V: Ha & Schmidhuber's ConvVAE, free-bits KL
wmlab/carracing/mdnrnn.py    M: LSTM + per-dimension Gaussian mixture, reward and done heads
wmlab/carracing/controller.py C: linear policy, batched dream rollouts, CMA-ES, video
tests/                       Fast correctness checks (pytest)
LOG.md                       The lab notebook. Commit the failures.
```

## Two deliberate departures from the plan

**Week 0 trains its own Othello-GPT** instead of cloning `likenneth/othello_world`. That repo's checkpoints and data sit behind Drive links that come and go. A self-contained engine generates random games at ~1,500 a second, and an 8-layer model trains on a T4 in about an hour. The probe and intervention follow Nanda's linear, mine/theirs formulation.

**M predicts reward and episode end.** The 2018 paper trained the CarRacing controller in the real environment and kept dream-only training for VizDoom. Dream training on CarRacing needs the dream to hand out rewards, so M carries two small extra heads. This is the harder experiment and the gap between dream and real scores is the thing to measure, not an embarrassment to hide.

## Datasets for weeks 3–6 (checked 29 September 2026)

The plan's Hugging Face IDs have drifted. These are the current ones:

| Use | Where | Access | Licence |
| --- | --- | --- | --- |
| 1X World Model Challenge, tokenised data (weeks 3–4) | HF `1x-technologies/world_model_tokenized_data` | Public | Apache-2.0 |
| 1X challenge code and baselines | GitHub `1x-technologies/1xgpt` (a code repo, not a dataset) | Public | See repo |
| LoViF 2026 PhyScore (weeks 5–6) | [Codabench competition 13622](https://www.codabench.org/competitions/13622/) and the [workshop site](https://lovif-cvpr-2026-workshop.github.io/). **Not on the Hub.** | Codabench registration | See competition terms |
| GigaBrain CVPR 2026 World Model Track | HF `open-gigaai/CVPR-2026-WorldModel-Track-Dataset` | Gated: team registration and approval | Custom challenge licence; non-commercial, no redistribution |
| AgiBot World Challenge 2026 | HF `agibot-world/AgiBotWorldChallenge-2026` | Open, not gated | CC-BY-NC-SA-4.0 |

Two notes for later. PhyScore's 1,554 videos split into 863 train (with scores on four dimensions and anomaly timestamps), 298 validation and 393 test, which have inputs only; the week 5–6 "does it beat the baseline" test can only be run on the 863 train videos, or through Codabench submissions. And three of these carry non-commercial terms (GigaBrain, AgiBot, and 1X's raw video, which is CC-BY-NC-SA-4.0; the tokenised set is Apache-2.0). Anything built toward a commercial product should rest on the Apache data or get permission first.

## Checkpoints

- **After week 0:** explain what a probe is, why the untrained-model control matters, and why an intervention proves more than a probe.
- **After week 2:** the side-by-side video exists, and `dream_vs_real.png` shows whether dream progress transferred. This is the hard gate.

## References

- Li et al., *Emergent World Representations* (ICLR 2023)
- Nanda, *Actually, Othello-GPT Has A Linear Emergent World Representation* (2023)
- Ha & Schmidhuber, *World Models* (2018), worldmodels.github.io
- Hafner et al., *Mastering Atari with Discrete World Models* (DreamerV2, 2021)
