# Lab log

One entry per session. What you tried, what happened, what you think it means. Paste the chart. Failures stay in.

## Template

**Date · notebook · runtime**
- Tried:
- Saw: (numbers, and a figure path under `results/`)
- Think:
- Next:

---

## Reference run · CPU only, deliberately undersized (scaffold check, not a result)

**Othello, 6 layers × 256, 150k games, 1 epoch, ~30 min on 2 CPU cores**
- Legal-move rate 0.866 (a well-trained model reaches ~0.99+). Undertrained.
- Probe accuracy rises with depth: 0.65 (embeddings) → 0.78 (layer 6). Untrained control: 0.66.
- Intervention, editing layers 2–6: errors against the *original* board rose 2.67 → 3.06; against the *edited* board fell 2.97 → 2.91. Right direction, small effect.
- Read: the pipeline works and the sign is right. Magnitude should grow sharply with the full Colab model. If it does not, that is the first thing to investigate.

**CarRacing, 60 random rollouts × 300 steps, VAE 6 epochs, M 30 epochs, CMA-ES 40 gens × 32, all CPU**
- Random policy: −10 reward over 300 steps.
- Dream-only controller on unseen real tracks, 300 steps: 26, 228, 27 (mean of CMA at gen 30: 276).
- Read: transfer is real even at toy scale, and noisy. Dream-predicted rewards (~2) run far below real ones (~200): M's reward head regresses sparse tile rewards toward the mean. It ranks controllers, it does not price them.
