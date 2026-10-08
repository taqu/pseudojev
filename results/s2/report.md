# S0 / S1 / S2 Score Formulation Comparison

Model: `bonsai.gguf`  
No calibration artifact. Temperature = 1.0.

---

## Dataset: score / original  (n = 12)

| Metric                    |    S0 (digits)  |  S1 (letters)   | S2 (ltr-rotate) |
|---------------------------|-----------------|-----------------|-----------------|
| Accuracy                  |    **0.833**    |      0.667      |      0.500      |
| MAE (argmax)              |    **0.333**    |      0.417      |      0.750      |
| Expected-score MAE        |      0.383      |    **0.352**    |      0.572      |
| QWK                       |      0.750      |    **0.817**    |      0.592      |
| NLL                       |    **0.431**    |      0.779      |      0.918      |
| Brier                     |    **0.237**    |      0.435      |      0.532      |
| ECE                       |    **0.109**    |      0.254      |    **0.207**    |
| Margin mean               |    **1.944**    |      1.086      |      0.902      |
| Margin p10                |     -0.146      |     -1.813      |    **-1.400**   |
| Large-error rate (≥2)     |      0.167      |    **0.083**    |      0.167      |
| Prompt tokens (total)     |      1 514      |      1 466      |      5 864      |
| Eval time (ms)            |      1 743      |      1 615      |      7 212      |
| Evaluations / item        |       1.0       |       1.0       |       4.0       |
| Latency vs S0             |       1.0×      |       0.93×     |       4.14×     |

### S2 rotation diagnostics (original)
| Metric                              | Value  |
|-------------------------------------|--------|
| Rotation disagreement rate          |  0.750 |
| Mean distinct winners               |  2.667 |
| Mean semantic logit variance        |  1.805 |
| Max semantic logit variance         |  6.083 |
| Mean rotation expected-score stddev |  0.515 |

**Summary**: On the "original" split S0 has the best accuracy and calibration (NLL, Brier). S1 achieves the best QWK (0.817 vs 0.750) and lowest large-error rate (0.083), as well as the best expected-score MAE. S2 underperforms both baselines on all primary metrics while costing 4× the wall-clock time. High rotation disagreement (75% of items have per-rotation winner disagreement) signals that the model's letter-token preferences are not stable across label permutations on this split.

---

## Dataset: score / hard  (n = 6)

| Metric                    |    S0 (digits)  |  S1 (letters)   | S2 (ltr-rotate) |
|---------------------------|-----------------|-----------------|-----------------|
| Accuracy                  |      0.000      |    **0.333**    |      0.000      |
| MAE (argmax)              |      1.833      |    **1.167**    |      1.833      |
| Expected-score MAE        |      1.059      |    **0.626**    |      1.115      |
| QWK                       |     -0.211      |    **0.093**    |     -0.211      |
| NLL                       |      2.225      |    **1.852**    |      2.352      |
| Brier                     |      1.176      |    **0.932**    |      1.260      |
| ECE                       |      0.604      |    **0.216**    |      0.622      |
| Margin mean               |     -1.705      |    **-1.016**   |     -1.841      |
| Margin p10                |     -2.483      |    **-2.172**   |     -2.761      |
| Large-error rate (≥2)     |      0.667      |    **0.500**    |      0.667      |
| Prompt tokens (total)     |     10 078      |     10 053      |     42 739      |
| Eval time (ms)            |      3 439      |      3 392      |     14 226      |
| Evaluations / item        |       1.0       |       1.0       |       4.167     |
| Latency vs S0             |       1.0×      |       0.99×     |       4.14×     |

### S2 rotation diagnostics (hard)
| Metric                              | Value  |
|-------------------------------------|--------|
| Rotation disagreement rate          |  0.667 |
| Mean distinct winners               |  2.500 |
| Mean semantic logit variance        |  1.000 |
| Max semantic logit variance         |  2.756 |
| Mean rotation expected-score stddev |  0.497 |

**Summary**: On the "hard" split S1 is the clear winner across every metric — accuracy, MAE, expected-score MAE, QWK, NLL, Brier, ECE, and margins all improve over S0. S2 is worse than S0 on all metrics here and costs 4× as much. The rotation disagreement rate (67%) and mean distinct winners (2.5 out of 4 levels) confirm high instability: the letter assigned to a given semantic level significantly changes which level the model selects.

---

## Cross-split ranking

| Split     | Best accuracy | Best QWK | Best NLL | Best Brier |
|-----------|---------------|----------|----------|------------|
| original  | S0            | S1       | S0       | S0         |
| hard      | S1            | S1       | S1       | S1         |

## Conclusions

1. **S1 ≥ S0 on hard items**: Letter labels improve every metric on the hard set, likely because the model uses ordinal letter cues (A < B < C < D) when numeric digit cues are ambiguous. On easier items S0 still leads on accuracy/NLL because digit labels are unambiguous and the model is well calibrated on them.

2. **S2 does not help**: Label rotation averaging hurts on both splits. The core assumption — that per-letter biases cancel when every letter appears at every level equally — fails here. High rotation disagreement (67–75%) and large semantic logit variance show the model's logits respond strongly to label identity, not just ordinal position, so averaging cross-rotation estimates introduces noise rather than removing bias.

3. **Cost**: S2 runs ~4.1× slower than S0/S1 with no accuracy benefit. For production use S1 is the recommended baseline.

4. **Next steps**: Try S2 with prior correction (`--prior-correction`) to suppress letter-frequency bias before averaging. Also test on a larger split (n > 12) to improve statistical reliability.
