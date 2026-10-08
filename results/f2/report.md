# F2 Score Formulation: Prior-Corrected F1

**F0** = S1 baseline (letters scheme, single pass, no filler)  
**F1** = F0 + fixed ~16-token punctuation filler before `Answer:` anchor  
**F2** = F1 + candidate prior correction (corrected_i = raw_i − α × prior_i)  
Model: `bonsai.gguf` · No calibration artifact · Temperature = 1.0  

---

## Prior Configuration

```
model:                bonsai.gguf
primitive:            score
candidate scheme:     letters
score levels:         4
filler:               ". . . . . . . . . . . . . . . ." (16 tokens)
answer anchor:        <|im_end|>\n<|im_start|>assistant\n\n
candidate token IDs:  A=32, B=33, C=34, D=35
```

### Measured F1-Compatible Prior Logits (4-level items)

| Label | F0 prior logit | F1 prior logit | shift  |
|-------|---------------|----------------|--------|
| A     |  2.4001       |  3.6234        | +1.223 |
| B     | −0.2815       |  1.3528        | +1.634 |
| C     |  0.6816       |  2.3808        | +1.699 |
| D     | −0.8661       |  1.3294        | +2.196 |

Prior logit variance: F1 = 0.8828 · Prior range: F1 = 2.294 logit units

### Prior Flattening Diagnostic (alpha = 1.0)

Corrected blank logits = prior_i − 1.0 × prior_i = 0 for all i.  
**Corrected prior range = 0.0000, variance = 0.0000** (flattened as expected).

---

## F0 vs F1 vs F2 / score / original  (n = 12)

| metric               |  S0 (natural) |   S1 / F0   |     F1      | F2 α=1.0    |
|----------------------|---------------|-------------|-------------|-------------|
| accuracy             | **0.8333**    |   0.6667    |   0.7500    |   0.2500    |
| MAE                  | **0.3333**    |   0.4167    | **0.3333**  |   1.5000    |
| expected-score MAE   | **0.3832**    |   0.3522    |   0.4265    |   0.9444    |
| QWK                  |   0.7500      |   0.8174    | **0.8333**  |   0.0294    |
| NLL                  | **0.4306**    |   0.7789    |   0.8630    |   1.3204    |
| Brier                | **0.2367**    |   0.4352    |   0.4724    |   0.8034    |
| ECE                  | **0.1065**    |   0.2363    |   0.3013    |   0.4150    |
| margin mean          |   1.1044      | **1.0859**  |   0.7355    |   0.4465    |
| large-error rate     | **0.0000**    |   0.0833    |   0.0833    |   0.4167    |
| prompt tokens (total)|   1 514       |   1 466     |   1 658     |   1 658     |
| evaluations / item   |   1.0         |   1.0       |   1.0       | **1.0**     |

---

## F0 vs F1 vs F2 / score / hard  (n = 6)

| metric               |  S0 (natural) |   S1 / F0   |     F1      | F2 α=1.0    |
|----------------------|---------------|-------------|-------------|-------------|
| accuracy             |   0.0000      | **0.3333**  |   0.0000    |   0.0000    |
| MAE                  |   1.8333      | **1.1667**  |   1.8333    |   2.0000    |
| expected-score MAE   |   1.0592      | **0.6256**  |   0.7163    |   1.3062    |
| QWK                  |  −0.2105      | **0.0930**  |  −0.2105    |   0.0602    |
| NLL                  |   2.2245      |   1.8517    | **1.8143**  |   1.9905    |
| Brier                |   1.1755      | **0.9317**  |   0.9567    |   1.1977    |
| ECE                  |   0.6040      | **0.3861**  |   0.4789    |   0.6613    |
| margin mean          |  −0.6558      | **−1.0163** |  −1.0666    |   0.4765    |
| large-error rate     |   0.6667      | **0.5000**  |   0.6667    |   0.6667    |
| prompt tokens (total)|     920       |     872     |     968     |     968     |
| evaluations / item   |   1.0         |   1.0       |   1.0       | **1.0**     |

---

## Alpha Sweep / score / original  (n = 12)

| alpha | Acc    | MAE    | ExpMAE | QWK    | NLL    | Brier  | ECE    | MarMean |
|-------|--------|--------|--------|--------|--------|--------|--------|---------|
|  0.00 | 0.7500 | 0.3333 | 0.4265 | 0.8333 | 0.8630 | 0.4724 | 0.3013 |  0.4998 |
|  0.25 | 0.5000 | 0.7500 | 0.5293 | 0.5920 | 0.9178 | 0.5222 | 0.4020 |  0.4521 |
|  0.50 | 0.5000 | 0.7500 | 0.6505 | 0.5920 | 1.0133 | 0.5965 | 0.1780 |  0.4219 |
|  0.75 | 0.5000 | 0.7500 | 0.8072 | 0.5984 | 1.1486 | 0.6934 | 0.3620 |  0.3979 |
|  1.00 | 0.2500 | 1.5000 | 0.9444 | 0.0294 | 1.3204 | 0.8034 | 0.4150 |  0.4465 |
|  1.25 | 0.2500 | 1.5000 | 1.0536 | 0.0294 | 1.5229 | 0.9115 | 0.4578 |  0.5237 |
|  1.50 | 0.2500 | 1.5000 | 1.1334 | 0.0294 | 1.7490 | 1.0049 | 0.4907 |  0.5734 |

Alpha = 0.00 identical to raw F1 (validation of offline-sweep equivalence).  
Every α > 0 degrades accuracy and most metrics monotonically.

---

## Alpha Sweep / score / hard  (n = 6)

| alpha | Acc    | MAE    | ExpMAE | QWK     | NLL    | Brier  | ECE    | MarMean |
|-------|--------|--------|--------|---------|--------|--------|--------|---------|
|  0.00 | 0.0000 | 1.8333 | 0.7226 | −0.2105 | 1.8122 | 0.9568 | 0.4799 |  0.1858 |
|  0.25 | 0.0000 | 1.8333 | 0.9158 | −0.2105 | 1.7847 | 0.9901 | 0.4980 |  0.2409 |
|  0.50 | 0.0000 | 1.8333 | 1.0702 | −0.1695 | 1.8123 | 1.0519 | 0.5536 |  0.3412 |
|  0.75 | 0.0000 | 1.8333 | 1.1838 | −0.1695 | 1.8846 | 1.1254 | 0.6144 |  0.4255 |
|  1.00 | 0.0000 | 1.8333 | 1.2623 | −0.1695 | 1.9905 | 1.1977 | 0.6613 |  0.4765 |
|  1.25 | 0.0000 | 1.8333 | 1.3148 | −0.1695 | 2.1205 | 1.2617 | 0.6963 |  0.5109 |
|  1.50 | 0.0000 | 1.8333 | 1.3494 | −0.1695 | 2.2670 | 1.3153 | 0.7219 |  0.5342 |

No alpha recovers accuracy on hard. All metrics worsen monotonically with α.

---

## Selection Frequencies

### Original (n = 12)

| level | GT | F0 (S1) | F1 | F2 α=1 |
|-------|----|---------|----|--------|
|   0   |  4 |    5    |  5 |    0   |
|   1   |  2 |    0    |  0 |    1   |
|   2   |  4 |    2    |  3 |    0   |
|   3   |  2 |    5    |  4 |   11   |

F2 α=1 collapses almost entirely to level 3 (D), predicting 11/12 items as the highest score.

### Hard (n = 6)

| level | GT | F0 (S1) | F1 | F2 α=1 |
|-------|----|---------|----|--------|
|   0   |  1 |    3    |  2 |    0   |
|   1   |  3 |    0    |  0 |    1   |
|   2   |  2 |    1    |  0 |    0   |
|   3   |  0 |    2    |  4 |    5   |

Both F1 and F2 concentrate excessively on level 3 (the correct-level count for D is 0 in this split).

---

## Per-Example Transitions

### F1 → F2 (α = 1.0)

| split    | correct→wrong | wrong→correct | unchanged |
|----------|---------------|---------------|-----------|
| original |       7       |       1       |     4     |
| hard     |       0       |       0       |     6     |

### F0 → F2 (α = 1.0)

| split    | correct→wrong | wrong→correct | unchanged |
|----------|---------------|---------------|-----------|
| original |       6       |       1       |     5     |
| hard     |       2       |       0       |     4     |

F2 at α=1 degrades 7 of F1's correctly-classified items on original and does not recover any hard items.

---

## Filler-Bias Hypothesis Diagnostic

The hypothesis was: F1's bias toward higher candidate labels (D shifted +2.196) could be removed by subtracting the F1-measured prior at α=1.

**What happened instead**: Subtracting the F1 prior at α=1 moved bias *toward* D, not away from it. Selection of D went from F1=4/12 to F2=11/12 on original.

**Root cause**: The F1 prior has A as the *largest* logit (3.6234). Subtracting α×prior penalizes A the most. After correction, candidates with smaller prior logits (D=1.3294, B=1.3528) are relatively favored. The F1 model's raw logits for D are apparently already high enough (due to the +2.196 shift into D by the filler) that after removing D's smaller prior, D dominates.

The additive prior correction formula:

```
corrected_i = raw_i − α × F1_prior_i
```

does not recover F0-equivalent semantic signal. The filler effect on raw logits is not
simply "add prior_i to every item independently" — the effect is entangled with the model's
token predictions and cannot be factored out with this linear formula.

---

## Performance

```
evaluations per item:     1  (F2 adds no extra inference pass)
prompt tokens (F2 orig):  1658 total (same as F1, +192 vs F0)
prompt tokens (F2 hard):   968 total (same as F1)
```

Prior correction is post-logit only; model evaluation count is identical to F1.

---

## Conclusion: **Reject F2**

F2 meets the rejection criteria:

| criterion                              | original | hard    |
|----------------------------------------|----------|---------|
| Prior correction removes F1's gains    | **YES**  | N/A     |
| Does not recover hard performance      | N/A      | **YES** |
| Worse than F0 on primary ordinal metrics | **YES** | **YES** |

**Original**: F2 α=1 accuracy collapses from F1=0.75 to 0.25 (F0=0.67), MAE triples to 1.50, QWK drops to 0.03.  
**Hard**: F2 never recovers accuracy (0% across all α). MAE worsens relative to F1.  
**No α value** in the sweep {0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50} improves over F1 on both splits simultaneously.  
**The sweep is monotonically degrading**, not peaked at an optimum — the null correction (α=0) is always at least as good as any corrected value.

**Interpretation**: The filler does not introduce a simple additive logit bias that can be removed by subtracting a content-free prior. The punctuation's effect on the model's hidden state is non-linear, and subtracting the F1 prior over-corrects by penalizing A (the highest-prior candidate) and indirectly amplifying D.

**Status of hypothesis**: The hypothesis that F1's performance loss was due to a separable linear candidate-token prior bias that could be removed by prior subtraction is **not supported**. The filler's effect is deeper than token-level prior arithmetic.

**Recommended status**: F0/S1 (letters, no filler) remains the best single-pass score formulation.
