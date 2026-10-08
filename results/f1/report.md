# F0 vs F1 Score Formulation Comparison

**F0** = S1 baseline (letters scheme, single pass, no filler)  
**F1** = F0 + fixed ~16-token punctuation filler before `Answer:` anchor  
Model: `bonsai.gguf` · No calibration artifact · Temperature = 1.0

---

## F1 Filler Diagnostics

```
filler_text:         ". . . . . . . . . . . . . . . ."
filler_char_count:   31  (16 × ". " pattern)
filler_token_count:  16  (each ". " = 1 token in Bonsai tokenizer)
insertion_position:  end of user content, immediately before ANSWER_ANCHOR
evaluations_per_item: 1  (no additional inference passes)
```

### Tokenization candidates tested

| Filler string                              | tokens |
|--------------------------------------------|--------|
| `................` (16 periods, merged BPE)  |    1   |
| `. . . . . . . .` (8 period-space pairs)   |    8   |
| `... ... ... ...` (4 ellipsis tokens)       |    4   |
| `. . . . . . . . . . . . . . . .` ← frozen |  **16** |

---

## Candidate Prior Shifts (4-level items)

The filler changes the hidden state before the answer anchor, altering content-free
candidate priors. All letter prior logits shift upward, but D shifts most.

| Label | F0 prior logit | F1 prior logit | shift  |
|-------|---------------|----------------|--------|
| A     |  2.4001       |  3.6234        | +1.223 |
| B     | -0.2815       |  1.3528        | +1.634 |
| C     |  0.6816       |  2.3808        | +1.699 |
| D     | -0.8661       |  1.3294        | +2.196 |

Max absolute shift: **+2.196 logit units** (D)  
Prior logit variance: F0 = 1.530, F1 = 0.883 (filler slightly flattens raw distribution, but D is still disproportionately boosted relative to F0)

**Interpretation**: The punctuation filler significantly shifts the model's prior toward D (highest level), which creates a new label bias. This is the primary driver of F1's behavior on the hard set.

---

## F0 vs F1 / score / original  (n = 12)

| metric                    |   F0 (S1)   |     F1      |   delta    |
|---------------------------|-------------|-------------|------------|
| accuracy                  |   0.6667    | **0.7500**  | +0.0833    |
| MAE                       |   0.4167    | **0.3333**  | −0.0833    |
| expected-score MAE        | **0.3522**  |   0.4265    | +0.0743    |
| QWK                       |   0.8174    | **0.8333**  | +0.0159    |
| NLL                       | **0.7789**  |   0.8630    | +0.0841    |
| Brier                     | **0.4352**  |   0.4724    | +0.0372    |
| ECE                       | **0.2536**  |   0.3264    | +0.0728    |
| margin mean               | **1.0859**  |   0.7355    | −0.3504    |
| margin median             | **0.8014**  |   0.4594    | −0.3421    |
| margin p10                |  −1.8125    | **−1.4121** | +0.4005    |
| margin min                | **−2.4734** |  −2.8245    | −0.3510    |
| large-error rate          |   0.0833    |   0.0833    |  0.0000    |
| prompt tokens (total)     |   1 466     |   1 658     | +192       |
| eval_ms                   |   1 493     |   1 811     | +318       |
| evaluations / item        |   1.0       |   1.0       |  0.0       |

Error histogram (|pred−gt| = 0,1,2,...):  
F0: [8, 3, 1]  
F1: [9, 2, 1]

### Per-example (original)
| transition          | count |
|---------------------|-------|
| F0 correct → F1 wrong | 0 |
| F0 wrong → F1 correct | 1 |
| unchanged             | 11 |

Margin gain: mean = −0.350, median = −0.351; improved 2/12, degraded 10/12  
Expected-score error change: mean = +0.074 (worse), improved 2/12, degraded 10/12  
GT probability gain: mean = −0.046, improved 2/12, degraded 10/12

---

## F0 vs F1 / score / hard  (n = 6)

| metric                    |   F0 (S1)   |     F1      |   delta    |
|---------------------------|-------------|-------------|------------|
| accuracy                  | **0.3333**  |   0.0000    | −0.3333    |
| MAE                       | **1.1667**  |   1.8333    | +0.6667    |
| expected-score MAE        | **0.6256**  |   0.7163    | +0.0906    |
| QWK                       | **0.0930**  |  −0.2105    | −0.3035    |
| NLL                       |   1.8517    | **1.8143**  | −0.0373    |
| Brier                     | **0.9317**  |   0.9567    | +0.0250    |
| ECE                       | **0.2161**  |   0.4789    | +0.2629    |
| margin mean               | **−1.0163** |  −1.0666    | −0.0502    |
| margin median             |  −1.3195    | **−1.0013** | +0.3181    |
| margin p10                |  −2.1724    | **−1.9522** | +0.2202    |
| margin min                |  −2.1962    | **−2.1319** | +0.0642    |
| large-error rate          | **0.5000**  |   0.6667    | +0.1667    |
| prompt tokens (total)     |  10 053     |  10 149     | +96        |
| eval_ms                   |   3 306     |   3 493     | +187       |
| evaluations / item        |   1.0       |   1.0       |  0.0       |

Error histogram:  
F0: [2, 1, 3] (0 correct, 1 off-by-1, 3 off-by-2)  
F1: [0, 2, 3, 1] (0 correct, 2 off-by-1, 3 off-by-2, 1 off-by-3)

### Per-example (hard)
| transition          | count |
|---------------------|-------|
| F0 correct → F1 wrong | 2 |
| F0 wrong → F1 correct | 0 |
| unchanged             | 4 |

Margin gain: mean = −0.050, median = +0.017; improved 3/6, degraded 3/6  
Expected-score error change: mean = +0.091 (worse), improved 3/6, degraded 3/6  
GT probability gain: mean = −0.025, improved 3/6, degraded 3/6

---

## Performance overhead

Prompt token delta per item: +16 tokens (consistent with filler token count)  
Eval latency overhead: ~300 ms on original (~18%), ~190 ms on hard (~6%)  
Evaluations per item: **1** (no extra inference pass)

---

## Conclusion: **Reject F1**

F1 meets the rejection criteria from the spec:

| criterion              | original | hard    |
|------------------------|----------|---------|
| MAE worse              | better   | **YES** |
| QWK worse              | better   | **YES** |
| NLL/Brier worse        | **YES**  | Brier↑  |
| ECE worse              | **YES**  | **YES** |

The filler does not improve score quality across both splits. While it produces a small argmax-accuracy gain on the original split (8→9 correct), calibration degrades substantially there (NLL +8%, Brier +9%, ECE +29%, margins shrink on 10/12 items). On the hard split the filler causes two items to flip from correct to wrong, degrading MAE by +0.667, QWK by −0.304, and ECE by +0.263.

**Root cause**: The ~16-token punctuation filler shifts all candidate letter prior logits upward by 1.2–2.2 logit units, with D (highest level) shifting most (+2.20). This biases the model toward higher-level predictions rather than neutrally improving discriminability.

**Do not increase filler length**: The spec says not to immediately try 32/64/300 tokens when a 16-token probe is clearly negative.

**Recommended next step**: F0/S1 remains the best single-pass score formulation. If filler is revisited, consider applying prior correction (`--prior-correction`) to suppress the letter-token bias introduced by the filler.
