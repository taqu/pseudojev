# noul evaluation metrics

`pjev run` reports, for the `noul` primitive, confidence / calibration / decision-margin
metrics in addition to `accuracy`, `labeled`, `n` and `eval_ms` (unchanged). They separate
configurations whose accuracy is identical, for example two configurations that both reach
100% on the `easy` split.

Implementation: `src/experiment/noul_metrics.{h,cpp}`. The flow is:

```text
DecisionOutput -> semantic normalization (NoulSample, runner.cpp) -> compute_noul_metrics
```

The metric code does not depend on the inference backend. Per-example records are stored in
the result JSON as `noul_items`, so `pjev compare` can recompute every metric without running
the model.

## Conventions

| item | definition |
|---|---|
| labeled sample | `expected` is `"yes"`/`"no"` (mapped to `true`/`false`). All ground-truth metrics use labeled samples only. |
| `P(True)` | semantic probability of `true`, located by candidate **key**, never by position |
| prediction | the semantic class the engine selected |
| confidence | probability of the predicted class: `max(P(True), P(False))` |
| epsilon | probabilities are clamped to `[1e-12, 1 - 1e-12]` before `log` |
| undefined | serialized as JSON `null`. `-1` is not used because margins can be negative. |

## Probability metrics

Top-level `nll`, `brier`, `ece` and `mean_confidence_correct` use the probabilities the run
reported. When a calibration artifact is loaded these are calibrated (temperature-scaled)
probabilities. The same metrics are also reported for both probability sets:

- `calibrated`: from the reported `P(True)`
- `raw`: the same decision at `T = 1` (after prior correction, before temperature)

| metric | definition | better |
|---|---|---|
| NLL | `mean(-log P(ground truth))` | lower |
| Brier | `mean((P(True) - y)^2)`, `y = 1` for True, `0` for False (binary form) | lower |
| ECE | `sum_b (n_b / N) * abs(mean_conf_b - acc_b)` over `ece_bins` (default 15) equal-width bins `[b/B, (b+1)/B)`. Confidence `1.0` goes in the last bin. Empty bins are skipped. | lower |
| mean_confidence_correct | mean confidence over correctly classified samples; `null` if none | higher |

The binary Brier above is half of the multi-class Brier (`sum over candidates`) used by
`pjev calibration evaluate`. The two values are not directly comparable.

## Margins

Margins come from **pre-temperature** logits. Temperature only rescales them, and it does not
change their sign or ranking.

- `semantic_margin = logit(true) - logit(false)`. Candidates are mapped back to semantic
  True/False first. With `A=False, B=True` it is `logit_B - logit_A`. With `A=True, B=False`
  it is `logit_A - logit_B`.
- `signed_margin = semantic_margin` if the ground truth is True, `-semantic_margin` if it is
  False. A positive value means the prediction is on the correct side of the decision
  boundary. Values near zero are fragile.
- Two sources are reported:
  - `corrected`: the margin the decision used, after prior correction. Top-level
    `*_signed_margin` fields use this source.
  - `raw`: the margin before prior correction. It is identical to `corrected` when prior
    correction is off.

For E1, both sources are the ensemble margin `(m1 + m2) / 2`.

Summary statistics (`mean`, `median`, `p10`, `min`) are taken over labeled samples.
Percentiles use **linear interpolation between closest ranks** (Hyndman & Fan type 7, the
numpy default): `h = (n-1)q`, `x[floor h] + (h - floor h)(x[floor h + 1] - x[floor h])`.
`p10` and `min` show lower-tail robustness. A configuration with a slightly lower mean but a
much higher minimum can be the more robust one.

## E1 (binary option-order ensemble)

| field | definition |
|---|---|
| `order_disagreement_count` | samples where ordering 1 and ordering 2 predict different semantic classes (`m1 > 0` vs `m2 > 0`) |
| `order_n` | samples with both order members. Ground truth is not required, so the denominator is all evaluated noul samples. |
| `order_disagreement_rate` | `count / order_n` |

Per item: `order1_semantic_margin`, `order2_semantic_margin`, `ensemble_semantic_margin`,
`order1_prediction`, `order2_prediction`, `order_disagreement`.

## Baseline vs target (margin gain)

`pjev compare --baseline A.json --target B.json` matches labeled samples by `source_id` and
computes `margin_gain = signed_margin_target - signed_margin_baseline`.

| field | definition |
|---|---|
| `mean_margin_gain`, `median_margin_gain` | over matched samples |
| `improved_margin_count` / `degraded_margin_count` | `gain > 0` / `gain < 0`, with `abs(gain) >= 1e-9` |
| `unchanged_margin_count` | `abs(gain) < 1e-9` |

## Per-example fields (`noul_items`)

`source_id, ground_truth, prediction, correct, p_true, p_true_raw, confidence,
raw_semantic_margin, semantic_margin, signed_margin` and, for E1, the order fields listed
above. `ground_truth`, `correct` and `signed_margin` are `null` for unlabeled samples.
