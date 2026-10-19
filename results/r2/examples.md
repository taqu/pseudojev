# R2 examples (held-out partition, frozen threshold 0.90398929312344334)

Representative cases, not cherry-picked successes: each category takes the first matches in file order, except the similarity extremes which take the highest / lowest similarity.

## Activated rescue (S1 wrong, R1 right, gate on)

#### community-diversity-v2/20260921/support/damaged_order/0/1/score

| Field | Value |
|---|---|
| Kind | score (5 levels) |
| Similarity (est. cosine) | 0.9482 (Hamming 79) |
| Gate | activated at 0.9040 |
| Ground truth | C |
| S1 / R1 / R2 | B / C / C |
| Demo answer label | C |

- Query: Candidate P27. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
- Exemplar 804d37c8ff218266: Candidate P30. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established. [family: policy]

#### community-diversity-v2/20260921/support/damaged_order/4/2/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.9402 (Hamming 85) |
| Gate | activated at 0.9040 |
| Ground truth | B |
| S1 / R1 / R2 | A / B / B |
| Demo answer label | D |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar 95ba7a22fb1c447d: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

#### community-diversity-v2/20260921/support/damaged_order/8/0/noul

| Field | Value |
|---|---|
| Kind | noul |
| Similarity (est. cosine) | 0.9270 (Hamming 94) |
| Gate | activated at 0.9040 |
| Ground truth | B |
| S1 / R1 / R2 | A / B / B |
| Demo answer label | B |

- Query: Candidate P13. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
- Exemplar ea073ff8743153f7: Candidate K27. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility. [family: policy]

## Activated harm (S1 right, R1 wrong, gate on)

#### community-diversity-v2/20260921/support/damaged_order/0/3/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.9415 (Hamming 84) |
| Gate | activated at 0.9040 |
| Ground truth | G |
| S1 / R1 / R2 | G / B / B |
| Demo answer label | D |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar 95ba7a22fb1c447d: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

#### community-diversity-v2/20260921/support/damaged_order/4/2/noul

| Field | Value |
|---|---|
| Kind | noul |
| Similarity (est. cosine) | 0.9330 (Hamming 90) |
| Gate | activated at 0.9040 |
| Ground truth | B |
| S1 / R1 / R2 | B / A / A |
| Demo answer label | B |

- Query: Candidate item_78. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
- Exemplar ea073ff8743153f7: Candidate K27. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility. [family: policy]

#### community-diversity-v2/20260921/support/damaged_order/4/3/score

| Field | Value |
|---|---|
| Kind | score (5 levels) |
| Similarity (est. cosine) | 0.9508 (Hamming 77) |
| Gate | activated at 0.9040 |
| Ground truth | B |
| S1 / R1 / R2 | B / D / D |
| Demo answer label | D |

- Query: Candidate K28. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
- Exemplar 64e7ac253a44c8f7: Candidate K45. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established. [family: policy]

## Rejected rescue opportunity (R1 would have rescued, gate off)

#### community-diversity-v2/20260921/contracts/governing_law_venue/1/1/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.8876 (Hamming 117) |
| Gate | rejected at 0.9040 |
| Ground truth | D |
| S1 / R1 / R2 | A / D / A |
| Demo answer label | D |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar d72d35c05d310473: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

#### community-diversity-v2/20260921/contracts/governing_law_venue/1/3/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.8876 (Hamming 117) |
| Gate | rejected at 0.9040 |
| Ground truth | D |
| S1 / R1 / R2 | A / D / A |
| Demo answer label | D |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar d72d35c05d310473: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

#### community-diversity-v2/20260921/contracts/governing_law_venue/3/1/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.8987 (Hamming 111) |
| Gate | rejected at 0.9040 |
| Ground truth | C |
| S1 / R1 / R2 | A / C / A |
| Demo answer label | C |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar 2742cd0cc9570a0e: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

## Prevented harm (R1 would have harmed, gate off)

#### community-diversity-v2/20260921/contracts/governing_law_venue/2/3/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.8987 (Hamming 111) |
| Gate | rejected at 0.9040 |
| Ground truth | B |
| S1 / R1 / R2 | B / A / B |
| Demo answer label | A |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar 6c4fd119242c236b: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

#### community-diversity-v2/20260921/contracts/governing_law_venue/3/3/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.8987 (Hamming 111) |
| Gate | rejected at 0.9040 |
| Ground truth | G |
| S1 / R1 / R2 | G / A / G |
| Demo answer label | C |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar 2742cd0cc9570a0e: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

#### community-diversity-v2/20260921/contracts/governing_law_venue/4/3/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.8895 (Hamming 116) |
| Gate | rejected at 0.9040 |
| Ground truth | B |
| S1 / R1 / R2 | B / A / B |
| Demo answer label | A |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar 6c4fd119242c236b: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

## High-similarity harm (R1 harms with the highest similarity)

#### wanli-decisions-v1:train:298043

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.9808 (Hamming 48) |
| Gate | activated at 0.9040 |
| Ground truth | C |
| S1 / R1 / R2 | C / B / B |
| Demo answer label | B |

- Query: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
- Exemplar cc10aeac04722797: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction. [family: natural_language_inference]

#### wanli-decisions-v1:train:7403

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.9808 (Hamming 48) |
| Gate | activated at 0.9040 |
| Ground truth | C |
| S1 / R1 / R2 | C / B / B |
| Demo answer label | C |

- Query: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
- Exemplar 24241ee064b62e9c: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction. [family: natural_language_inference]

#### wanli-decisions-v1:train:25317

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.9800 (Hamming 49) |
| Gate | activated at 0.9040 |
| Ground truth | C |
| S1 / R1 / R2 | C / B / B |
| Demo answer label | A |

- Query: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
- Exemplar e7f31a7797c8cf81: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction. [family: natural_language_inference]

## Low-similarity rescue (R1 rescues with the lowest similarity)

#### community-diversity-v2/20260921/contracts/governing_law_venue/26/3/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.8800 (Hamming 121) |
| Gate | rejected at 0.9040 |
| Ground truth | G |
| S1 / R1 / R2 | A / G / A |
| Demo answer label | D |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar c4c92fd98b1a0b5a: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

#### community-diversity-v2/20260921/contracts/governing_law_venue/35/1/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.8800 (Hamming 121) |
| Gate | rejected at 0.9040 |
| Ground truth | D |
| S1 / R1 / R2 | B / D / B |
| Demo answer label | D |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar c4c92fd98b1a0b5a: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

#### community-diversity-v2/20260921/contracts/governing_law_venue/35/3/choice

| Field | Value |
|---|---|
| Kind | choice |
| Similarity (est. cosine) | 0.8800 (Hamming 121) |
| Gate | rejected at 0.9040 |
| Ground truth | A |
| S1 / R1 / R2 | B / A / B |
| Demo answer label | D |

- Query: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
- Exemplar c4c92fd98b1a0b5a: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported. [family: policy]

