# R0 Retrieval Sanity Evaluation — Summary

Run date: 2026-10-12T19:50:17  
Database: models/exsamplar.db (109600 examples)  
Input: datasets/train.jsonl  
Model: models/nomic-embed-text-v1.5.Q8_0.gguf  
Top-K: 5  
Distance metric: cosine  
Note: No dedicated validation split found in input; all English records used as queries. Records in the exemplar DB were excluded (--skip-db-records).  

## Overall Statistics

| Kind | Queries | Mean Top-1 Sim | Median Top-1 Sim | Same-Group@1 | Same-Template@1 |
|------|---------|----------------|-----------------|--------------|----------------|
| choice | 611 | 0.4662 | 0.3394 | 0.005 | 0.000 |
| noul | 862 | 0.3070 | 0.3032 | 0.000 | 0.000 |
| score | 602 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| **all** | **2075** | **0.3730** | **0.3108** | **0.001** | **0.000** |

## Per-Kind Details

### choice (611 queries)

#### Top-1 Similarity

| Metric | Value |
|--------|-------|
| Mean   | 0.4662 |
| Median | 0.3394 |
| Std    | 0.1840 |
| P10    | 0.2653 |
| P25    | 0.2977 |
| P75    | 0.6575 |
| P90    | 0.6957 |
| Mean Top-5 sim | 0.4576 |
| Mean Top1-Top2 gap | 0.0070 |

#### Retrieval Structure

| Metric | Rate |
|--------|------|
| Same-group @ Top-1 | 0.005 |
| Same-group (any in Top-5) | 0.002 |
| Same-group fraction in Top-5 | 0.003 |
| Same-template @ Top-1 | 0.000 |
| Same-template (any in Top-5) | 0.000 |
| Same-source-instance @ Top-1 | 0.000 |
| Queries with < 5 results | 0 |

#### Most Retrieved Examples (top-5)

| ID | Count |
|-------|-------|
| community-diversity-v2/20260921/support/subscription_canc... | 336 |
| community-diversity-v2/20260921/support/subscription_canc... | 186 |
| community-diversity-v2/20260921/support/subscription_canc... | 186 |
| community-diversity-v2/20260921/support/subscription_canc... | 185 |
| community-diversity-v2/20260921/support/subscription_canc... | 184 |

#### Ground-Truth Distribution

| Label | Query | Top-1 Retrieved |
|-------|-------|----------------|
| (empty) | 23 | 0 |
| The premise contradicts the... | 42 | 67 |
| The premise leaves the clai... | 137 | 164 |
| The premise supports the cl... | 106 | 119 |
| abstain | 0 | 261 |
| account: Login, permissions... | 39 | 0 |
| account_statement: A period... | 12 | 0 |
| airline: An air carrier's o... | 8 | 0 |
| banking: A financial instit... | 6 | 0 |
| billing: Charges, invoices,... | 53 | 0 |
| bug_report: The user is rep... | 66 | 0 |
| education: Education servic... | 2 | 0 |
| electricity: Electrical pow... | 7 | 0 |
| feature_request: The user i... | 43 | 0 |
| insurance: An insurer's own... | 1 | 0 |
| invoice: An issued bill, in... | 22 | 0 |
| newsletter: A recurring inf... | 4 | 0 |
| other: A provider's bill fo... | 17 | 0 |
| other: Other mail, such as ... | 7 | 0 |
| payment_confirmation: Confi... | 8 | 0 |
| promotion: An offer, discou... | 4 | 0 |
| telecom: Mobile or fixed te... | 4 | 0 |

Query-Top1 answer agreement: 111 / 611 (18.2%)

### noul (862 queries)

#### Top-1 Similarity

| Metric | Value |
|--------|-------|
| Mean   | 0.3070 |
| Median | 0.3032 |
| Std    | 0.0278 |
| P10    | 0.2773 |
| P25    | 0.2903 |
| P75    | 0.3210 |
| P90    | 0.3485 |
| Mean Top-5 sim | 0.3060 |
| Mean Top1-Top2 gap | 0.0006 |

#### Retrieval Structure

| Metric | Rate |
|--------|------|
| Same-group @ Top-1 | 0.000 |
| Same-group (any in Top-5) | 0.000 |
| Same-group fraction in Top-5 | 0.000 |
| Same-template @ Top-1 | 0.000 |
| Same-template (any in Top-5) | 0.000 |
| Same-source-instance @ Top-1 | 0.000 |
| Queries with < 5 results | 0 |

#### Most Retrieved Examples (top-5)

| ID | Count |
|-------|-------|
| community-diversity-v2/20260921/support/subscription_canc... | 709 |
| community-diversity-v2/20260921/support/subscription_canc... | 339 |
| community-diversity-v2/20260921/support/subscription_canc... | 327 |
| community-diversity-v2/20260921/support/subscription_canc... | 267 |
| community-diversity-v2/20260921/support/refund_window/4/0... | 241 |

#### Ground-Truth Distribution

| Label | Query | Top-1 Retrieved |
|-------|-------|----------------|
| no | 707 | 510 |
| yes | 155 | 352 |

Query-Top1 answer agreement: 465 / 862 (53.9%)

### score (602 queries)

#### Top-1 Similarity

| Metric | Value |
|--------|-------|
| Mean   | 0.0000 |
| Median | 0.0000 |
| Std    | 0.0000 |
| P10    | 0.0000 |
| P25    | 0.0000 |
| P75    | 0.0000 |
| P90    | 0.0000 |
| Mean Top-5 sim | 0.0000 |
| Mean Top1-Top2 gap | 0.0000 |

#### Retrieval Structure

| Metric | Rate |
|--------|------|
| Same-group @ Top-1 | 0.000 |
| Same-group (any in Top-5) | 0.000 |
| Same-group fraction in Top-5 | 0.000 |
| Same-template @ Top-1 | 0.000 |
| Same-template (any in Top-5) | 0.000 |
| Same-source-instance @ Top-1 | 0.000 |
| Queries with < 5 results | 602 |

#### Most Retrieved Examples (top-5)

| ID | Count |
|-------|-------|

#### Ground-Truth Distribution

| Label | Query | Top-1 Retrieved |
|-------|-------|----------------|
| (empty) | 56 | 0 |
| Blocking issue; no workarou... | 18 | 0 |
| Broken or degraded feature;... | 16 | 0 |
| Calm, matter-of-fact | 57 | 0 |
| Cosmetic; no impact to func... | 149 | 0 |
| Frustrated but civil | 63 | 0 |
| The customer explicitly int... | 52 | 0 |
| The customer explicitly req... | 60 | 0 |
| The customer is undecided a... | 66 | 0 |
| Very angry | 65 | 0 |

Query-Top1 answer agreement: 0 / 602 (0.0%)

## Performance

| Metric | Mean | P50 | P90 |
|--------|------|-----|-----|
| Embed (ms)  | 254.5 | 264.8 | 333.0 |
| Search (ms) | 262.7 | 256.8 | 295.2 |
| Total (ms)  | 517.2 | 520.9 | 599.1 |

Queries/sec: 1.9

## Kind/Level Filtering Verification

Wrong-kind retrieval count: **0**  
Score-level mismatch count: **0**  

_Kind and score-level filters are working correctly._
