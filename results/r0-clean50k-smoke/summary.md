# R0 Retrieval Sanity Evaluation — Summary

Run date: 2026-10-14T00:22:44  
Database: models/exsamplar.db (88300 examples)  
Input: datasets/clean50k-r0.jsonl  
Model: models/nomic-embed-text-v1.5.Q8_0.gguf  
Top-K: 5  
Distance metric: cosine  
Note: No dedicated validation split found in input; all English records used as queries. Records in the exemplar DB were NOT excluded.  

## Overall Statistics

| Kind | Queries | Mean Top-1 Sim | Median Top-1 Sim | Same-Group@1 | Same-Template@1 |
|------|---------|----------------|-----------------|--------------|----------------|
| choice | 10 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| **all** | **10** | **0.0000** | **0.0000** | **0.000** | **0.000** |

## Per-Kind Details

### choice (10 queries)

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
| Queries with < 5 results | 10 |

#### Most Retrieved Examples (top-5)

| ID | Count |
|-------|-------|

#### Ground-Truth Distribution

| Label | Query | Top-1 Retrieved |
|-------|-------|----------------|
| Type: action
Name: get_chil... | 1 | 0 |
| Type: action
Name: get_fan_... | 1 | 0 |
| Type: action
Name: get_item... | 1 | 0 |
| Type: action
Name: get_rese... | 1 | 0 |
| Type: action
Name: get_user... | 1 | 0 |
| Type: action
Name: process_... | 1 | 0 |
| Type: action
Name: transfer... | 1 | 0 |
| Type: action
Name: transfer... | 1 | 0 |
| Type: action
Name: transfer... | 1 | 0 |
| Type: action
Name: transfer... | 1 | 0 |

Query-Top1 answer agreement: 0 / 10 (0.0%)

## Performance

| Metric | Mean | P50 | P90 |
|--------|------|-----|-----|
| Embed (ms)  | 553.4 | 109.6 | 608.2 |
| Search (ms) | 0.1 | 0.1 | 0.2 |
| Total (ms)  | 553.5 | 109.7 | 608.3 |

Queries/sec: 1.8

## Kind/Level Filtering Verification

Wrong-kind retrieval count: **0**  
Score-level mismatch count: **0**  

_Kind and score-level filters are working correctly._
