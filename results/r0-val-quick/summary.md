# R0 Retrieval Sanity Evaluation — Summary

Run date: 2026-10-14T00:24:40  
Database: models/exsamplar.db (89900 examples)  
Input: datasets/validation.jsonl  
Model: models/nomic-embed-text-v1.5.Q8_0.gguf  
Top-K: 5  
Distance metric: cosine  
Note: No dedicated validation split found in input; all English records used as queries. Records in the exemplar DB were NOT excluded.  

## Overall Statistics

| Kind | Queries | Mean Top-1 Sim | Median Top-1 Sim | Same-Group@1 | Same-Template@1 |
|------|---------|----------------|-----------------|--------------|----------------|
| choice | 3 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| **all** | **3** | **0.0000** | **0.0000** | **0.000** | **0.000** |

## Per-Kind Details

### choice (3 queries)

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
| Queries with < 5 results | 3 |

#### Most Retrieved Examples (top-5)

| ID | Count |
|-------|-------|

#### Ground-Truth Distribution

| Label | Query | Top-1 Retrieved |
|-------|-------|----------------|
| K46 | 1 | 0 |
| K83 | 1 | 0 |
| item_73 | 1 | 0 |

Query-Top1 answer agreement: 0 / 3 (0.0%)

## Performance

| Metric | Mean | P50 | P90 |
|--------|------|-----|-----|
| Embed (ms)  | 84.8 | 69.0 | 116.9 |
| Search (ms) | 0.1 | 0.1 | 0.1 |
| Total (ms)  | 84.9 | 69.0 | 117.0 |

Queries/sec: 11.8

## Kind/Level Filtering Verification

Wrong-kind retrieval count: **0**  
Score-level mismatch count: **0**  

_Kind and score-level filters are working correctly._
