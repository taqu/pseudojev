# R0 Retrieval Sanity Evaluation — Summary

Run date: 2026-10-16T02:02:32  
Database: models/examplar.db (136311 examples)  
Input: datasets/validation.jsonl  
Model: models/nomic-embed-text-v1.5.gguf  
Top-K: 5  
Distance metric: cosine  
Note: No dedicated validation split found in input; all English records used as queries. Records in the exemplar DB were NOT excluded.  

## Overall Statistics

| Kind | Queries | Mean Top-1 Sim | Median Top-1 Sim | Same-Group@1 | Same-Template@1 |
|------|---------|----------------|-----------------|--------------|----------------|
| choice | 7516 | 0.0000 | 0.0000 | 0.651 | 0.356 |
| noul | 2921 | 0.0000 | 0.0000 | 0.201 | 0.997 |
| score | 1250 | 0.0000 | 0.0000 | 0.204 | 0.984 |
| **all** | **11687** | **0.0000** | **0.0000** | **0.491** | **0.583** |

## Per-Kind Details

### choice (7516 queries)

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
| Same-group @ Top-1 | 0.651 |
| Same-group (any in Top-5) | 0.000 |
| Same-group fraction in Top-5 | 0.148 |
| Same-template @ Top-1 | 0.356 |
| Same-template (any in Top-5) | 0.000 |
| Same-source-instance @ Top-1 | 0.007 |
| Queries with < 5 results | 0 |

#### Most Retrieved Examples (top-5)

| ID | Count |
|-------|-------|
| 60e4d7b102784ef | 640 |
| d5f951ba8ff11d34 | 640 |
| 7bc75214a9f48ce3 | 640 |
| a5a7fceef43066e6 | 640 |
| 547f5b878f6fb86f | 376 |

#### Ground-Truth Distribution

| Label | Query | Top-1 Retrieved |
|-------|-------|----------------|
| (empty) | 5 | 13 |
| K10 | 5 | 0 |
| K11 | 5 | 0 |
| K12 | 6 | 0 |
| K13 | 3 | 0 |
| K14 | 4 | 0 |
| K15 | 4 | 0 |
| K16 | 7 | 0 |
| K17 | 6 | 0 |
| K18 | 2 | 0 |
| K19 | 7 | 0 |
| K20 | 4 | 0 |
| K21 | 7 | 0 |
| K22 | 8 | 0 |
| K23 | 8 | 0 |
| K24 | 7 | 0 |
| K25 | 5 | 0 |
| K26 | 8 | 0 |
| K27 | 8 | 0 |
| K28 | 6 | 0 |
| K29 | 6 | 0 |
| K30 | 4 | 0 |
| K31 | 7 | 0 |
| K32 | 3 | 0 |
| K33 | 8 | 0 |
| K34 | 8 | 0 |
| K35 | 9 | 0 |
| K36 | 8 | 0 |
| K37 | 11 | 0 |
| K38 | 9 | 0 |
| K39 | 11 | 0 |
| K40 | 8 | 0 |
| K41 | 8 | 0 |
| K42 | 9 | 0 |
| K43 | 4 | 0 |
| K44 | 7 | 0 |
| K45 | 8 | 0 |
| K46 | 7 | 0 |
| K47 | 6 | 0 |
| K48 | 11 | 0 |
| K49 | 5 | 0 |
| K50 | 6 | 0 |
| K51 | 7 | 0 |
| K52 | 10 | 0 |
| K53 | 4 | 0 |
| K54 | 9 | 0 |
| K55 | 5 | 0 |
| K56 | 8 | 0 |
| K57 | 8 | 0 |
| K58 | 6 | 0 |
| K59 | 7 | 0 |
| K60 | 4 | 0 |
| K61 | 8 | 0 |
| K62 | 4 | 0 |
| K63 | 8 | 0 |
| K64 | 6 | 0 |
| K65 | 4 | 0 |
| K66 | 5 | 0 |
| K67 | 9 | 0 |
| K68 | 4 | 0 |
| K69 | 7 | 0 |
| K70 | 6 | 0 |
| K71 | 6 | 0 |
| K72 | 12 | 0 |
| K73 | 4 | 0 |
| K74 | 5 | 0 |
| K75 | 8 | 0 |
| K76 | 1 | 0 |
| K77 | 8 | 0 |
| K78 | 3 | 0 |
| K79 | 7 | 0 |
| K80 | 6 | 0 |
| K81 | 3 | 0 |
| K82 | 5 | 0 |
| K83 | 2 | 0 |
| K84 | 2 | 0 |
| K85 | 8 | 0 |
| K86 | 4 | 0 |
| K87 | 8 | 0 |
| K88 | 7 | 0 |
| K89 | 4 | 0 |
| K90 | 8 | 0 |
| K91 | 6 | 0 |
| K92 | 7 | 0 |
| K93 | 6 | 0 |
| K94 | 4 | 0 |
| K95 | 11 | 0 |
| K96 | 3 | 0 |
| K97 | 4 | 0 |
| K98 | 5 | 0 |
| P10 | 6 | 0 |
| P11 | 6 | 0 |
| P12 | 9 | 0 |
| P13 | 9 | 0 |
| P14 | 14 | 0 |
| P15 | 5 | 0 |
| P16 | 6 | 0 |
| P17 | 4 | 0 |
| P18 | 12 | 0 |
| P19 | 5 | 0 |
| P20 | 4 | 0 |
| P21 | 10 | 0 |
| P22 | 4 | 0 |
| P23 | 5 | 0 |
| P24 | 5 | 0 |
| P25 | 8 | 0 |
| P26 | 4 | 0 |
| P27 | 1 | 0 |
| P28 | 4 | 0 |
| P29 | 6 | 0 |
| P30 | 5 | 0 |
| P31 | 9 | 0 |
| P32 | 5 | 0 |
| P33 | 5 | 0 |
| P34 | 5 | 0 |
| P35 | 6 | 0 |
| P36 | 5 | 0 |
| P37 | 6 | 0 |
| P38 | 5 | 0 |
| P39 | 3 | 0 |
| P40 | 12 | 0 |
| P41 | 9 | 0 |
| P42 | 10 | 0 |
| P43 | 6 | 0 |
| P44 | 10 | 0 |
| P45 | 8 | 0 |
| P46 | 11 | 0 |
| P47 | 9 | 0 |
| P48 | 9 | 0 |
| P49 | 6 | 0 |
| P50 | 7 | 0 |
| P51 | 7 | 0 |
| P52 | 9 | 0 |
| P53 | 9 | 0 |
| P54 | 6 | 0 |
| P55 | 7 | 0 |
| P56 | 9 | 0 |
| P57 | 5 | 0 |
| P58 | 10 | 0 |
| P59 | 8 | 0 |
| P60 | 10 | 0 |
| P61 | 5 | 0 |
| P62 | 5 | 0 |
| P63 | 7 | 0 |
| P64 | 5 | 0 |
| P65 | 5 | 0 |
| P66 | 8 | 0 |
| P67 | 3 | 0 |
| P68 | 6 | 0 |
| P69 | 8 | 0 |
| P70 | 5 | 0 |
| P71 | 11 | 0 |
| P72 | 5 | 0 |
| P73 | 6 | 0 |
| P74 | 5 | 0 |
| P75 | 7 | 0 |
| P76 | 3 | 0 |
| P77 | 5 | 0 |
| P78 | 6 | 0 |
| P79 | 5 | 0 |
| P80 | 7 | 0 |
| P81 | 6 | 0 |
| P82 | 8 | 0 |
| P83 | 9 | 0 |
| P84 | 8 | 0 |
| P85 | 8 | 0 |
| P86 | 5 | 0 |
| P87 | 6 | 0 |
| P88 | 5 | 0 |
| P89 | 5 | 0 |
| P90 | 5 | 0 |
| P91 | 4 | 0 |
| P92 | 7 | 0 |
| P93 | 10 | 0 |
| P94 | 11 | 0 |
| P95 | 9 | 0 |
| P96 | 5 | 0 |
| P97 | 10 | 0 |
| P98 | 5 | 0 |
| The premise contradicts the... | 717 | 715 |
| The premise leaves the clai... | 2307 | 2318 |
| The premise supports the cl... | 1806 | 1797 |
| abstain | 580 | 2320 |
| account: Login, permissions... | 4 | 1 |
| account_statement: A period... | 21 | 21 |
| airline: An air carrier's o... | 7 | 7 |
| banking: A financial instit... | 12 | 12 |
| billing: Charges, invoices,... | 9 | 4 |
| bug_report: The user is rep... | 8 | 6 |
| education: Education servic... | 9 | 9 |
| electricity: Electrical pow... | 10 | 10 |
| feature_request: The user i... | 4 | 6 |
| insurance: An insurer's own... | 3 | 0 |
| invoice: An issued bill, in... | 84 | 82 |
| item_10 | 6 | 0 |
| item_11 | 6 | 0 |
| item_12 | 8 | 0 |
| item_13 | 11 | 0 |
| item_14 | 6 | 0 |
| item_15 | 6 | 0 |
| item_16 | 12 | 0 |
| item_17 | 3 | 0 |
| item_18 | 7 | 0 |
| item_19 | 4 | 0 |
| item_20 | 3 | 0 |
| item_21 | 4 | 0 |
| item_22 | 8 | 0 |
| item_23 | 10 | 0 |
| item_24 | 6 | 0 |
| item_25 | 9 | 0 |
| item_26 | 7 | 0 |
| item_27 | 5 | 0 |
| item_28 | 7 | 0 |
| item_29 | 11 | 0 |
| item_30 | 5 | 0 |
| item_31 | 2 | 0 |
| item_32 | 5 | 0 |
| item_33 | 3 | 0 |
| item_34 | 4 | 0 |
| item_35 | 4 | 0 |
| item_36 | 7 | 0 |
| item_37 | 6 | 0 |
| item_38 | 5 | 0 |
| item_39 | 6 | 0 |
| item_40 | 6 | 0 |
| item_41 | 10 | 0 |
| item_42 | 9 | 0 |
| item_43 | 10 | 0 |
| item_44 | 5 | 0 |
| item_45 | 10 | 0 |
| item_46 | 5 | 0 |
| item_47 | 6 | 0 |
| item_48 | 7 | 0 |
| item_49 | 5 | 0 |
| item_50 | 4 | 0 |
| item_51 | 5 | 0 |
| item_52 | 5 | 0 |
| item_53 | 4 | 0 |
| item_54 | 6 | 0 |
| item_55 | 6 | 0 |
| item_56 | 8 | 0 |
| item_57 | 5 | 0 |
| item_58 | 3 | 0 |
| item_59 | 6 | 0 |
| item_60 | 6 | 0 |
| item_61 | 7 | 0 |
| item_62 | 8 | 0 |
| item_63 | 7 | 0 |
| item_64 | 5 | 0 |
| item_65 | 11 | 0 |
| item_66 | 4 | 0 |
| item_67 | 7 | 0 |
| item_68 | 10 | 0 |
| item_69 | 7 | 0 |
| item_70 | 6 | 0 |
| item_71 | 6 | 0 |
| item_72 | 6 | 0 |
| item_73 | 8 | 0 |
| item_74 | 11 | 0 |
| item_75 | 7 | 0 |
| item_76 | 5 | 0 |
| item_77 | 3 | 0 |
| item_78 | 8 | 0 |
| item_79 | 6 | 0 |
| item_80 | 8 | 0 |
| item_81 | 7 | 0 |
| item_82 | 10 | 0 |
| item_83 | 6 | 0 |
| item_84 | 7 | 0 |
| item_85 | 5 | 0 |
| item_86 | 10 | 0 |
| item_87 | 8 | 0 |
| item_88 | 6 | 0 |
| item_89 | 6 | 0 |
| item_90 | 7 | 0 |
| item_91 | 7 | 0 |
| item_92 | 9 | 0 |
| item_93 | 5 | 0 |
| item_94 | 8 | 0 |
| item_95 | 7 | 0 |
| item_96 | 3 | 0 |
| item_97 | 4 | 0 |
| item_98 | 8 | 0 |
| newsletter: A recurring inf... | 21 | 21 |
| other: A provider's bill fo... | 70 | 73 |
| other: Other mail, such as ... | 21 | 21 |
| payment_confirmation: Confi... | 21 | 23 |
| promotion: An offer, discou... | 21 | 21 |
| telecom: Mobile or fixed te... | 36 | 36 |

Query-Top1 answer agreement: 5675 / 7516 (75.5%)

### noul (2921 queries)

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
| Same-group @ Top-1 | 0.201 |
| Same-group (any in Top-5) | 0.000 |
| Same-group fraction in Top-5 | 0.128 |
| Same-template @ Top-1 | 0.997 |
| Same-template (any in Top-5) | 0.000 |
| Same-source-instance @ Top-1 | 0.075 |
| Queries with < 5 results | 0 |

#### Most Retrieved Examples (top-5)

| ID | Count |
|-------|-------|
| 7dd370d1b3ce6915 | 267 |
| 689f2cd7c5dab086 | 160 |
| bda7b962933ae580 | 120 |
| a8ff7eeae751d983 | 118 |
| 4cc4a5c4099aa00e | 110 |

#### Ground-Truth Distribution

| Label | Query | Top-1 Retrieved |
|-------|-------|----------------|
| no | 2292 | 2319 |
| yes | 629 | 602 |

Query-Top1 answer agreement: 2558 / 2921 (87.6%)

### score (1250 queries)

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
| Same-group @ Top-1 | 0.204 |
| Same-group (any in Top-5) | 0.001 |
| Same-group fraction in Top-5 | 0.103 |
| Same-template @ Top-1 | 0.984 |
| Same-template (any in Top-5) | 0.001 |
| Same-source-instance @ Top-1 | 0.197 |
| Queries with < 5 results | 0 |

#### Most Retrieved Examples (top-5)

| ID | Count |
|-------|-------|
| 26764efc02321a90 | 191 |
| 7ccf94898e933248 | 132 |
| 3c7d987b69f96bb7 | 113 |
| 33171739d314f1d4 | 96 |
| f8e62317c69c139f | 89 |

#### Ground-Truth Distribution

| Label | Query | Top-1 Retrieved |
|-------|-------|----------------|
| (empty) | 8 | 2 |
| Blocking issue; no workarou... | 2 | 3 |
| Broken or degraded feature;... | 3 | 11 |
| Calm, matter-of-fact | 9 | 9 |
| Cosmetic; no impact to func... | 23 | 16 |
| Exactly 0 of the 4 numbered... | 15 | 3 |
| Exactly 1 of the 4 numbered... | 101 | 121 |
| Exactly 2 of the 4 numbered... | 249 | 218 |
| Exactly 3 of the 4 numbered... | 474 | 545 |
| Exactly 4 of the 4 numbered... | 321 | 273 |
| Frustrated but civil | 12 | 7 |
| The customer explicitly int... | 12 | 16 |
| The customer explicitly req... | 8 | 9 |
| The customer is undecided a... | 7 | 4 |
| Very angry | 6 | 13 |

Query-Top1 answer agreement: 575 / 1250 (46.0%)

## Performance

| Metric | Mean | P50 | P90 |
|--------|------|-----|-----|
| Embed (ms)  | 21.6 | 20.3 | 28.5 |
| Search (ms) | 16.0 | 15.3 | 19.6 |
| Total (ms)  | 37.6 | 35.8 | 44.9 |

Queries/sec: 26.6

## Kind/Level Filtering Verification

Wrong-kind retrieval count: **0**  
Score-level mismatch count: **0**  

_Kind and score-level filters are working correctly._
