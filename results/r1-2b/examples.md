# R1 — Representative examples

Per transition and kind: the highest-similarity (smallest Hamming distance) and the lowest-similarity case, chosen deterministically (ties broken by file order). Successes and failures are shown alike.

## rescue

### wanli-decisions-v1:train:171040

*Selected as: choice, highest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 38 / 0.988 |
| Ground truth | C (The premise supports the claim.) |
| S1 prediction | B (The premise leaves the claim undetermined.) (margin 6.10) |
| R1 prediction | C (The premise supports the claim.) (margin 4.19) |
| Demo answer | A (The premise leaves the claim undetermined.) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / natural_language_inference |

**Query**

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"This interpretation is not correct.","premise":"A careful examination of the evidence shows that this interpretation is not correct."}
    A: The premise contradicts the claim.
    B: The premise leaves the claim undetermined.
    C: The premise supports the claim.

**Retrieved exemplar** (a151cf6f77000fa2)

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"The findings are not correct.","premise":"As the paper is not peer-reviewed, we cannot be sure that the findings are correct."}
    A: The premise leaves the claim undetermined.
    B: The premise contradicts the claim.
    C: The premise supports the claim.
    Answer: A

Relevant differences: R1 changed the prediction, not to the demo label.

### community-diversity-v2/20260921/contracts/governing_law_venue/26/3/choice

*Selected as: choice, lowest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 121 / 0.880 |
| Ground truth | G (abstain) |
| S1 prediction | A (K22) (margin 0.35) |
| R1 prediction | G (abstain) (margin 1.83) |
| Demo answer | D (P12) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | contracts / policy |

**Query**

    Question: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
    State: {"candidates":"K22\t{\"amendment_signed_by_both\": false, \"forum\": \"local_courts\", \"in_amendment\": true, \"in_signed_master\": true, \"jurisdiction\": \"Ontario\", \"section_number\": 62, \"superseded\": true}\nP63\t{\"amendment_signed_by_both\": true, \"forum\": \"local_courts\", \"in_amendment\": false, \"in_signed_master\": false, \"jurisdiction\": \"other_value\", \"section_number\": 66, \"superseded\": false}\nitem_88\t{\"amendment_signed_by_both\": true, \"forum\": null, \"in_amendment\": true, \"in_signed_master\": false, \"jurisdiction\": \"Ontario\", \"section_number\": 64, \"superseded\": false}\nP71\t{\"amendment_signed_by_both\": true, \"forum\": \"other_value\", \"in_amend ...[truncated 1645 bytes]
    A: K22
    B: P71
    C: P88
    D: K44
    E: P63
    F: item_88
    G: abstain

**Retrieved exemplar** (c4c92fd98b1a0b5a)

    Question: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
    State: {"candidates":[{"candidate":"P75","facts":{"annual_fees_cents":144,"cap_cents":89,"cap_disclaimed_in_amendment":true,"carve_outs":["intentional_misconduct"],"section_number":60}},{"candidate":"P12","facts":{"annual_fees_cents":0,"cap_cents":8,"cap_disclaimed_in_amendment":false,"carve_outs":["fraud","intentional_misconduct"],"section_number":56}},{"candidate":"item_37","facts":{"annual_fees_cents":null,"cap_cents":144,"cap_disclaimed_in_amendment":false,"carve_outs":["fraud","intentional_misconduct","unrelated_item"],"section_number":62}},{"candidate":"P81","facts":{"annual_fees_cents":2,"cap_cents":1,"cap_disclaimed_in_amendment":false,"carve_outs":["fraud","intentional_misconduct","unrelat ...[truncated 1262 bytes]
    A: P81
    B: P75
    C: abstain
    D: P12
    E: item_37
    F: K21
    Answer: D

Relevant differences: R1 changed the prediction, not to the demo label; option counts differ (7 vs 6).

### mailroom:9a8e84c8e68b:en:promotion:base:about_banking

*Selected as: noul, highest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 42 / 0.985 |
| Ground truth | A (no) |
| S1 prediction | B (yes) (margin 1.18) |
| R1 prediction | A (no) (margin 9.08) |
| Demo answer | A (no) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / evidence |

**Query**

    Question: Is the subject matter of this email a bank, payment card or financial institution's own products; not a utility merely paid through a bank? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns electrical power supplied by the electricity utility itself.\nA promotional offer: buy now for EUR 486.40 and save EUR 20. You have placed no order and owe nothing.","date":"2026-09-20","from":{"display_name":"Luma-9a8e84c8e68b","domain":"luma-9a8e84c8e68b.example","email":"contact@luma-9a8e84c8e68b.example"},"subject":"Luma-9a8e84c8e68b — service correspondence"}}
    A: no
    B: yes

**Retrieved exemplar** (821f4eb10746f943)

    Question: Is the subject matter of this email a bank, payment card or financial institution's own products; not a utility merely paid through a bank? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns electrical power supplied by the electricity utility itself.\nA promotional offer: buy now for EUR 207.40 and save EUR 20. You have placed no order and owe nothing.","date":"2026-09-20","from":{"display_name":"Luma-54b7bf60e7d1","domain":"luma-54b7bf60e7d1.example","email":"contact@luma-54b7bf60e7d1.example"},"subject":"Luma-54b7bf60e7d1 — service correspondence"}}
    A: no
    B: yes
    Answer: A

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/browser_tools/idempotent_retry/61/3/noul

*Selected as: noul, lowest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 111 / 0.899 |
| Ground truth | A (no) |
| S1 prediction | B (yes) (margin 1.37) |
| R1 prediction | A (no) (margin 2.12) |
| Demo answer | A (no) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | browser_tools / routing |

**Query**

    Question: Candidate item_92. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":{"K28":{"idempotency_key":"record_B","interaction_count":53,"original_idempotency_key":"record_B","original_payload_hash":null,"payload_hash":"record_B","previous_attempt_not_started":true,"previous_result_unknown":false,"server_supports_idempotency":true},"K41":{"idempotency_key":"record_B","interaction_count":50,"original_idempotency_key":"record_B","original_payload_hash":"record_C","payload_hash":"record_B","previous_attempt_not_started":true,"previous_result_unknown":false,"server_supports_idempotency":true},"P76":{"idempotency_key":"record_C","interaction_count":51,"original_idempotency_key":"record_D","original_payload_hash":"record_B","payload_hash":"record_B","previous ...[truncated 1722 bytes]
    A: no
    B: yes

**Retrieved exemplar** (b7903de13571727d)

    Question: Candidate K62. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":[{"candidate":"K66","facts":{"capabilities":[],"endpoint_available":false,"granted_scopes":["unrelated_item"],"interaction_count":73,"writes_remote_state":false}},{"candidate":"item_80","facts":{"capabilities":["inspect"],"endpoint_available":true,"granted_scopes":["project_read"],"interaction_count":65,"writes_remote_state":false}},{"candidate":"item_62","facts":{"capabilities":["inspect"],"endpoint_available":true,"granted_scopes":[],"interaction_count":71,"writes_remote_state":true}},{"candidate":"item_69","facts":{"capabilities":["inspect"],"endpoint_available":false,"granted_scopes":["unrelated_item"],"interaction_count":67,"writes_remote_state":false}},{"candidate":"K81", ...[truncated 1588 bytes]
    A: no
    B: yes
    Answer: A

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/shell_history/checksum_verify/52/1/score

*Selected as: score, highest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 53 / 0.977 |
| Ground truth | D (Exactly 3 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.78) |
| R1 prediction | D (Exactly 3 of the 4 numbered requirements are established.) (margin 2.06) |
| Demo answer | D (Exactly 3 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | shell_history / routing |

**Query**

    Question: Candidate item_78. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K48":{"algorithm":"other_value","comparison_performed":true,"expected_digest_source":"trusted_manifest","history_sequence":95,"target_file":"weights.bin"},"P60":{"algorithm":"sha256","comparison_performed":true,"expected_digest_source":"trusted_manifest","history_sequence":93,"target_file":"weights.bin"},"P86":{"algorithm":null,"comparison_performed":true,"expected_digest_source":"trusted_manifest","history_sequence":89,"target_file":"weights.bin"},"item_45":{"algorithm":"other_value","comparison_performed":false,"expected_digest_source":"trusted_manifest","history_sequence":91,"target_file":"weights.bin"},"item_78":{"algorithm":"sha256","comparison_performed":true,"expected_ ...[truncated 1135 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (976fd9706372973c)

    Question: Candidate item_25. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K18":{"history_sequence":41,"operation":"inspect","parsed_path":"report[final].csv","path_argument_count":1,"shell_expansion_present":null},"K86":{"history_sequence":43,"operation":"other_value","parsed_path":"other_value","path_argument_count":34,"shell_expansion_present":false},"P32":{"history_sequence":37,"operation":"inspect","parsed_path":"report[final].csv","path_argument_count":8,"shell_expansion_present":true},"item_23":{"history_sequence":45,"operation":null,"parsed_path":"report[final].csv","path_argument_count":1,"shell_expansion_present":false},"item_25":{"history_sequence":48,"operation":"inspect","parsed_path":"report[final].csv","path_argument_count":2,"shell_e ...[truncated 1368 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: D

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/games/racing_overtake/38/3/score

*Selected as: score, lowest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 104 / 0.911 |
| Ground truth | D (Exactly 3 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.90) |
| R1 prediction | D (Exactly 3 of the 4 numbered requirements are established.) (margin 1.44) |
| Demo answer | D (Exactly 3 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | games / policy |

**Query**

    Question: Candidate K17. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":"K17\t{\"completion_ticks\": 63, \"overtake_cancelled\": true, \"passing_distance\": 1, \"relative_speed\": 144, \"return_lane_clear\": true, \"straight_remaining\": 0, \"yellow_flag\": true}\nitem_65\t{\"completion_ticks\": 68, \"overtake_cancelled\": false, \"passing_distance\": 144, \"relative_speed\": 34, \"return_lane_clear\": false, \"straight_remaining\": 13, \"yellow_flag\": false}\nP49\t{\"completion_ticks\": 66, \"overtake_cancelled\": false, \"passing_distance\": 5, \"relative_speed\": null, \"return_lane_clear\": true, \"straight_remaining\": 55, \"yellow_flag\": false}\nP29\t{\"completion_ticks\": 64, \"overtake_cancelled\": false, \"passing_distance\": 0, \"relati ...[truncated 1338 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (c0087cd399ad3467)

    Question: Candidate K22. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":[{"candidate":"item_93","facts":{"blocked_cells_crossed":0,"completion_ticks":72,"end_cell":15,"moves":18,"path_is_connected":true}},{"candidate":"P46","facts":{"blocked_cells_crossed":0,"completion_ticks":69,"end_cell":16,"moves":19,"path_is_connected":true}},{"candidate":"P94","facts":{"blocked_cells_crossed":0,"completion_ticks":76,"end_cell":15,"moves":null,"path_is_connected":true}},{"candidate":"K22","facts":{"blocked_cells_crossed":0,"completion_ticks":70,"end_cell":15,"moves":1,"path_is_connected":false}},{"candidate":"K18","facts":{"blocked_cells_crossed":8,"completion_ticks":74,"end_cell":5,"moves":34,"path_is_connected":true}}],"evidence_policy":"Each numbered requir ...[truncated 971 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: D

Relevant differences: R1 moved to the demonstration's answer label.

## harm

### customer-control-v1:42:652:category

*Selected as: choice, highest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 41 / 0.986 |
| Ground truth | B (feature_request: The user is requesting new functionality) |
| S1 prediction | B (feature_request: The user is requesting new functionality) (margin 3.50) |
| R1 prediction | C (billing: Charges, invoices, refunds, subscriptions) (margin 0.21) |
| Demo answer | C (feature_request: The user is requesting new functionality) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / routing |

**Query**

    Question: Determine the broad category of this support ticket. For this controlled task, select the customer's expressly prioritized request. If two requests explicitly have equal priority, allocate equal support to their two categories. A malfunction of an existing feature is bug_report; a payment or refund request is billing; a request for absent functionality is feature_request; an access or profile request is account. Secondary context does not override an explicit priority.
    State: Conversation with account holder Harbor-bbcb7e7173.
Customer message: I would like an offline reading mode, which this product does not currently offer. That is the request I want you to handle first.
Support: Please tell us more.
Customer response: I am furious. This treatment is outrageous and completely unacceptable! I will keep using the service and renew my subscription.
    A: account: Login, permissions, profile, security
    B: feature_request: The user is requesting new functionality
    C: billing: Charges, invoices, refunds, subscriptions
    D: bug_report: The user is reporting something that is broken or producing errors

**Retrieved exemplar** (aa9b5598ec8e895c)

    Question: Determine the broad category of this support ticket. For this controlled task, select the customer's expressly prioritized request. If two requests explicitly have equal priority, allocate equal support to their two categories. A malfunction of an existing feature is bug_report; a payment or refund request is billing; a request for absent functionality is feature_request; an access or profile request is account. Secondary context does not override an explicit priority.
    State: Customer Harbor-df0cb7cbd3: I would like an offline reading mode, which this product does not currently offer. That is the request I want you to handle first.
Agent: Could you add the details?
Customer: I am furious. This treatment is outrageous and completely unacceptable! I will keep using the service and renew my subscription.
    A: billing: Charges, invoices, refunds, subscriptions
    B: bug_report: The user is reporting something that is broken or producing errors
    C: feature_request: The user is requesting new functionality
    D: account: Login, permissions, profile, security
    Answer: C

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/contracts/governing_law_venue/32/0/choice

*Selected as: choice, lowest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 121 / 0.880 |
| Ground truth | A (K98) |
| S1 prediction | A (K98) (margin 0.09) |
| R1 prediction | D (K50) (margin 1.43) |
| Demo answer | D (P12) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | contracts / policy |

**Query**

    Question: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
    State: {"candidates":"K50\t{\"amendment_signed_by_both\": true, \"forum\": \"local_courts\", \"in_amendment\": false, \"in_signed_master\": true, \"jurisdiction\": \"other_value\", \"section_number\": 65, \"superseded\": false}\nK98\t{\"amendment_signed_by_both\": true, \"forum\": \"local_courts\", \"in_amendment\": true, \"in_signed_master\": false, \"jurisdiction\": \"Ontario\", \"section_number\": 66, \"superseded\": false}\nK77\t{\"amendment_signed_by_both\": true, \"forum\": \"local_courts\", \"in_amendment\": true, \"in_signed_master\": false, \"jurisdiction\": \"Ontario\", \"section_number\": 68, \"superseded\": false}\nK60\t{\"amendment_signed_by_both\": true, \"forum\": \"local_courts\", \ ...[truncated 1462 bytes]
    A: K98
    B: abstain
    C: K77
    D: K50
    E: K82
    F: K60

**Retrieved exemplar** (c4c92fd98b1a0b5a)

    Question: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
    State: {"candidates":[{"candidate":"P75","facts":{"annual_fees_cents":144,"cap_cents":89,"cap_disclaimed_in_amendment":true,"carve_outs":["intentional_misconduct"],"section_number":60}},{"candidate":"P12","facts":{"annual_fees_cents":0,"cap_cents":8,"cap_disclaimed_in_amendment":false,"carve_outs":["fraud","intentional_misconduct"],"section_number":56}},{"candidate":"item_37","facts":{"annual_fees_cents":null,"cap_cents":144,"cap_disclaimed_in_amendment":false,"carve_outs":["fraud","intentional_misconduct","unrelated_item"],"section_number":62}},{"candidate":"P81","facts":{"annual_fees_cents":2,"cap_cents":1,"cap_disclaimed_in_amendment":false,"carve_outs":["fraud","intentional_misconduct","unrelat ...[truncated 1262 bytes]
    A: P81
    B: P75
    C: abstain
    D: P12
    E: item_37
    F: K21
    Answer: D

Relevant differences: R1 moved to the demonstration's answer label.

### mailroom:9a8e84c8e68b:en:receipt:base:about_electricity

*Selected as: noul, highest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 54 / 0.976 |
| Ground truth | B (yes) |
| S1 prediction | B (yes) (margin 0.42) |
| R1 prediction | A (no) (margin 3.86) |
| Demo answer | A (no) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / evidence |

**Query**

    Question: Is the subject matter of this email an electricity supplier or supplied electrical power; not a hardware retailer's equipment? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns electrical power supplied by the electricity utility itself.\nWe are a document delivery service for Luma-9a8e84c8e68b; we did not provide the billed service and these are not our own charges.\nPayment of EUR 486.40 was already received. This is your payment receipt, with no new charges and nothing left to pay.\nReference number: INV-9a8e84c8e68b.","date":"2026-09-20","from":{"display_name":"PaperRelay-9a8e84c8e68b","domain":"delivery-9a8e84c8e68b.example","email":"relay@delivery-9a8e84c8e68b.example"},"subject":"Luma-9a8e84c8e68b — service correspondence"}}
    A: no
    B: yes

**Retrieved exemplar** (e5e68513b6513c30)

    Question: Is the subject matter of this email an electricity supplier or supplied electrical power; not a hardware retailer's equipment? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns water supply from the water utility.\nWe are a document delivery service for Luma-f75888e6b823; we did not provide the billed service and these are not our own charges.\nPayment of EUR 240.40 was already received. This is your payment receipt, with no new charges and nothing left to pay.\nReference number: INV-f75888e6b823.","date":"2026-09-20","from":{"display_name":"PaperRelay-f75888e6b823","domain":"delivery-f75888e6b823.example","email":"relay@delivery-f75888e6b823.example"},"subject":"Luma-f75888e6b823 — service correspondence"}}
    A: no
    B: yes
    Answer: A

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/browser_tools/idempotent_retry/2/2/noul

*Selected as: noul, lowest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 111 / 0.899 |
| Ground truth | B (yes) |
| S1 prediction | B (yes) (margin 1.89) |
| R1 prediction | A (no) (margin 5.73) |
| Demo answer | A (no) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | browser_tools / routing |

**Query**

    Question: Candidate P91. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":"K17\t{\"idempotency_key\": \"record_C\", \"interaction_count\": 58, \"original_idempotency_key\": \"record_B\", \"original_payload_hash\": \"record_B\", \"payload_hash\": \"record_C\", \"previous_attempt_not_started\": true, \"previous_result_unknown\": true, \"server_supports_idempotency\": false}\nP94\t{\"idempotency_key\": \"record_B\", \"interaction_count\": 53, \"original_idempotency_key\": \"record_B\", \"original_payload_hash\": \"record_B\", \"payload_hash\": \"record_B\", \"previous_attempt_not_started\": false, \"previous_result_unknown\": true, \"server_supports_idempotency\": true}\nP91\t{\"idempotency_key\": \"record_D\", \"interaction_count\": 54, \"original_idem ...[truncated 2521 bytes]
    A: no
    B: yes

**Retrieved exemplar** (b7903de13571727d)

    Question: Candidate K62. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":[{"candidate":"K66","facts":{"capabilities":[],"endpoint_available":false,"granted_scopes":["unrelated_item"],"interaction_count":73,"writes_remote_state":false}},{"candidate":"item_80","facts":{"capabilities":["inspect"],"endpoint_available":true,"granted_scopes":["project_read"],"interaction_count":65,"writes_remote_state":false}},{"candidate":"item_62","facts":{"capabilities":["inspect"],"endpoint_available":true,"granted_scopes":[],"interaction_count":71,"writes_remote_state":true}},{"candidate":"item_69","facts":{"capabilities":["inspect"],"endpoint_available":false,"granted_scopes":["unrelated_item"],"interaction_count":67,"writes_remote_state":false}},{"candidate":"K81", ...[truncated 1588 bytes]
    A: no
    B: yes
    Answer: A

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/shell_history/checksum_verify/75/2/score

*Selected as: score, highest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 57 / 0.973 |
| Ground truth | B (Exactly 1 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.44) |
| R1 prediction | D (Exactly 3 of the 4 numbered requirements are established.) (margin 0.53) |
| Demo answer | D (Exactly 3 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | shell_history / routing |

**Query**

    Question: Candidate K10. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":[{"candidate":"P52","facts":{"algorithm":"sha256","comparison_performed":true,"expected_digest_source":"trusted_manifest","history_sequence":77,"target_file":"archive.tar"}},{"candidate":"item_16","facts":{"algorithm":"sha256","comparison_performed":false,"expected_digest_source":"other_value","history_sequence":68,"target_file":"archive.tar"}},{"candidate":"P59","facts":{"algorithm":"sha256","comparison_performed":true,"expected_digest_source":"trusted_manifest","history_sequence":74,"target_file":"archive.tar"}},{"candidate":"K23","facts":{"algorithm":"sha256","comparison_performed":true,"expected_digest_source":"trusted_manifest","history_sequence":76,"target_file":"archive. ...[truncated 1587 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (fceb1d354fdea285)

    Question: Candidate K52. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":"K52\t{\"history_sequence\": 40, \"operation\": \"inspect\", \"parsed_path\": \"report[final].csv\", \"path_argument_count\": 1, \"shell_expansion_present\": null}\nitem_90\t{\"history_sequence\": 44, \"operation\": \"inspect\", \"parsed_path\": \"report[final].csv\", \"path_argument_count\": 1, \"shell_expansion_present\": false}\nP15\t{\"history_sequence\": 46, \"operation\": \"inspect\", \"parsed_path\": \"report[final].csv\", \"path_argument_count\": 1, \"shell_expansion_present\": false}\nK69\t{\"history_sequence\": 47, \"operation\": \"inspect\", \"parsed_path\": \"other_value\", \"path_argument_count\": 1, \"shell_expansion_present\": false}\nK50\t{\"history_sequence\":  ...[truncated 1186 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: D

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/games/racing_overtake/55/0/score

*Selected as: score, lowest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 102 / 0.914 |
| Ground truth | B (Exactly 1 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.91) |
| R1 prediction | E (Exactly 4 of the 4 numbered requirements are established.) (margin 0.15) |
| Demo answer | E (Exactly 4 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | games / policy |

**Query**

    Question: Candidate K50. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K42":{"completion_ticks":31,"overtake_cancelled":true,"passing_distance":144,"relative_speed":null,"return_lane_clear":true,"straight_remaining":144,"yellow_flag":true},"K50":{"completion_ticks":29,"overtake_cancelled":false,"passing_distance":144,"relative_speed":144,"return_lane_clear":false,"straight_remaining":1,"yellow_flag":true},"P88":{"completion_ticks":24,"overtake_cancelled":true,"passing_distance":34,"relative_speed":34,"return_lane_clear":false,"straight_remaining":144,"yellow_flag":true},"item_72":{"completion_ticks":27,"overtake_cancelled":true,"passing_distance":144,"relative_speed":89,"return_lane_clear":true,"straight_remaining":144,"yellow_flag":true},"item_ ...[truncated 1207 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (1b0092e40fc3b0b5)

    Question: Candidate item_14. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K66":{"blocked_cells_crossed":55,"completion_ticks":84,"end_cell":89,"moves":2,"path_is_connected":true},"P11":{"blocked_cells_crossed":0,"completion_ticks":82,"end_cell":23,"moves":null,"path_is_connected":true},"P75":{"blocked_cells_crossed":0,"completion_ticks":80,"end_cell":1,"moves":30,"path_is_connected":false},"item_14":{"blocked_cells_crossed":0,"completion_ticks":78,"end_cell":23,"moves":5,"path_is_connected":true},"item_15":{"blocked_cells_crossed":0,"completion_ticks":75,"end_cell":23,"moves":1,"path_is_connected":true},"item_97":{"blocked_cells_crossed":0,"completion_ticks":76,"end_cell":23,"moves":0,"path_is_connected":true}},"evidence_policy":"Each numbered requ ...[truncated 973 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: E

Relevant differences: R1 moved to the demonstration's answer label.

## both_correct

### customer-control-v1:42:665:category

*Selected as: choice, highest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 38 / 0.988 |
| Ground truth | A (feature_request: The user is requesting new functionality) |
| S1 prediction | A (feature_request: The user is requesting new functionality) (margin 1.55) |
| R1 prediction | A (feature_request: The user is requesting new functionality) (margin 1.35) |
| Demo answer | C (feature_request: The user is requesting new functionality) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / routing |

**Query**

    Question: Determine the broad category of this support ticket. For this controlled task, select the customer's expressly prioritized request. If two requests explicitly have equal priority, allocate equal support to their two categories. A malfunction of an existing feature is bug_report; a payment or refund request is billing; a request for absent functionality is feature_request; an access or profile request is account. Secondary context does not override an explicit priority.
    State: Customer Harbor-aaabb55dbf: I would like an offline reading mode, which this product does not currently offer. That is the request I want you to handle first.
Agent: Could you add the details?
Customer: This is frustrating, but I appreciate your help and will keep this constructive. Whether I renew depends on whether this is fixed before Friday.
    A: feature_request: The user is requesting new functionality
    B: billing: Charges, invoices, refunds, subscriptions
    C: bug_report: The user is reporting something that is broken or producing errors
    D: account: Login, permissions, profile, security

**Retrieved exemplar** (4c74050bf2489124)

    Question: Determine the broad category of this support ticket. For this controlled task, select the customer's expressly prioritized request. If two requests explicitly have equal priority, allocate equal support to their two categories. A malfunction of an existing feature is bug_report; a payment or refund request is billing; a request for absent functionality is feature_request; an access or profile request is account. Secondary context does not override an explicit priority.
    State: Customer Harbor-58080683e9: I would like an offline reading mode, which this product does not currently offer. That is the request I want you to handle first.
Agent: Could you add the details?
Customer: This is frustrating, but I appreciate your help and will keep this constructive. I have decided to leave. Please cancel my subscription at the end of this month.
    A: bug_report: The user is reporting something that is broken or producing errors
    B: billing: Charges, invoices, refunds, subscriptions
    C: feature_request: The user is requesting new functionality
    D: account: Login, permissions, profile, security
    Answer: C

### wanli-decisions-v1:train:225422

*Selected as: choice, lowest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 127 / 0.868 |
| Ground truth | C (The premise supports the claim.) |
| S1 prediction | C (The premise supports the claim.) (margin 0.32) |
| R1 prediction | C (The premise supports the claim.) (margin 2.38) |
| Demo answer | B (The premise leaves the claim undetermined.) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / natural_language_inference |

**Query**

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"The howdah is a cushion for the rider.","premise":"An elephant-ride, which allows you to swing through the jungle and grassland while reclining on a cushioned howdah, is a great joy."}
    A: The premise contradicts the claim.
    B: The premise leaves the claim undetermined.
    C: The premise supports the claim.

**Retrieved exemplar** (0717f10cacd9c8ba)

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"She wants to travel to India to meet her grandparents.","premise":"She wants to travel to India to meet her parents."}
    A: The premise contradicts the claim.
    B: The premise leaves the claim undetermined.
    C: The premise supports the claim.
    Answer: B

### mailroom:f04e451a9469:en:relay_invoice:base:about_telecom

*Selected as: noul, highest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 42 / 0.985 |
| Ground truth | A (no) |
| S1 prediction | A (no) (margin 0.81) |
| R1 prediction | A (no) (margin 10.64) |
| Demo answer | A (no) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / evidence |

**Query**

    Question: Is the subject matter of this email a telephone operator or mobile/fixed line service; not a retailer's handset? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns water supply from the water utility.\nWe are a document delivery service for Luma-f04e451a9469; we did not provide the billed service and these are not our own charges.\nWe have issued your invoice. Amount now owed: EUR 838.40.\nReference number: INV-f04e451a9469.","date":"2026-09-20","from":{"display_name":"PaperRelay-f04e451a9469","domain":"delivery-f04e451a9469.example","email":"relay@delivery-f04e451a9469.example"},"subject":"Luma-f04e451a9469 — service correspondence"}}
    A: no
    B: yes

**Retrieved exemplar** (ddc7dba7457b3d93)

    Question: Is the subject matter of this email a telephone operator or mobile/fixed line service; not a retailer's handset? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns water supply from the water utility.\nWe are a document delivery service for Luma-4c808f2025a0; we did not provide the billed service and these are not our own charges.\nWe have issued your invoice. Amount now owed: EUR 504.40.\nReference number: INV-4c808f2025a0.","date":"2026-09-20","from":{"display_name":"PaperRelay-4c808f2025a0","domain":"delivery-4c808f2025a0.example","email":"relay@delivery-4c808f2025a0.example"},"subject":"Luma-4c808f2025a0 — service correspondence"}}
    A: no
    B: yes
    Answer: A

### community-diversity-v2/20260921/games/racing_overtake/0/0/noul

*Selected as: noul, lowest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 108 / 0.904 |
| Ground truth | A (no) |
| S1 prediction | A (no) (margin 0.61) |
| R1 prediction | A (no) (margin 7.91) |
| Demo answer | A (no) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | games / policy |

**Query**

    Question: Candidate P63. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":[{"candidate":"P63","facts":{"completion_ticks":47,"overtake_cancelled":false,"passing_distance":89,"relative_speed":89,"return_lane_clear":false,"straight_remaining":13,"yellow_flag":true}},{"candidate":"P40","facts":{"completion_ticks":45,"overtake_cancelled":false,"passing_distance":0,"relative_speed":89,"return_lane_clear":false,"straight_remaining":2,"yellow_flag":true}},{"candidate":"item_64","facts":{"completion_ticks":39,"overtake_cancelled":true,"passing_distance":1,"relative_speed":5,"return_lane_clear":true,"straight_remaining":3,"yellow_flag":true}},{"candidate":"item_42","facts":{"completion_ticks":41,"overtake_cancelled":false,"passing_distance":21,"relative_speed ...[truncated 1692 bytes]
    A: no
    B: yes

**Retrieved exemplar** (6ad5334e7be36a99)

    Question: Candidate K36. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":{"K15":{"blocked_cells_crossed":0,"completion_ticks":32,"end_cell":15,"moves":18,"path_is_connected":true},"K36":{"blocked_cells_crossed":0,"completion_ticks":38,"end_cell":15,"moves":8,"path_is_connected":null},"K97":{"blocked_cells_crossed":0,"completion_ticks":34,"end_cell":15,"moves":8,"path_is_connected":true},"P25":{"blocked_cells_crossed":89,"completion_ticks":36,"end_cell":21,"moves":2,"path_is_connected":true},"item_72":{"blocked_cells_crossed":0,"completion_ticks":31,"end_cell":14,"moves":0,"path_is_connected":true}},"evidence_policy":"Each numbered requirement must be established by the visible candidate facts. null means unreported, not false. AND is false if any op ...[truncated 857 bytes]
    A: no
    B: yes
    Answer: A

### community-diversity-v2/20260921/shell_history/checksum_verify/74/3/score

*Selected as: score, highest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 54 / 0.976 |
| Ground truth | B (Exactly 1 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.34) |
| R1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.17) |
| Demo answer | B (Exactly 1 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | shell_history / routing |

**Query**

    Question: Candidate K22. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":"item_45\t{\"algorithm\": \"sha256\", \"comparison_performed\": true, \"expected_digest_source\": \"other_value\", \"history_sequence\": 46, \"target_file\": \"weights.bin\"}\nP29\t{\"algorithm\": \"other_value\", \"comparison_performed\": true, \"expected_digest_source\": \"other_value\", \"history_sequence\": 37, \"target_file\": \"weights.bin\"}\nP38\t{\"algorithm\": \"other_value\", \"comparison_performed\": true, \"expected_digest_source\": \"other_value\", \"history_sequence\": 41, \"target_file\": \"weights.bin\"}\nK22\t{\"algorithm\": \"sha256\", \"comparison_performed\": false, \"expected_digest_source\": \"other_value\", \"history_sequence\": 35, \"target_file\": \"ot ...[truncated 1579 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (dd1b1e6d5810e75c)

    Question: Candidate item_25. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":[{"candidate":"item_25","facts":{"history_sequence":19,"operation":"other_value","parsed_path":"other_value","path_argument_count":5,"shell_expansion_present":false}},{"candidate":"item_92","facts":{"history_sequence":21,"operation":"other_value","parsed_path":"other_value","path_argument_count":3,"shell_expansion_present":true}},{"candidate":"P15","facts":{"history_sequence":25,"operation":"inspect","parsed_path":"other_value","path_argument_count":21,"shell_expansion_present":false}},{"candidate":"P29","facts":{"history_sequence":30,"operation":"inspect","parsed_path":"draft notes.txt","path_argument_count":2,"shell_expansion_present":false}},{"candidate":"item_50","facts":{" ...[truncated 1487 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: B

### community-diversity-v2/20260921/contracts/governing_law_venue/73/2/score

*Selected as: score, lowest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 100 / 0.917 |
| Ground truth | B (Exactly 1 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.07) |
| R1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 2.50) |
| Demo answer | B (Exactly 1 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | contracts / policy |

**Query**

    Question: Candidate item_25. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K95":{"amendment_signed_by_both":false,"forum":"commercial_court","in_amendment":false,"in_signed_master":true,"jurisdiction":"Ontario","section_number":25,"superseded":false},"P56":{"amendment_signed_by_both":true,"forum":"commercial_court","in_amendment":false,"in_signed_master":true,"jurisdiction":"Ontario","section_number":23,"superseded":false},"P70":{"amendment_signed_by_both":true,"forum":"commercial_court","in_amendment":true,"in_signed_master":false,"jurisdiction":"Ontario","section_number":22,"superseded":false},"item_25":{"amendment_signed_by_both":true,"forum":"other_value","in_amendment":true,"in_signed_master":true,"jurisdiction":"other_value","section_number":2 ...[truncated 1334 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (cd9ec9cdf340b389)

    Question: Candidate P91. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"P28":{"accessible_records":[],"annual_audits_used":21,"notice_days":7,"permitted_auditors":["customer"],"section_number":35,"suspected_breach_exception":true},"P76":{"accessible_records":[],"annual_audits_used":13,"notice_days":2,"permitted_auditors":["unrelated_item"],"section_number":39,"suspected_breach_exception":true},"P86":{"accessible_records":[],"annual_audits_used":5,"notice_days":34,"permitted_auditors":["customer"],"section_number":37,"suspected_breach_exception":true},"P91":{"accessible_records":["unrelated_customers"],"annual_audits_used":21,"notice_days":1,"permitted_auditors":[],"section_number":45,"suspected_breach_exception":true},"item_29":{"accessible_recor ...[truncated 1606 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: B

## both_incorrect

### customer-control-v1:42:871:category

*Selected as: choice, highest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 33 / 0.991 |
| Ground truth | C (feature_request: The user is requesting new functionality) |
| S1 prediction | A (account: Login, permissions, profile, security) (margin 0.88) |
| R1 prediction | B (billing: Charges, invoices, refunds, subscriptions) (margin 1.01) |
| Demo answer | C (feature_request: The user is requesting new functionality) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / routing |

**Query**

    Question: Determine the broad category of this support ticket. For this controlled task, select the customer's expressly prioritized request. If two requests explicitly have equal priority, allocate equal support to their two categories. A malfunction of an existing feature is bug_report; a payment or refund request is billing; a request for absent functionality is feature_request; an access or profile request is account. Secondary context does not override an explicit priority.
    State: Customer Harbor-bbf8157d35: I would like an offline reading mode, which this product does not currently offer. That is the request I want you to handle first.
Agent: Could you add the details?
Customer: Thanks for checking. I am simply reporting what I observed. I will keep using the service and renew my subscription.
    A: account: Login, permissions, profile, security
    B: billing: Charges, invoices, refunds, subscriptions
    C: feature_request: The user is requesting new functionality
    D: bug_report: The user is reporting something that is broken or producing errors

**Retrieved exemplar** (4c74050bf2489124)

    Question: Determine the broad category of this support ticket. For this controlled task, select the customer's expressly prioritized request. If two requests explicitly have equal priority, allocate equal support to their two categories. A malfunction of an existing feature is bug_report; a payment or refund request is billing; a request for absent functionality is feature_request; an access or profile request is account. Secondary context does not override an explicit priority.
    State: Customer Harbor-58080683e9: I would like an offline reading mode, which this product does not currently offer. That is the request I want you to handle first.
Agent: Could you add the details?
Customer: This is frustrating, but I appreciate your help and will keep this constructive. I have decided to leave. Please cancel my subscription at the end of this month.
    A: bug_report: The user is reporting something that is broken or producing errors
    B: billing: Charges, invoices, refunds, subscriptions
    C: feature_request: The user is requesting new functionality
    D: account: Login, permissions, profile, security
    Answer: C

Relevant differences: R1 changed the prediction, not to the demo label.

### wanli-decisions-v1:train:34718

*Selected as: choice, lowest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 124 / 0.874 |
| Ground truth | A (The premise leaves the claim undetermined.) |
| S1 prediction | B (The premise supports the claim.) (margin 5.18) |
| R1 prediction | B (The premise supports the claim.) (margin 5.63) |
| Demo answer | C (The premise supports the claim.) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / natural_language_inference |

**Query**

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"State-level innovation is a function of state-level characteristics and state-level policies.","premise":"The purpose of the study is to assess the extent to which state-level innovation is a function of state-level characteristics and the extent to which it is a function of state-level policies."}
    A: The premise leaves the claim undetermined.
    B: The premise supports the claim.
    C: The premise contradicts the claim.

**Retrieved exemplar** (4019ce0ed9c77ac4)

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"They are a major contributor to the local economy.","premise":"They have a long history of being a major contributor to the local economy."}
    A: The premise leaves the claim undetermined.
    B: The premise contradicts the claim.
    C: The premise supports the claim.
    Answer: C

### mailroom:1ad8237f7b82:en:relay_invoice:base:about_telecom

*Selected as: noul, highest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 51 / 0.978 |
| Ground truth | B (yes) |
| S1 prediction | A (no) (margin 0.09) |
| R1 prediction | A (no) (margin 0.42) |
| Demo answer | A (no) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / evidence |

**Query**

    Question: Is the subject matter of this email a telephone operator or mobile/fixed line service; not a retailer's handset? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns mobile line service supplied by the telephone operator for a student's phone.\nWe are a document delivery service for Luma-1ad8237f7b82; we did not provide the billed service and these are not our own charges.\nWe have issued your invoice. Amount now owed: EUR 592.40.\nReference number: INV-1ad8237f7b82.","date":"2026-09-20","from":{"display_name":"PaperRelay-1ad8237f7b82","domain":"delivery-1ad8237f7b82.example","email":"relay@delivery-1ad8237f7b82.example"},"subject":"Luma-1ad8237f7b82 — service correspondence"}}
    A: no
    B: yes

**Retrieved exemplar** (125431d971a71dfe)

    Question: Is the subject matter of this email a telephone operator or mobile/fixed line service; not a retailer's handset? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns a phone handset sold by a general retailer, without any telephone service.\nWe are a document delivery service for Luma-d8b6d1f483ea; we did not provide the billed service and these are not our own charges.\nWe have issued your invoice. Amount now owed: EUR 398.40.\nReference number: INV-d8b6d1f483ea.","date":"2026-09-20","from":{"display_name":"PaperRelay-d8b6d1f483ea","domain":"delivery-d8b6d1f483ea.example","email":"relay@delivery-d8b6d1f483ea.example"},"subject":"Luma-d8b6d1f483ea — service correspondence"}}
    A: no
    B: yes
    Answer: A

### community-diversity-v2/20260921/games/racing_overtake/16/2/noul

*Selected as: noul, lowest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 106 / 0.907 |
| Ground truth | B (yes) |
| S1 prediction | A (no) (margin 0.11) |
| R1 prediction | A (no) (margin 6.68) |
| Demo answer | A (no) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | games / policy |

**Query**

    Question: Candidate K82. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":{"K82":{"completion_ticks":101,"overtake_cancelled":true,"passing_distance":3,"relative_speed":55,"return_lane_clear":true,"straight_remaining":13,"yellow_flag":true},"P43":{"completion_ticks":104,"overtake_cancelled":true,"passing_distance":8,"relative_speed":2,"return_lane_clear":true,"straight_remaining":13,"yellow_flag":false},"P76":{"completion_ticks":108,"overtake_cancelled":true,"passing_distance":8,"relative_speed":2,"return_lane_clear":null,"straight_remaining":13,"yellow_flag":false},"item_17":{"completion_ticks":102,"overtake_cancelled":true,"passing_distance":1,"relative_speed":10,"return_lane_clear":true,"straight_remaining":8,"yellow_flag":true},"item_43":{"comple ...[truncated 1537 bytes]
    A: no
    B: yes

**Retrieved exemplar** (386ace2e9a24f4fe)

    Question: Candidate K31. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":[{"candidate":"K69","facts":{"blocked_cells_crossed":null,"completion_ticks":92,"end_cell":15,"moves":3,"path_is_connected":true}},{"candidate":"K97","facts":{"blocked_cells_crossed":0,"completion_ticks":88,"end_cell":15,"moves":3,"path_is_connected":true}},{"candidate":"item_51","facts":{"blocked_cells_crossed":0,"completion_ticks":86,"end_cell":15,"moves":3,"path_is_connected":true}},{"candidate":"K31","facts":{"blocked_cells_crossed":144,"completion_ticks":90,"end_cell":5,"moves":1,"path_is_connected":true}},{"candidate":"P30","facts":{"blocked_cells_crossed":1,"completion_ticks":85,"end_cell":15,"moves":18,"path_is_connected":true}}],"evidence_policy":"Each numbered require ...[truncated 970 bytes]
    A: no
    B: yes
    Answer: A

### community-diversity-v2/20260921/shell_history/checksum_verify/72/3/score

*Selected as: score, highest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 54 / 0.976 |
| Ground truth | D (Exactly 3 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.70) |
| R1 prediction | C (Exactly 2 of the 4 numbered requirements are established.) (margin 1.47) |
| Demo answer | C (Exactly 2 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | shell_history / routing |

**Query**

    Question: Candidate item_56. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":[{"candidate":"item_56","facts":{"algorithm":"sha256","comparison_performed":false,"expected_digest_source":"trusted_manifest","history_sequence":54,"target_file":"weights.bin"}},{"candidate":"K98","facts":{"algorithm":"other_value","comparison_performed":true,"expected_digest_source":"other_value","history_sequence":50,"target_file":"weights.bin"}},{"candidate":"K72","facts":{"algorithm":"sha256","comparison_performed":true,"expected_digest_source":"trusted_manifest","history_sequence":52,"target_file":null}},{"candidate":"item_60","facts":{"algorithm":"sha256","comparison_performed":true,"expected_digest_source":"trusted_manifest","history_sequence":48,"target_file":null}},{" ...[truncated 1582 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (f69e15ccabfc06d8)

    Question: Candidate item_51. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":[{"candidate":"P70","facts":{"history_sequence":46,"operation":"inspect","parsed_path":"report[final].csv","path_argument_count":1,"shell_expansion_present":null}},{"candidate":"K11","facts":{"history_sequence":42,"operation":"other_value","parsed_path":"other_value","path_argument_count":55,"shell_expansion_present":true}},{"candidate":"P69","facts":{"history_sequence":52,"operation":"inspect","parsed_path":"report[final].csv","path_argument_count":1,"shell_expansion_present":false}},{"candidate":"P87","facts":{"history_sequence":50,"operation":"inspect","parsed_path":"report[final].csv","path_argument_count":1,"shell_expansion_present":false}},{"candidate":"item_51","facts":{ ...[truncated 1517 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: C

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/contracts/governing_law_venue/37/2/score

*Selected as: score, lowest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 107 / 0.906 |
| Ground truth | E (Exactly 4 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.10) |
| R1 prediction | D (Exactly 3 of the 4 numbered requirements are established.) (margin 2.50) |
| Demo answer | D (Exactly 3 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | contracts / policy |

**Query**

    Question: Candidate K30. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K30":{"amendment_signed_by_both":false,"forum":"local_courts","in_amendment":true,"in_signed_master":true,"jurisdiction":"Ontario","section_number":69,"superseded":false},"K76":{"amendment_signed_by_both":false,"forum":null,"in_amendment":true,"in_signed_master":true,"jurisdiction":"Ontario","section_number":73,"superseded":false},"K82":{"amendment_signed_by_both":false,"forum":"other_value","in_amendment":true,"in_signed_master":true,"jurisdiction":"Ontario","section_number":77,"superseded":true},"P40":{"amendment_signed_by_both":true,"forum":"local_courts","in_amendment":false,"in_signed_master":true,"jurisdiction":"Ontario","section_number":66,"superseded":false},"item_31" ...[truncated 1648 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (f8d68d07af297c61)

    Question: Candidate item_92. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":"P82\t{\"clause_kind\": \"noncompete\", \"executed\": true, \"restriction_months\": 7, \"section_number\": 56, \"specified_terms\": [\"territory\", \"unrelated_item\"], \"waived\": false}\nP60\t{\"clause_kind\": \"noncompete\", \"executed\": true, \"restriction_months\": 1, \"section_number\": 60, \"specified_terms\": [\"unrelated_item\"], \"waived\": true}\nK67\t{\"clause_kind\": \"noncompete\", \"executed\": null, \"restriction_months\": 6, \"section_number\": 62, \"specified_terms\": [\"territory\"], \"waived\": false}\nitem_92\t{\"clause_kind\": \"noncompete\", \"executed\": false, \"restriction_months\": 3, \"section_number\": 55, \"specified_terms\": [\"territory\", \"unr ...[truncated 1237 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: D

Relevant differences: R1 moved to the demonstration's answer label.

