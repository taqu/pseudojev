# R1 — Representative examples

Per transition and kind: the highest-similarity (smallest Hamming distance) and the lowest-similarity case, chosen deterministically (ties broken by file order). Successes and failures are shown alike.

## rescue

### community-diversity-v2/20260921/support/damaged_order/0/1/score

*Selected as: score, highest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 79 / 0.948 |
| Ground truth | C (Exactly 2 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 0.88) |
| R1 prediction | C (Exactly 2 of the 4 numbered requirements are established.) (margin 2.06) |
| Demo answer | C (Exactly 2 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | support / policy |

**Query**

    Question: Candidate P27. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":[{"candidate":"item_73","facts":{"claim_days":28,"damage_evidence_received":true,"electrical_hazard":false,"refund_available":false,"replacement_in_stock":true,"response_minutes":42,"specialist_authorized":true}},{"candidate":"K46","facts":{"claim_days":31,"damage_evidence_received":true,"electrical_hazard":true,"refund_available":true,"replacement_in_stock":true,"response_minutes":40,"specialist_authorized":true}},{"candidate":"item_43","facts":{"claim_days":2,"damage_evidence_received":false,"electrical_hazard":true,"refund_available":false,"replacement_in_stock":false,"response_minutes":48,"specialist_authorized":false}},{"candidate":"P27","facts":{"claim_days":2,"damage_evi ...[truncated 1679 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (804d37c8ff218266)

    Question: Candidate P30. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K54":{"accidental_cover":true,"diagnosis":"consumable_wear","liquid_damage":false,"months_owned":13,"response_minutes":75,"serial_matches_receipt":false},"K78":{"accidental_cover":true,"diagnosis":"other_value","liquid_damage":false,"months_owned":25,"response_minutes":64,"serial_matches_receipt":true},"P30":{"accidental_cover":false,"diagnosis":"consumable_wear","liquid_damage":true,"months_owned":0,"response_minutes":73,"serial_matches_receipt":true},"P88":{"accidental_cover":false,"diagnosis":null,"liquid_damage":false,"months_owned":21,"response_minutes":67,"serial_matches_receipt":true},"P89":{"accidental_cover":true,"diagnosis":"consumable_wear","liquid_damage":true,"mo ...[truncated 1406 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: C

Relevant differences: R1 moved to the demonstration's answer label.

## harm

### community-diversity-v2/20260921/support/damaged_order/0/3/choice

*Selected as: choice, highest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 84 / 0.942 |
| Ground truth | G (abstain) |
| S1 prediction | G (abstain) (margin 0.52) |
| R1 prediction | B (K46) (margin 4.29) |
| Demo answer | D (K49) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | support / policy |

**Query**

    Question: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
    State: {"candidates":[{"candidate":"item_73","facts":{"claim_days":28,"damage_evidence_received":null,"electrical_hazard":false,"refund_available":false,"replacement_in_stock":true,"response_minutes":42,"specialist_authorized":true}},{"candidate":"K46","facts":{"claim_days":31,"damage_evidence_received":true,"electrical_hazard":true,"refund_available":true,"replacement_in_stock":true,"response_minutes":40,"specialist_authorized":true}},{"candidate":"item_43","facts":{"claim_days":2,"damage_evidence_received":false,"electrical_hazard":true,"refund_available":false,"replacement_in_stock":false,"response_minutes":48,"specialist_authorized":false}},{"candidate":"P27","facts":{"claim_days":2,"damage_evi ...[truncated 1679 bytes]
    A: K83
    B: K46
    C: P27
    D: item_21
    E: item_73
    F: item_43
    G: abstain

**Retrieved exemplar** (95ba7a22fb1c447d)

    Question: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
    State: {"candidates":[{"candidate":"item_78","facts":{"administrator_account":false,"manager_approval":true,"recovery_channel_compromised":false,"response_minutes":96,"verified_identity_factors":0,"verified_ownership_factors":0}},{"candidate":"K26","facts":{"administrator_account":true,"manager_approval":true,"recovery_channel_compromised":false,"response_minutes":94,"verified_identity_factors":11,"verified_ownership_factors":1}},{"candidate":"P85","facts":{"administrator_account":true,"manager_approval":true,"recovery_channel_compromised":null,"response_minutes":98,"verified_identity_factors":11,"verified_ownership_factors":1}},{"candidate":"P37","facts":{"administrator_account":true,"manager_appr ...[truncated 1830 bytes]
    A: abstain
    B: P37
    C: P33
    D: K49
    E: item_78
    F: K26
    G: K38
    H: P85
    Answer: D

Relevant differences: R1 changed the prediction, not to the demo label; option counts differ (7 vs 8).

## both_correct

### community-diversity-v2/20260921/support/damaged_order/0/0/noul

*Selected as: noul, highest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 92 / 0.930 |
| Ground truth | A (no) |
| S1 prediction | A (no) (margin 0.54) |
| R1 prediction | A (no) (margin 6.49) |
| Demo answer | B (yes) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | support / policy |

**Query**

    Question: Candidate item_43. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":[{"candidate":"item_73","facts":{"claim_days":28,"damage_evidence_received":true,"electrical_hazard":false,"refund_available":false,"replacement_in_stock":true,"response_minutes":42,"specialist_authorized":true}},{"candidate":"K46","facts":{"claim_days":29,"damage_evidence_received":true,"electrical_hazard":true,"refund_available":true,"replacement_in_stock":true,"response_minutes":40,"specialist_authorized":true}},{"candidate":"item_43","facts":{"claim_days":2,"damage_evidence_received":false,"electrical_hazard":true,"refund_available":false,"replacement_in_stock":false,"response_minutes":48,"specialist_authorized":false}},{"candidate":"P27","facts":{"claim_days":2,"damage_evi ...[truncated 1679 bytes]
    A: no
    B: yes

**Retrieved exemplar** (ea073ff8743153f7)

    Question: Candidate K27. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":"K60\t{\"days_since_delivery\": 5, \"defect_confirmed\": true, \"payment_settled\": true, \"refund_already_paid\": false, \"response_minutes\": 87, \"seal_intact\": false}\nitem_39\t{\"days_since_delivery\": 31, \"defect_confirmed\": true, \"payment_settled\": true, \"refund_already_paid\": true, \"response_minutes\": 95, \"seal_intact\": false}\nitem_88\t{\"days_since_delivery\": 28, \"defect_confirmed\": true, \"payment_settled\": false, \"refund_already_paid\": true, \"response_minutes\": 89, \"seal_intact\": false}\nK29\t{\"days_since_delivery\": 1, \"defect_confirmed\": true, \"payment_settled\": true, \"refund_already_paid\": false, \"response_minutes\": 84, \"seal_intact ...[truncated 1550 bytes]
    A: no
    B: yes
    Answer: B

### community-diversity-v2/20260921/support/damaged_order/4/0/noul

*Selected as: noul, lowest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 93 / 0.929 |
| Ground truth | A (no) |
| S1 prediction | A (no) (margin 0.71) |
| R1 prediction | A (no) (margin 4.90) |
| Demo answer | B (yes) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | support / policy |

**Query**

    Question: Candidate P47. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":{"K28":{"claim_days":89,"damage_evidence_received":true,"electrical_hazard":true,"refund_available":false,"replacement_in_stock":false,"response_minutes":50,"specialist_authorized":false},"K92":{"claim_days":null,"damage_evidence_received":true,"electrical_hazard":false,"refund_available":true,"replacement_in_stock":true,"response_minutes":46,"specialist_authorized":false},"K93":{"claim_days":3,"damage_evidence_received":true,"electrical_hazard":false,"refund_available":true,"replacement_in_stock":true,"response_minutes":42,"specialist_authorized":false},"K97":{"claim_days":34,"damage_evidence_received":false,"electrical_hazard":false,"refund_available":true,"replacement_in_sto ...[truncated 1733 bytes]
    A: no
    B: yes

**Retrieved exemplar** (ea073ff8743153f7)

    Question: Candidate K27. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":"K60\t{\"days_since_delivery\": 5, \"defect_confirmed\": true, \"payment_settled\": true, \"refund_already_paid\": false, \"response_minutes\": 87, \"seal_intact\": false}\nitem_39\t{\"days_since_delivery\": 31, \"defect_confirmed\": true, \"payment_settled\": true, \"refund_already_paid\": true, \"response_minutes\": 95, \"seal_intact\": false}\nitem_88\t{\"days_since_delivery\": 28, \"defect_confirmed\": true, \"payment_settled\": false, \"refund_already_paid\": true, \"response_minutes\": 89, \"seal_intact\": false}\nK29\t{\"days_since_delivery\": 1, \"defect_confirmed\": true, \"payment_settled\": true, \"refund_already_paid\": false, \"response_minutes\": 84, \"seal_intact ...[truncated 1550 bytes]
    A: no
    B: yes
    Answer: B

## both_incorrect

### community-diversity-v2/20260921/support/damaged_order/0/0/choice

*Selected as: choice, highest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 84 / 0.942 |
| Ground truth | G (K46) |
| S1 prediction | A (item_21) (margin 0.47) |
| R1 prediction | B (item_73) (margin 2.52) |
| Demo answer | D (K49) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | support / policy |

**Query**

    Question: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
    State: {"candidates":[{"candidate":"item_73","facts":{"claim_days":28,"damage_evidence_received":true,"electrical_hazard":false,"refund_available":false,"replacement_in_stock":true,"response_minutes":42,"specialist_authorized":true}},{"candidate":"K46","facts":{"claim_days":29,"damage_evidence_received":true,"electrical_hazard":true,"refund_available":true,"replacement_in_stock":true,"response_minutes":40,"specialist_authorized":true}},{"candidate":"item_43","facts":{"claim_days":2,"damage_evidence_received":false,"electrical_hazard":true,"refund_available":false,"replacement_in_stock":false,"response_minutes":48,"specialist_authorized":false}},{"candidate":"P27","facts":{"claim_days":2,"damage_evi ...[truncated 1679 bytes]
    A: item_21
    B: item_73
    C: K83
    D: item_43
    E: P27
    F: abstain
    G: K46

**Retrieved exemplar** (95ba7a22fb1c447d)

    Question: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
    State: {"candidates":[{"candidate":"item_78","facts":{"administrator_account":false,"manager_approval":true,"recovery_channel_compromised":false,"response_minutes":96,"verified_identity_factors":0,"verified_ownership_factors":0}},{"candidate":"K26","facts":{"administrator_account":true,"manager_approval":true,"recovery_channel_compromised":false,"response_minutes":94,"verified_identity_factors":11,"verified_ownership_factors":1}},{"candidate":"P85","facts":{"administrator_account":true,"manager_approval":true,"recovery_channel_compromised":null,"response_minutes":98,"verified_identity_factors":11,"verified_ownership_factors":1}},{"candidate":"P37","facts":{"administrator_account":true,"manager_appr ...[truncated 1830 bytes]
    A: abstain
    B: P37
    C: P33
    D: K49
    E: item_78
    F: K26
    G: K38
    H: P85
    Answer: D

Relevant differences: R1 changed the prediction, not to the demo label; option counts differ (7 vs 8).

### community-diversity-v2/20260921/support/damaged_order/4/0/choice

*Selected as: choice, lowest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 85 / 0.940 |
| Ground truth | E (item_78) |
| S1 prediction | A (abstain) (margin 0.66) |
| R1 prediction | G (P47) (margin 0.91) |
| Demo answer | D (K49) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | support / policy |

**Query**

    Question: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
    State: {"candidates":{"K28":{"claim_days":89,"damage_evidence_received":true,"electrical_hazard":true,"refund_available":false,"replacement_in_stock":false,"response_minutes":50,"specialist_authorized":false},"K92":{"claim_days":null,"damage_evidence_received":true,"electrical_hazard":false,"refund_available":true,"replacement_in_stock":true,"response_minutes":46,"specialist_authorized":false},"K93":{"claim_days":3,"damage_evidence_received":true,"electrical_hazard":false,"refund_available":true,"replacement_in_stock":true,"response_minutes":42,"specialist_authorized":false},"K97":{"claim_days":34,"damage_evidence_received":false,"electrical_hazard":false,"refund_available":true,"replacement_in_sto ...[truncated 1733 bytes]
    A: abstain
    B: K92
    C: K93
    D: K97
    E: item_78
    F: K28
    G: P47
    H: item_39

**Retrieved exemplar** (95ba7a22fb1c447d)

    Question: Which candidate is supported by all four requirements? Apply the stated tie-break; abstain if none is fully supported.
    State: {"candidates":[{"candidate":"item_78","facts":{"administrator_account":false,"manager_approval":true,"recovery_channel_compromised":false,"response_minutes":96,"verified_identity_factors":0,"verified_ownership_factors":0}},{"candidate":"K26","facts":{"administrator_account":true,"manager_approval":true,"recovery_channel_compromised":false,"response_minutes":94,"verified_identity_factors":11,"verified_ownership_factors":1}},{"candidate":"P85","facts":{"administrator_account":true,"manager_approval":true,"recovery_channel_compromised":null,"response_minutes":98,"verified_identity_factors":11,"verified_ownership_factors":1}},{"candidate":"P37","facts":{"administrator_account":true,"manager_appr ...[truncated 1830 bytes]
    A: abstain
    B: P37
    C: P33
    D: K49
    E: item_78
    F: K26
    G: K38
    H: P85
    Answer: D

Relevant differences: R1 changed the prediction, not to the demo label.

### community-diversity-v2/20260921/support/damaged_order/0/3/score

*Selected as: score, highest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 78 / 0.950 |
| Ground truth | D (Exactly 3 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 0.57) |
| R1 prediction | C (Exactly 2 of the 4 numbered requirements are established.) (margin 3.60) |
| Demo answer | C (Exactly 2 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | support / policy |

**Query**

    Question: Candidate K83. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":[{"candidate":"item_73","facts":{"claim_days":28,"damage_evidence_received":null,"electrical_hazard":false,"refund_available":false,"replacement_in_stock":true,"response_minutes":42,"specialist_authorized":true}},{"candidate":"K46","facts":{"claim_days":31,"damage_evidence_received":true,"electrical_hazard":true,"refund_available":true,"replacement_in_stock":true,"response_minutes":40,"specialist_authorized":true}},{"candidate":"item_43","facts":{"claim_days":2,"damage_evidence_received":false,"electrical_hazard":true,"refund_available":false,"replacement_in_stock":false,"response_minutes":48,"specialist_authorized":false}},{"candidate":"P27","facts":{"claim_days":2,"damage_evi ...[truncated 1679 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (804d37c8ff218266)

    Question: Candidate P30. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K54":{"accidental_cover":true,"diagnosis":"consumable_wear","liquid_damage":false,"months_owned":13,"response_minutes":75,"serial_matches_receipt":false},"K78":{"accidental_cover":true,"diagnosis":"other_value","liquid_damage":false,"months_owned":25,"response_minutes":64,"serial_matches_receipt":true},"P30":{"accidental_cover":false,"diagnosis":"consumable_wear","liquid_damage":true,"months_owned":0,"response_minutes":73,"serial_matches_receipt":true},"P88":{"accidental_cover":false,"diagnosis":null,"liquid_damage":false,"months_owned":21,"response_minutes":67,"serial_matches_receipt":true},"P89":{"accidental_cover":true,"diagnosis":"consumable_wear","liquid_damage":true,"mo ...[truncated 1406 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: C

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/support/damaged_order/4/1/score

*Selected as: score, lowest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 79 / 0.948 |
| Ground truth | D (Exactly 3 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.05) |
| R1 prediction | C (Exactly 2 of the 4 numbered requirements are established.) (margin 4.84) |
| Demo answer | C (Exactly 2 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | support / policy |

**Query**

    Question: Candidate item_78. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K28":{"claim_days":89,"damage_evidence_received":true,"electrical_hazard":true,"refund_available":false,"replacement_in_stock":false,"response_minutes":50,"specialist_authorized":false},"K92":{"claim_days":null,"damage_evidence_received":true,"electrical_hazard":false,"refund_available":true,"replacement_in_stock":true,"response_minutes":46,"specialist_authorized":false},"K93":{"claim_days":3,"damage_evidence_received":true,"electrical_hazard":false,"refund_available":true,"replacement_in_stock":true,"response_minutes":42,"specialist_authorized":false},"K97":{"claim_days":34,"damage_evidence_received":false,"electrical_hazard":false,"refund_available":true,"replacement_in_sto ...[truncated 1734 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (0479891c35888738)

    Question: Candidate P27. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K13":{"accidental_cover":true,"diagnosis":"other_value","liquid_damage":true,"months_owned":89,"response_minutes":61,"serial_matches_receipt":false},"P12":{"accidental_cover":false,"diagnosis":"other_value","liquid_damage":false,"months_owned":null,"response_minutes":59,"serial_matches_receipt":true},"P27":{"accidental_cover":true,"diagnosis":"other_value","liquid_damage":false,"months_owned":55,"response_minutes":63,"serial_matches_receipt":false},"P56":{"accidental_cover":true,"diagnosis":"other_value","liquid_damage":true,"months_owned":13,"response_minutes":52,"serial_matches_receipt":true},"P78":{"accidental_cover":true,"diagnosis":"consumable_wear","liquid_damage":false ...[truncated 1410 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: C

Relevant differences: R1 moved to the demonstration's answer label.

