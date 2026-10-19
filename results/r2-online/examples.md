# R1 — Representative examples

Per transition and kind: the highest-similarity (smallest Hamming distance) and the lowest-similarity case, chosen deterministically (ties broken by file order). Successes and failures are shown alike.

## rescue

### wanli-decisions-v1:train:192221

*Selected as: choice, highest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 47 / 0.982 |
| Ground truth | B (The premise supports the claim.) |
| S1 prediction | A (The premise leaves the claim undetermined.) (margin 2.90) |
| R1 prediction | B (The premise supports the claim.) (margin 1.41) |
| Demo answer | B (The premise contradicts the claim.) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / natural_language_inference |

**Query**

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"We are not getting anything out of this.","premise":"It's not like we're getting anything out of this."}
    A: The premise leaves the claim undetermined.
    B: The premise supports the claim.
    C: The premise contradicts the claim.

**Retrieved exemplar** (a0e668de4d53d548)

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"I don't think we're going to get anything out of it.","premise":"They say that we're not going to get anything out of it, but I don't think that's true."}
    A: The premise leaves the claim undetermined.
    B: The premise contradicts the claim.
    C: The premise supports the claim.
    Answer: B

Relevant differences: R1 moved to the demonstration's answer label.

### wanli-decisions-v1:train:311044

*Selected as: choice, lowest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 108 / 0.904 |
| Ground truth | C (The premise contradicts the claim.) |
| S1 prediction | B (The premise supports the claim.) (margin 5.18) |
| R1 prediction | C (The premise contradicts the claim.) (margin 2.50) |
| Demo answer | C (The premise leaves the claim undetermined.) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / natural_language_inference |

**Query**

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"The big companies that control the most land and the most water, such as the American Farm Bureau Federation, the National Cattlemen's Beef Association, and the National Farmers Union, have been fighting to give up their influence and power.","premise":"The big companies that control the most land and the most water, such as the American Farm Bureau Federation, the National Cattlemen's Beef Association, and the National Farmers Union, have been fighting to maintain their influence and power."}
    A: The premise leaves the claim undetermined.
    B: The premise supports the claim.
    C: The premise contradicts the claim.

**Retrieved exemplar** (9d14798491feac01)

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"The view of some American farmers is the same as the one that was once taken by most American farmers.","premise":"This view is the same as the one that was once taken by most American farmers."}
    A: The premise supports the claim.
    B: The premise contradicts the claim.
    C: The premise leaves the claim undetermined.
    Answer: C

Relevant differences: R1 moved to the demonstration's answer label.

### mailroom:c930d77401db:en:newsletter:base:about_banking

*Selected as: noul, highest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 45 / 0.983 |
| Ground truth | A (no) |
| S1 prediction | B (yes) (margin 0.52) |
| R1 prediction | A (no) (margin 8.08) |
| Demo answer | A (no) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / evidence |

**Query**

    Question: Is the subject matter of this email a bank, payment card or financial institution's own products; not a utility merely paid through a bank? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns an insurance policy supplied by the insurer.\nOur monthly informational bulletin contains service news only, with no offer and no payment owed.","date":"2026-09-20","from":{"display_name":"Luma-c930d77401db","domain":"luma-c930d77401db.example","email":"contact@luma-c930d77401db.example"},"subject":"Luma-c930d77401db — service correspondence"}}
    A: no
    B: yes

**Retrieved exemplar** (a5cd132a0780b782)

    Question: Is the subject matter of this email a bank, payment card or financial institution's own products; not a utility merely paid through a bank? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns an insurance policy supplied by the insurer.\nOur monthly informational bulletin contains service news only, with no offer and no payment owed.","date":"2026-09-20","from":{"display_name":"Luma-80e5ae1d4b19","domain":"luma-80e5ae1d4b19.example","email":"contact@luma-80e5ae1d4b19.example"},"subject":"Luma-80e5ae1d4b19 — service correspondence"}}
    A: no
    B: yes
    Answer: A

Relevant differences: R1 moved to the demonstration's answer label.

### customer-control-v1:42:285:has_reproducible_steps

*Selected as: noul, lowest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 107 / 0.906 |
| Ground truth | A (no) |
| S1 prediction | B (yes) (margin 3.60) |
| R1 prediction | A (no) (margin 1.53) |
| Demo answer | A (no) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / evidence |

**Query**

    Question: The user describes specific steps to reproduce the issue. For this controlled task, require at least two ordered user actions and the observed result. A vague report, a resolved historical issue, or an instruction to support staff is insufficient.
    State: Customer Harbor-4dd7cf41d6: I would like an offline reading mode, which this product does not currently offer. I need access restored after changing the email address used to sign in. Both of those requests matter equally to me; I have not chosen one to address first.
Agent: Could you add the details?
Customer: Thanks for checking. I am simply reporting what I observed. Whether I renew depends on whether this is fixed before Friday.
    A: no
    B: yes

**Retrieved exemplar** (1c2c99f262e50196)

    Question: The user describes specific steps to reproduce the issue. For this controlled task, require at least two ordered user actions and the observed result. A vague report, a resolved historical issue, or an instruction to support staff is insufficient.
    State: Conversation with account holder Harbor-a3d7f1492e.
Customer message: I need access restored after changing the email address used to sign in. I would like an offline reading mode, which this product does not currently offer. Both of those requests matter equally to me; I have not chosen one to address first.
Support: Please tell us more.
Customer response: I am furious. This treatment is outrageous and completely unacceptable! I will keep using the service and renew my subscription.
    A: no
    B: yes
    Answer: A

Relevant differences: R1 moved to the demonstration's answer label.

### customer-control-v1:42:88:bug_severity

*Selected as: score, highest similarity*

| | |
|---|---|
| Kind | score (3 levels) |
| Hamming / est. cosine | 61 / 0.969 |
| Ground truth | B (Broken or degraded feature; workaround exists) |
| S1 prediction | A (Cosmetic; no impact to functionality) (margin 1.29) |
| R1 prediction | B (Broken or degraded feature; workaround exists) (margin 0.80) |
| Demo answer | B (Broken or degraded feature; workaround exists) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / rubric |

**Query**

    Question: How severe is the reported issue? Use the stated functional impact and the existence of a workaround. Use no functional impact when no malfunction is described. In this synthetic control, if a broken feature is described but the availability of a workaround is explicitly unknown, the two latent possibilities (workaround exists; no workaround exists) are equally likely.
    State: Customer Harbor-b1a63d8326: Exporting a report fails, but I can download the CSV and finish in the desktop app. That is the request I want you to handle first.
Agent: Could you add the details?
Customer: It happens sometimes, but I cannot recall the sequence that led to it. This is frustrating, but I appreciate your help and will keep this constructive. I will keep using the service and renew my subscription.
    A: Cosmetic; no impact to functionality
    B: Broken or degraded feature; workaround exists
    C: Blocking issue; no workaround exists

**Retrieved exemplar** (cec67aae953674ac)

    Question: How severe is the reported issue? Use the stated functional impact and the existence of a workaround. Use no functional impact when no malfunction is described. In this synthetic control, if a broken feature is described but the availability of a workaround is explicitly unknown, the two latent possibilities (workaround exists; no workaround exists) are equally likely.
    State: Customer Harbor-43d231017b: My card was charged twice for the same monthly invoice. Exporting a report fails, but I can download the CSV and finish in the desktop app. Both of those requests matter equally to me; I have not chosen one to address first.
Agent: Could you add the details?
Customer: I open a saved report, then click Export; the download panel disappears and nothing is saved. Last month's refund has arrived. Today I only want the invoice explained. This is frustrating, but I appreciate your help and will keep this constructive. I will keep using the service and renew my subscription.
    A: Cosmetic; no impact to functionality
    B: Broken or degraded feature; workaround exists
    C: Blocking issue; no workaround exists
    Answer: B

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/contracts/governing_law_venue/56/3/score

*Selected as: score, lowest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 100 / 0.917 |
| Ground truth | D (Exactly 3 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.03) |
| R1 prediction | D (Exactly 3 of the 4 numbered requirements are established.) (margin 1.70) |
| Demo answer | D (Exactly 3 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | contracts / policy |

**Query**

    Question: Candidate K19. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":"K19\t{\"amendment_signed_by_both\": false, \"forum\": \"other_value\", \"in_amendment\": false, \"in_signed_master\": true, \"jurisdiction\": \"Ontario\", \"section_number\": 91, \"superseded\": false}\nK37\t{\"amendment_signed_by_both\": true, \"forum\": \"arbitration\", \"in_amendment\": false, \"in_signed_master\": true, \"jurisdiction\": null, \"section_number\": 94, \"superseded\": false}\nK67\t{\"amendment_signed_by_both\": true, \"forum\": \"other_value\", \"in_amendment\": false, \"in_signed_master\": true, \"jurisdiction\": \"other_value\", \"section_number\": 102, \"superseded\": false}\nP72\t{\"amendment_signed_by_both\": true, \"forum\": \"arbitration\", \"in_amend ...[truncated 1849 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (5ae312f269153c16)

    Question: Candidate item_16. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":"P20\t{\"clause_kind\": \"noncompete\", \"executed\": false, \"restriction_months\": 144, \"section_number\": 90, \"specified_terms\": [\"territory\", \"unrelated_item\"], \"waived\": false}\nitem_16\t{\"clause_kind\": \"noncompete\", \"executed\": false, \"restriction_months\": 12, \"section_number\": 85, \"specified_terms\": [\"territory\"], \"waived\": false}\nK30\t{\"clause_kind\": \"noncompete\", \"executed\": true, \"restriction_months\": 8, \"section_number\": 92, \"specified_terms\": null, \"waived\": false}\nitem_32\t{\"clause_kind\": \"noncompete\", \"executed\": true, \"restriction_months\": 13, \"section_number\": 86, \"specified_terms\": [\"territory\"], \"waived\" ...[truncated 1385 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: D

Relevant differences: R1 moved to the demonstration's answer label.

## harm

### wanli-decisions-v1:train:298043

*Selected as: choice, highest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 48 / 0.981 |
| Ground truth | C (The premise supports the claim.) |
| S1 prediction | C (The premise supports the claim.) (margin 1.12) |
| R1 prediction | B (The premise contradicts the claim.) (margin 0.55) |
| Demo answer | B (The premise leaves the claim undetermined.) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / natural_language_inference |

**Query**

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"You are right.","premise":"No, you're right."}
    A: The premise leaves the claim undetermined.
    B: The premise contradicts the claim.
    C: The premise supports the claim.

**Retrieved exemplar** (cc10aeac04722797)

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"Yes, you're wrong.","premise":"No, you're wrong."}
    A: The premise contradicts the claim.
    B: The premise leaves the claim undetermined.
    C: The premise supports the claim.
    Answer: B

Relevant differences: R1 moved to the demonstration's answer label.

### wanli-decisions-v1:train:102662

*Selected as: choice, lowest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 107 / 0.906 |
| Ground truth | B (The premise leaves the claim undetermined.) |
| S1 prediction | B (The premise leaves the claim undetermined.) (margin 5.12) |
| R1 prediction | C (The premise contradicts the claim.) (margin 2.33) |
| Demo answer | C (The premise contradicts the claim.) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / natural_language_inference |

**Query**

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"James Dean is an archetypal hero.","premise":"The main character, played by James Dean, is an archetypal anti-hero, the kind of guy who doesn't really exist."}
    A: The premise supports the claim.
    B: The premise leaves the claim undetermined.
    C: The premise contradicts the claim.

**Retrieved exemplar** (2aa777a7813cd84f)

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"In the film, he is a hero, a former cop who has killed a man and is now a gangster.","premise":"In the film, he is an anti-hero, a gangster who can't quite keep his hand out of the till. In the novel, he is a hero, a former cop who has killed a man and is now a gangster."}
    A: The premise leaves the claim undetermined.
    B: The premise supports the claim.
    C: The premise contradicts the claim.
    Answer: C

Relevant differences: R1 moved to the demonstration's answer label.

### mailroom:1ad8237f7b82:en:newsletter:base:about_telecom

*Selected as: noul, highest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 56 / 0.974 |
| Ground truth | B (yes) |
| S1 prediction | B (yes) (margin 0.13) |
| R1 prediction | A (no) (margin 1.97) |
| Demo answer | A (no) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / evidence |

**Query**

    Question: Is the subject matter of this email a telephone operator or mobile/fixed line service; not a retailer's handset? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns mobile line service supplied by the telephone operator for a student's phone.\nOur monthly informational bulletin contains service news only, with no offer and no payment owed.","date":"2026-09-20","from":{"display_name":"Luma-1ad8237f7b82","domain":"luma-1ad8237f7b82.example","email":"contact@luma-1ad8237f7b82.example"},"subject":"Luma-1ad8237f7b82 — service correspondence"}}
    A: no
    B: yes

**Retrieved exemplar** (ffcd00207240200d)

    Question: Is the subject matter of this email a telephone operator or mobile/fixed line service; not a retailer's handset? Inspect email.from, email.subject and email.body. This concerns subject matter regardless of whether money is owed; promotions and newsletters can qualify.
    State: {"email":{"body":"This message concerns water supply from the water utility.\nWe, Luma-c83255d00198, are the provider and issue our own charges.\nYour issued invoice is available at https://documents.example/c83255d00198. Open the invoice to see its amount; no amount is stated in this email.","date":"2026-09-20","from":{"display_name":"Luma-c83255d00198","domain":"luma-c83255d00198.example","email":"contact@luma-c83255d00198.example"},"subject":"Luma-c83255d00198 — service correspondence"}}
    A: no
    B: yes
    Answer: A

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/contracts/governing_law_venue/61/1/noul

*Selected as: noul, lowest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 105 / 0.909 |
| Ground truth | B (yes) |
| S1 prediction | B (yes) (margin 0.95) |
| R1 prediction | A (no) (margin 3.61) |
| Demo answer | A (no) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | contracts / policy |

**Query**

    Question: Candidate item_92. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":{"K28":{"amendment_signed_by_both":true,"forum":null,"in_amendment":false,"in_signed_master":true,"jurisdiction":"Scotland","section_number":48,"superseded":false},"K39":{"amendment_signed_by_both":true,"forum":"local_courts","in_amendment":false,"in_signed_master":true,"jurisdiction":"other_value","section_number":41,"superseded":false},"K43":{"amendment_signed_by_both":false,"forum":"other_value","in_amendment":false,"in_signed_master":false,"jurisdiction":"Scotland","section_number":52,"superseded":false},"P63":{"amendment_signed_by_both":true,"forum":"local_courts","in_amendment":true,"in_signed_master":true,"jurisdiction":"other_value","section_number":42,"superseded":fals ...[truncated 1664 bytes]
    A: no
    B: yes

**Retrieved exemplar** (06beefcfb568d6cb)

    Question: Candidate P94. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":{"K55":{"accessible_records":["unrelated_customers","unrelated_item"],"annual_audits_used":13,"notice_days":2,"permitted_auditors":["regulator","unrelated_item"],"section_number":74,"suspected_breach_exception":false},"P63":{"accessible_records":["unrelated_item"],"annual_audits_used":1,"notice_days":21,"permitted_auditors":[],"section_number":69,"suspected_breach_exception":true},"P94":{"accessible_records":null,"annual_audits_used":10,"notice_days":89,"permitted_auditors":["regulator","unrelated_item"],"section_number":72,"suspected_breach_exception":true},"item_13":{"accessible_records":[],"annual_audits_used":5,"notice_days":55,"permitted_auditors":["unrelated_item"],"secti ...[truncated 1762 bytes]
    A: no
    B: yes
    Answer: A

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/rubric_judge/trace_consistency/4/3/score

*Selected as: score, highest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 62 / 0.968 |
| Ground truth | B (Exactly 1 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 2.55) |
| R1 prediction | D (Exactly 3 of the 4 numbered requirements are established.) (margin 1.85) |
| Demo answer | D (Exactly 3 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | rubric_judge / rubric |

**Query**

    Question: Candidate item_46. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K31":{"action_pending":false,"action_sequence":8,"claimed_complete":false,"ignored_failure":true,"reported_result":"record_B","response_words":40,"verification_sequence":13,"verified_result":"record_B"},"K32":{"action_pending":true,"action_sequence":8,"claimed_complete":true,"ignored_failure":false,"reported_result":"record_A","response_words":44,"verification_sequence":55,"verified_result":"record_D"},"K41":{"action_pending":false,"action_sequence":34,"claimed_complete":true,"ignored_failure":false,"reported_result":"record_C","response_words":42,"verification_sequence":null,"verified_result":"record_C"},"P81":{"action_pending":null,"action_sequence":34,"claimed_complete":tr ...[truncated 1817 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (844aff74b4bdaf20)

    Question: Candidate K85. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K19":{"left_denominator":"record_C","reported_sides":["left","right","unrelated_item"],"response_words":50,"right_denominator":"record_C","same_metric_definition":false,"unmeasured_superiority_claim":false},"K85":{"left_denominator":"record_D","reported_sides":["left","right","unrelated_item"],"response_words":56,"right_denominator":"record_D","same_metric_definition":true,"unmeasured_superiority_claim":null},"P16":{"left_denominator":"record_C","reported_sides":["left","right"],"response_words":54,"right_denominator":"record_D","same_metric_definition":true,"unmeasured_superiority_claim":true},"P78":{"left_denominator":"record_D","reported_sides":["left","right","unrelated_i ...[truncated 1420 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: D

Relevant differences: R1 moved to the demonstration's answer label.

### community-diversity-v2/20260921/contracts/governing_law_venue/24/1/score

*Selected as: score, lowest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 96 / 0.924 |
| Ground truth | B (Exactly 1 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.03) |
| R1 prediction | D (Exactly 3 of the 4 numbered requirements are established.) (margin 0.80) |
| Demo answer | D (Exactly 3 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | contracts / policy |

**Query**

    Question: Candidate K67. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":[{"candidate":"K67","facts":{"amendment_signed_by_both":true,"forum":"commercial_court","in_amendment":false,"in_signed_master":false,"jurisdiction":"other_value","section_number":87,"superseded":true}},{"candidate":"K93","facts":{"amendment_signed_by_both":true,"forum":"commercial_court","in_amendment":true,"in_signed_master":true,"jurisdiction":"Scotland","section_number":85,"superseded":false}},{"candidate":"item_14","facts":{"amendment_signed_by_both":false,"forum":"commercial_court","in_amendment":false,"in_signed_master":false,"jurisdiction":"Scotland","section_number":82,"superseded":false}},{"candidate":"item_83","facts":{"amendment_signed_by_both":false,"forum":"commer ...[truncated 1448 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (43b8d2709fc4f693)

    Question: Candidate item_12. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":"P66\t{\"accessible_records\": [\"unrelated_customers\", \"unrelated_item\"], \"annual_audits_used\": 2, \"notice_days\": 7, \"permitted_auditors\": [\"regulator\"], \"section_number\": 41, \"suspected_breach_exception\": true}\nitem_12\t{\"accessible_records\": [\"unrelated_customers\", \"unrelated_item\"], \"annual_audits_used\": 5, \"notice_days\": 144, \"permitted_auditors\": [\"regulator\", \"unrelated_item\"], \"section_number\": 40, \"suspected_breach_exception\": true}\nK69\t{\"accessible_records\": null, \"annual_audits_used\": 2, \"notice_days\": 6, \"permitted_auditors\": [\"regulator\"], \"section_number\": 47, \"suspected_breach_exception\": true}\nitem_29\t{\"acce ...[truncated 1745 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: D

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

### wanli-decisions-v1:train:161255

*Selected as: choice, lowest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 108 / 0.904 |
| Ground truth | A (The premise supports the claim.) |
| S1 prediction | A (The premise supports the claim.) (margin 5.42) |
| R1 prediction | A (The premise supports the claim.) (margin 1.41) |
| Demo answer | A (The premise leaves the claim undetermined.) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / natural_language_inference |

**Query**

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"The La Rambla is the most attractive of the two.","premise":"The La Rambla, with its narrow, tree-lined, central section, is the most attractive of the two."}
    A: The premise supports the claim.
    B: The premise contradicts the claim.
    C: The premise leaves the claim undetermined.

**Retrieved exemplar** (2f5bdaf54f471ceb)

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"I'm not sure if I'm seeing the village for the first time, but I'm sure it's the most wonderful thing I've ever seen.","premise":"This picture of a small village, which you are seeing for the first time, is to me the most wonderful of all the things I have ever seen."}
    A: The premise leaves the claim undetermined.
    B: The premise supports the claim.
    C: The premise contradicts the claim.
    Answer: A

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

### community-diversity-v2/20260921/contracts/governing_law_venue/55/1/noul

*Selected as: noul, lowest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 106 / 0.907 |
| Ground truth | A (no) |
| S1 prediction | A (no) (margin 0.03) |
| R1 prediction | A (no) (margin 6.90) |
| Demo answer | A (no) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | contracts / policy |

**Query**

    Question: Candidate K18. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":{"K18":{"amendment_signed_by_both":false,"forum":"local_courts","in_amendment":false,"in_signed_master":true,"jurisdiction":null,"section_number":108,"superseded":false},"P40":{"amendment_signed_by_both":true,"forum":"local_courts","in_amendment":true,"in_signed_master":true,"jurisdiction":"Ontario","section_number":101,"superseded":true},"P96":{"amendment_signed_by_both":true,"forum":"local_courts","in_amendment":false,"in_signed_master":false,"jurisdiction":"Ontario","section_number":106,"superseded":true},"item_16":{"amendment_signed_by_both":true,"forum":"other_value","in_amendment":true,"in_signed_master":false,"jurisdiction":"Ontario","section_number":102,"superseded":fal ...[truncated 1307 bytes]
    A: no
    B: yes

**Retrieved exemplar** (28be4bb99af79361)

    Question: Candidate K17. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":"item_60\t{\"annual_fees_cents\": 144, \"cap_cents\": 1, \"cap_disclaimed_in_amendment\": true, \"carve_outs\": [\"fraud\", \"intentional_misconduct\"], \"section_number\": 81}\nK17\t{\"annual_fees_cents\": 5, \"cap_cents\": 1, \"cap_disclaimed_in_amendment\": false, \"carve_outs\": [], \"section_number\": 85}\nK45\t{\"annual_fees_cents\": 145, \"cap_cents\": 144, \"cap_disclaimed_in_amendment\": false, \"carve_outs\": [\"fraud\", \"intentional_misconduct\"], \"section_number\": 77}\nP23\t{\"annual_fees_cents\": 5, \"cap_cents\": 89, \"cap_disclaimed_in_amendment\": false, \"carve_outs\": [\"fraud\", \"intentional_misconduct\"], \"section_number\": 79}\nitem_42\t{\"annual_fees_ ...[truncated 1387 bytes]
    A: no
    B: yes
    Answer: A

### community-diversity-v2/20260921/rubric_judge/trace_consistency/22/3/score

*Selected as: score, highest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 66 / 0.964 |
| Ground truth | B (Exactly 1 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 2.11) |
| R1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 0.77) |
| Demo answer | C (Exactly 2 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | rubric_judge / rubric |

**Query**

    Question: Candidate P85. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"P18":{"action_pending":false,"action_sequence":2,"claimed_complete":false,"ignored_failure":false,"reported_result":"record_A","response_words":87,"verification_sequence":89,"verified_result":"record_D"},"P85":{"action_pending":true,"action_sequence":55,"claimed_complete":true,"ignored_failure":true,"reported_result":"record_B","response_words":91,"verification_sequence":21,"verified_result":"record_B"},"item_36":{"action_pending":true,"action_sequence":21,"claimed_complete":false,"ignored_failure":true,"reported_result":"record_B","response_words":86,"verification_sequence":34,"verified_result":"record_B"},"item_84":{"action_pending":false,"action_sequence":null,"claimed_com ...[truncated 1406 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (c9d40d41d078779e)

    Question: Candidate K96. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":[{"candidate":"item_89","facts":{"answer_supported":true,"extra_fields":null,"present_fields":["answer","evidence"],"response_words":88,"schema_valid":true}},{"candidate":"K96","facts":{"answer_supported":true,"extra_fields":34,"present_fields":["answer","evidence"],"response_words":86,"schema_valid":false}},{"candidate":"P82","facts":{"answer_supported":true,"extra_fields":0,"present_fields":["answer","evidence"],"response_words":82,"schema_valid":false}},{"candidate":"K75","facts":{"answer_supported":true,"extra_fields":0,"present_fields":["evidence","unrelated_item"],"response_words":81,"schema_valid":true}},{"candidate":"P69","facts":{"answer_supported":true,"extra_fields": ...[truncated 1295 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: C

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

### wanli-decisions-v1:train:314227

*Selected as: choice, highest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 38 / 0.988 |
| Ground truth | B (The premise leaves the claim undetermined.) |
| S1 prediction | A (The premise supports the claim.) (margin 1.10) |
| R1 prediction | C (The premise contradicts the claim.) (margin 1.68) |
| Demo answer | C (The premise supports the claim.) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / natural_language_inference |

**Query**

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"The owner of each unit should install and operate CEMS to affect sulfur dioxide in the air.","premise":"(3) By the later of January 1, 2009 or the date on which the unit commences operation, the owner or operator of each affected unit under subpart 2 of part B shall install and operate CEMS, quality assure the data, and keep records and reports in accordance with the regulations issued under paragraph (a)(2) with regard to sulfur dioxide, opacity, and volumetric flow."}
    A: The premise supports the claim.
    B: The premise leaves the claim undetermined.
    C: The premise contradicts the claim.

**Retrieved exemplar** (9b44865102ee1fef)

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"The owner of each unit should install and operate CEMS to affect sulfur dioxide in the air.","premise":"(1) By the later of January 1, 2009 or the date on which the unit commences operation, the owner or operator of each affected unit under subpart 2 of part B shall install and operate CEMS, quality assure the data, and keep records and reports in accordance with the regulations issued under paragraph (a)(2) with regard to sulfur dioxide, opacity, and volumetric flow."}
    A: The premise contradicts the claim.
    B: The premise leaves the claim undetermined.
    C: The premise supports the claim.
    Answer: C

Relevant differences: R1 moved to the demonstration's answer label.

### wanli-decisions-v1:train:182645

*Selected as: choice, lowest similarity*

| | |
|---|---|
| Kind | choice |
| Hamming / est. cosine | 108 / 0.904 |
| Ground truth | C (The premise leaves the claim undetermined.) |
| S1 prediction | B (The premise supports the claim.) (margin 5.58) |
| R1 prediction | B (The premise supports the claim.) (margin 7.63) |
| Demo answer | C (The premise leaves the claim undetermined.) |
| Same domain / family / group | no / yes / no |
| Demo domain / family |  / natural_language_inference |

**Query**

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"Behaviorism is the idea that behavior is controlled by external stimuli.","premise":"The third example is a quote from an interview with Dr. John B. Watson, who is known as the father of behaviorism."}
    A: The premise contradicts the claim.
    B: The premise supports the claim.
    C: The premise leaves the claim undetermined.

**Retrieved exemplar** (c2be20898689b4d4)

    Question: Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.
    State: {"claim":"The president did not like the meeting.","premise":"The meeting ended with a brief, impassioned speech by the president, who called for unity and for the spirit of cooperation to prevail."}
    A: The premise supports the claim.
    B: The premise contradicts the claim.
    C: The premise leaves the claim undetermined.
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

### community-diversity-v2/20260921/contracts/governing_law_venue/43/3/noul

*Selected as: noul, lowest similarity*

| | |
|---|---|
| Kind | noul |
| Hamming / est. cosine | 104 / 0.911 |
| Ground truth | A (no) |
| S1 prediction | B (yes) (margin 0.17) |
| R1 prediction | B (yes) (margin 2.44) |
| Demo answer | B (yes) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | contracts / policy |

**Query**

    Question: Candidate item_67. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":{"K25":{"amendment_signed_by_both":true,"forum":"other_value","in_amendment":true,"in_signed_master":false,"jurisdiction":"other_value","section_number":44,"superseded":true},"K55":{"amendment_signed_by_both":false,"forum":"other_value","in_amendment":false,"in_signed_master":true,"jurisdiction":"Scotland","section_number":39,"superseded":false},"P49":{"amendment_signed_by_both":true,"forum":"other_value","in_amendment":false,"in_signed_master":false,"jurisdiction":"Scotland","section_number":48,"superseded":true},"P60":{"amendment_signed_by_both":true,"forum":"arbitration","in_amendment":false,"in_signed_master":true,"jurisdiction":null,"section_number":42,"superseded":false}, ...[truncated 1471 bytes]
    A: no
    B: yes

**Retrieved exemplar** (066425035dcee98a)

    Question: Candidate P42. Is this candidate supported by all four requirements? Unknown evidence does not establish eligibility.
    State: {"candidates":{"K45":{"annual_fees_cents":0,"cap_cents":2,"cap_disclaimed_in_amendment":true,"carve_outs":["fraud","intentional_misconduct","unrelated_item"],"section_number":37},"K75":{"annual_fees_cents":8,"cap_cents":55,"cap_disclaimed_in_amendment":false,"carve_outs":["intentional_misconduct","unrelated_item"],"section_number":36},"P42":{"annual_fees_cents":21,"cap_cents":89,"cap_disclaimed_in_amendment":false,"carve_outs":["fraud","intentional_misconduct","unrelated_item"],"section_number":39},"item_13":{"annual_fees_cents":2,"cap_cents":5,"cap_disclaimed_in_amendment":true,"carve_outs":["intentional_misconduct"],"section_number":41},"item_31":{"annual_fees_cents":3,"cap_cents":144,"cap ...[truncated 1424 bytes]
    A: no
    B: yes
    Answer: B

### community-diversity-v2/20260921/rubric_judge/trace_consistency/67/0/score

*Selected as: score, highest similarity*

| | |
|---|---|
| Kind | score (5 levels) |
| Hamming / est. cosine | 60 / 0.970 |
| Ground truth | D (Exactly 3 of the 4 numbered requirements are established.) |
| S1 prediction | B (Exactly 1 of the 4 numbered requirements are established.) (margin 1.91) |
| R1 prediction | A (Exactly 0 of the 4 numbered requirements are established.) (margin 3.00) |
| Demo answer | A (Exactly 0 of the 4 numbered requirements are established.) |
| Same domain / family / group | yes / yes / no |
| Demo domain / family | rubric_judge / rubric |

**Query**

    Question: Candidate K83. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K48":{"action_pending":false,"action_sequence":8,"claimed_complete":false,"ignored_failure":true,"reported_result":"record_A","response_words":29,"verification_sequence":8,"verified_result":"record_A"},"K83":{"action_pending":false,"action_sequence":3,"claimed_complete":false,"ignored_failure":null,"reported_result":"record_D","response_words":36,"verification_sequence":89,"verified_result":"record_D"},"P49":{"action_pending":false,"action_sequence":3,"claimed_complete":false,"ignored_failure":false,"reported_result":"record_D","response_words":32,"verification_sequence":89,"verified_result":"record_D"},"P78":{"action_pending":false,"action_sequence":13,"claimed_complete":tru ...[truncated 1405 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.

**Retrieved exemplar** (8802cb03ac8a73b6)

    Question: Candidate K25. How many of the four requirements are established for this candidate? Count each numbered requirement once; unknown is not established.
    State: {"candidates":{"K25":{"left_denominator":"record_B","reported_sides":["right"],"response_words":103,"right_denominator":"record_C","same_metric_definition":false,"unmeasured_superiority_claim":true},"K58":{"left_denominator":null,"reported_sides":["left","right","unrelated_item"],"response_words":105,"right_denominator":"record_C","same_metric_definition":true,"unmeasured_superiority_claim":false},"P66":{"left_denominator":"record_C","reported_sides":["right","unrelated_item"],"response_words":109,"right_denominator":"record_A","same_metric_definition":true,"unmeasured_superiority_claim":false},"P92":{"left_denominator":"record_C","reported_sides":["left","right","unrelated_item"],"response_ ...[truncated 1794 bytes]
    A: Exactly 0 of the 4 numbered requirements are established.
    B: Exactly 1 of the 4 numbered requirements are established.
    C: Exactly 2 of the 4 numbered requirements are established.
    D: Exactly 3 of the 4 numbered requirements are established.
    E: Exactly 4 of the 4 numbered requirements are established.
    Answer: A

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

