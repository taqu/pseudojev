# R0 Retrieval Sanity Evaluation — Examples

This document shows representative query/retrieval pairs for manual inspection.
Similarity = 1 - cosine_distance.

---

## Kind: choice

### Medium Similarity (0.60 ≤ Top-1 < 0.80)

**ID:** `wanli-decisions-v1:train:102918`  
**Kind:** choice  
**Group:** `wanli-decisions-v1:3e30638fea667f63ee2c4d7b6bc19e83eb83b85da66b65fe3dc5f72fa73646ab`  
**Template:** ``  
**Answer:** `The premise supports the claim.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.6214 dist=0.3786  
  ID: `wanli-decisions-v1:train:131888`  
  Answer: `The premise supports the claim.` (index=0)  
  Scenario:   

  **Rank 2** — sim=0.6156 dist=0.3844  
  ID: `wanli-decisions-v1:train:43542`  
  Answer: `The premise supports the claim.` (index=0)  
  Scenario:   

  **Rank 3** — sim=0.6121 dist=0.3879  
  ID: `wanli-decisions-v1:train:336044`  
  Answer: `The premise supports the claim.` (index=0)  
  Scenario:   

  **Rank 4** — sim=0.6100 dist=0.3900  
  ID: `wanli-decisions-v1:train:325341`  
  Answer: `The premise leaves the claim undetermined.` (index=1)  
  Scenario:   

  **Rank 5** — sim=0.6053 dist=0.3947  
  ID: `wanli-decisions-v1:train:182507`  
  Answer: `The premise supports the claim.` (index=1)  
  Scenario:   

**ID:** `wanli-decisions-v1:train:103817`  
**Kind:** choice  
**Group:** `wanli-decisions-v1:4dc937d4859aa072d47909fd70487944b291a668a5fc1da404fc094eada3067e`  
**Template:** ``  
**Answer:** `The premise leaves the claim undetermined.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.6204 dist=0.3796  
  ID: `wanli-decisions-v1:train:235729`  
  Answer: `The premise supports the claim.` (index=1)  
  Scenario:   

  **Rank 2** — sim=0.6172 dist=0.3828  
  ID: `wanli-decisions-v1:train:96403`  
  Answer: `The premise supports the claim.` (index=0)  
  Scenario:   

  **Rank 3** — sim=0.6163 dist=0.3837  
  ID: `wanli-decisions-v1:train:118834`  
  Answer: `The premise supports the claim.` (index=0)  
  Scenario:   

  **Rank 4** — sim=0.6162 dist=0.3838  
  ID: `wanli-decisions-v1:train:216099`  
  Answer: `The premise supports the claim.` (index=2)  
  Scenario:   

  **Rank 5** — sim=0.6154 dist=0.3846  
  ID: `wanli-decisions-v1:train:365982`  
  Answer: `The premise leaves the claim undetermined.` (index=2)  
  Scenario:   

### Low Similarity (Top-1 < 0.50)

**ID:** `customer-control-v1:42:103:category`  
**Kind:** choice  
**Group:** `customer-control-v1:42:103`  
**Template:** `customer-control-v1:id:0`  
**Answer:** `account: Login, permissions, profile, security`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.3309 dist=0.6691  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/64/3/choice`  
  Answer: `abstain` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 2** — sim=0.3309 dist=0.6691  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/0/choice`  
  Answer: `K36` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 3** — sim=0.3309 dist=0.6691  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/1/choice`  
  Answer: `K15` (index=3)  
  Scenario: support/subscription_cancellation  

  **Rank 4** — sim=0.3309 dist=0.6691  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/2/choice`  
  Answer: `K55` (index=6)  
  Scenario: support/subscription_cancellation  

  **Rank 5** — sim=0.3309 dist=0.6691  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/3/choice`  
  Answer: `abstain` (index=1)  
  Scenario: support/subscription_cancellation  

**ID:** `customer-control-v1:42:107:category`  
**Kind:** choice  
**Group:** `customer-control-v1:42:107`  
**Template:** `customer-control-v1:id:0`  
**Answer:** `account: Login, permissions, profile, security`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.3188 dist=0.6812  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/64/3/choice`  
  Answer: `abstain` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 2** — sim=0.3188 dist=0.6812  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/0/choice`  
  Answer: `K36` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 3** — sim=0.3188 dist=0.6812  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/1/choice`  
  Answer: `K15` (index=3)  
  Scenario: support/subscription_cancellation  

  **Rank 4** — sim=0.3188 dist=0.6812  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/2/choice`  
  Answer: `K55` (index=6)  
  Scenario: support/subscription_cancellation  

  **Rank 5** — sim=0.3188 dist=0.6812  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/3/choice`  
  Answer: `abstain` (index=1)  
  Scenario: support/subscription_cancellation  

### Different-Template Top-1 (cross-template match)

**ID:** `customer-control-v1:42:103:category`  
**Kind:** choice  
**Group:** `customer-control-v1:42:103`  
**Template:** `customer-control-v1:id:0`  
**Answer:** `account: Login, permissions, profile, security`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.3309 dist=0.6691  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/64/3/choice`  
  Answer: `abstain` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 2** — sim=0.3309 dist=0.6691  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/0/choice`  
  Answer: `K36` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 3** — sim=0.3309 dist=0.6691  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/1/choice`  
  Answer: `K15` (index=3)  
  Scenario: support/subscription_cancellation  

  **Rank 4** — sim=0.3309 dist=0.6691  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/2/choice`  
  Answer: `K55` (index=6)  
  Scenario: support/subscription_cancellation  

  **Rank 5** — sim=0.3309 dist=0.6691  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/3/choice`  
  Answer: `abstain` (index=1)  
  Scenario: support/subscription_cancellation  

**ID:** `customer-control-v1:42:107:category`  
**Kind:** choice  
**Group:** `customer-control-v1:42:107`  
**Template:** `customer-control-v1:id:0`  
**Answer:** `account: Login, permissions, profile, security`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.3188 dist=0.6812  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/64/3/choice`  
  Answer: `abstain` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 2** — sim=0.3188 dist=0.6812  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/0/choice`  
  Answer: `K36` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 3** — sim=0.3188 dist=0.6812  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/1/choice`  
  Answer: `K15` (index=3)  
  Scenario: support/subscription_cancellation  

  **Rank 4** — sim=0.3188 dist=0.6812  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/2/choice`  
  Answer: `K55` (index=6)  
  Scenario: support/subscription_cancellation  

  **Rank 5** — sim=0.3188 dist=0.6812  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/68/3/choice`  
  Answer: `abstain` (index=1)  
  Scenario: support/subscription_cancellation  

---

## Kind: noul

### Low Similarity (Top-1 < 0.50)

**ID:** `customer-control-v1:42:101:has_reproducible_steps`  
**Kind:** noul  
**Group:** `customer-control-v1:42:101`  
**Template:** `customer-control-v1:id:0`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.3452 dist=0.6548  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/60/0/noul`  
  Answer: `no` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 2** — sim=0.3451 dist=0.6549  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/0/2/noul`  
  Answer: `no` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 3** — sim=0.3433 dist=0.6567  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/4/2/noul`  
  Answer: `no` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 4** — sim=0.3432 dist=0.6568  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/24/0/noul`  
  Answer: `no` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 5** — sim=0.3427 dist=0.6573  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/0/0/noul`  
  Answer: `no` (index=0)  
  Scenario: support/subscription_cancellation  

**ID:** `customer-control-v1:42:102:has_reproducible_steps`  
**Kind:** noul  
**Group:** `customer-control-v1:42:102`  
**Template:** `customer-control-v1:id:0`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.3544 dist=0.6456  
  ID: `community-diversity-v2/20260921/support/duplicate_charge/44/2/noul`  
  Answer: `no` (index=0)  
  Scenario: support/duplicate_charge  

  **Rank 2** — sim=0.3536 dist=0.6464  
  ID: `community-diversity-v2/20260921/support/duplicate_charge/20/2/noul`  
  Answer: `yes` (index=1)  
  Scenario: support/duplicate_charge  

  **Rank 3** — sim=0.3531 dist=0.6469  
  ID: `community-diversity-v2/20260921/support/duplicate_charge/8/2/noul`  
  Answer: `yes` (index=1)  
  Scenario: support/duplicate_charge  

  **Rank 4** — sim=0.3530 dist=0.6470  
  ID: `community-diversity-v2/20260921/support/duplicate_charge/8/0/noul`  
  Answer: `no` (index=0)  
  Scenario: support/duplicate_charge  

  **Rank 5** — sim=0.3528 dist=0.6472  
  ID: `community-diversity-v2/20260921/support/duplicate_charge/16/0/noul`  
  Answer: `no` (index=0)  
  Scenario: support/duplicate_charge  

### Different-Template Top-1 (cross-template match)

**ID:** `customer-control-v1:42:101:has_reproducible_steps`  
**Kind:** noul  
**Group:** `customer-control-v1:42:101`  
**Template:** `customer-control-v1:id:0`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.3452 dist=0.6548  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/60/0/noul`  
  Answer: `no` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 2** — sim=0.3451 dist=0.6549  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/0/2/noul`  
  Answer: `no` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 3** — sim=0.3433 dist=0.6567  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/4/2/noul`  
  Answer: `no` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 4** — sim=0.3432 dist=0.6568  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/24/0/noul`  
  Answer: `no` (index=0)  
  Scenario: support/subscription_cancellation  

  **Rank 5** — sim=0.3427 dist=0.6573  
  ID: `community-diversity-v2/20260921/support/subscription_cancellation/0/0/noul`  
  Answer: `no` (index=0)  
  Scenario: support/subscription_cancellation  

**ID:** `customer-control-v1:42:102:has_reproducible_steps`  
**Kind:** noul  
**Group:** `customer-control-v1:42:102`  
**Template:** `customer-control-v1:id:0`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.3544 dist=0.6456  
  ID: `community-diversity-v2/20260921/support/duplicate_charge/44/2/noul`  
  Answer: `no` (index=0)  
  Scenario: support/duplicate_charge  

  **Rank 2** — sim=0.3536 dist=0.6464  
  ID: `community-diversity-v2/20260921/support/duplicate_charge/20/2/noul`  
  Answer: `yes` (index=1)  
  Scenario: support/duplicate_charge  

  **Rank 3** — sim=0.3531 dist=0.6469  
  ID: `community-diversity-v2/20260921/support/duplicate_charge/8/2/noul`  
  Answer: `yes` (index=1)  
  Scenario: support/duplicate_charge  

  **Rank 4** — sim=0.3530 dist=0.6470  
  ID: `community-diversity-v2/20260921/support/duplicate_charge/8/0/noul`  
  Answer: `no` (index=0)  
  Scenario: support/duplicate_charge  

  **Rank 5** — sim=0.3528 dist=0.6472  
  ID: `community-diversity-v2/20260921/support/duplicate_charge/16/0/noul`  
  Answer: `no` (index=0)  
  Scenario: support/duplicate_charge  

---

## Kind: score

