# R0 Retrieval Sanity Evaluation — Examples

This document shows representative query/retrieval pairs for manual inspection.
Similarity = 1 - cosine_distance.

---

## Kind: choice

### Low Similarity (Top-1 < 0.50)

**ID:** `8001ab3e52b1f12c`  
**Kind:** choice  
**Group:** `5b8aa84cbd2cf205`  
**Template:** `community-diversity-v2/contracts/governing_law_venue`  
**Answer:** `P76`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=55.0000 [SAME-TEMPLATE]  
  ID: `715a060d55e515c0`  
  Answer: `abstain` (index=3)  
  Scenario: contracts/governing_law_venue  

  **Rank 2** — sim=0.0000 dist=55.0000 [SAME-TEMPLATE]  
  ID: `a8198d8400a742c9`  
  Answer: `K28` (index=0)  
  Scenario: contracts/governing_law_venue  

  **Rank 3** — sim=0.0000 dist=55.0000 [SAME-TEMPLATE]  
  ID: `e89183ddf4062d7`  
  Answer: `item_75` (index=2)  
  Scenario: contracts/governing_law_venue  

  **Rank 4** — sim=0.0000 dist=55.0000 [SAME-TEMPLATE]  
  ID: `bd8218b0db005967`  
  Answer: `P31` (index=2)  
  Scenario: contracts/governing_law_venue  

  **Rank 5** — sim=0.0000 dist=55.0000 [SAME-TEMPLATE]  
  ID: `68a54c1722a6e887`  
  Answer: `abstain` (index=0)  
  Scenario: contracts/governing_law_venue  

**ID:** `801647c81b006ab5`  
**Kind:** choice  
**Group:** `74d4882b4e345cdc`  
**Template:** ``  
**Answer:** `The premise supports the claim.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=44.0000 [SAME-GROUP]  
  ID: `801647c81b006ab5`  
  Answer: `The premise supports the claim.` (index=1)  
  Scenario:   

  **Rank 2** — sim=0.0000 dist=71.0000  
  ID: `eebaa97f4f278a89`  
  Answer: `The premise leaves the claim undetermined.` (index=2)  
  Scenario:   

  **Rank 3** — sim=0.0000 dist=73.0000  
  ID: `d80955f3c0d3c50b`  
  Answer: `The premise contradicts the claim.` (index=1)  
  Scenario:   

  **Rank 4** — sim=0.0000 dist=76.0000  
  ID: `976209ab1304411`  
  Answer: `The premise leaves the claim undetermined.` (index=0)  
  Scenario:   

  **Rank 5** — sim=0.0000 dist=77.0000  
  ID: `e1332033e096924b`  
  Answer: `The premise supports the claim.` (index=2)  
  Scenario:   

### Same-Template Top-1 (potential boilerplate bias)

**ID:** `8001ab3e52b1f12c`  
**Kind:** choice  
**Group:** `5b8aa84cbd2cf205`  
**Template:** `community-diversity-v2/contracts/governing_law_venue`  
**Answer:** `P76`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=55.0000 [SAME-TEMPLATE]  
  ID: `715a060d55e515c0`  
  Answer: `abstain` (index=3)  
  Scenario: contracts/governing_law_venue  

  **Rank 2** — sim=0.0000 dist=55.0000 [SAME-TEMPLATE]  
  ID: `a8198d8400a742c9`  
  Answer: `K28` (index=0)  
  Scenario: contracts/governing_law_venue  

  **Rank 3** — sim=0.0000 dist=55.0000 [SAME-TEMPLATE]  
  ID: `e89183ddf4062d7`  
  Answer: `item_75` (index=2)  
  Scenario: contracts/governing_law_venue  

  **Rank 4** — sim=0.0000 dist=55.0000 [SAME-TEMPLATE]  
  ID: `bd8218b0db005967`  
  Answer: `P31` (index=2)  
  Scenario: contracts/governing_law_venue  

  **Rank 5** — sim=0.0000 dist=55.0000 [SAME-TEMPLATE]  
  ID: `68a54c1722a6e887`  
  Answer: `abstain` (index=0)  
  Scenario: contracts/governing_law_venue  

**ID:** `806984dc10bbd811`  
**Kind:** choice  
**Group:** `254a8007e23b9ef8`  
**Template:** `community-diversity-v2/ir/access_schedule`  
**Answer:** `P26`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=58.0000 [SAME-TEMPLATE]  
  ID: `76fb87d70ba23d46`  
  Answer: `abstain` (index=4)  
  Scenario: ir/access_schedule  

  **Rank 2** — sim=0.0000 dist=58.0000 [SAME-TEMPLATE]  
  ID: `30b77ae1554aca74`  
  Answer: `P61` (index=6)  
  Scenario: ir/access_schedule  

  **Rank 3** — sim=0.0000 dist=58.0000 [SAME-TEMPLATE]  
  ID: `b034079c4de12ec0`  
  Answer: `P54` (index=2)  
  Scenario: ir/access_schedule  

  **Rank 4** — sim=0.0000 dist=58.0000 [SAME-TEMPLATE]  
  ID: `200a1c62f90688f3`  
  Answer: `item_75` (index=0)  
  Scenario: ir/access_schedule  

  **Rank 5** — sim=0.0000 dist=58.0000 [SAME-TEMPLATE]  
  ID: `891a12d418f165de`  
  Answer: `abstain` (index=1)  
  Scenario: ir/access_schedule  

### Different-Template Top-1 (cross-template match)

**ID:** `801647c81b006ab5`  
**Kind:** choice  
**Group:** `74d4882b4e345cdc`  
**Template:** ``  
**Answer:** `The premise supports the claim.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=44.0000 [SAME-GROUP]  
  ID: `801647c81b006ab5`  
  Answer: `The premise supports the claim.` (index=1)  
  Scenario:   

  **Rank 2** — sim=0.0000 dist=71.0000  
  ID: `eebaa97f4f278a89`  
  Answer: `The premise leaves the claim undetermined.` (index=2)  
  Scenario:   

  **Rank 3** — sim=0.0000 dist=73.0000  
  ID: `d80955f3c0d3c50b`  
  Answer: `The premise contradicts the claim.` (index=1)  
  Scenario:   

  **Rank 4** — sim=0.0000 dist=76.0000  
  ID: `976209ab1304411`  
  Answer: `The premise leaves the claim undetermined.` (index=0)  
  Scenario:   

  **Rank 5** — sim=0.0000 dist=77.0000  
  ID: `e1332033e096924b`  
  Answer: `The premise supports the claim.` (index=2)  
  Scenario:   

**ID:** `801fedeccd92b610`  
**Kind:** choice  
**Group:** `c3137b3507be2bc7`  
**Template:** ``  
**Answer:** `The premise leaves the claim undetermined.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=46.0000 [SAME-GROUP]  
  ID: `801fedeccd92b610`  
  Answer: `The premise leaves the claim undetermined.` (index=0)  
  Scenario:   

  **Rank 2** — sim=0.0000 dist=78.0000  
  ID: `41e5adbf6b1db0c5`  
  Answer: `The premise supports the claim.` (index=0)  
  Scenario:   

  **Rank 3** — sim=0.0000 dist=80.0000  
  ID: `44841b27fca144d8`  
  Answer: `The premise supports the claim.` (index=0)  
  Scenario:   

  **Rank 4** — sim=0.0000 dist=81.0000  
  ID: `fa53158e96a98430`  
  Answer: `The premise supports the claim.` (index=1)  
  Scenario:   

  **Rank 5** — sim=0.0000 dist=81.0000  
  ID: `b2786a9388da1a2e`  
  Answer: `The premise supports the claim.` (index=1)  
  Scenario:   

### Same Source Instance, Different Answer (counterfactual candidates)

**ID:** `865d3aca34662053`  
**Kind:** choice  
**Group:** `f25b1b53cb2cb9c5`  
**Template:** `community-diversity-v2/contracts/governing_law_venue`  
**Answer:** `P36`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=56.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `e7ddf7952616e244`  
  Answer: `abstain` (index=4)  
  Scenario: contracts/governing_law_venue  

  **Rank 2** — sim=0.0000 dist=56.0000 [SAME-TEMPLATE]  
  ID: `49ebcbb649aa450f`  
  Answer: `K94` (index=3)  
  Scenario: contracts/governing_law_venue  

  **Rank 3** — sim=0.0000 dist=56.0000 [SAME-TEMPLATE]  
  ID: `a570a3403c904b9c`  
  Answer: `P64` (index=6)  
  Scenario: contracts/governing_law_venue  

  **Rank 4** — sim=0.0000 dist=56.0000 [SAME-TEMPLATE]  
  ID: `cde532227b888c84`  
  Answer: `item_26` (index=0)  
  Scenario: contracts/governing_law_venue  

  **Rank 5** — sim=0.0000 dist=56.0000 [SAME-TEMPLATE]  
  ID: `a72fc3e459c029e`  
  Answer: `abstain` (index=6)  
  Scenario: contracts/governing_law_venue  

**ID:** `899b87ce49bf5cd8`  
**Kind:** choice  
**Group:** `2b7a502ebb9827d`  
**Template:** `community-diversity-v2/contracts/governing_law_venue`  
**Answer:** `P11`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=52.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `58673cef2e0222d2`  
  Answer: `abstain` (index=3)  
  Scenario: contracts/governing_law_venue  

  **Rank 2** — sim=0.0000 dist=52.0000 [SAME-TEMPLATE]  
  ID: `ca848351b39e99fd`  
  Answer: `P63` (index=0)  
  Scenario: contracts/governing_law_venue  

  **Rank 3** — sim=0.0000 dist=52.0000 [SAME-TEMPLATE]  
  ID: `2798f8cbc88c6db5`  
  Answer: `item_92` (index=2)  
  Scenario: contracts/governing_law_venue  

  **Rank 4** — sim=0.0000 dist=52.0000 [SAME-TEMPLATE]  
  ID: `92fb2bfc19cad424`  
  Answer: `K39` (index=6)  
  Scenario: contracts/governing_law_venue  

  **Rank 5** — sim=0.0000 dist=52.0000 [SAME-TEMPLATE]  
  ID: `2af5335278083ae`  
  Answer: `abstain` (index=2)  
  Scenario: contracts/governing_law_venue  

---

## Kind: noul

### Low Similarity (Top-1 < 0.50)

**ID:** `8024fd2460f9a7d4`  
**Kind:** noul  
**Group:** `66e616d5eee0bd51`  
**Template:** `mailroom-control-v1:normal_id`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=54.0000 [SAME-TEMPLATE]  
  ID: `4a8cb5803652c2d3`  
  Answer: `no` (index=0)  
  Scenario: mailroom-control-v1:normal_id  

  **Rank 2** — sim=0.0000 dist=58.0000 [SAME-TEMPLATE]  
  ID: `aac1c048bd4b3a33`  
  Answer: `no` (index=0)  
  Scenario: mailroom-control-v1:normal_id  

  **Rank 3** — sim=0.0000 dist=60.0000 [SAME-TEMPLATE]  
  ID: `6c3416e4f3112f7c`  
  Answer: `no` (index=0)  
  Scenario: mailroom-control-v1:normal_id  

  **Rank 4** — sim=0.0000 dist=60.0000 [SAME-TEMPLATE]  
  ID: `18886382dd29cd9e`  
  Answer: `no` (index=0)  
  Scenario: mailroom-control-v1:normal_id  

  **Rank 5** — sim=0.0000 dist=61.0000 [SAME-TEMPLATE]  
  ID: `b9eb77b06a231eb6`  
  Answer: `no` (index=0)  
  Scenario: mailroom-control-v1:normal_id  

**ID:** `803561f67fb2ba9c`  
**Kind:** noul  
**Group:** `b23dfc2842012ed3`  
**Template:** `community-diversity-v2/contracts/governing_law_venue`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=44.0000 [SAME-TEMPLATE]  
  ID: `b8c00fa7dc4a2f27`  
  Answer: `no` (index=0)  
  Scenario: contracts/governing_law_venue  

  **Rank 2** — sim=0.0000 dist=46.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `803561f67fb2ba9c`  
  Answer: `no` (index=0)  
  Scenario: contracts/governing_law_venue  

  **Rank 3** — sim=0.0000 dist=46.0000 [SAME-TEMPLATE]  
  ID: `e34031251d6f506d`  
  Answer: `yes` (index=1)  
  Scenario: contracts/governing_law_venue  

  **Rank 4** — sim=0.0000 dist=46.0000 [SAME-TEMPLATE]  
  ID: `faa58306b669236`  
  Answer: `no` (index=0)  
  Scenario: contracts/governing_law_venue  

  **Rank 5** — sim=0.0000 dist=46.0000 [SAME-TEMPLATE]  
  ID: `caa4557c9382eb9d`  
  Answer: `no` (index=0)  
  Scenario: contracts/governing_law_venue  

### Same-Template Top-1 (potential boilerplate bias)

**ID:** `8024fd2460f9a7d4`  
**Kind:** noul  
**Group:** `66e616d5eee0bd51`  
**Template:** `mailroom-control-v1:normal_id`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=54.0000 [SAME-TEMPLATE]  
  ID: `4a8cb5803652c2d3`  
  Answer: `no` (index=0)  
  Scenario: mailroom-control-v1:normal_id  

  **Rank 2** — sim=0.0000 dist=58.0000 [SAME-TEMPLATE]  
  ID: `aac1c048bd4b3a33`  
  Answer: `no` (index=0)  
  Scenario: mailroom-control-v1:normal_id  

  **Rank 3** — sim=0.0000 dist=60.0000 [SAME-TEMPLATE]  
  ID: `6c3416e4f3112f7c`  
  Answer: `no` (index=0)  
  Scenario: mailroom-control-v1:normal_id  

  **Rank 4** — sim=0.0000 dist=60.0000 [SAME-TEMPLATE]  
  ID: `18886382dd29cd9e`  
  Answer: `no` (index=0)  
  Scenario: mailroom-control-v1:normal_id  

  **Rank 5** — sim=0.0000 dist=61.0000 [SAME-TEMPLATE]  
  ID: `b9eb77b06a231eb6`  
  Answer: `no` (index=0)  
  Scenario: mailroom-control-v1:normal_id  

**ID:** `803561f67fb2ba9c`  
**Kind:** noul  
**Group:** `b23dfc2842012ed3`  
**Template:** `community-diversity-v2/contracts/governing_law_venue`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=44.0000 [SAME-TEMPLATE]  
  ID: `b8c00fa7dc4a2f27`  
  Answer: `no` (index=0)  
  Scenario: contracts/governing_law_venue  

  **Rank 2** — sim=0.0000 dist=46.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `803561f67fb2ba9c`  
  Answer: `no` (index=0)  
  Scenario: contracts/governing_law_venue  

  **Rank 3** — sim=0.0000 dist=46.0000 [SAME-TEMPLATE]  
  ID: `e34031251d6f506d`  
  Answer: `yes` (index=1)  
  Scenario: contracts/governing_law_venue  

  **Rank 4** — sim=0.0000 dist=46.0000 [SAME-TEMPLATE]  
  ID: `faa58306b669236`  
  Answer: `no` (index=0)  
  Scenario: contracts/governing_law_venue  

  **Rank 5** — sim=0.0000 dist=46.0000 [SAME-TEMPLATE]  
  ID: `caa4557c9382eb9d`  
  Answer: `no` (index=0)  
  Scenario: contracts/governing_law_venue  

### Different-Template Top-1 (cross-template match)

**ID:** `d587bda0d7526d1e`  
**Kind:** noul  
**Group:** `f13f90654f9bce98`  
**Template:** `customer-control-v1:id:0`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=77.0000  
  ID: `eb91c598cdf03dcf`  
  Answer: `no` (index=0)  
  Scenario: customer-control-v1:id:1  

  **Rank 2** — sim=0.0000 dist=78.0000 [SAME-TEMPLATE]  
  ID: `f04a0f27a85a10da`  
  Answer: `no` (index=0)  
  Scenario: customer-control-v1:id:0  

  **Rank 3** — sim=0.0000 dist=81.0000 [SAME-GROUP] [SAME-TEMPLATE]  
  ID: `d587bda0d7526d1e`  
  Answer: `no` (index=0)  
  Scenario: customer-control-v1:id:0  

  **Rank 4** — sim=0.0000 dist=83.0000 [SAME-TEMPLATE]  
  ID: `fef3744011471d7b`  
  Answer: `no` (index=0)  
  Scenario: customer-control-v1:id:0  

  **Rank 5** — sim=0.0000 dist=86.0000 [SAME-TEMPLATE]  
  ID: `f4379dbd94d5cd2a`  
  Answer: `no` (index=0)  
  Scenario: customer-control-v1:id:0  

**ID:** `f1bbe77fc577e895`  
**Kind:** noul  
**Group:** `8795ab3a54b5efdf`  
**Template:** `customer-control-v1:id:1`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=89.0000  
  ID: `33598cfc8e76a8dd`  
  Answer: `no` (index=0)  
  Scenario: customer-control-v1:id:0  

  **Rank 2** — sim=0.0000 dist=94.0000 [SAME-TEMPLATE]  
  ID: `326bb3756a268786`  
  Answer: `no` (index=0)  
  Scenario: customer-control-v1:id:1  

  **Rank 3** — sim=0.0000 dist=95.0000 [SAME-TEMPLATE]  
  ID: `4d2a602741f1f65e`  
  Answer: `no` (index=0)  
  Scenario: customer-control-v1:id:1  

  **Rank 4** — sim=0.0000 dist=95.0000 [SAME-TEMPLATE]  
  ID: `960067b0c142ff8d`  
  Answer: `no` (index=0)  
  Scenario: customer-control-v1:id:1  

  **Rank 5** — sim=0.0000 dist=96.0000 [SAME-TEMPLATE]  
  ID: `f45ba45f5e04423`  
  Answer: `no` (index=0)  
  Scenario: customer-control-v1:id:1  

### Same Source Instance, Different Answer (counterfactual candidates)

**ID:** `88f7fbee505b7a58`  
**Kind:** noul  
**Group:** `4c7154e33615af6d`  
**Template:** `community-diversity-v2/shell_history/checksum_verify`  
**Answer:** `no`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=46.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `662bc2b94f1c6e1d`  
  Answer: `yes` (index=1)  
  Scenario: shell_history/checksum_verify  

  **Rank 2** — sim=0.0000 dist=47.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `88f7fbee505b7a58`  
  Answer: `no` (index=0)  
  Scenario: shell_history/checksum_verify  

  **Rank 3** — sim=0.0000 dist=48.0000 [SAME-TEMPLATE]  
  ID: `faaf967cae548f2e`  
  Answer: `yes` (index=1)  
  Scenario: shell_history/checksum_verify  

  **Rank 4** — sim=0.0000 dist=48.0000 [SAME-TEMPLATE]  
  ID: `1404aa477ea32258`  
  Answer: `no` (index=0)  
  Scenario: shell_history/checksum_verify  

  **Rank 5** — sim=0.0000 dist=48.0000 [SAME-TEMPLATE]  
  ID: `cf183bcab19f57d3`  
  Answer: `yes` (index=1)  
  Scenario: shell_history/checksum_verify  

**ID:** `92326f2b0618d74b`  
**Kind:** noul  
**Group:** `5e9411cd6bc58f1c`  
**Template:** `community-diversity-v2/shell_history/checksum_verify`  
**Answer:** `yes`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=52.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `d7f154dc6f3d3434`  
  Answer: `no` (index=0)  
  Scenario: shell_history/checksum_verify  

  **Rank 2** — sim=0.0000 dist=52.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `92326f2b0618d74b`  
  Answer: `yes` (index=1)  
  Scenario: shell_history/checksum_verify  

  **Rank 3** — sim=0.0000 dist=53.0000 [SAME-TEMPLATE]  
  ID: `e2a4d6f9a2190346`  
  Answer: `no` (index=0)  
  Scenario: shell_history/checksum_verify  

  **Rank 4** — sim=0.0000 dist=54.0000 [SAME-TEMPLATE]  
  ID: `fdf31c5de5e1a651`  
  Answer: `no` (index=0)  
  Scenario: shell_history/checksum_verify  

  **Rank 5** — sim=0.0000 dist=54.0000 [SAME-TEMPLATE]  
  ID: `74dff5fbeb0641af`  
  Answer: `no` (index=0)  
  Scenario: shell_history/checksum_verify  

---

## Kind: score

### Low Similarity (Top-1 < 0.50)

**ID:** `80447bcfcd98c680`  
**Kind:** score (levels=5)  
**Group:** `9e41454467677f0d`  
**Template:** `community-diversity-v2/games/racing_overtake`  
**Answer:** `Exactly 2 of the 4 numbered requirements are established.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=59.0000 [SAME-TEMPLATE]  
  ID: `33171739d314f1d4`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: games/racing_overtake  

  **Rank 2** — sim=0.0000 dist=59.0000 [SAME-TEMPLATE]  
  ID: `1e0c6fc6e7ae7ed3`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: games/racing_overtake  

  **Rank 3** — sim=0.0000 dist=60.0000 [SAME-TEMPLATE]  
  ID: `6ef0dc3888b142b2`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: games/racing_overtake  

  **Rank 4** — sim=0.0000 dist=61.0000 [SAME-TEMPLATE]  
  ID: `43d7e282a10ad1e1`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: games/racing_overtake  

  **Rank 5** — sim=0.0000 dist=61.0000 [SAME-TEMPLATE]  
  ID: `9d1089e80d6fd9c0`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: games/racing_overtake  

**ID:** `809447d8795872f8`  
**Kind:** score (levels=5)  
**Group:** `8d8e0f3fdcf3c3ef`  
**Template:** `community-diversity-v2/games/racing_overtake`  
**Answer:** `Exactly 4 of the 4 numbered requirements are established.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=56.0000 [SAME-TEMPLATE]  
  ID: `f96df6132a4a094`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: games/racing_overtake  

  **Rank 2** — sim=0.0000 dist=57.0000 [SAME-TEMPLATE]  
  ID: `d050f2e7d425b242`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: games/racing_overtake  

  **Rank 3** — sim=0.0000 dist=57.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `809447d8795872f8`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: games/racing_overtake  

  **Rank 4** — sim=0.0000 dist=57.0000 [SAME-TEMPLATE]  
  ID: `2834593fa191840`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: games/racing_overtake  

  **Rank 5** — sim=0.0000 dist=57.0000 [SAME-TEMPLATE]  
  ID: `69e558e83426fcaa`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: games/racing_overtake  

### Same-Template Top-1 (potential boilerplate bias)

**ID:** `80447bcfcd98c680`  
**Kind:** score (levels=5)  
**Group:** `9e41454467677f0d`  
**Template:** `community-diversity-v2/games/racing_overtake`  
**Answer:** `Exactly 2 of the 4 numbered requirements are established.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=59.0000 [SAME-TEMPLATE]  
  ID: `33171739d314f1d4`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: games/racing_overtake  

  **Rank 2** — sim=0.0000 dist=59.0000 [SAME-TEMPLATE]  
  ID: `1e0c6fc6e7ae7ed3`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: games/racing_overtake  

  **Rank 3** — sim=0.0000 dist=60.0000 [SAME-TEMPLATE]  
  ID: `6ef0dc3888b142b2`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: games/racing_overtake  

  **Rank 4** — sim=0.0000 dist=61.0000 [SAME-TEMPLATE]  
  ID: `43d7e282a10ad1e1`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: games/racing_overtake  

  **Rank 5** — sim=0.0000 dist=61.0000 [SAME-TEMPLATE]  
  ID: `9d1089e80d6fd9c0`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: games/racing_overtake  

**ID:** `809447d8795872f8`  
**Kind:** score (levels=5)  
**Group:** `8d8e0f3fdcf3c3ef`  
**Template:** `community-diversity-v2/games/racing_overtake`  
**Answer:** `Exactly 4 of the 4 numbered requirements are established.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=56.0000 [SAME-TEMPLATE]  
  ID: `f96df6132a4a094`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: games/racing_overtake  

  **Rank 2** — sim=0.0000 dist=57.0000 [SAME-TEMPLATE]  
  ID: `d050f2e7d425b242`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: games/racing_overtake  

  **Rank 3** — sim=0.0000 dist=57.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `809447d8795872f8`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: games/racing_overtake  

  **Rank 4** — sim=0.0000 dist=57.0000 [SAME-TEMPLATE]  
  ID: `2834593fa191840`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: games/racing_overtake  

  **Rank 5** — sim=0.0000 dist=57.0000 [SAME-TEMPLATE]  
  ID: `69e558e83426fcaa`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: games/racing_overtake  

### Different-Template Top-1 (cross-template match)

**ID:** `85a5410ebaf33048`  
**Kind:** score (levels=3)  
**Group:** `8ef698da333a3f42`  
**Template:** `customer-control-v1:id:1`  
**Answer:** `The customer explicitly intends to keep using the service and continue their subscription.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=87.0000  
  ID: `c5ccb99b0d8202e9`  
  Answer: `The customer explicitly intends to keep using the service and continue their subscription.` (index=0)  
  Scenario: customer-control-v1:id:0  

  **Rank 2** — sim=0.0000 dist=87.0000 [SAME-TEMPLATE]  
  ID: `d43ec5cdf22df1a2`  
  Answer: `The customer explicitly intends to keep using the service and continue their subscription.` (index=0)  
  Scenario: customer-control-v1:id:1  

  **Rank 3** — sim=0.0000 dist=88.0000  
  ID: `ae2fb0a0899e7482`  
  Answer: `The customer explicitly intends to keep using the service and continue their subscription.` (index=0)  
  Scenario: customer-control-v1:id:0  

  **Rank 4** — sim=0.0000 dist=89.0000 [SAME-TEMPLATE]  
  ID: `a85ae2c459527793`  
  Answer: `The customer explicitly requests cancellation or states a definite decision to stop using the service.` (index=2)  
  Scenario: customer-control-v1:id:1  

  **Rank 5** — sim=0.0000 dist=89.0000  
  ID: `a474c395f4991402`  
  Answer: `The customer is undecided about continuing, or makes departure conditional on whether the issue is resolved.` (index=1)  
  Scenario: customer-control-v1:id:0  

**ID:** `85b3d5dd43836200`  
**Kind:** score (levels=3)  
**Group:** `f6c73448aac803aa`  
**Template:** `customer-control-v1:id:1`  
**Answer:** `The customer explicitly intends to keep using the service and continue their subscription.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=81.0000  
  ID: `477dc03508df076d`  
  Answer: `The customer explicitly intends to keep using the service and continue their subscription.` (index=0)  
  Scenario: customer-control-v1:id:0  

  **Rank 2** — sim=0.0000 dist=82.0000 [SAME-TEMPLATE]  
  ID: `284c876eb676e389`  
  Answer: `The customer is undecided about continuing, or makes departure conditional on whether the issue is resolved.` (index=1)  
  Scenario: customer-control-v1:id:1  

  **Rank 3** — sim=0.0000 dist=82.0000 [SAME-TEMPLATE]  
  ID: `9964a226e1c4ad78`  
  Answer: `The customer explicitly intends to keep using the service and continue their subscription.` (index=0)  
  Scenario: customer-control-v1:id:1  

  **Rank 4** — sim=0.0000 dist=82.0000 [SAME-TEMPLATE]  
  ID: `6f36ae76df96a7c4`  
  Answer: `The customer explicitly intends to keep using the service and continue their subscription.` (index=0)  
  Scenario: customer-control-v1:id:1  

  **Rank 5** — sim=0.0000 dist=85.0000  
  ID: `5413c69d6689f0f6`  
  Answer: `The customer is undecided about continuing, or makes departure conditional on whether the issue is resolved.` (index=1)  
  Scenario: customer-control-v1:id:0  

### Same Source Instance, Different Answer (counterfactual candidates)

**ID:** `a68a3d49125b0fbd`  
**Kind:** score (levels=5)  
**Group:** `2c078da5b2250949`  
**Template:** `community-diversity-v2/contracts/governing_law_venue`  
**Answer:** `Exactly 4 of the 4 numbered requirements are established.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=38.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `3535b7b910171d40`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: contracts/governing_law_venue  

  **Rank 2** — sim=0.0000 dist=38.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `a68a3d49125b0fbd`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: contracts/governing_law_venue  

  **Rank 3** — sim=0.0000 dist=38.0000 [SAME-TEMPLATE]  
  ID: `91034ecdc060ea18`  
  Answer: `Exactly 2 of the 4 numbered requirements are established.` (index=2)  
  Scenario: contracts/governing_law_venue  

  **Rank 4** — sim=0.0000 dist=38.0000 [SAME-TEMPLATE]  
  ID: `9fec07ac43a64299`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: contracts/governing_law_venue  

  **Rank 5** — sim=0.0000 dist=40.0000 [SAME-TEMPLATE]  
  ID: `edcd5bb2b1db9a83`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: contracts/governing_law_venue  

**ID:** `c05e7b088b9792d7`  
**Kind:** score (levels=5)  
**Group:** `a70ef6121391beb8`  
**Template:** `community-diversity-v2/shell_history/checksum_verify`  
**Answer:** `Exactly 3 of the 4 numbered requirements are established.`  
**Retrieved:** 5 results  

  **Rank 1** — sim=0.0000 dist=30.0000 [SAME-GROUP] [SAME-TEMPLATE] [SAME-SOURCE-INSTANCE]  
  ID: `f776a02a68d5a5a`  
  Answer: `Exactly 1 of the 4 numbered requirements are established.` (index=1)  
  Scenario: shell_history/checksum_verify  

  **Rank 2** — sim=0.0000 dist=30.0000 [SAME-TEMPLATE]  
  ID: `a5cc951e19d62f08`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: shell_history/checksum_verify  

  **Rank 3** — sim=0.0000 dist=30.0000 [SAME-TEMPLATE]  
  ID: `29a27dc1baf970fe`  
  Answer: `Exactly 4 of the 4 numbered requirements are established.` (index=4)  
  Scenario: shell_history/checksum_verify  

  **Rank 4** — sim=0.0000 dist=30.0000 [SAME-TEMPLATE]  
  ID: `fe78f8ce3656dc30`  
  Answer: `Exactly 3 of the 4 numbered requirements are established.` (index=3)  
  Scenario: shell_history/checksum_verify  

  **Rank 5** — sim=0.0000 dist=31.0000 [SAME-TEMPLATE]  
  ID: `f3a688d5537d2f6a`  
  Answer: `Exactly 1 of the 4 numbered requirements are established.` (index=1)  
  Scenario: shell_history/checksum_verify  

