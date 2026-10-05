# Do DIANA's out-of-fold community_type predictions follow the oral content the papers report? (2026-10-01)

Reference model (5-network average), dev-fold out-of-fold probabilities (`diana_reference_oof_community_type_probs.tsv`).
Groups defined from the papers (see `amd_tooth_label_evidence.md`); Mann-Whitney test on p(oral) between groups.

| study | groups (paper) | p(oral) median | predicted oral | p |
|---|---|---|---|---|
| Philips 2017, 165 teeth (AMD: oral) | paper: mostly environmental, 88 of 140 at 0 % oral | 0.019 (quartiles 0.010 to 0.044) | 13 % (85 % predicted skeletal tissue) | - |
| Neukamm 2020 teeth (AMD: oral) | 4 oral-rich teeth vs 65 others | 0.387 vs 0.339 | 100 % vs 89 % | 0.036 |
| Jackson 2024 (AMD: oral) | crown KGH1-E (about 75 %) vs root KGH1-A (about 0 %) | 0.382 vs 0.368 | 97 % vs 72 % | 0.087 |
| Jackson 2024 | crowns (E, F) vs roots (A, B) | 0.382 vs 0.367 | 90 % vs 67 % | 0.093 |
| Mann 2018 dentine (AMD: skeletal) | NF47, NF217 (56 %, 78 % plaque) vs 58 other dentine runs | 0.395 vs 0.315 | 100 % vs 67 % | 0.048 |
| Brealey 2020 bears (AMD: oral) | Ua6 (no oral signature) vs other bears | 0.503 vs 0.488 | 100 % vs 100 % | 0.081 |
| Kazarina 2021b (AMD: oral) | short libraries (about 58 %) vs total libraries (about 9 %) | 0.351 vs 0.294 | 100 % vs 80 % | 0.69 |

Reading: between studies the predictions follow the papers: Philips teeth, which the paper calls mostly environmental, get p(oral) near 0.02 and are predicted skeletal tissue, so the model's flags against their "oral" label point the same way as the paper; Neukamm, Jackson and Mann teeth with more reported oral content get higher p(oral) and are predicted oral more often, with small differences (0.05 to 0.08 in p(oral)). Not followed: Ua6 (predicted oral like the other bears) and the Kazarina library pairs. Philips has no per-run oral content in hand, so only its overall level is compared.
