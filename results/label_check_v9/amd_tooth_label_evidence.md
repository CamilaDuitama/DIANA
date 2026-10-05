# Tooth labels in AncientMetagenomeDir: evidence from the original papers

(Camila's document, received 2026-10-01, stored verbatim. Verification notes: amd_tooth_label_evidence_check.md.)

**What this is.** Most of DIANA's community_type false alarms are tooth runs. The benchmark assumes AMD labels are correct. This document checks that assumption against the original papers for the eight training-set studies with teeth. AMD release v26.03.0; training set only. "est." = read from a figure by eye.

## AMD's own rule

- AMD defines community_type as "the type of the original host's community the sample contains", which is content-based (`amd_v26.03.0_community_type_definition.md`).
- There is no rule for teeth.
- In v20.09, AMD re-listed teeth as "skeletal tissue" for Willmann 2018 and Brealey 2020. The other tooth studies were not revisited (same file, changelog).

## Evidence

| Study (project) | AMD label | Sampled | Oral content | Verdict |
|---|---|---|---|---|
| Philips 2017 (PRJNA354503) | oral | Tooth roots, surface bleached (paper, Data Description; Methods) | 140 teeth: median 0% oral, 88 at 0%, 21 >20% (`gigascience…s2.xlsx`, Suppl. Table 1, column O) | Mostly mismatch |
| Neukamm 2020 (PRJEB33848) | oral | Inside the pulp (`ncomms15694.pdf`, Methods p. 8) | 4 teeth >10%, the rest 0–5.8% (paper, "Oral microbiome assessment") | Mostly mismatch |
| Kazarina 2021 (PRJEB47251) | oral | Whole tooth, calculus removed (`…main.pdf`, §2.2) | Short libraries ≈58%, Total libraries ≈9% (`…mmc2.xlsx`; my classification) | Mixed |
| Jackson 2024 (PRJEB64128) | oral | Crowns and roots, pulp removed (paper, Results) | KGH1-E ≈75%, KGH2-B ≈35%, KGH2-F ≈12%, **KGH1-A ≈0%** (`supplement.pdf`, Fig. S1; est.) | KGH1-A mismatch |
| Brealey 2020, Ua6 (PRJEB33363) | oral | Bear calculus | No oral signature; excluded by authors (`msaa135.pdf`, p. 3004) | Mismatch |
| Hansen 2017 (PRJEB18722) | skeletal | Root cementum (`file.pdf`, "Targeted sampling" p. 3) | Not measured | Same material as Philips, opposite label |
| Mann 2018 (PRJNA445215) | skeletal | Pulp chamber (`s41598…pdf`, Methods p. 12) | Mostly soil; 7 of 46 >20% oral (p. 10) | Mostly fits |
| Ozga 2016, Norris Farms (PRJNA293758) | skeletal | Dentine (`Ozga…pdf`, p. 222) | **NF47 56%, NF217 78% oral**, possible caries (`s41598…pdf`, p. 5) | Mismatch: actually oral |
| Brealey 2020, Rt_tooth (PRJEB33363) | skeletal | Tooth segment without calculus, a control (`msaa135.pdf`, p. 3004) | ≈0% oral (SuppMat Fig. S3; est.) | Fits |

**Caveats.**
- Some oral DNA in dentine may come from after death (Mann, p. 10).
- For bears and reindeer, a low oral share can reflect gaps in human-based reference databases (Brealey, pp. 3011–3012).

## Other metadata errors

- ENA describes Ozga's NF-217D as "dental calculus"; it is dentine (`Ozga…pdf`, Table 1).
- Brealey's Table S1 swaps the sampling sites of Ua7 and Ua9 relative to the main text (pp. 3004, 3006).
- Jackson's KGH2 radiocarbon date is called problematic by the authors (`supplement.pdf`, Suppl. Note 2). Check AMD's recorded age.
- PRJEB42014 has `sample_age = 100` for all runs, while the collection years are 1842–2016.

## Conclusion

AMD labels teeth inconsistently, and in several studies the label does not match what the authors measured, in both directions. Some of DIANA's tooth "false alarms" are therefore likely correct flags.
