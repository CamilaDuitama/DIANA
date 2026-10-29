# Tooth labels in AncientMetagenomeDir: evidence from the original papers

(Camila's document, version of 2026-10-05, stored verbatim; supersedes the 2026-10-01 version,
kept as amd_tooth_label_evidence_v2_superseded.md. Verification notes:
amd_tooth_label_evidence_check.md.)

**What this is.** Most of DIANA's community_type false alarms are tooth runs. The benchmark assumes AMD labels are correct. This document checks that assumption against the original papers for the eight training-set studies with teeth. AMD release v26.03.0; training set only. "est." = read from a figure by eye.

## AMD's own rule

- AMD defines community_type as "the type of the original host's community the sample contains", which is content-based (`amd_v26.03.0_community_type_definition.md`).
- There is no rule for teeth.
- In v20.09, AMD re-listed teeth as "skeletal tissue" for Willmann 2018 and Brealey 2020. The other tooth studies were not revisited (same file, changelog).

## How the labels were decided (AMD GitHub history)

- **August 2020, one week, same curator** (issues #33, #35, #36; release v20.09):
  - **Philips (#35):** kept "oral", on the grounds that the teeth carry an oral microbiome signature.
  - **Willmann (#36):** moved to skeletal, on the grounds that the sample is not calculus.
  - **Brealey tooth control (#33):** tied to the same discussion.
- **So two different criteria were applied the same week:** content for Philips, material for Willmann.
- **The outcome is the reverse of the measured content.** Philips teeth have a median of 0% oral, yet are labelled oral. Willmann teeth carry oral pathogens in 5 of 6, yet are labelled skeletal.
- **Neukamm, Jackson and Kazarina were never discussed** for this label. No issue or pull request on community_type and teeth exists after 2020.
- **People to ask:** jfy133, ivelsko.

## How the inconsistency is shown

The same kind of sample gets opposite labels in different studies:

| Material sampled | Labelled "oral" | Labelled "skeletal tissue" |
|---|---|---|
| Tooth root | Philips 2017 | Hansen 2017 |
| Pulp chamber | Neukamm 2020 | Mann 2018 |
| Whole tooth, calculus removed | Kazarina 2021 | Willmann 2018 |

No single rule (part sampled, reason for sampling, or measured content) reproduces AMD's labels. In the third pair, the labels are the opposite of the content: Willmann's teeth carry oral pathogens but are labelled skeletal; Kazarina's are mostly soil but labelled oral.

## Evidence

| Study (project) | AMD label | Sampled | Oral content | Verdict |
|---|---|---|---|---|
| Philips 2017 (PRJNA354503) | oral | Tooth roots, surface bleached (paper, Data Description; Methods) | 140 teeth: median 0% oral, 88 at 0%, 21 >20% (`gigascience…s2.xlsx`, Suppl. Table 1, column O) | Mostly mismatch |
| Neukamm 2020 (PRJEB33848) | oral | Inside the pulp (`ncomms15694.pdf`, Methods p. 8) | 4 teeth >10%, the rest 0–5.8% (paper, "Oral microbiome assessment") | Mostly mismatch |
| Kazarina 2021 (PRJEB47251) | oral | Whole tooth, calculus removed (`…main.pdf`, §2.2) | Short libraries ≈58%, Total libraries ≈9% (`…mmc2.xlsx`; my classification) | Mixed |
| Jackson 2024 (PRJEB64128) | oral | Crowns and roots, pulp removed (paper, Results) | KGH1-E ≈75%, KGH2-B ≈35%, KGH2-F ≈12%, **KGH1-A ≈0%** (`supplement.pdf`, Fig. S1; est.) | KGH1-A mismatch |
| Brealey 2020, Ua6 (PRJEB33363) | oral | Bear calculus | No oral signature; excluded by authors (`msaa135.pdf`, p. 3004) | Mismatch |
| Willmann 2018 (PRJNA302605; held-out) | skeletal (changed from oral in v20.09) | Whole tooth, calculus abraded, bleached, powdered; an oral-health study (`Hm1uQ7-file.pdf`, Methods "Tooth and bone preparation", p. 5) | Oral pathogens in 5 of 6 teeth, none in tooth 308; bones = soil controls (p. 9) | Mismatch: mostly oral, labelled skeletal |
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

## Does DIANA follow the content?

DIANA's out-of-fold predictions (study held out) compared with what the papers measured. Per-run data: `tooth_content_check_runs.tsv`.

| Check | Result | Follows content? |
|---|---|---|
| Philips: p(oral) vs measured oral, 140 teeth | ρ = 0.59 [0.44, 0.70]; 0.31 [0.13, 0.47] after controlling for depth. 16 of 21 oral-rich teeth predicted oral; 0 of 88 zero-oral teeth | Yes |
| Philips vs Hansen (opposite labels, same material) | Same predictions: 85% vs 86% skeletal | Yes |
| Norris Farms (labelled skeletal) | NF47 and NF217 predicted oral (one Ozga run has almost no data) | Yes |
| Mann dentine | ρ = 0.33, gone after controlling for depth | No |
| Neukamm | 62 of 69 teeth predicted oral, although most contain no oral DNA | No |
| Jackson KGH1-A | Not lower than the other samples | No |
| Brealey Ua6 | Predicted oral, like the other bears | No |

## Conclusion

AMD labels teeth inconsistently, and in several studies the label does not match what the authors measured, in both directions. In the largest tooth study (Philips), DIANA's disagreements with the AMD label match the authors' own measurements, so the benchmark counts correct flags as errors. Within other studies, DIANA mostly gives all samples the same answer, so it cannot yet verify individual samples everywhere.

## Evidence table for the current split (v19, added 2026-10-27)

Rows = BioProjects with tooth runs in the v19 split (`data/splits_v19/{train,test}_metadata.tsv`, `material == tooth`). Prediction = released DIANA (shared network, 5 seeds), held-out community_type, top class (`results/heldout_v19/diana_net/probs_DIANA_community_type.tsv`). The six BioProjects not covered above were read on 2026-10-27, one agent per paper (paper, supplementary files, ENA run to sample map); their findings are summarised below the table.

**Our labels equal AMD's.** Every run was traced through its library row to its sample row in AMD v26.03.0 (run to library: one run per row, no lists; sample accession lists expanded within one sample). All 544 tooth runs carry the community_type of their own AMD sample. In AMD's raw sample rows, all tooth samples of a BioProject share one community_type. The other differences between our labels and AMD over all 3,638 runs are the deliberate rules of `scripts/data_prep/05_build_v9_labels.py` (12 McDonough2018 runs masked, 67 runs with a rare material masked, 14 Rampelli2021 runs keeping both tables' labels).

| BioProject | Paper | Split | Tooth runs | AMD community label | Part sampled (paper) | Oral content (paper) | Label vs content | DIANA held-out prediction (oral / skeletal / soft tissue) |
|---|---|---|---|---|---|---|---|---|
| PRJNA354503 | Philips 2017 | train | 165 | oral | tooth root | median 0 %; 88 of 140 teeth at 0 %, 21 above 20 % | mostly mismatch | |
| PRJEB41240 | Seguin-Orlando 2021 | train | 104 | oral | tooth cementum; tooth and calculus removal not reported | 3 of 10 teeth 4.8 to 39.5 % (68 runs); 7 teeth 0 % at the paper's 1 % cutoff (36 runs) | 7 of 10 teeth mismatch | |
| PRJEB33848 | Neukamm 2020 | train | 69 | oral | pulp | 4 teeth above 10 %, the rest 0 to 5.8 % | mostly mismatch | |
| PRJNA445215 | Mann 2018 | train | 60 | skeletal tissue | pulp chamber | mostly soil; 7 of 46 above 20 % | mostly fits | |
| PRJEB67998 | Bozzi 2024 (samples described in Cruz Dávalos 2022, preprint) | train | 18 | skeletal tissue | dentine (preprint) | preprint: above 10 % of reads from eight oral species in 20 of 22 individuals (group statement); Bozzi SourceTracker2: at most 2.1 % per tooth (est.) | cannot tell (sources disagree) | |
| PRJNA302605 | Willmann 2018 | train | 3 | skeletal tissue | whole tooth, calculus abraded | oral pathogens in 5 of 6 teeth | mismatch | |
| PRJNA293758 | Ozga 2016 | train | 2 | skeletal tissue | dentine | 56 % and 78 % | mismatch | |
| PRJEB64128 | Jackson 2024 | held-out | 68 | oral | crown and root, pulp removed | 12 to 75 %; KGH1-A about 0 % (est.) | KGH1-A mismatch | 54 / 14 / 0 |
| PRJEB18722 | Hansen 2017 (data reanalysed by Bozzi 2024, under whose name AMD lists it) | held-out | 21 | skeletal tissue | root cementum | Hansen: not measured. Bozzi SourceTracker2 (est.): V9T 30 %, V10T 23 %, H4T 8 %, the other 18 at most 2 % | 2 mismatch, 1 cannot tell, 18 fit | 15 / 6 / 0 |
| PRJEB47251 | Kazarina 2021 | held-out | 10 | oral | whole tooth, calculus removed | short libraries about 58 %, total about 9 % | mixed | 9 / 1 / 0 |
| PRJEB78615 | Guinet 2025 (woolly mammoth) | held-out | 9 | skeletal tissue | tooth; part not reported | no proportion measured. Actinobacillus/Basfia-like clade in 8 teeth, which the authors propose was part of the mammoth oral microbiome | 8 mismatch by the authors' interpretation, 1 cannot tell | 7 / 0 / 2 |
| PRJEB56776 | Ķimsis 2023 | held-out | 9 | oral | molar or canine; part not reported | no proportion measured. Damage-authenticated oral taxa in 4 teeth (LZP10T, LZP26T, LZP27T, LZP40T); none reported in 5 | 4 fit, 5 cannot tell | 4 / 5 / 0 |
| PRJEB33363 | Brealey 2020 (Rt_tooth, all 4 runs) | held-out | 4 | skeletal tissue | reindeer tooth surface without calculus (control) | about 0 % | fits | 0 / 0 / 4 |
| PRJEB44723 | Sarhan 2023 | held-out | 1 | oral | whole premolar, ground | no proportion measured; five oral species named (2.1 % of reads by the agent's count) | fits (not quantified) | 1 / 0 / 0 |
| PRJEB58818 | Fontani 2023 | held-out | 1 | oral | root cementum and dentine of a molar | not measured; pathogen screen including three oral species below the authors' threshold | cannot tell | 1 / 0 / 0 |

**Pattern.** Each BioProject gives all its teeth one community label, in AMD itself. The label does not follow the part sampled (root or cementum is "oral" in Philips, Seguin-Orlando and Fontani, "skeletal tissue" in Hansen) nor the measured content: within one BioProject, teeth with and without oral content carry the same label (Seguin-Orlando, Ķimsis, Hansen, Jackson).

**DIANA on held-out teeth against the papers' content.**

| BioProject | Teeth with oral content reported | of which DIANA says oral | Teeth without | of which DIANA says oral |
|---|---|---|---|---|
| Ķimsis 2023 | 4 (damage-authenticated oral taxa) | 4 | 5 (none reported) | 0 |
| PRJEB18722 | 3 (SourceTracker2 8 to 30 %, est.) | 3 | 18 (at most 2 %, est.) | 12 |
| Guinet 2025 | 8 (Actinobacillus/Basfia-like clade) | 6 | 1 (L422) | 1 |

In PRJEB18722 DIANA says oral for all 11 UK Viking teeth (V1T to V11T, recorded by Hansen as poorly preserved) and for 4 of the 10 Danish teeth.

**Other metadata findings (2026-10-27 checks).**
- Sarhan 2023 sample 2132: AMD material "stomach"; paper and ENA "gut tissue" (masked by us as a rare material). Tooth 2077 has 5 ENA runs, AMD lists 1.
- AMD omits samples: PRJEB67998 lists 19 of 24 (MN00019, MN00066, MN00118, MN00119, MN00316 missing); Guinet 2025 MD228 and P033 are in AMD but not among our runs.
- sample_age: Seguin-Orlando, AMD equals the uncalibrated radiocarbon age rounded to 100 for 22 of 25 samples, GBVPK about 500 years off; Guinet 2025, three ages (FK012, FK033, MD213) have no date in the paper.
- Read counts differing from the paper: Fontani tooth run, Ķimsis LZP22K.
- No wrong host, wrong material class we use, or run to sample swap found in any of the six.
