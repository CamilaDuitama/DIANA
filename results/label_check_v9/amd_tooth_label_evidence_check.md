# Check of amd_tooth_label_evidence.md against sources read on 2026-10-01

Sources I read directly: AMD v26.03.0 tables, schema, documentation and CHANGELOG (local git clone); ENA/BioSample records (API); the open-access full texts of Philips 2017, Neukamm 2020, Jackson 2024, Brealey 2020, Hansen 2017, Mann 2018, Ozga 2016 (Europe PMC); the Kazarina 2021b PDF. Supplements were not read.

| claim | status |
|---|---|
| AMD definition is content-based; no rule for teeth; v20.09 re-listed Willmann 2018 and Brealey 2020 teeth as skeletal tissue | verified (schema, documentation, CHANGELOG lines 884-885) |
| Philips: tooth roots (dentine + cementum), surfaces bleached; most microbes environmental | verified in the main text. The per-tooth shares (140 teeth, 88 at 0 %, 21 > 20 %) come from Suppl. Table 1, not read by me. Note: the archive and our training set hold 165 runs; 25 are outside the 140 that passed the paper's selection (Table 1: 161 analysed, 140 passed). |
| Neukamm: 4 teeth > 10 % oral, rest 0 to 5.8 % | verified in the main text. "Inside the pulp" is cited from ncomms15694 (Schuenemann 2017, the earlier Abusir study), not read by me. |
| Kazarina: whole tooth, calculus removed; two libraries per tooth | verified (PDF §2.2, §2.3; ENA read counts match Table 2: _nesk = short library, _sk/_skelt = total library). The 58 % / 9 % oral shares are Camila's classification of mmc2, not checked. The main text lists six oral species among the top ten of the short fraction. |
| Jackson: crowns and roots, pulp removed; KGH1-E about 75 % oral | verified in the main text. The main text also says "good oral microbiome preservation in both crowns and one root" and reports T. forsythia at 1x and T. denticola at 1.95x from KGH2-F, so the KGH2-F ≈ 12 % estimate from Fig. S1 sits uneasily with the authors' wording; not resolved. KGH1-A ≈ 0 % is consistent with "one root". |
| Brealey Ua6: no oral signature, excluded from microbial analyses | verified: "one bear sample (Ua6) contained high levels of contaminants (>70 %) and no detectable oral microbiome signature after contamination filtering and was therefore excluded from all microbial analyses". Ua6 is in our training set (4 runs, labelled oral); DIANA predicts oral for all four, mean p 0.50. |
| Hansen: root cementum; P = petrous, S = parietal, T = tooth | verified |
| Mann: dentin from the pulp chamber; 15 % of dentin samples > 20 % oral | verified (7 of 46 = 15.2 %) |
| Ozga NF47 / NF217 dentine 56 % and 78 % from dental plaque, caries not excluded, samples dropped from Mann's downstream analyses | verified in Mann 2018 ("56.2 % and 78.4 % derived from dental plaque", SourceTracker). Both individuals are in our training set twice: Ozga 2016 (fold 0; dentine runs with 404 and 18 non-zero unitigs) and Mann 2018 (fold 2; 78k and 83k). DIANA predicts oral for three of the four dentine runs. |
| Rt_tooth distinct from calculus | verified (main text; the ≈ 0 % figure is from Fig. S3, not read). DIANA predicts soft tissue for all four runs, p(oral) 0.004 to 0.011. |
| ENA free text of NF-217D says dental calculus; BioSample and paper say dentine | verified |
| Brealey Table S1 swaps Ua7 and Ua9 sites | not checked (supplement) |
| Jackson KGH2 date problematic (Suppl. Note 2) | not checked (supplement). AMD v26.03.0 records sample_age 3800 for all four KGH samples. |
| PRJEB42014 sample_age = 100 for all runs | verified: all 79 AMD samples of Brealey2021 carry 100 |

Our labels agree with AMD for every run of the eight studies (checked 2026-10-01, see the ena_*_run_sample_map.tsv files). The disagreement is between AMD's per-study conventions and, in the cases above, between the label and what the authors measured.

## Provenance of the tooth convention in AMD itself (checked 2026-10-02, GitHub issues and git history)

- **The split was a deliberate curator judgement in August 2020, made per study and never revisited.**
  Issue #35 (jfy133, 2020-08-03, about the Philips data): the teeth are "_not_ calculus but derived
  from teeth **with signature of oral microbiome**. A new entry in `material` will need to be made."
  Issue #36 (Willmann 2018, same day, same curator): "Note this is _not_ from calculus, and will
  need a new category", with a follow-up comment assigning the bone samples group "skeleton".
  Issue #33 (Brealey 2020): the tooth control "will contain the tooth sample as mentioned in the
  comments of #35". So Philips teeth were judged oral-by-content, Willmann teeth not oral, in the
  same week by the same curators; the v20.09 changelog lines "re-list the community_type of tooth
  samples as skeletal tissue" (Willmann2018, Brealey2020) record that decision.
- **The curators' belief about Philips conflicts with Philips' own supplement** (median 0 % oral,
  88 of 140 teeth at 0 %), which is the crux of the paper's discussion point: the oral label was a
  content judgement, and the measured content does not support it.
- **Philips, Neukamm, Jackson and Kazarina were never re-discussed for this label.** Every later
  issue touching them is about dates, duplicates or missing libraries (#1525, #1521 revised dates
  2025; #928 the Willman/Willmann spelling; #1160, #447, #634 additions). No issue or PR mentions
  community_type for teeth after 2020; a search for "community_type tooth" returns nothing.
- No issue or PR number is attached to the changelog lines themselves; the trail is issues #33,
  #35, #36 and the v20.09 release.
