#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --job-name=net_rescore_v19
#SBATCH --output=logs/net_v19/rescore_%a_%j.out
#SBATCH --error=logs/net_v19/rescore_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# DIANA without logistic regression (PROTOCOLS.md v19 amendment 2026-10-26): every result that uses DIANA's outputs, recomputed
# from stored outputs into new folders (*net*). Nothing refitted.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python; DN=results/heldout_v19/diana_net; DV=results/devfold_probs_v19net; H=results/heldout_v19/baselines/{task}; B=results/baselines_v19/{task}
M=results/multiplicity_v19
# The network-only tables are the shared network's own tables (held-out: 49_ on the refit; dev folds: 42_), copied once,
# never overwritten; an existing copy that differs stops the job.
for t in community_type feature sample_host material; do
    for pair in "results/heldout_v19/diana_refit/probs_Shared_$t.tsv $DN/probs_DIANA_$t.tsv" "results/devfold_probs_v19/probs_Shared_$t.tsv $DV/probs_DIANA_$t.tsv"; do
        read -r SRC DST <<< "$pair"; mkdir -p "$(dirname "$DST")"
        [ -e "$DST" ] || { cp "$SRC" "$DST.$SLURM_ARRAY_TASK_ID.tmp" && mv "$DST.$SLURM_ARRAY_TASK_ID.tmp" "$DST"; }
        cmp -s "$SRC" "$DST" || { echo "$DST differs from $SRC" >&2; exit 1; }
    done
done
HMODELS=(--model "DIANA=$DN/probs_DIANA_{task}.tsv" --model "LogReg=$H/probs_LogisticRegression_Bal.tsv"
         --model "RandomForest=$H/probs_RandomForest_Bal.tsv" --model "kNN=$H/probs_kNN.tsv" --model "Majority=$H/probs_MajorityClass.tsv")
PAIRS=(--pair DIANA:LogReg --pair DIANA:RandomForest)
HT=(--heldout --plantings results/planted_v19/heldout)
DEVFLAG() { $PY scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 --plantings results/planted_v19/dev --pattern "planted_train_${1}_r0.1_s*.tsv" "${@:2}"; }
case $SLURM_ARRAY_TASK_ID in
 0) $PY scripts/analysis/75_classification_metrics_v19.py --heldout --diana "$DN/probs_DIANA_{task}.tsv" --out "$M/net_classification_heldout" ;;
 1) for kind in mixed record; do
      $PY scripts/analysis/41_flagging_v19.py "${HT[@]}" --split-dir data/splits_v19 --pattern "planted_test_${kind}_r0.1_s*.tsv" \
          "${HMODELS[@]}" "${PAIRS[@]}" --out "results/heldout_v19/scores_net/flag_$([ $kind = mixed ] && echo single || echo record)"
    done ;;
 2) $PY scripts/analysis/68_regime_tables_v19.py "${HT[@]}" --pattern "planted_test_mixed_r0.1_s*.tsv" "${HMODELS[@]}" "${PAIRS[@]}" --out "$M/net_regime_single.tsv" ;;
 3) $PY scripts/analysis/68_regime_tables_v19.py --whole-field "${HT[@]}" --pattern "planted_test_mixed_r0.1_s*.tsv" "${HMODELS[@]}" "${PAIRS[@]}" --out "$M/net_whole_field.tsv" ;;
 4) $PY scripts/analysis/73_real_mislabel_replay_v19.py --diana "$DN/probs_DIANA_{task}.tsv" --cutoffs results/posthoc_v19/abstention_v19_coverage80.json \
        --out results/heldout_v19/real_mislabels_net
    $PY scripts/analysis/83_prior_correction_v19.py --net
    $PY scripts/analysis/85_prior_confidence_v19.py --net
    for kind in mixed record; do
      DEVFLAG $kind --model "DIANA=$DV/probs_DIANA_{task}.tsv" --model "DIANAprior=results/prior_v19net/probs_DIANAprior_{task}.tsv" \
          --pair DIANAprior:DIANA --out "results/prior_v19net/flag_${kind}"
    done ;;
 5) $PY scripts/analysis/81_depth_tuned_v19.py --net ;;
 6) $PY scripts/analysis/76_consensus_disagreement_v19.py --net
    $PY scripts/analysis/74_class_studies_f1_v19.py --net
    $PY scripts/analysis/78_faure_amd_table_v19.py --net --faure data/external/faure/metadata_correction.tsv
    $PY scripts/analysis/46_posthoc_v19.py temperature --net
    DEVFLAG mixed --model "DIANA=$DV/probs_DIANA_{task}.tsv" --model "DIANAcal=$DV/probs_DIANAcal_{task}.tsv" --pair DIANAcal:DIANA \
        --out results/rescore_v19/dev_single_calibration_net ;;
 7) for kind in mixed record; do
      DEVFLAG $kind --model "DIANA=$DV/probs_DIANA_{task}.tsv" --model "DIANA_CL=$DV/probs_DIANA_{task}.tsv" --cl DIANA_CL --pair DIANA_CL:DIANA \
          --out "results/flagging_v19/dev_cl_net_${kind}"
    done
    $PY scripts/analysis/43_classification_v19.py --model "DIANA=$DV/probs_DIANA_{task}.tsv" --model "DIANAW=results/studyweight_v19/probs_SharedW_{task}.tsv" \
        --pair DIANAW:DIANA --out results/classification_v19/dev_studyweight_net
    DEVFLAG mixed --model "DIANA=$DV/probs_DIANA_{task}.tsv" --model "DIANAW=results/studyweight_v19/probs_SharedW_{task}.tsv" --pair DIANAW:DIANA \
        --out results/rescore_v19/dev_single_studyweight_net ;;
esac
echo "=== COMPLETE task $SLURM_ARRAY_TASK_ID $(date) ==="
