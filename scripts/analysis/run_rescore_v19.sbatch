#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-27
#SBATCH --job-name=rescore_v19
#SBATCH --output=logs/rescore_v19/task_%a_%j.out
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Re-score every stored v19 mislabel-detection and regime table with the scoring fixes of
# 2026-10-15/16 (code review: unattainable budgets are "can't check", attainability fixed on
# the full data, regime resamples as 41_/43_, regime pairs over plantings both models reach).
# Same stored predictions, same plantings; nothing is refitted and no new prediction is read.
# Outputs go to results/rescore_v19/ beside the originals for comparison.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python; O=results/rescore_v19; mkdir -p "$O"
D=results/devfold_probs_v19; B=results/baselines_v19/{task}; W=results/studyweight_v19
H=results/baselines_hv_v19/{task}; HD=results/baselines_hvdb_v19/{task}
V=results/baselines_sim_v19/{task}; P=results/baselines_simprofile_v19/{task}
TU=results/train_fit_v19/baselines/{task}; TH=results/train_fit_v19/baselines_hvdb/{task}
HU=results/heldout_v19/baselines/{task}; HH=results/heldout_v19/baselines_hvdb/{task}; HDI=results/heldout_v19/diana
DEV=(--split-dir data/splits_v19 --plantings results/planted_v19/dev --pattern "planted_train_mixed_r0.1_s*.tsv")
F41() { $PY scripts/analysis/41_flagging_v19.py "$@"; }
R68() { $PY scripts/analysis/68_regime_tables_v19.py "$@"; }
LR="LogReg=$B/oof_probs_LogisticRegression_Bal.tsv"; RF="RandomForest=$B/oof_probs_RandomForest_Bal.tsv"
KNN="kNN=$B/oof_probs_kNN.tsv"; MAJ="Majority=$B/oof_probs_MajorityClass.tsv"; SVM="LinearSVM=$B/oof_pred_LinearSVM_Bal.tsv"
hvm() {  # unitig and hypervector/similarity models: $1 = folder template of the second input, $2 = suffix
    echo --model "$LR" --model "LogReg$2=$1/oof_probs_LogisticRegression_Bal.tsv" --model "$RF" \
         --model "RandomForest$2=$1/oof_probs_RandomForest_Bal.tsv" --model "$KNN" --model "kNN$2=$1/oof_probs_kNN.tsv"
}
hvp() { echo --pair "LogReg$1:LogReg" --pair "RandomForest$1:RandomForest" --pair "kNN$1:kNN"; }
held() {  # $1 unitig folder, $2 second folder (hvdb) or empty, $3 DIANA folder or empty
    local a=(--model "LogReg=$1/probs_LogisticRegression_Bal.tsv" --model "RandomForest=$1/probs_RandomForest_Bal.tsv"
             --model "kNN=$1/probs_kNN.tsv")
    [ -n "$3" ] && a=(--model "DIANA=$3/probs_DIANA_{task}.tsv" "${a[@]}" --model "Majority=$1/probs_MajorityClass.tsv")
    [ -n "$2" ] && a+=(--model "LogRegHV=$2/probs_LogisticRegression_Bal.tsv" --model "RandomForestHV=$2/probs_RandomForest_Bal.tsv" --model "kNNHV=$2/probs_kNN.tsv")
    [ -z "$2" ] && [ -z "$3" ] && a+=(--model "Majority=$1/probs_MajorityClass.tsv")
    echo "${a[@]}"
}
HT=(--heldout --split-dir data/splits_v19 --plantings results/planted_v19/heldout)
case $SLURM_ARRAY_TASK_ID in
 1) F41 "${DEV[@]}" --model "Separate=$D/probs_Separate_{task}.tsv" --model "Shared=$D/probs_Shared_{task}.tsv" \
       --model "Separate1=$D/probs_Separate1_{task}.tsv" --model "Shared1=$D/probs_Shared1_{task}.tsv" --model "$LR" --model "$RF" --model "$KNN" --model "$MAJ" \
       --pair Separate:Shared --pair Separate:Separate1 --pair Shared:Shared1 --pair Separate:LogReg --pair Shared:LogReg \
       --pair Separate:RandomForest --pair Shared:RandomForest --out $O/dev_single ;;
 2) R68 --model "Separate=$D/probs_Separate_{task}.tsv" --model "Shared=$D/probs_Shared_{task}.tsv" --model "$LR" --model "$RF" --model "$KNN" \
       --pred-only "$SVM" --out $O/regime_dev.tsv ;;
 3) F41 "${DEV[@]}" --model "$LR" --model "$RF" --model "$KNN" --model "$MAJ" --out $O/dev_single_baselines ;;
 4) R68 --model "$LR" --model "$RF" --model "$KNN" --model "$MAJ" --pred-only "$SVM" --out $O/regime_baselines_dev.tsv ;;
 5) F41 "${DEV[@]}" --model "Separate=$D/probs_Separate_{task}.tsv" --model "SeparateLR=$D/probs_SeparateLR_{task}.tsv" \
       --model "Shared=$D/probs_Shared_{task}.tsv" --model "SharedLR=$D/probs_SharedLR_{task}.tsv" \
       --pair SeparateLR:Separate --pair SharedLR:Shared --pair SeparateLR:SharedLR --out $O/dev_single_composition ;;
 6) F41 "${DEV[@]}" --model "DIANA=$D/probs_DIANA_{task}.tsv" --model "Shared=$D/probs_Shared_{task}.tsv" --model "$LR" --model "$RF" \
       --pair DIANA:LogReg --pair DIANA:RandomForest --pair DIANA:Shared --out $O/dev_single_diana ;;
 7) R68 --model "DIANA=$D/probs_DIANA_{task}.tsv" --model "$LR" --model "$RF" --pred-only "$SVM" --out $O/regime_diana_dev.tsv ;;
 8) F41 "${DEV[@]}" --model "DIANA=$D/probs_DIANA_{task}.tsv" --model "DIANAcal=$D/probs_DIANAcal_{task}.tsv" --pair DIANAcal:DIANA \
       --out $O/dev_single_calibration ;;
 9|10) SW=(--model "DIANA=$D/probs_DIANA_{task}.tsv" --model "DIANAW=$W/probs_DIANAW_{task}.tsv" --model "$LR" \
           --model "LogRegW=$W/baselines/{task}/oof_probs_LogisticRegression_Bal.tsv")
       if [ $SLURM_ARRAY_TASK_ID = 9 ]; then F41 "${DEV[@]}" "${SW[@]}" --pair DIANAW:DIANA --pair LogRegW:LogReg --out $O/dev_single_studyweight
       else R68 "${SW[@]}" --out $O/regime_studyweight_dev.tsv; fi ;;
 11) F41 "${DEV[@]}" $(hvm "$H" HV) $(hvp HV) --out $O/dev_single_hv ;;
 12) R68 $(hvm "$H" HV) --pred-only "$SVM" --pred-only "LinearSVMHV=$H/oof_pred_LinearSVM_Bal.tsv" $(hvp HV) --pair LinearSVMHV:LinearSVM --out $O/regime_hv_dev.tsv ;;
 13) F41 "${DEV[@]}" $(hvm "$HD" HV) $(hvp HV) --out $O/dev_single_hvdb ;;
 14) R68 $(hvm "$HD" HV) --pred-only "$SVM" --pred-only "LinearSVMHV=$HD/oof_pred_LinearSVM_Bal.tsv" $(hvp HV) --pair LinearSVMHV:LinearSVM --out $O/regime_hvdb_dev.tsv ;;
 15|16) SIM=(--model "$LR" --model "$RF" --model "$KNN" --model "SimVote=$V/oof_probs_SimVote.tsv"
             --model "LogRegSP=$P/oof_probs_LogisticRegression_Bal.tsv" --model "RandomForestSP=$P/oof_probs_RandomForest_Bal.tsv" --model "kNNSP=$P/oof_probs_kNN.tsv")
        SP=(--pair SimVote:kNN --pair SimVote:LogReg --pair LogRegSP:LogReg --pair RandomForestSP:RandomForest --pair kNNSP:kNN)
        if [ $SLURM_ARRAY_TASK_ID = 15 ]; then F41 "${DEV[@]}" "${SIM[@]}" "${SP[@]}" --out $O/dev_single_sim
        else R68 "${SIM[@]}" --pred-only "$SVM" --pred-only "LinearSVMSP=$P/oof_pred_LinearSVM_Bal.tsv" "${SP[@]}" --pair LinearSVMSP:LinearSVM --out $O/regime_sim_dev.tsv; fi ;;
 17) R68 --train-fit --model "DIANA=results/train_fit_v19/probs_DIANA_{task}.tsv" --model "LogReg=$TU/probs_LogisticRegression_Bal.tsv" \
        --model "RandomForest=$TU/probs_RandomForest_Bal.tsv" --model "kNN=$TU/probs_kNN.tsv" --model "Majority=$TU/probs_MajorityClass.tsv" \
        --pred-only "LinearSVM=$TU/pred_LinearSVM_Bal.tsv" --out $O/train_fit_regime_single.tsv ;;
 18) R68 --train-fit --model "LogReg=$TU/probs_LogisticRegression_Bal.tsv" --model "LogRegHV=$TH/probs_LogisticRegression_Bal.tsv" \
        --model "RandomForest=$TU/probs_RandomForest_Bal.tsv" --model "RandomForestHV=$TH/probs_RandomForest_Bal.tsv" \
        --model "kNN=$TU/probs_kNN.tsv" --model "kNNHV=$TH/probs_kNN.tsv" --pred-only "LinearSVM=$TU/pred_LinearSVM_Bal.tsv" \
        --pred-only "LinearSVMHV=$TH/pred_LinearSVM_Bal.tsv" $(hvp HV) --pair LinearSVMHV:LinearSVM --out $O/train_fit_regime_hvdb.tsv ;;
 19) F41 "${HT[@]}" --pattern "planted_test_mixed_r0.1_s*.tsv" $(held $HU "" "") --pair LogReg:RandomForest --out $O/read1_flag_single ;;
 20) F41 "${HT[@]}" --pattern "planted_test_record_r0.1_s*.tsv" $(held $HU "" "") --pair LogReg:RandomForest --out $O/read1_flag_record ;;
 21) R68 --heldout --plantings results/planted_v19/heldout --pattern "planted_test_mixed_r0.1_s*.tsv" $(held $HU "" "") \
        --pred-only "LinearSVM=$HU/pred_LinearSVM_Bal.tsv" --out $O/read1_regime_single.tsv ;;
 22) F41 "${HT[@]}" --pattern "planted_test_mixed_r0.1_s*.tsv" $(held $HU "" $HDI) --pair DIANA:LogReg --pair DIANA:RandomForest --out $O/read2_flag_single ;;
 23) F41 "${HT[@]}" --pattern "planted_test_record_r0.1_s*.tsv" $(held $HU "" $HDI) --pair DIANA:LogReg --pair DIANA:RandomForest --out $O/read2_flag_record ;;
 24) R68 --heldout --plantings results/planted_v19/heldout --pattern "planted_test_mixed_r0.1_s*.tsv" $(held $HU "" $HDI) \
        --pred-only "LinearSVM=$HU/pred_LinearSVM_Bal.tsv" --pair DIANA:LogReg --pair DIANA:RandomForest --pair DIANA:LinearSVM --pair DIANA:kNN \
        --out $O/read2_regime_single.tsv ;;
 25) F41 "${HT[@]}" --pattern "planted_test_mixed_r0.1_s*.tsv" $(held $HU $HH "") $(hvp HV) --out $O/read3_flag_single ;;
 26) F41 "${HT[@]}" --pattern "planted_test_record_r0.1_s*.tsv" $(held $HU $HH "") $(hvp HV) --out $O/read3_flag_record ;;
 27) R68 --heldout --plantings results/planted_v19/heldout --pattern "planted_test_mixed_r0.1_s*.tsv" $(held $HU $HH "") \
        --pred-only "LinearSVM=$HU/pred_LinearSVM_Bal.tsv" --pred-only "LinearSVMHV=$HH/pred_LinearSVM_Bal.tsv" $(hvp HV) --pair LinearSVMHV:LinearSVM \
        --out $O/read3_regime_single.tsv ;;
esac
echo "=== COMPLETE task $SLURM_ARRAY_TASK_ID $(date) ==="
