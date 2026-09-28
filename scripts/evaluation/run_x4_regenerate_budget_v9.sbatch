#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=diana_x4_budget
#SBATCH --output=logs/final_eval_budget_v9/x4_%j.out
#SBATCH --error=logs/final_eval_budget_v9/x4_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X4: every Table 2 / Figure 1 / Figure 2 artefact, regenerated from the
# epoch-budget arms' held-out read (results/final_eval_budget_v9/heldout_*),
# after that read has been taken. Re-scorings of stored predictions only: no
# model runs here.
#
# Same baselines, same 922 runs, same planted set, same eligibility table and
# metric as read 1. Detection is scored like-for-like (--encoders), which read 1
# was not (PROJECT.md section 3, detector). Nothing under results/final_eval_v9/
# or results/paper/{paired_deltas,detector_operating_curve}.png is touched.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python
EV=results/final_eval_budget_v9
ENC='results/final_budget_v9/{task}/final_model/label_encoders.json'
for t in community_type feature sample_host material; do
  [ -f "$EV/heldout_$t/test_metrics.json" ] || { echo "held-out read 9 not taken for $t" >&2; exit 1; }
done

echo "=== whole-task paired test (Figure 1, classification) ==="
$PY scripts/analysis/18_paired_superiority.py --eval-dir $EV --single-pattern "heldout_{t}" \
    --output results/paired_superiority_budget_v9
echo "=== per-regime paired test (Table 2, verdict column) ==="
$PY scripts/analysis/25_regime_paired_test.py --eval-dir $EV --single-pattern "heldout_{task}" \
    --out $EV/regime_paired.tsv
echo "=== classification by regime, reproducible rule ==="
$PY scripts/analysis/30_heldout_regime_table.py --arm $EV --label budget
echo "=== detector, like-for-like ==="
for t in community_type feature sample_host material; do
  $PY scripts/analysis/15_anomaly_detection_roc.py --predictions $EV/heldout_$t/test_predictions.tsv \
      --planted results/planted_mislabels_v9/planted_test_mixed_r0.1.tsv \
      --encoders results/final_budget_v9/$t/final_model/label_encoders.json \
      --label DIANA_$t --output results/anomaly_detection_budget_v9
done
$PY scripts/analysis/19_baseline_anomaly_detection.py --diana-dir results/anomaly_detection_budget_v9 \
    --output results/anomaly_detection_baselines_budget_v9
$PY scripts/analysis/39_detector_paired_test.py --eval-dir $EV --single-pattern "heldout_{task}" \
    --encoders-pattern "$ENC" --out $EV/detector_paired.tsv
echo "=== Table 2 ==="
$PY scripts/analysis/23_regime_stratified_table.py --eval-dir $EV --single-pattern "heldout_{task}" \
    --encoders-pattern "$ENC" --out $EV/regime_table.tsv
echo "=== MCC and the all-models table ==="
$PY scripts/evaluation/13_mcc_heldout_v9.py --eval-dir $EV --single-pattern "heldout_{task}" \
    --output $EV/mcc.tsv
$PY scripts/evaluation/40_heldout_all_models.py --eval-dir $EV --single-pattern "heldout_{task}" \
    --output $EV/heldout_all_models.tsv
echo "=== figures ==="
$PY scripts/paper/plot_paired_deltas.py --paired results/paired_superiority_budget_v9/paired_deltas.tsv \
    --detection $EV/detector_paired.tsv --out results/paper/paired_deltas_budget.png
$PY scripts/paper/plot_detector_operating_curve.py --diana-dir results/anomaly_detection_budget_v9 \
    --out results/paper/detector_operating_curve_budget.png
echo "=== COMPLETE — $(date) ==="
