#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=diana_v13_tables
#SBATCH --output=logs/final_eval_v13/x4_%j.out
#SBATCH --error=logs/final_eval_v13/x4_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# K6: Table 2 / Figure 1 / Figure 2 for v13, from held-out read 10
# (results/final_eval_v13/heldout_*, results/baseline_predictions_v13/). Re-scorings of
# stored predictions only: no model runs here. Same 922 runs, same planted set, same
# metric; eligibility and training support from data/splits_v13/. Nothing under
# results/final_eval_budget_v9/ or results/paper/*_budget.png is touched.
# afterok run_heldout_v13.sbatch and run_heldout_baselines_v13.sbatch.
# 30_heldout_regime_table.py is not run: it is a reproduction check against the v9
# 38-class table and does not apply to another class set.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python
EV=results/final_eval_v13
BP=results/baseline_predictions_v13
SP=data/splits_v13
PL=results/planted_mislabels_v9/planted_test_mixed_r0.1.tsv
ENC='results/final_budget_v13/{task}/final_model/label_encoders.json'
for t in multitask community_type feature sample_host material; do
  [ -f "$EV/heldout_$t/test_metrics.json" ] || { echo "held-out read 10 not taken for $t" >&2; exit 1; }
done
[ -f "$BP/heldout_predictions.tsv" ] || { echo "baseline held-out predictions missing" >&2; exit 1; }
echo "=== whole-task paired test (Figure 1, classification) ==="
$PY scripts/analysis/18_paired_superiority.py --eval-dir $EV --single-pattern "heldout_{t}" \
    --baselines $BP/heldout_predictions.tsv --splits $SP --output results/paired_superiority_v13
echo "=== per-regime paired test (Table 2, verdict column) ==="
$PY scripts/analysis/25_regime_paired_test.py --eval-dir $EV --single-pattern "heldout_{task}" \
    --splits $SP --baselines $BP/heldout_predictions.tsv --out $EV/regime_paired.tsv
echo "=== detector, like-for-like ==="
for t in community_type feature sample_host material; do
  $PY scripts/analysis/15_anomaly_detection_roc.py --predictions $EV/heldout_$t/test_predictions.tsv \
      --planted $PL --encoders results/final_budget_v13/$t/final_model/label_encoders.json \
      --label DIANA_$t --output results/anomaly_detection_v13
done
$PY scripts/analysis/19_baseline_anomaly_detection.py --diana-dir results/anomaly_detection_v13 \
    --baseline-dir $BP --planted $PL --output results/anomaly_detection_baselines_v13
$PY scripts/analysis/39_detector_paired_test.py --eval-dir $EV --single-pattern "heldout_{task}" \
    --encoders-pattern "$ENC" --baseline-dir $BP --planted $PL --out $EV/detector_paired.tsv
echo "=== Table 2 ==="
$PY scripts/analysis/23_regime_stratified_table.py --eval-dir $EV --single-pattern "heldout_{task}" \
    --encoders-pattern "$ENC" --splits $SP --planted $PL --baseline-dir $BP --out $EV/regime_table.tsv
echo "=== MCC and the all-models table ==="
$PY scripts/evaluation/13_mcc_heldout_v9.py --eval-dir $EV --single-pattern "heldout_{task}" \
    --splits $SP --baselines $BP/heldout_predictions.tsv --output $EV/mcc.tsv
$PY scripts/evaluation/40_heldout_all_models.py --eval-dir $EV --single-pattern "heldout_{task}" \
    --baselines $BP/summary.csv --output $EV/heldout_all_models.tsv
echo "=== figures ==="
$PY scripts/paper/plot_paired_deltas.py --paired results/paired_superiority_v13/paired_deltas.tsv \
    --detection $EV/detector_paired.tsv --baseline-summary $BP/summary.csv --out results/paper/paired_deltas_v13.png
$PY scripts/paper/plot_detector_operating_curve.py --diana-dir results/anomaly_detection_v13 \
    --baseline-dir $BP --planted $PL --out results/paper/detector_operating_curve_v13.png
echo "=== COMPLETE — $(date) ==="
