Discarded 2026-09-16, not adjusted.

dnamax_final_v9_foldscaling: the first S6 final fits and held-out read. The models were
trained on unitigs.frac.dnamax.fold0.mat, whose appended block is standardised on folds 1-4
(80% of train), and then evaluated on a held-out matrix standardised on all 2,716 training
runs. The model was therefore tested on a different scaling than it was fitted on. Measured
shift: mean 0.09 of the appended block's own standard deviation, max 0.25.

No leakage was involved: the held-out block used training statistics (verified, max
deviation 1.6e-05 against the all-train recomputation, and 0.87 away from a held-out-own
scaling), and the unitig rows are identical between the train and held-out matrices.

The numbers from this run must not be quoted. They are kept only so the discarded state is
on the record rather than deleted.
