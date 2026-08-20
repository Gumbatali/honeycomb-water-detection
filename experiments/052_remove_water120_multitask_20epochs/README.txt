Experiment 052: multitask/ordinal model with water120 removed
================================================================

This run adds the semantic, binary-defect and ordinal heads from experiment
049 to the exact experiment-051 removal protocol.  There is no label merge,
no combined-class reweighting and no independent per-cell temporal warp.

Loss is multiclass CE + 0.5 Dice + 0.30 binary defect loss + 0.20 ordinal BCE.
The ordinal thresholds cover the five retained levels water20..water100.
water120 pixels are neutralized in both input streams and ignored in loss and
metrics.  Model selection uses water2 only.
