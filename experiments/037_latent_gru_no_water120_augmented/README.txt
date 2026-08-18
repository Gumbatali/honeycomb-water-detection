Experiment 037: continue winner with neutral/patch augmentation, remove water120
================================================================================

Warm-start from experiment 033 (frozen U-Net v1 e4-only + one 128-channel
ConvGRU).  The previous 7-class final convolution is deliberately replaced by
a new 6-class layer: background plus water20, water40, water60, water80 and
water100.  Pixels labelled water120 are ignore_index=255 in cross-entropy,
Dice and macro-IoU.  They are not relabelled as 20/40 because that would impose
a false physical ordering.

Training data: water1, its seven geometric/radiometric augmentation videos,
neutral_water1 and three translated defect-patch videos.  Validation remains
unaltered water2; water4 stays untouched.  Optimisation continues at lr=1e-4
for at most 50 epochs with ReduceLROnPlateau and 15-epoch early stopping.

