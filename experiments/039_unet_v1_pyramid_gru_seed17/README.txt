Experiment 039: U-Net v1 multiscale pyramid ConvGRU, seed 17
=============================================================

Purpose
-------
Correct the unstable e4-only warm-start series.  Train from scratch and log
both optimization losses and segmentation metrics so overfitting and optimizer
instability can be distinguished.

Data protocol
-------------
Train: water1, horizontal flip, aggressive rotation, aggressive affine, and
three translated-defect patched videos.  Validation: unchanged water2.
water4 is untouched.  water120 (label 6) is ignored rather than assigned an
unsupported intermediate physical class.

Architecture and input
----------------------
Frozen models/segmentation/v1/unet_water_v2.pth encoder features e2/e3/e4 are
gated into a full-panel thermal encoder at three spatial scales.  A one-layer
128-channel ConvGRU processes 60 equally spaced frames covering 0--29.5 s.
The decoder uses multiscale thermal skip features.  Output classes are
background plus water20, water40, water60, water80 and water100.

Optimization
------------
Fresh initialization, seed 17, AdamW lr 1e-4, weight decay 1e-4, dropout 0.20,
BF16 mixed precision (v1 activations overflowed FP16 ConvGRU gradients),
five-epoch linear warmup followed by cosine decay, maximum 100 epochs and
early-stopping patience 30.  History includes epoch-0 validation, train/valid
CE, Dice loss, combined loss, gradient norm, per-class IoU, predicted class
fractions, macro-IoU, macro-Dice and ordinal MAE.
