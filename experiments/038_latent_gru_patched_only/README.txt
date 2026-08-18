Experiment 038: patched defects only, low-LR continuation
==========================================================

Controlled response to experiment 037.  Warm-start its best six-output model,
but remove the all-background neutral video from training.  Keep the three
translated defect-patch videos, all original water1 augmentations, and ignore
water120.  Fine-tune at lr=3e-5 for at most 30 epochs with the same water2
validation and early stopping.  water4 remains untouched.

