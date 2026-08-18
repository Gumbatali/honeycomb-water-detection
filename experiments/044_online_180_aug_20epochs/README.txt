Experiment 044: online domain augmentation, 20-epoch diagnostic
================================================================

Short controlled run requested to test whether validation dynamics approach
training dynamics after adding online augmentation.

Every training sample applies identity/180 branch (50/50), scale/translation
jitter with p=0.7, independent cell permutation with p=0.5, and local defect
gain 0.9--1.1 with p=0.7.  Sixteen stochastic water1 samples are generated per
epoch.  water2 validation is unchanged and water4 remains untouched.

Architecture and preprocessing match experiment 043: frozen U-Net v1 pyramid,
one 128-channel ConvGRU, ROI, pixel-peak normalization, BF16, dropout 0.20,
AdamW lr=1e-4 and weight decay 1e-4.  Training is capped at exactly 20 epochs;
patience 30 cannot stop it early.  Train/validation combined loss, CE, Dice,
macro-IoU and per-class IoU are saved every epoch.
