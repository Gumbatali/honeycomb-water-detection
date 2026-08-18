Experiment 031: U-Net feature-pyramid ConvGRU, 96 channels
============================================================

First hyperparameter trial for the new architecture from experiment 030.

Train: water1/water2 plus horizontal flip, aggressive rotation and aggressive
affine versions (8 independent video samples).  Validation: unchanged water4.
Temporal input: 60 equally spaced frames through 30 seconds.  U-Net v2 e2/e3/
e4 features are frozen and fused with the full thermal frame features.

Parameters: ConvGRU hidden 96, one 3x3 recurrent layer, dropout 0.10, AdamW
lr 3e-4, weight decay 1e-4, maximum 80 epochs, patience 20.  Select the best
checkpoint by water-class macro-IoU.

