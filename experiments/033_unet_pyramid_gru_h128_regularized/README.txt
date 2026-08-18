Experiment 033: 128-channel ConvGRU with stronger regularisation
=================================================================

Controlled regularisation ablation of experiment 032.  Same 128-channel
frozen-U-Net feature-pyramid ConvGRU and same video split, but dropout rises
from 0.10 to 0.20 and AdamW lr falls from 3e-4 to 1e-4.  This tests whether the
semantic IoU gain of 128 channels can be retained with lower ordinal error.

Maximum 80 epochs, early stopping patience 20, checkpoint metric macro-IoU.

