Experiment 032: U-Net feature-pyramid ConvGRU, 128 channels
=============================================================

Controlled capacity ablation of experiment 031.  All data, sequence length,
loss, optimiser and split are identical; only ConvGRU hidden channels change
from 96 to 128.  Maximum 80 epochs and patience 20.  Select by validation
macro-IoU, then compare against experiment 031.

