Experiment 054: absolute plus local-contrast thermal channels
=============================================================

Controlled counterpart to experiments 051 and 053.  Each of the 60 ConvGRU
time steps has two channels:

  channel 0: experiment-051 per-pixel-peak normalized temperature;
  channel 1: experiment-053 signed local contrast.

This lets the network retain absolute temporal shape while using a separately
normalized, position-invariant comparison with nearby panel material.  The
first residual block is the only architecture component widened from one to
two input channels.  water120 removal, U-Net input, augmentations, optimizer
and 20-epoch protocol exactly match experiment 051.
