Experiment 040: individual cell permutation augmentation
========================================================

Controlled response to experiment 039 class-position memorization.  Four new
30-second videos place every water1 defect cell into a different original cell
slot independently.  Thermal pixels and masks receive the identical integer
translation.  Unlike the older group-shift patching, relative water-class
positions change in every variant.

Training uses the same fresh U-Net v1 e2/e3/e4 pyramid ConvGRU, BF16, seed 17,
loss, optimizer, split and stopping rule as experiment 039.  The only data
change is replacing the three rigid-group patched videos with four independent
cell-permutation videos.  water120 remains ignored and water4 remains untouched.
