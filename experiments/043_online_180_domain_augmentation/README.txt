Experiment 043: online 0/180-degree domain augmentation
========================================================

Purpose
-------
Close the measured geometry/radiometry gap between water1 training and water2
validation without adding water2 to training.  water1->water2 cell centroids
are described by -179.3 degrees, scale 0.981 and about 2.9-pixel fit error.

Every train sample independently applies:

  identity or 180-degree branch:       p=1.0 (50/50)
  scale/translation jitter:            p=0.7
  independent cell permutation:        p=0.5
  local defect contrast gain 0.9--1.1: p=0.7

The 180-degree branch uses the measured base scale 0.98 and upward panel shift
14.5%; optional jitter adds scale 0.95--1.05 and translation +/-3%.  The exact
same affine transform is used for all 60 thermal frames, seven U-Net frames,
ROI and semantic mask.  Images use linear interpolation; labels use nearest
neighbour and label 255 outside the transformed ROI.

Cell permutation selects one of four independently permuted neutral-canvas
videos.  Local gain uses neutral_water1 and applies the physical residual rule:

  neutral + gain * (defect - neutral)

Sixteen stochastic samples are generated per epoch.  Validation water2 is
never augmented.  Model and optimization match experiment 042: fresh U-Net v1
pyramid 1x128 ConvGRU, ROI, pixel-peak normalization, BF16, dropout 0.20,
AdamW 1e-4, maximum 100 epochs and patience 30.  water4 remains untouched.
