Experiment 045: online spatial + temporal augmentation, 20 epochs
=================================================================

Controlled extension of experiment 044.  Spatial/radiometric probabilities are
unchanged: identity/180 branch 1.0 (50/50), scale/translation p=0.7,
independent cell permutation p=0.5 and local defect gain 0.9--1.1 p=0.7.

New temporal augmentation is applied synchronously to the complete panel after
frame-0 subtraction and before normalization:

  response onset shift +/-0.5 seconds: p=0.5
  global time warp 0.9--1.1:           p=0.5
  cooling warp 0.85--1.15 after 3.7 s: p=0.3

The parameters cover the measured water1/water2 differences: 0.1--0.2 s onset
shift, 6--8% heating-duration difference and class-dependent cooling rates.
Temporal transforms are shared by every pixel and class; semantic masks do not
change in time.  They affect both the 60-frame ConvGRU input and the seven
early U-Net frames.  Validation water2 is never augmented; water4 is untouched.

All model, ROI, pixel-peak, optimizer and 20-epoch settings match experiment
044 for a direct train/validation-gap comparison.
