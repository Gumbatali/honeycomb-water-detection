Experiment 047: balanced merge plus independent cell dynamics
==============================================================

Controlled extension of experiment 046.  water120 is still mapped to
water100, but the combined class receives cross-entropy weight 0.5 because it
contains two cells.  In addition to the global temporal transform, each defect
cell independently receives, with sample probability 0.7:

  onset shift:   -0.25 .. +0.25 seconds
  time scale:     0.95 .. 1.05
  cooling scale:  0.90 .. 1.10 after 3.7 seconds

The same cell parameters are applied to the 60-frame thermal sequence and the
seven-frame U-Net input.  Masks are static in time and spatial transforms stay
synchronized.  Architecture and all other settings match experiment 046.

Training/model selection uses water1/water2 only.  water4 is excluded until
the checkpoint is fixed.
