Experiment 041: ROI + pixel-peak temporal normalization
=======================================================

Continue the successful independent-cell permutation training from experiment
040, but train a fresh model with two physically motivated preprocessing fixes:

1. Documented panel ROI: Otsu on corrected frame 50, close/fill/open, largest
   component and 12-pixel erosion.  Inputs outside ROI are zero and labels are
   ignored; deployment prediction is constrained to the same ROI.
2. Every thermal pixel's 60-frame curve is divided by its own temporal peak.
   This suppresses lamp/camera spatial gain while retaining heating/cooling
   curve shape.  The frozen U-Net v1 still receives its documented globally
   normalized seven early frames, masked by ROI.

All other settings match 040: water1 plus three geometric augmentations and
four independent cell permutations, water2 validation, water4 untouched,
water120 ignored, BF16, fresh seed 17, hidden 128, dropout 0.20, AdamW 1e-4,
100-epoch cap and patience 30.
