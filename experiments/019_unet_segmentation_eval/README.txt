Experiment 019: frozen U-Net segmentation evaluation
=====================================================

Purpose
-------
Measure the supplied models/segmentation/unet_water_v2.pth without fitting any
weights.  The checkpoint has a 7-channel input, so each inference sample is a
stack of seven neighbouring thermal frames.  The centre frame is labelled.

Protocol
--------
Evaluation uses the documented U-Net stack at 10 Hz: 1, 3, 5, 7, 10, 15 and
20 seconds (frames 10, 30, 50, 70, 100, 150, 200).  Radiometry is exactly
frame0 subtraction, clipping below zero, then one global maximum over the
0..20-second window.  The ROI is derived from the 5-second difference map by
Otsu, closing/fill/opening/largest component and 12 px erosion.  The six water
labels are merged into the binary target.  Metrics are measured inside ROI.
The decision threshold is 0.5 and is fixed before evaluation.

The supplied documented preprocessing is required: arbitrary neighbouring-frame
stacks or per-stack p01-p99 normalisation are incompatible with this checkpoint.
The command writes metrics.json, summary.txt and a qualitative probability/mask
diagram under this experiment.
