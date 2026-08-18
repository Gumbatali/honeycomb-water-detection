Experiment 020: full-frame mask-conditioned temporal semantic segmentation
==========================================================================

Purpose
-------
Replace ROI-by-ROI classification with one full-frame model.  The frozen U-Net
first produces a soft binary defect probability map.  A larger temporal network
then observes the entire panel and predicts a 7-class map: background plus
water20, water40, water60, water80, water100 and water120.

Input and temporal interval
---------------------------
The input is 60 equally spaced frames over the first 30 seconds at 10 Hz:
frames 0, 5, 10, ..., 295 (0.5-second interval).  For each time point the
network receives two full-frame channels after resizing to 240 x 320:

  1. thermal frame after frame0 subtraction, clipping, and one global 0..30 s
     maximum normalisation;
  2. frozen U-Net soft probability, obtained with the documented 1/3/5/7/10/
     15/20-second segmentation preprocessing.

Architecture
------------
Shared 2D residual encoder (2 -> 48 -> 96 -> 160 channels) creates a 30 x 40
feature map for every frame.  A 160-channel ConvGRU compares feature maps over
all 60 time steps.  A residual decoder upsamples to 240 x 320 and predicts the
seven semantic classes.  This is materially larger than the ROI BiLSTM and
keeps spatial correspondence between a segment's mask and its heating/cooling
curve.  The model does not crop around the labelled boxes.

Split and training
------------------
Training uses water1/water2 and their materialised augmentations; validation is
always untouched water4.  The loss combines weighted cross-entropy (to counter
the large background) and soft Dice over six water classes.  The run script
uses 100 epochs, batch size 1 and saves the best validation macro-Dice.

Outputs
-------
artifacts/best.pt, metrics.json, summary.txt and a predicted semantic map.

