Experiment 056: segmentation model v2 evaluation
=================================================

Measure models/segmentation/v2/unet_water_v2.pth on water1, water2 and water4.
The checkpoint is a binary 7-channel U-Net: defect versus normal panel, not a
water-level classifier. Inputs use the documented U-Net pipeline: frame-0
subtraction, negative clipping, a global 0..20 s maximum scale and an
Otsu/morphology panel ROI. Channels are frames at 1, 3, 5, 7, 10, 15 and 20 s.

Primary metrics use fixed threshold 0.50 on masks_binary inside ROI: Dice,
IoU, precision, recall, specificity and pixel accuracy. Threshold sweep is a
calibration diagnostic only; its per-video optimum is not a validation metric.
