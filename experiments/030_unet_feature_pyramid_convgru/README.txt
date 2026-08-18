Experiment 030: U-Net-feature-pyramid ConvGRU
===============================================

This is the new clean baseline.  Earlier mask-channel classifiers and full-frame
mask experiments are retained only as history and must not determine choices in
this experiment.

Architecture
------------
The binary U-Net probability mask is not an input.  Instead, the frozen U-Net
encoder from models/segmentation/v2/unet_water_v2.pth processes the documented seven-frame input at
1/3/5/7/10/15/20 seconds and exposes its e2, e3 and e4 feature maps.  These
maps contain learned texture, cell-boundary and defect-context features.

The classifier/segmenter receives a full thermal sequence of 60 frames over
0..29.5 seconds at a uniform 0.5-second interval.  The thermal branch keeps the
entire panel but resamples it to 192 x 256 for feasible 60-step backpropagation
on an 8 GB GPU; U-Net features remain calculated at original 480 x 640.  A
shared thermal encoder creates three feature scales.  U-Net e2/e3/e4 features are projected and fused
at matching scales through gated residual addition.  Only the deepest fused
24x32 map is recurrently aggregated by ConvGRU.  A multi-scale decoder outputs
seven semantic classes: background and six ordered water levels.

This keeps the full image and temporal behaviour, but avoids teaching the
second model to copy a hard segmentation mask.  U-Net is frozen in the initial
study; its strong water4 Dice 0.8597 protects the small temporal dataset from
overfitting the spatial encoder.

Training protocol
-----------------
Independent unit: a video, never a frame.  Train water1/water2 and their
selected augmentations; validate only the untouched water4.  Report macro-IoU
over water classes, macro Dice and ordinal MAE over matched predicted segments;
do not use background pixel accuracy as the primary metric.

See HYPERPARAMETERS.txt for the selected base configuration and ablations.
