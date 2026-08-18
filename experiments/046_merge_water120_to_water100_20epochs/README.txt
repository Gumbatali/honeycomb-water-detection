Experiment 046: merge water120 into water100
============================================

Purpose
-------
Remove water120 as a separate detection class without teaching the model that
this real defect is neutral background.  Original label 6 (water120) is mapped
to output label 5 (water100) in training, validation and test targets.

Why water100
------------
Water100 is physically adjacent to water120.  On water2 their normalized
30-second curves correlate by 0.987; on untouched water4 by 0.995.  Across all
three recordings it gives the strongest curve correlation (0.981) and the
smallest derivative error among candidate classes.  Absolute curve RMSE on
water1 alone is inconsistent because of its known heating/domain shift and is
not used to map water120 to the physically implausible water20 class.

Controlled protocol
-------------------
The architecture, seed, 20-epoch schedule and online spatial/temporal
augmentations exactly match experiment 045.  The only intended change is that
label 6 is supervised as label 5 rather than ignored.  The output remains six
channels: background plus water20, water40, water60, water80 and a combined
water100/water120 class.

Training: water1 with 16 independently augmented samples per epoch.
Validation/model selection: unchanged water2 with label 6 mapped to label 5.
water4 is not used during training or checkpoint selection.  It has already
been opened by experiment 045, so any later water4 score is comparative rather
than a new pristine final-test estimate.
