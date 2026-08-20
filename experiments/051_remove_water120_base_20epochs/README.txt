Experiment 051: remove water120 from inputs, targets and metrics
================================================================

Controlled extension of experiment 045.  water120 (label 6) is assumed
physically impossible in deployment and is removed rather than merged:

  input: every label-6 pixel in every thermal/U-Net frame is replaced by the
         response of its nearest normal panel pixel from a local 31x31 ring;
  target: label-6 pixels become ignore=255;
  loss/metrics: ignored pixels contribute to neither training nor evaluation.

Original arrays and masks are not modified.  All architecture, spatial,
radiometric and global temporal augmentation settings match experiment 045.
There is no independent per-cell temporal warp and no water120 output class;
six logits represent background plus water20..water100.

water1 trains, water2 selects the checkpoint.  water4 is only a later
comparative check because it has already been opened in prior experiments.
