Experiment 048: full augmentation with multitask ordinal supervision
=====================================================================

Final combined experiment.  It contains every change from experiment 047 and
adds two auxiliary heads to the shared full-frame decoder:

  defect head: one binary logit for defect versus panel background;
  ordinal head: four logits for level thresholds y>1, y>2, y>3 and y>4.

The original six-channel semantic head remains the deployment output.  Loss:

  multiclass CE + 0.5 Dice
  + 0.30 * (binary BCE + 0.5 binary Dice)
  + 0.20 * ordinal BCE on defect pixels

water120 is mapped to water100 and their combined CE weight is 0.5.  Spatial,
radiometric, global temporal and independent per-cell temporal augmentation
are all enabled.  Training/model selection uses water1/water2 only; water4 is
not consulted until the best validation checkpoint has been selected.
