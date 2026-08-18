Experiment 018: curriculum fine-tuning on all augmentations
=============================================================

Start from experiment 017's best affine+flip+patch checkpoint.  Fine-tune for
100 epochs on original water1/water2 and all seven materialised augmentation
families: mild rotation, flip, mild affine, patching, location shift, aggressive
rotation and aggressive affine.  Validation is unchanged water4 only.

