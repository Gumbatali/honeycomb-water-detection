Experiment 057: online random defect pasting, 20 epochs
========================================================

Controlled extension of experiment 051.  All architecture, seed, optimiser,
water120 neutralization, ROI and existing online domain/temporal augmentation
are unchanged.

With probability 0.5, an online training draw replaces water1 by its neutral
background and pastes one to three selected water20..water100 thermal residual
patches at independently sampled, non-overlapping locations inside the panel.
The semantic target is built from the same transformed patch masks.  A patch is
accepted only if all of its pixels remain inside an eroded panel ROI.  water120
is never pasted.  Random patch pasting and the existing pre-materialized cell
permutation are mutually exclusive in a draw, so the new transform has a clear
meaning.

Training uses water1; water2 selects the checkpoint.  water4 is a comparative
post-selection check only.
