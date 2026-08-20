Experiment 062: merge water60 into water80 with full validated augmentation
==========================================================================

Deployment-oriented extension of experiment 051.  The unstable water60 label
(class 3) is merged into water80 (class 4), yielding five output classes:
background, water20, water40, combined water60/water80, water100.  This is the
only label merge.  water120 (class 6) remains neutralized in inputs and ignored
in supervision/metrics, not merged into any class.  The combined class weight
is 0.5 so the double-area class does not dominate CE.

Training retains the effective online augmentation from experiment 051:
identity/180-degree view, scale/translation, cell permutation, local defect
gain, shared onset/time/cooling warps.  It additionally includes the three
translated defect-patch videos and four neutral-canvas gain-permutation videos.
These are augmentations that had previously shown useful or neutral robustness.
Known harmful transforms (horizontal flip, aggressive affine, unconstrained
random defect pasting) are deliberately excluded.

water2 selects the checkpoint.  water4 is a comparative external check.
