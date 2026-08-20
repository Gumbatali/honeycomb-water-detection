Experiment 049: multitask model without independent cell time warp
==================================================================

This is the first controlled follow-up to experiment 048.  It retains:

  water120 -> water100 target mapping;
  combined-class cross-entropy weight 0.5;
  semantic, binary-defect and ordinal heads;
  identity/180, geometry, cell permutation and local gain augmentation;
  global onset/time/cooling augmentation from experiment 045.

The only removed component is independent temporal warping of each defect
cell.  This tests whether the broad local curve perturbation caused the class
overlap observed in experiments 047/048.  water1 trains, water2 selects the
checkpoint, and water4 is not opened until selection is complete.
