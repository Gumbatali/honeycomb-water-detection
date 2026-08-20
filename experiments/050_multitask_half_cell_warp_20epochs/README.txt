Experiment 050: multitask model with half-range cell time warp
================================================================

Controlled counterpart to experiments 048 and 049.  All experiment-049
settings are retained, while independent per-cell temporal augmentation is
enabled with half of the experiment-048 ranges:

  onset shift:   -0.125 .. +0.125 seconds
  time scale:     0.975 .. 1.025
  cooling scale:  0.95 .. 1.05 after 3.7 seconds
  sample probability: 0.7

These ranges add within-class variability without deliberately spanning the
larger water1/water2 domain shift already handled by the global transform.
Training/model selection uses water1/water2 only.
