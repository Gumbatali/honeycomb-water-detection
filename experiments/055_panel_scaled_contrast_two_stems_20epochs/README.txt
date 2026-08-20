Experiment 055: ROI-scaled local contrast with separate thermal stems
====================================================================

Question
--------
Experiments 053 and 054 added local thermal contrast but degraded validation.
Their contrast was normalised independently at every pixel over time.  That
operation makes a shallow and a deep cell have the same peak magnitude,
discarding a physical water-level cue.

Controlled change from champion experiment 051
----------------------------------------------
- The original absolute, pixel-peak-normalised 60-frame signal is retained.
- A second channel is signed local contrast: frame minus morphological closing
  (41 px), computed without masks.
- It is scaled once by the 99th percentile of its absolute value inside the
  Otsu panel ROI over all 60 frames, then clipped to [-3, 3].
- Absolute and contrast channels have separate 24-channel stride-2 residual
  stems, are concatenated, then fused to 48 channels.  All later U-Net fusion,
  128-channel ConvGRU, decoder and optimisation settings match 051.
- water120 (label 6) is physically replaced by nearby normal-panel response,
  then excluded from target/loss/metrics, as in 051.

Protocol
--------
Train: water1, 16 independent online augmented draws per epoch.
Selection validation: water2.  Comparative external evaluation: water4.
The latter is no longer an untouched test because it informed prior analysis.

Decision rule
-------------
Retain the representation only if water2 macro-IoU is competitive with 051
(0.6966) and the water4 result does not regress materially.  Artifacts,
plots and a final RESULTs file are written atomically in this directory.
