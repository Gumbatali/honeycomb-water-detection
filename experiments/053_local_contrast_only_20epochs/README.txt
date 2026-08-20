Experiment 053: local thermal contrast only
===========================================

Controlled extension of the best experiment 051.  water120 remains physically
neutralized and ignored.  The 60-frame ConvGRU input is replaced by a signed
local contrast map.  For every resized frame, a 41x41 morphological closing
estimates the slowly varying normal panel/lamp field without using labels:

  local_contrast(t) = normalized_temperature(t) - local_background(t)

Each pixel's signed contrast curve is divided by its maximum absolute temporal
response.  This removes position-dependent lamp intensity while preserving
heating/cooling sign and shape.  The seven U-Net frames and all experiment-051
augmentations are unchanged.  water1 trains and water2 selects the checkpoint.
