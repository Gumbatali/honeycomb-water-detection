Experiment 016_aggressive_affine: Aggressive affine geometry
============================================================

Stronger viewpoint variation: rotation, 10% scale change, shear and sub-cell translation.

Unit of augmentation: a whole video.  Parameters are deterministic and
identical for every frame, preserving each cell's heating/cooling curve.
Semantic masks use nearest-neighbour interpolation; thermal frames use
linear interpolation.  Bounding boxes are calculated from transformed
corners and saved in the manifest.

Data product: data/synthetic/video_augmentation_manifests/<experiment>/
contains one manifest per source video and plots/ contains QA diagrams.
Full cached frames and masks, when requested, are under
data/honeycomb/synthetic/materialized/.

Training: train_lstm_cuda.sh runs 100 CUDA epochs on original water1/water2
plus their aggressive affine copies (-12 degrees, 0.90 scale, 0.08 shear),
validating on unchanged water4.  The best validation epoch is retained in
artifacts/ and initialises the curriculum experiment 017.
