Experiment 015_aggressive_rotation: Aggressive rotation
=======================================================

Large but still plausible ±15-degree camera/panel rotation for robust spatial invariance.

Unit of augmentation: a whole video.  Parameters are deterministic and
identical for every frame, preserving each cell's heating/cooling curve.
Semantic masks use nearest-neighbour interpolation; thermal frames use
linear interpolation.  Bounding boxes are calculated from transformed
corners and saved in the manifest.

Data product: data/synthetic/video_augmentation_manifests/<experiment>/
contains one manifest per source video and plots/ contains QA diagrams.
Full cached frames and masks, when requested, are under
data/honeycomb/synthetic/materialized/.

Training: train_lstm_cuda.sh runs the selected BiLSTM configuration for 100
epochs on original water1/water2 plus their +15-degree rotations, validating
on unchanged water4.  The best validation epoch is retained in artifacts/.
