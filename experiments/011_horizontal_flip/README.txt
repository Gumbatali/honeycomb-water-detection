Experiment 011_horizontal_flip: Horizontal flip
===============================================

Mirror view.  Class IDs are preserved, while every mask and bbox is mirrored.

Unit of augmentation: a whole video.  Parameters are deterministic and
identical for every frame, preserving each cell's heating/cooling curve.
Semantic masks use nearest-neighbour interpolation; thermal frames use
linear interpolation.  Bounding boxes are calculated from transformed
corners and saved in the manifest.

Data product: data/synthetic/video_augmentation_manifests/<experiment>/
contains one manifest per source video and plots/ contains QA diagrams.
Full cached frames and masks, when requested, are under
data/honeycomb/synthetic/materialized/.

Training protocol: train_lstm_cuda.sh runs 40 CUDA epochs with original
water1/water2 plus their flipped copies (24 sequences), validating only on
unaltered water4.  The best validation checkpoint is saved in artifacts/.
