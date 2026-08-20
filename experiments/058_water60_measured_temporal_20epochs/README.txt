Experiment 058: measured water60 temporal augmentation
=======================================================

Controlled extension of experiment 051.  The normal online domain and global
temporal augmentation are unchanged.  On 70% of training draws, only water60
(class 3) receives an additional temporal transform: onset delay 0.20--0.60 s,
heating time scale 0.78--0.90 and post-peak cooling scale 0.70--0.85.  The
ranges are derived from the measured later peak and slower water4-water60
cooling curve; every other class is unchanged by this transform.

The transform is applied consistently to the complete 60-frame ConvGRU input
and all seven U-Net frames.  water2 remains selection validation; water4 is a
comparative, now diagnostic, check.
