Experiment 036: neutral background and defect patch synthesis
==============================================================

Goal
----
Create a defect-free thermal video from water1, then create new labelled videos
by adding the original defect cells back at new panel positions.

Neutral video
-------------
For each labelled water class, the original defect pixels are replaced in every
frame by donor pixels from a dilated annulus around that class.  Donors are
selected once from the brighter 40% of the local background in frame 50, then
the same donor coordinates are read in every frame.  This preserves the local
background's temporal heating/cooling behaviour and does not insert a constant
or black artificial patch.  Its semantic mask is all background.

Patched videos
--------------
Three variants begin from the neutral video.  All six original class patches
are copied with one rigid translation per video; the semantic mask is warped
with nearest-neighbour interpolation using exactly the same transform.  This
creates new defect locations without changing their thermal time profiles.

Outputs
-------
data/honeycomb/synthetic/neutral_patch/<video_id>/ contains images/, masks/,
metadata.json and MANIFEST.txt.  Every video is generated in a .partial folder
and renamed only after all 3000 frames and masks were written.

