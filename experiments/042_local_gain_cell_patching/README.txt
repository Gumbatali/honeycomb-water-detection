Experiment 042: local +/-10% gain cell patching
================================================

The defect-free neutral_water1 video is retained unchanged and used only as a
synthesis canvas.  Four independently permuted cell videos are rebuilt with a
different class-balanced gain from 0.90 to 1.10 for every defect patch:

  output = neutral + gain * (source defect - source local neutral)

This changes defect thermal contrast without unrealistically multiplying the
absolute Celsius baseline.  Masks use exactly the same cell translations.
The range covers the measured 7.9% difference in mean 5-second response between
water1 and water2.

Training is otherwise identical to experiment 041: fresh seed 17, U-Net v1
e2/e3/e4 pyramid, ROI, per-pixel temporal normalization, water1 plus flip,
aggressive rotation/affine and four gain-permuted patches; water2 validation,
water4 untouched, water120 ignored, BF16, 100 epochs and patience 30.
