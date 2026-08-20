Experiment 060: water60-only onset shift up to 30 seconds
==========================================================

Controlled experiment requested to test a much wider water60 temporal shift.
On 70% of training draws, water60 is delayed uniformly in [0, 30] seconds;
heating and cooling scales remain 1.0.  The full model input spans 0--29.5 s,
so delays near 30 s intentionally create little or no visible water60 response
while the semantic label remains water60.  This tests robustness to late onset,
but is not physically equivalent to the measured water4 0.5-second shift.

All other experiment-051 settings are unchanged.
