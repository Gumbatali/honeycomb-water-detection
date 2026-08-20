Experiment 059: water60-only onset shift, 20 epochs
====================================================

Controlled extension of experiment 051.  On 70% of water1 training draws,
only the water60 thermal response is delayed by 0.30--0.70 seconds.  Heating
and cooling scales remain exactly 1.0.  The range is centred on the measured
0.5-second later water4-water60 peak.  The same transform is applied to the
60-frame ConvGRU input and seven U-Net frames.

All other settings are identical to experiment 051; water2 selects the model
and water4 is the diagnostic comparative check.
