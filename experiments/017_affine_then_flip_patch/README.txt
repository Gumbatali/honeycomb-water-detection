Experiment 017: curriculum fine-tuning after aggressive affine
===============================================================

Start from the best checkpoint of experiment 016 (trained with aggressive
affine geometry).  Fine-tune for 100 epochs on original water1/water2 plus
three augmentation variants: aggressive affine, horizontal flip and background
patching.  Validation remains unchanged water4.  This curriculum introduces
additional invariances after geometry has been learned rather than from random
initialisation.

