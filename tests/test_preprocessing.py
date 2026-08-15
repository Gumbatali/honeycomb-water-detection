import numpy as np

from src.preprocessing.background_subtraction import subtract_background


def test_subtract_background_zeroes_first_frame():
    data = np.random.rand(4, 4, 10).astype(np.float32)
    result = subtract_background(data)
    assert np.allclose(result[:, :, 0], 0)


def test_subtract_background_preserves_shape():
    data = np.random.rand(4, 4, 10).astype(np.float32)
    result = subtract_background(data)
    assert result.shape == data.shape
