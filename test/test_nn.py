import unittest
import numpy as np

from numpy.testing import assert_array_equal
from simplenn import nn


img = np.array(
    [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    ]
).reshape(1, 3, 3, 1)


class Conv2dTest(unittest.TestCase):
    @unittest.skip("")
    def test_forward(self):
        conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
        conv.filters = np.ones((2, 2, 1, 2))
        conv.filters[:, :, :, 1] = 2
        pred = conv.forward(img)
        assert_array_equal(pred[0, :, :, 0], np.array([[2, 6], [2, 6]]))
        assert_array_equal(pred[0, :, :, 1], np.array([[4, 12], [4, 12]]))

    def test_backward(self):
        conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
        conv.filters = np.ones((2, 2, 1, 2))
        # filter 0 is [[1 2][3 4]] and filter 1 is all 1s
        conv.filters[:, :, 0, 0] = np.array(
            [
                [1, 2],
                [3, 4],
            ]
        )
        conv.forward(img)
        output_grad = np.ones((1, 2, 2, 2))
        output_grad[0, :, :, 1] = 2
        grad = conv.backward(output_grad)
        assert_array_equal(conv._filter_grads[:, :, 0, 0], np.array([[2, 6], [2, 6]]))
        assert_array_equal(conv._filter_grads[:, :, 0, 1], np.array([[4, 12], [4, 12]]))
        assert_array_equal(
            grad[0, :, :, 0], np.array([[3, 7, 4], [8, 18, 10], [5, 11, 6]])
        )

    def test_update_params(self):
        conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
        conv.filters = np.ones((2, 2, 1, 2))
        conv._filter_grads[:, :, 0, 0] = np.array([[1, 1], [1, 1]])
        conv._filter_grads[:, :, 0, 1] = np.array([[-1, -1], [-1, -1]])
        conv.update_params(lr=0.1)
        assert_array_equal(conv.filters[:, :, 0, 0], np.array([[0.9, 0.9], [0.9, 0.9]]))
        assert_array_equal(conv.filters[:, :, 0, 1], np.array([[1.1, 1.1], [1.1, 1.1]]))
