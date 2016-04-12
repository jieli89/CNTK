# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for non-linear operators.
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, C, AA, I, precision
from ...graph import *
from ...reader import *
from ..non_linear import sigmoid

TENSORS = [
    ([[0, -0.1]]),
    ([[-100, -10], [-1, -0.1], [-0.01, -0.001],
      [0.001, 0.01], [0.1, 1], [10, 100]]),
]

@pytest.mark.parametrize("tensor", TENSORS)
def test_op_sigmoid(tensor, device_id, precision):

    def numpy_sigmoid(x):
        return 1.0 / (1.0 + np.exp(-AA(x)))

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample

    expected = [[numpy_sigmoid(tensor)]]

    input_node = I([tensor], has_sequence_dimension=False)
    op_node = sigmoid(input_node)

    unittest_helper(op_node, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=False, backward_pass=False)

    # Backward pass test
    # ==================
    # The expected results for the backward pass is sigmoid(x)*(1-sigmoid(x))
    s = numpy_sigmoid(tensor)
    expected = [[s * (1 - s)]]

    unittest_helper(op_node, expected, device_id=device_id,
                    precision=precision, clean_up=False, backward_pass=True,
                    input_node=input_node)
