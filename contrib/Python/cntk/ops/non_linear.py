# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Non-linear operators
"""

from cntk.ops.cntk1 import Sigmoid


def sigmoid(x, name=None):
    """
    computes the element-wise sigmoid of `x`: 

    :math:`sigmoid(x) = {1 \over {1+\exp(-x)}}`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Sigmoid(x, var_name=name)
