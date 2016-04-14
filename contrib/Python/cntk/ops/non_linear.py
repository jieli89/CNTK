# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Non-linear operators
"""

from cntk.ops.cntk1 import Exp, RectifiedLinear, Sigmoid, Softmax, Tanh

def rectified_linear(x, name=None):
    """
    computes the element-wise rectified linear of `x`: `max(x, 0)`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return RectifiedLinear(x, var_name=name)

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

def tanh(x, name=None):
    """
    computes the element-wise tanh of `x`: 

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Tanh(x, var_name=name)

def softmax(x, name=None):
    """
    computes the element-wise sigmoid of `x`: 

    :math:`softmax(X) = {\exp(X_ \over {1+\exp(-x)}}`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Softmax(x, var_name=name)

def exp(x, name=None):
    """
    computes the element-wise exponential of `x`: 

    :math:`exp(x) = {e^x}`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Exp(x, var_name=name)
