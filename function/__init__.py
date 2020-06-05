"""
This module will provides the mathematical function
required for the deep learning
"""
import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    sigmoid(z)
    """
    return 1 / (1 + np.exp(-z))
