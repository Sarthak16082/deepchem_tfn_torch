"""
Utility functions for Tensor Field Networks in PyTorch.

This module provides utility functions for implementing Tensor Field Networks
as described in the paper "Tensor field networks: Rotation- and translation-equivariant
neural networks for 3D point clouds" (https://arxiv.org/abs/1802.08219).
"""

import torch
import numpy as np
import scipy.linalg

# Constants
FLOAT_TYPE = torch.float32
EPSILON = 1e-8


def get_eijk() -> torch.Tensor:
    """
    Constant Levi-Civita tensor.

    Returns:
        torch.Tensor of shape [3, 3, 3]
    """
    eijk_ = np.zeros((3, 3, 3))
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
    return torch.tensor(eijk_, dtype=FLOAT_TYPE)


def norm_with_epsilon(input_tensor: torch.Tensor, axis=None, keep_dims=False) -> torch.Tensor:
    """
    Regularized norm.

    Args:
        input_tensor: torch.Tensor
        axis: Dimension along which to compute the norm
        keep_dims: Whether to keep the dimensions

    Returns:
        torch.Tensor normed over axis
    """
    if axis is None:
        return torch.sqrt(torch.clamp(torch.sum(torch.square(input_tensor)), min=EPSILON))
    else:
        return torch.sqrt(torch.clamp(torch.sum(torch.square(input_tensor), dim=axis, keepdim=keep_dims), min=EPSILON))


def ssp(x: torch.Tensor) -> torch.Tensor:
    """
    Shifted soft plus nonlinearity.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor of same shape as x 
    """
    return torch.log(0.5 * torch.exp(x) + 0.5)


def rotation_equivariant_nonlinearity(x: torch.Tensor, nonlin=ssp, biases_initializer=None) -> torch.Tensor:
    """
    Rotation equivariant nonlinearity.

    The -1 axis is assumed to be M index (of which there are 2L + 1 for given L).

    Args:
        x: torch.Tensor with channels as -2 axis and M as -1 axis.
        nonlin: Nonlinearity function to apply
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)

    Returns:
        torch.Tensor of same shape as x with 3d rotation-equivariant nonlinearity applied.
    """
    shape = x.shape
    representation_index = shape[-1]

    if representation_index == 1:
        return nonlin(x)
    else:
        norm = norm_with_epsilon(x, axis=-1, keep_dims=True)
        nonlin_out = nonlin(norm)
        factor = torch.div(nonlin_out, norm)
        # Apply factor to each component
        return torch.multiply(x, factor)


def difference_matrix(geometry: torch.Tensor) -> torch.Tensor:
    """
    Get relative vector matrix for array of shape [N, 3].

    Args:
        geometry: torch.Tensor with Cartesian coordinates and shape [N, 3]

    Returns:
        Relative vector matrix with shape [N, N, 3]
    """
    # [N, 1, 3]
    ri = geometry.unsqueeze(1)
    # [1, N, 3]
    rj = geometry.unsqueeze(0)
    # [N, N, 3]
    rij = ri - rj
    return rij


def distance_matrix(geometry: torch.Tensor) -> torch.Tensor:
    """
    Get relative distance matrix for array of shape [N, 3].

    Args:
        geometry: torch.Tensor with Cartesian coordinates and shape [N, 3]

    Returns:
        Relative distance matrix with shape [N, N]
    """
    # [N, N, 3]
    rij = difference_matrix(geometry)
    # [N, N]
    dij = norm_with_epsilon(rij, axis=-1)
    return dij


def random_rotation_matrix(numpy_random_state=None):
    """
    Generates a random 3D rotation matrix from axis and angle.

    Args:
        numpy_random_state: numpy random state object

    Returns:
        Random rotation matrix.
    """
    if numpy_random_state is None:
        rng = np.random.RandomState()
    else:
        rng = numpy_random_state
        
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + EPSILON
    theta = 2 * np.pi * rng.uniform(0.0, 1.0)
    return rotation_matrix(axis, theta)


def rotation_matrix(axis, theta):
    """
    Create a rotation matrix for rotation around an axis by theta.
    
    Args:
        axis: 3D axis to rotate around
        theta: Angle to rotate by
        
    Returns:
        3x3 rotation matrix
    """
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))