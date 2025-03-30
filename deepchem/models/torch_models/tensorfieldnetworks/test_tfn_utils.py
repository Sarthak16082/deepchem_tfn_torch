"""
Tests for Tensor Field Networks utility functions.
"""
import pytest
import numpy as np
import torch
from deepchem.models.torch_models.tensorfieldnetworks.utils import (
    get_eijk, norm_with_epsilon, ssp, rotation_equivariant_nonlinearity,
    difference_matrix, distance_matrix, random_rotation_matrix, rotation_matrix
)


def test_get_eijk():
    """Test the Levi-Civita tensor."""
    eijk = get_eijk()
    
    # Check shape
    assert eijk.shape == (3, 3, 3)
    
    # Check specific values
    assert eijk[0, 1, 2].item() == 1.0
    assert eijk[0, 2, 1].item() == -1.0
    assert eijk[0, 0, 0].item() == 0.0


def test_norm_with_epsilon():
    """Test the norm_with_epsilon function."""
    # Test with 1D tensor
    x = torch.tensor([3.0, 4.0])
    norm = norm_with_epsilon(x)
    assert torch.isclose(norm, torch.tensor(5.0))
    
    # Test with 2D tensor
    x = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    norm = norm_with_epsilon(x, axis=1)
    assert torch.allclose(norm, torch.tensor([5.0, 13.0]))
    
    # Test with small values
    x = torch.tensor([1e-10, 1e-10])
    norm = norm_with_epsilon(x)
    assert norm > 0  # Should not be zero due to epsilon


def test_ssp():
    """Test the shifted soft plus function."""
    # Test with scalar
    x = torch.tensor(0.0)
    y = ssp(x)
    assert torch.isclose(y, torch.log(torch.tensor(1.0)))
    
    # Test with tensor
    x = torch.tensor([0.0, 1.0, 2.0])
    y = ssp(x)
    expected = torch.log(0.5 * torch.exp(x) + 0.5)
    assert torch.allclose(y, expected)


def test_rotation_equivariant_nonlinearity():
    """Test the rotation equivariant nonlinearity."""
    # Test with l=0 tensor (scalar representation)
    x = torch.tensor([[[1.0]], [[2.0]]])  # [2, 1, 1]
    y = rotation_equivariant_nonlinearity(x)
    assert torch.allclose(y, ssp(x))
    
    # Test with l=1 tensor (vector representation)
    x = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]])  # [2, 1, 3]
    y = rotation_equivariant_nonlinearity(x)
    
    # Instead of comparing norms directly, we check that:
    # 1. The direction (unit vector) is preserved
    x_normalized = x / norm_with_epsilon(x, axis=-1, keep_dims=True)
    y_normalized = y / norm_with_epsilon(y, axis=-1, keep_dims=True)
    assert torch.allclose(x_normalized, y_normalized)
    
    # 2. The magnitude has been transformed by the nonlinearity
    x_norm = norm_with_epsilon(x, axis=-1)
    y_norm = norm_with_epsilon(y, axis=-1)
    expected_norm = ssp(x_norm)
    assert torch.allclose(y_norm, expected_norm)
    
    # Test that the function respects batch dimensions
    x = torch.rand(5, 4, 3)  # batch_size=5, channels=4, representation_dim=3
    y = rotation_equivariant_nonlinearity(x)
    assert y.shape == x.shape
    
    # Check edge case: very small inputs should not cause numerical issues
    x_small = torch.ones(2, 1, 3) * 1e-10
    y_small = rotation_equivariant_nonlinearity(x_small)
    assert not torch.isnan(y_small).any()
    assert not torch.isinf(y_small).any()


def test_difference_matrix():
    """Test the difference_matrix function."""
    # Test with simple coordinates
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    diff = difference_matrix(coords)
    
    # Expected differences
    expected = torch.tensor([
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, -1.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    ])
    
    assert torch.allclose(diff, expected)


def test_distance_matrix():
    """Test the distance_matrix function."""
    # Test with simple coordinates
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dist = distance_matrix(coords)
    
    sqrt2 = 1.41421356237  # sqrt(2) value

    # Expected distances
    expected = torch.tensor([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, sqrt2],
        [1.0,sqrt2, 0.0]
    ])
    
    assert torch.allclose(dist, expected, atol=1e-3) 


def test_random_rotation_matrix():
    """Test the random_rotation_matrix function."""
    # Test with fixed seed
    rng = np.random.RandomState(42)
    rot_mat = random_rotation_matrix(rng)
    
    # Check that it's a valid rotation matrix
    # Determinant should be 1
    assert np.isclose(np.linalg.det(rot_mat), 1.0)
    
    # Rotation matrix should be orthogonal
    identity = np.eye(3)
    assert np.allclose(np.dot(rot_mat, rot_mat.T), identity, atol=1e-5)


def test_rotation_matrix():
    """Test the rotation_matrix function."""
    # Test rotation around z-axis
    axis = np.array([0, 0, 1])
    theta = np.pi/2  # 90 degrees
    rot_mat = rotation_matrix(axis, theta)
    
    # Rotating [1, 0, 0] around z by 90 degrees should give [0, 1, 0]
    point = np.array([1, 0, 0])
    rotated = np.dot(rot_mat, point)
    assert np.allclose(rotated, np.array([0, 1, 0]), atol=1e-5)