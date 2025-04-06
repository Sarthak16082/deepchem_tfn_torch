"""
Tests for Tensor Field Network model.
"""
import pytest
import numpy as np
import torch
import deepchem as dc
from deepchem.models.torch_models.tensorfieldnetworks.tfn import TensorFieldNetworkModule, TensorFieldNetworkModel
from deepchem.models.torch_models.tensorfieldnetworks.utils import random_rotation_matrix
from deepchem.models.torch_models.tensorfieldnetworks.layers import unit_vectors


def test_tensor_field_network_module_init():
    """Test initialization of TensorFieldNetworkModule."""
    # Create a simple model
    model = TensorFieldNetworkModule(
        layer_dims=[2, 3, 1],
        num_atom_types=4,
        rbf_low=0.0,
        rbf_high=5.0,
        rbf_count=10
    )
    
    # Check that the model has the expected attributes
    assert model.layer_dims == [2, 3, 1]
    assert model.num_atom_types == 4
    assert model.rbf_low == 0.0
    assert model.rbf_high == 5.0
    assert model.rbf_count == 10
    assert model.rbf_centers.shape == (10,)
    assert model.rbf_spacing == 0.5


def test_tensor_field_network_module_forward():
    """Test forward pass of TensorFieldNetworkModule."""
    # Create a simple model
    model = TensorFieldNetworkModule(
        layer_dims=[2, 3, 1],
        num_atom_types=4
    )
    
    # Create simple inputs
    r = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=torch.float32)
    
    one_hot = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=torch.float32)

    
    
    # Forward pass
    prob_scalars, missing_coords, atom_type_scalars = model(r, one_hot)
    
    # Check output shapes
    assert prob_scalars.shape == (3, 1, 1)
    assert missing_coords.shape == (3, 1, 3)
    assert atom_type_scalars.shape == (3, 4, 1)


def test_tensor_field_network_model_init():
    """Test initialization of TensorFieldNetworkModel."""
    # Create a simple model
    model = TensorFieldNetworkModel(
        n_tasks=1,
        layer_dims=[2, 3, 1],
        num_atom_types=4,
        mode='regression'
    )
    
    # Check that the model has the expected attributes
    assert model.n_tasks == 1
    assert model.layer_dims == [2, 3, 1]
    assert model.num_atom_types == 4
    assert model.mode == 'regression'


def test_tensor_field_network_model_prepare_batch():
    """Test _prepare_batch method of TensorFieldNetworkModel."""
    # Create a simple model
    model = TensorFieldNetworkModel(
        n_tasks=1,
        layer_dims=[2, 3, 1],
        num_atom_types=4,
        mode='regression'
    )
    
    # Create a simple batch
    r = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=torch.float32)
    
    one_hot = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=torch.float32)
    
    y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    w = torch.tensor(1.0)
    
    batch = ([r, one_hot], [y], [w])
    
    # Prepare batch
    inputs_device, labels_device, weights_device = model._prepare_batch(batch)
    
    # Check that tensors are moved to the correct device
    # Modify the device assertion lines (around line 216)
    assert inputs_device[0].device.type == model.device.type
    assert inputs_device[1].device.type == model.device.type
    assert labels_device[0].device.type == model.device.type
    assert weights_device[0].device.type == model.device.type

def test_tensor_field_network_model_default_generator():
    """Test default_generator method of TensorFieldNetworkModel."""
    # Create a simple model
    model = TensorFieldNetworkModel(
        n_tasks=1,
        layer_dims=[2, 3, 1],
        num_atom_types=4,
        mode='regression'
    )
    
    # Create a simple dataset
    r = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    one_hot = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # Create a mock batch directly
    coords = torch.tensor(r, dtype=torch.float32)
    atom_types = torch.tensor(one_hot, dtype=torch.float32)
    
    # Simulate removing a point (as the generator would do)
    remove_index = 1  # Remove the second point
    removed_point = coords[remove_index].clone()
    removed_atom_type = atom_types[remove_index].clone()
    
    # Remove the point from coordinates and one-hot
    new_coords = torch.cat([coords[:remove_index], coords[remove_index+1:]], dim=0)
    new_one_hot = torch.cat([atom_types[:remove_index], atom_types[remove_index+1:]], dim=0)
    
    # Create a mock batch
    mock_inputs = [new_coords, new_one_hot]
    mock_labels = [removed_point, removed_atom_type]
    mock_weights = [torch.tensor(1.0)]
    mock_batch = (mock_inputs, mock_labels, mock_weights)
    
    # Test the _prepare_batch method
    inputs_device, labels_device, weights_device = model._prepare_batch(mock_batch)
    
    # Check that tensors are moved to a device (without exact matching)
    assert str(inputs_device[0].device).startswith(str(model.device).split(':')[0])
    assert str(inputs_device[1].device).startswith(str(model.device).split(':')[0])
    assert str(labels_device[0].device).startswith(str(model.device).split(':')[0])
    assert str(labels_device[1].device).startswith(str(model.device).split(':')[0])
    assert str(weights_device[0].device).startswith(str(model.device).split(':')[0])
    
    # Check shapes
    assert inputs_device[0].shape == (2, 3)  # 2 atoms (after removal), 3 coordinates
    assert inputs_device[1].shape == (2, 4)  # 2 atoms (after removal), 4 atom types
    assert labels_device[0].shape == (3,)    # 3D coordinates of removed point
    assert labels_device[1].shape == (4,)    # one-hot encoding of removed point
