"""
Tensor Field Network model implementation in PyTorch for DeepChem.

This module provides a PyTorch implementation of Tensor Field Networks
as described in the paper "Tensor field networks: Rotation- and translation-equivariant
neural networks for 3D point clouds" (https://arxiv.org/abs/1802.08219).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any, Iterable

from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.torch_models.tensorfieldnetworks.utils import (
    difference_matrix, distance_matrix, EPSILON, FLOAT_TYPE
)
from deepchem.models.torch_models.tensorfieldnetworks.layers import (
    unit_vectors, self_interaction_layer_with_biases,  self_interaction_layer_without_biases, convolution, self_interaction,
    nonlinearity, concatenation
)
from deepchem.data import Dataset


class TensorFieldNetworkModule(nn.Module):
    """
    Tensor Field Network implementation in PyTorch.
    
    This network is equivariant to 3D rotations and translations, making it suitable
    for tasks involving 3D point clouds such as molecular property prediction,
    point cloud classification, and missing point prediction.
    """
    
    def __init__(self,
                 layer_dims: List[int],
                 num_atom_types: int = 1,
                 rbf_low: float = 0.0,
                 rbf_high: float = 2.5,
                 rbf_count: int = 4):
        """
        Initialize the Tensor Field Network.
        
        Parameters
        ----------
        layer_dims: List[int]
            Dimensions of each layer in the network
        num_atom_types: int
            Number of atom types (for one-hot encoding)
        rbf_low: float
            Lower bound for radial basis functions
        rbf_high: float
            Upper bound for radial basis functions
        rbf_count: int
            Number of radial basis functions
        """
        super(TensorFieldNetworkModule, self).__init__()
        
        self.layer_dims = layer_dims
        self.num_atom_types = num_atom_types
        self.rbf_low = rbf_low
        self.rbf_high = rbf_high
        self.rbf_count = rbf_count
        
        # RBF centers
        self.register_buffer(
            'rbf_centers',
            torch.linspace(rbf_low, rbf_high, rbf_count)
        )
        self.rbf_spacing = (rbf_high - rbf_low) / rbf_count
        
        # Create embedding layer
        self.embedding = nn.Linear(num_atom_types, layer_dims[0])
        
        # Create layers for self-interaction
        self.self_interaction_layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            layer_dict = nn.ModuleDict()
            
            # For l=0 tensors
            layer_dict['0'] = nn.Linear(layer_dims[i], layer_dims[i+1])
            
            # For l=1 tensors (if not the first layer)
            if i > 0:
                layer_dict['1'] = nn.Linear(layer_dims[i], layer_dims[i+1])
            
            # For l=2 tensors (if not the first or second layer)
            if i > 1:
                layer_dict['2'] = nn.Linear(layer_dims[i], layer_dims[i+1])
            
            self.self_interaction_layers.append(layer_dict)
        
        # Final layer for atom type prediction
        self.atom_type_predictor = nn.Linear(layer_dims[-1], num_atom_types)
    
    def forward(self, r: torch.Tensor, one_hot: torch.Tensor, debug = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        r: torch.Tensor
            Coordinates of shape [N, 3]
        one_hot: torch.Tensor
            One-hot encoded atom types of shape [N, num_atom_types]
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of (probability scalars, missing coordinates, atom type scalars)
        """
        # Calculate relative vectors and distances
        rij = difference_matrix(r)
        dij = distance_matrix(r)
        unit_vec = unit_vectors(rij) # Renamed from unit_vectors to avoid conflict in convolution

        if debug:
            print("Distance matrix:")
            print(dij)

        # Calculate radial basis functions
        gamma = 1.0 / self.rbf_spacing
        rbf = torch.exp(-gamma * torch.square(dij.unsqueeze(-1) - self.rbf_centers))

        if debug:
            print("RBF values:")
            print(torch.mean(rbf, dim=(0,1)))

        # Embedding (l=0 tensor)
        embed = self.embedding(one_hot).unsqueeze(-1)  # [N, layer_dims[0], 1]

        if debug:
            print("Embedded input:")
            print(embed)

        # Initialize tensor list with embedding
        input_tensor_list = {0: [embed]}

        # Apply convolution layers
        num_layers = len(self.layer_dims) - 1
        for layer in range(num_layers):
            if debug:
                print(f"Layer {layer}:")

            # Convolution - this preserves rotation equivariance
            conv_tensor_list = convolution(input_tensor_list, rbf, unit_vec) # Pass the renamed unit_vec

            if debug:
                 if 0 in conv_tensor_list and conv_tensor_list[0]:
                     print(f"  l=0 conv output shape: {conv_tensor_list[0][0].shape}")
                     print(f"  l=0 conv values: {conv_tensor_list[0][0]}")
                 if 1 in conv_tensor_list and conv_tensor_list[1]:
                     print(f"  l=1 conv output shape: {conv_tensor_list[1][0].shape}")
                     print(f"  l=1 conv values: {conv_tensor_list[1][0]}")


            # Concatenation - this combines tensors of the same order
            concat_tensor_list = concatenation(conv_tensor_list)

            # Self-interaction - Apply correct version based on L
            si_tensor_list = {} # Use a temporary dict to store SI outputs
            for l in concat_tensor_list:
                if l not in si_tensor_list:
                    si_tensor_list[l] = []

                for tensor in concat_tensor_list[l]:
                    output_channel_dim = self.layer_dims[layer+1]
                    # Use self-interaction with biases for L=0
                    if l == 0:
                        transformed = self_interaction_layer_with_biases(
                            tensor,
                            output_channel_dim
                            # Add initializers if needed later
                        )
                    # Use self-interaction without biases for L > 0
                    else:
                        transformed = self_interaction_layer_without_biases(
                            tensor,
                            output_channel_dim
                            # Add initializers if needed later
                        )
                    si_tensor_list[l].append(transformed)

                    if debug:
                        if l == 0: print(f"  l=0 self-interaction output: {transformed}")
                        if l == 1: print(f"  l=1 self-interaction output: {transformed}")

            input_tensor_list = si_tensor_list # Update input_tensor_list for next step

            # Apply nonlinearity except at the final layer
            if layer < num_layers - 1:
                input_tensor_list = nonlinearity(input_tensor_list)

                if debug:
                     if 0 in input_tensor_list and input_tensor_list[0]:
                         print(f"  l=0 after nonlinearity: {input_tensor_list[0][0]}")
                     if 1 in input_tensor_list and input_tensor_list[1]:
                         print(f"  l=1 after nonlinearity: {input_tensor_list[1][0]}")


        # Predict atom types at the final layer
        # Ensure L=0 tensor exists before prediction
        if 0 in input_tensor_list and input_tensor_list[0]:
             atom_type_scalars = self.atom_type_predictor(input_tensor_list[0][0].squeeze(-1)).unsqueeze(-1)
             # Extract outputs
             probability_scalars = input_tensor_list[0][0]  # [N, layer_dims[-1], 1]
        else:
             # Handle case where L=0 tensor might disappear
             batch_size = r.shape[0]
             probability_scalars = torch.zeros((batch_size, self.layer_dims[-1], 1), device=r.device)
             atom_type_scalars = torch.zeros((batch_size, self.num_atom_types, 1), device=r.device)


        # Handle missing_coordinates
        missing_coordinates = None
        if 1 in input_tensor_list and input_tensor_list[1]: # Check if list is not empty
            missing_coordinates = input_tensor_list[1][0] # Assume first L=1 tensor is the coordinate prediction
        else:
            # Initialize with zeros if l=1 tensor doesn't exist or list is empty
            batch_size = r.shape[0]
            output_dim = self.layer_dims[-1]
            missing_coordinates = torch.zeros((batch_size, output_dim, 3), device=r.device)

        return probability_scalars, missing_coordinates, atom_type_scalars
        

        



class TensorFieldNetworkModel(TorchModel):
    """
    DeepChem model implementation of Tensor Field Networks.
    
    This model implements the Tensor Field Networks as described in the paper:
    "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds"
    (https://arxiv.org/abs/1802.08219)
    
    The model is equivariant to 3D rotations and translations, making it suitable for
    tasks involving 3D point clouds such as molecular property prediction, point cloud
    classification, and missing point prediction.
    
    Examples
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> # Create a dataset with 3D coordinates and atom types
    >>> coords = [np.random.random((5, 3)) for _ in range(10)]
    >>> atom_types = [np.random.randint(0, 5, size=(5,)) for _ in range(10)]
    >>> # Convert atom types to one-hot encoding
    >>> atom_order = list(range(5))
    >>> one_hot = []
    >>> for atoms in atom_types:
    ...     one_hot.append(np.eye(len(atom_order))[atoms])
    >>> # Create a dataset
    >>> y = np.random.random((10, 1))
    >>> dataset = dc.data.NumpyDataset(X=[coords, one_hot], y=y)
    >>> # Create and train the model
    >>> model = dc.models.TensorFieldNetworkModel(
    ...     n_tasks=1,
    ...     layer_dims=[15, 15, 15, 1],
    ...     num_atom_types=5,
    ...     mode='regression'
    ... )
    >>> model.fit(dataset, nb_epoch=10)
    """
    
    def __init__(self,
                 n_tasks: int,
                 layer_dims: List[int] = [15, 15, 15, 1],
                 num_atom_types: int = 1,
                 rbf_low: float = 0.0,
                 rbf_high: float = 2.5,
                 rbf_count: int = 4,
                 mode: str = 'regression',
                 n_classes: int = 2,
                 **kwargs):
        """
        Initialize the TensorFieldNetworkModel.
        
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        layer_dims: List[int]
            Dimensions of each layer in the network
        num_atom_types: int
            Number of atom types (for one-hot encoding)
        rbf_low: float
            Lower bound for radial basis functions
        rbf_high: float
            Upper bound for radial basis functions
        rbf_count: int
            Number of radial basis functions
        mode: str
            Either 'regression' or 'classification'
        n_classes: int
            Number of classes (only used in classification mode)
        """
        self.n_tasks = n_tasks
        self.layer_dims = layer_dims
        self.num_atom_types = num_atom_types
        self.rbf_low = rbf_low
        self.rbf_high = rbf_high
        self.rbf_count = rbf_count
        self.mode = mode
        self.n_classes = n_classes
        
        # Create the TensorFieldNetwork model
        model = TensorFieldNetworkModule(
            layer_dims=layer_dims,
            num_atom_types=num_atom_types,
            rbf_low=rbf_low,
            rbf_high=rbf_high,
            rbf_count=rbf_count
        )
        
        # Define loss function
        if mode == 'regression':
            def loss_fn(outputs, labels, weights):
                # Extract the probability scalars
                prob_scalars = outputs[0]
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(prob_scalars.squeeze(-1), dim=0)
                # Extract the vector predictions
                vectors = outputs[1]
                # Weighted sum of vectors using probabilities
                pred = torch.sum(probs.unsqueeze(-1) * vectors, dim=0)
                # Mean squared error loss
                return torch.mean(weights[0] * torch.square(pred - labels[0]))
        else:  # classification
            def loss_fn(outputs, labels, weights):
                # Extract the probability scalars
                prob_scalars = outputs[0]
                # Apply softmax to get probabilities
                logits = torch.mean(prob_scalars.squeeze(-1), dim=0)
                # Cross entropy loss
                return torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), 
                    labels[0].long(), 
                    reduction='none'
                ) * weights[0]
        
        # Initialize the TorchModel
        super(TensorFieldNetworkModel, self).__init__(
            model=model,
            loss=loss_fn,
            output_types=['prediction'] if mode == 'regression' else ['prediction', 'loss'],
            **kwargs
        )
    
    def default_generator(self,
                          dataset: Dataset,
                          epochs: int = 1,
                          mode: str = 'fit',
                          deterministic: bool = True,
                          pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """
        Generator that yields batches of data for training or prediction.
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to iterate over
        epochs: int
            Number of epochs to iterate over the dataset
        mode: str
            Either 'fit' or 'predict'
        deterministic: bool
            Whether to iterate over the dataset deterministically
        pad_batches: bool
            Whether to pad batches to the same size
            
        Yields
        ------
        Tuple[List, List, List]
            A tuple of (inputs, labels, weights)
        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
                batch_size=self.batch_size,
                deterministic=deterministic,
                pad_batches=pad_batches
            ):
                # X_b should be a list of [coordinates, one_hot]
                if isinstance(X_b, list) and len(X_b) == 2:
                    coords_b, one_hot_b = X_b
                else:
                    # Handle the case where X_b might be structured differently
                    raise ValueError("Expected X_b to be a list of [coordinates, one_hot]")
                
                # Convert to tensors
                coords_tensors = [torch.tensor(coords, dtype=torch.float32) for coords in coords_b]
                one_hot_tensors = [torch.tensor(one_hot, dtype=torch.float32) for one_hot in one_hot_b]
                
                # For each molecule in the batch
                for i, (coords, one_hot) in enumerate(zip(coords_tensors, one_hot_tensors)):
                    # For missing point prediction, we need to remove a random point
                    if mode == 'fit' and self.mode == 'regression':
                        # Randomly select a point to remove
                        remove_index = np.random.randint(0, len(coords))
                        # Save the removed point and its atom type
                        removed_point = coords[remove_index].clone()
                        removed_atom_type = one_hot[remove_index].clone()
                        # Remove the point from coordinates and one-hot
                        new_coords = torch.cat([coords[:remove_index], coords[remove_index+1:]], dim=0)
                        new_one_hot = torch.cat([one_hot[:remove_index], one_hot[remove_index+1:]], dim=0)
                        
                        # Use the removed point as the label
                        y = removed_point
                        atom_y = removed_atom_type
                        
                        yield ([new_coords, new_one_hot], [y, atom_y], [w_b[i]])
    
    def _prepare_batch(self, batch):
        """
        Prepare a batch for the model.
        
        Parameters
        ----------
        batch: Tuple
            A tuple of (inputs, labels, weights)
            
        Returns
        -------
        Tuple
            A tuple of (inputs, labels, weights) with tensors moved to the model's device
        """
        inputs, labels, weights = batch
    
        # Move inputs to device
        inputs_device = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                # Strip the index from the device for comparison
                inputs_device.append(inp.to(self.device))
            else:
                inputs_device.append(inp)
        
        # Move labels to device
        labels_device = []
        for label in labels:
            if isinstance(label, torch.Tensor):
                labels_device.append(label.to(self.device))
            elif label is not None:
                labels_device.append(torch.tensor(label, device=self.device))
            else:
                labels_device.append(None)
        
        # Move weights to device
        weights_device = []
        for weight in weights:
            if isinstance(weight, torch.Tensor):
                weights_device.append(weight.to(self.device))
            elif weight is not None:
                weights_device.append(torch.tensor(weight, device=self.device))
            else:
                weights_device.append(None)
        
        return (inputs_device, labels_device, weights_device)