"""
Layer implementations for Tensor Field Networks in PyTorch.

This module provides layer implementations for Tensor Field Networks
as described in the paper "Tensor field networks: Rotation- and translation-equivariant
neural networks for 3D point clouds" (https://arxiv.org/abs/1802.08219).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
import math

from deepchem.models.torch_models.tensorfieldnetworks.utils import (
    EPSILON, FLOAT_TYPE, norm_with_epsilon, ssp, get_eijk, rotation_equivariant_nonlinearity
)


def R(inputs: torch.Tensor, 
      nonlin: Callable = F.relu, 
      hidden_dim: Optional[int] = None, 
      output_dim: int = 1, 
      weights_initializer=None, 
      biases_initializer=None) -> torch.Tensor: 
    """
    Radial function network for tensor field networks.
    
    Args:
        inputs: Input tensor of shape [N, N, input_dim]
        nonlin: Nonlinearity function
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output
        weights_initializer: Weight initialization function
        biases_initializer: Bias initialization function
        
    Returns:
        Output tensor of shape [N, N, output_dim]
    """
    input_dim = inputs.shape[-1]
    
    if hidden_dim is None:
        hidden_dim = input_dim
    
    # Create weights and biases manually to match TF implementation
    w1 = torch.empty(hidden_dim, input_dim, dtype=inputs.dtype, device=inputs.device)
    b1 = torch.empty(hidden_dim, dtype=inputs.dtype, device=inputs.device)
    w2 = torch.empty(output_dim, hidden_dim, dtype=inputs.dtype, device=inputs.device)
    b2 = torch.empty(output_dim, dtype=inputs.dtype, device=inputs.device)
    
    # Initialize weights and biases
    if weights_initializer is None:
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(w1)
        nn.init.xavier_uniform_(w2)
    else:
        # Custom initializer if provided
        weights_initializer(w1)
        weights_initializer(w2)
        
    if biases_initializer is None:
        # Zero initialization
        nn.init.zeros_(b1)
        nn.init.zeros_(b2)
    else:
        # Custom initializer if provided
        biases_initializer(b1)
        biases_initializer(b2)
    
    # Register parameters (optional, if you want them to be part of a module)
    w1 = nn.Parameter(w1)
    b1 = nn.Parameter(b1)
    w2 = nn.Parameter(w2)
    b2 = nn.Parameter(b2)
    
    # Apply operations exactly as in TF implementation
    # TF: hidden_layer = nonlin(b1 + tf.tensordot(inputs, w1, [[2], [1]]))
    hidden_layer = nonlin(b1.unsqueeze(0).unsqueeze(0) + torch.tensordot(inputs, w1, dims=([2], [1])))
    
    # TF: radial = b2 + tf.tensordot(hidden_layer, w2, [[2], [1]])
    radial = b2.unsqueeze(0).unsqueeze(0) + torch.tensordot(hidden_layer, w2, dims=([2], [1]))
    
    return radial


def unit_vectors(v: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """
    Normalize vectors along specified dimension.
    
    Args:
        v: Input tensor
        axis: Axis along which to normalize
        
    Returns:
        Normalized vectors
    """
    return v / (norm_with_epsilon(v, axis=axis, keep_dims=True))


def Y_2(rij: torch.Tensor) -> torch.Tensor:
    """
    Spherical harmonics of degree l=2.
    
    Args:
        rij: Relative position vectors of shape [N, N, 3]
        
    Returns:
        Spherical harmonics of shape [N, N, 5]
    """
    # rij : [N, N, 3]
    # x, y, z : [N, N]
    x = rij[..., 0]
    y = rij[..., 1]
    z = rij[..., 2]
    
    # Match TensorFlow's epsilon handling
    r2 = torch.maximum(torch.sum(torch.square(rij), dim=-1), torch.tensor(EPSILON, device=rij.device, dtype=rij.dtype))
    
    # Match TensorFlow's component order and formulas
    output = torch.stack([
        x * y / r2,
        y * z / r2,
        (-torch.square(x) - torch.square(y) + 2. * torch.square(z)) / (2 * math.sqrt(3) * r2),
        z * x / r2,
        (torch.square(x) - torch.square(y)) / (2. * r2)
    ], dim=-1)
    
    return output


def F_0(inputs: torch.Tensor, 
        nonlin: Callable = F.relu, 
        hidden_dim: Optional[int] = None, 
        output_dim: int = 1,
        weights_initializer=None, 
        biases_initializer=None) -> torch.Tensor:
    """
    Filter function for l=0 tensor field.
    
    Args:
        inputs: Input tensor of shape [N, N, input_dim]
        nonlin: Nonlinearity function
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output
        weights_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Output tensor of shape [N, N, output_dim, 1]
    """
    # [N, N, output_dim]
    outputs = R(inputs, nonlin, hidden_dim, output_dim, weights_initializer, biases_initializer)
    # [N, N, output_dim, 1]
    return outputs.unsqueeze(-1)


def F_1(inputs: torch.Tensor, 
        rij: torch.Tensor, 
        nonlin: Callable = F.relu, 
        hidden_dim: Optional[int] = None, 
        output_dim: int = 1,
        weights_initializer=None, 
        biases_initializer=None) -> torch.Tensor:
    """
    Filter function for l=1 tensor field.
    
    Args:
        inputs: Input tensor of shape [N, N, input_dim]
        rij: Relative position vectors of shape [N, N, 3]
        nonlin: Nonlinearity function
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output
        weights_initializer: Not used in PyTorch implementation
        biases_initializer: Not used in PyTorch implementation
        
    Returns:
        Output tensor of shape [N, N, output_dim, 3]
    """
    # [N, N, output_dim]
    radial = R(inputs, nonlin, hidden_dim, output_dim, weights_initializer, biases_initializer)
    
    # Mask out for dij = 0, matching TensorFlow implementation
    dij = torch.norm(rij, dim=-1)
    condition = (dij < EPSILON).unsqueeze(-1).expand(-1, -1, output_dim)
    masked_radial = torch.where(condition, torch.zeros_like(radial), radial)
    
    # [N, N, 3]
    unit_vecs = unit_vectors(rij)
    
    # [N, N, output_dim, 3]
    return unit_vecs.unsqueeze(-2) * masked_radial.unsqueeze(-1)


def F_2(inputs: torch.Tensor, 
        rij: torch.Tensor, 
        nonlin: Callable = F.relu, 
        hidden_dim: Optional[int] = None, 
        output_dim: int = 1,
        weights_initializer=None, 
        biases_initializer=None) -> torch.Tensor:
    """
    Filter function for l=2 tensor field.
    
    Args:
        inputs: Input tensor of shape [N, N, input_dim]
        rij: Relative position vectors of shape [N, N, 3]
        nonlin: Nonlinearity function
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output
        weights_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Output tensor of shape [N, N, output_dim, 5]
    
    """
    # [N, N, output_dim]
    radial = R(inputs, nonlin, hidden_dim, output_dim, weights_initializer, biases_initializer)

    # Mask out for dij = 0
    dij = torch.norm(rij, dim=-1)
    condition = (dij < EPSILON).unsqueeze(-1).expand(-1, -1, output_dim)
    masked_radial = torch.where(condition, torch.zeros_like(radial), radial)

    # [N, N, 5]
    Y2 = Y_2(rij)

    # [N, N, output_dim, 5]
    return Y2.unsqueeze(-2) * masked_radial.unsqueeze(-1)


def filter_0(layer_input: torch.Tensor,
             rbf_inputs: torch.Tensor,
             nonlin: Callable = F.relu,
             hidden_dim: Optional[int] = None,
             output_dim: int = 1,
             weights_initializer=None,
             biases_initializer=None) -> torch.Tensor:
    """
    Filter for l=0 tensor field.
    
    Args:
        layer_input: Input tensor of shape [N, input_dim, 1]
        rbf_inputs: Radial basis function inputs of shape [N, N, rbf_dim]
        nonlin: Nonlinearity function
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output
        weights_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Output tensor of shape [N, output_dim, 1]
    """

    # [N, input_dim, 1]
    layer_input_shape = layer_input.shape
    N = layer_input_shape[0]
    input_dim = layer_input_shape[1]
    
    # [N, N, input_dim, 1]
    layer_input_tiled = layer_input.unsqueeze(0).expand(N, -1, -1, -1)
    
    # [N, N, output_dim * input_dim, 1]
    filter_output = F_0(rbf_inputs, nonlin, hidden_dim, output_dim * input_dim, 
                        weights_initializer, biases_initializer)
    # Reshape to [N, N, output_dim, input_dim, 1]
    filter_output = filter_output.reshape(N, N, output_dim, input_dim, 1)
    
    # Multiply and sum over neighbors and input channels
    # [N, output_dim, 1]
    return torch.sum(filter_output * layer_input_tiled.unsqueeze(2), dim=(1, 3))

    
   
def filter_1_output_0(layer_input: torch.Tensor,
                      rbf_inputs: torch.Tensor,
                      rij: torch.Tensor,
                      nonlin: Callable = F.relu,
                      hidden_dim: Optional[int] = None,
                      output_dim: int = 1,
                      weights_initializer=None,
                      biases_initializer=None) -> torch.Tensor:
    """
    Filter for l=1 tensor field to l=0 tensor field.
    
    Args:
        layer_input: Input tensor of shape [N, input_dim, 3]
        rbf_inputs: Radial basis function inputs of shape [N, N, rbf_dim]
        rij: Relative position vectors of shape [N, N, 3]
        nonlin: Nonlinearity function
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output
        weights_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Output tensor of shape [N, output_dim, 1]
    """
    # Check input dimensions
    # Call the F_1 function 
    # Expected shape of F_1_out: [N, N, output_dim, 3]
    F_1_out = F_1(rbf_inputs,
                  rij,
                  nonlin=nonlin,
                  hidden_dim=hidden_dim,
                  output_dim=output_dim,
                  weights_initializer=weights_initializer,
                  biases_initializer=biases_initializer)
    
    # Check the last dimension of layer_input
    # Expected shape of layer_input: [N, output_dim, 3]
    if layer_input.shape[-1] == 1:
        raise ValueError("0 x 1 cannot yield 0")
    elif layer_input.shape[-1] == 3:
        # Create a constant tensor: [1, 3, 3]
        cg = torch.eye(3, device=layer_input.device).unsqueeze(0)
        # Perform the equivalent einsum operation:
        # 'ijk,abfj,bfk->afi'
        return torch.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
    else:
        raise NotImplementedError("Other Ls not implemented")


def filter_1_output_1(layer_input: torch.Tensor,
                      rbf_inputs: torch.Tensor,
                      rij: torch.Tensor,
                      nonlin: Callable = F.relu,
                      hidden_dim: Optional[int] = None,
                      output_dim: int = 1,
                      weights_initializer=None,
                      biases_initializer=None) -> torch.Tensor:
    """
    Filter for l=1 tensor field to l=1 tensor field.
    
    Args:
        layer_input: Input tensor of shape [N, input_dim, 3]
        rbf_inputs: Radial basis function inputs of shape [N, N, rbf_dim]
        rij: Relative position vectors of shape [N, N, 3]
        nonlin: Nonlinearity function
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output
        weights_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Output tensor of shape [N, output_dim, 3]
    """
    
    # Call F_1 to get a tensor of shape [N, N, output_dim, 3]
    F_1_out = F_1(rbf_inputs,
                  rij,
                  nonlin=nonlin,
                  hidden_dim=hidden_dim,
                  output_dim=output_dim,
                  weights_initializer=weights_initializer,
                  biases_initializer=biases_initializer)
    
    if layer_input.shape[-1] == 1:
        # Branch for "0 x 1 -> 1": create cg with shape [3,3,1]
        cg = torch.eye(3, device=layer_input.device).unsqueeze(-1)
        return torch.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
    elif layer_input.shape[-1] == 3:
        # Branch for "1 x 1 -> 1"
        return torch.einsum('ijk,abfj,bfk->afi', get_eijk(), F_1_out, layer_input)
    else:
        raise NotImplementedError("Other Ls not implemented")


def filter_2_output_2(layer_input: torch.Tensor,
                      rbf_inputs: torch.Tensor,
                      rij: torch.Tensor,
                      nonlin: Callable = F.relu,
                      hidden_dim: Optional[int] = None,
                      output_dim: int = 1,
                      weights_initializer=None,
                      biases_initializer=None) -> torch.Tensor:
    """
    Filter for l=2 tensor field to l=2 tensor field.
    
    Args:
        layer_input: Input tensor of shape [N, input_dim, 5]
        rbf_inputs: Radial basis function inputs of shape [N, N, rbf_dim]
        rij: Relative position vectors of shape [N, N, 3]
        nonlin: Nonlinearity function
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output
        weights_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Output tensor of shape [N, output_dim, 5]
    """
    # F_2 returns a tensor of shape [N, N, output_dim, 5]
    F_2_out = F_2(rbf_inputs,
                  rij,
                  nonlin=nonlin,
                  hidden_dim=hidden_dim,
                  output_dim=output_dim,
                  weights_initializer=weights_initializer,
                  biases_initializer=biases_initializer)
    
    if layer_input.shape[-1] == 1:
        # Branch for "0 x 2 -> 2":
        # Create constant tensor: identity matrix of size 5 expanded along last axis,
        # resulting in shape [5, 5, 1]
        cg = torch.eye(5, device=layer_input.device).unsqueeze(-1)
        # Perform the contraction:
        #   cg: [5, 5, 1]
        #   F_2_out: [N, N, output_dim, 5]
        #   layer_input: [N, output_dim, 1]
        # einsum indices 'ijk,abfj,bfk->afi' contract over the common dimensions,
        # yielding an output of shape [N, output_dim, 5]
        return torch.einsum('ijk,abfj,bfk->afi', cg, F_2_out, layer_input)
    else:
        raise NotImplementedError("Other Ls not implemented")

def self_interaction_layer_without_biases(inputs: torch.Tensor, 
                                         output_dim: int, 
                                         weights_initializer=None, 
                                         biases_initializer=None) -> torch.Tensor:
    """
    Self-interaction layer without biases.
    
    Args:
        inputs: Input tensor of shape [N, C, 2L+1]
        output_dim: Dimension of output
        weights_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Output tensor of shape [N, output_dim, 2L+1]
    """
    # Get the number of channels (input_dim)
    input_dim = inputs.shape[-2]

    # Create weight variable of shape [output_dim, input_dim]
    # If no initializer is provided, use orthogonal initialization.
    w_si = torch.empty(output_dim, input_dim, device=inputs.device, dtype=inputs.dtype)
    if weights_initializer is None:
        torch.nn.init.orthogonal_(w_si)
    else:
        weights_initializer(w_si)
    
    # Compute the self-interaction:
    #   Using einsum 'afi, gf -> aig' where:
    #     - inputs: [N, C, 2L+1]   (a, f, i)
    #     - w_si:   [output_dim, C] (g, f)
    #   The resulting tensor is of shape [N, 2L+1, output_dim].
    # Then, transpose to get shape [N, output_dim, 2L+1].
    out = torch.einsum('afi,gf->aig', inputs, w_si).transpose(1, 2)
    return out


def self_interaction_layer_with_biases(inputs: torch.Tensor, 
                                      output_dim: int, 
                                      weights_initializer=None, 
                                      biases_initializer=None) -> torch.Tensor:
    """
    Self-interaction layer with biases.
    
    Args:
        inputs: Input tensor of shape [N, C, 2L+1]
        output_dim: Dimension of output
        weights_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Output tensor of shape [N, output_dim, 2L+1]
    """
    # inputs shape: [N, C, 2L+1]
    input_dim = inputs.shape[-2]  # number of channels (C)

    # Create weight tensor of shape [output_dim, input_dim]
    w_si = torch.empty(output_dim, input_dim, device=inputs.device, dtype=inputs.dtype)
    if weights_initializer is None:
        torch.nn.init.orthogonal_(w_si)
    else:
        weights_initializer(w_si)
    
    # Create bias tensor of shape [output_dim]
    b_si = torch.empty(output_dim, device=inputs.device, dtype=inputs.dtype)
    if biases_initializer is None:
        torch.nn.init.constant_(b_si, 0.0)
    else:
        biases_initializer(b_si)
    
    # Compute einsum:
    #   inputs: [N, C, 2L+1] (indices: a, f, i)
    #   w_si:   [output_dim, C] (indices: g, f)
    # tf.einsum('afi,gf->aig', inputs, w_si) yields a tensor of shape [N, 2L+1, output_dim]
    #inputs = inputs / (torch.norm(inputs, dim=-1, keepdim=True) + 1e-8)
    out = torch.einsum('afi,gf->aig', inputs, w_si)
    
    # Add bias b_si of shape [output_dim] to the last dimension (output_dim axis)
    out = out + b_si  # broadcasting adds b_si along [N, 2L+1, output_dim]
    
    # Transpose the output to match tf.transpose(..., perm=[0,2,1]) so that the final shape is [N, output_dim, 2L+1]
    return out.transpose(1, 2)


def convolution(input_tensor_list: Dict[int, List[torch.Tensor]], 
                rbf: torch.Tensor, 
                unit_vectors: torch.Tensor, 
                weights_initializer=None, 
                biases_initializer=None) -> Dict[int, List[torch.Tensor]]:
    """
    Convolution operation for tensor field networks.
    
    Args:
        input_tensor_list: Dictionary mapping tensor order to list of tensors
        rbf: Radial basis function values of shape [N, N, rbf_dim]
        unit_vectors: Unit vectors of shape [N, N, 3]
        weights_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Dictionary mapping tensor order to list of tensors
    """
    output_tensor_list = {0: [], 1: []}
    for key in input_tensor_list:
        for tensor in input_tensor_list[key]:
            output_dim = tensor.shape[1]  # Get the channel dimension
            
            # Always apply filter_0 (L x 0 -> L)
            tensor_out = filter_0(tensor, rbf, 
                                 output_dim=output_dim,
                                 weights_initializer=weights_initializer,
                                 biases_initializer=biases_initializer)
            

            # Determine output tensor order based on output shape
            m = 0 if tensor_out.shape[-1] == 1 else 1
            output_tensor_list[m].append(tensor_out)
            
            if key == 1:
                # Only for l=1 tensors, apply filter_1_output_0 (1 x 1 -> 0)
                tensor_out = filter_1_output_0(tensor, rbf, unit_vectors,
                                              output_dim=output_dim,
                                              weights_initializer=weights_initializer,
                                              biases_initializer=biases_initializer)
                
            

                m = 0 if tensor_out.shape[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)
            
            if key == 0 or key == 1:
                # For l=0 and l=1 tensors, apply filter_1_output_1 (L x 1 -> 1)
                tensor_out = filter_1_output_1(tensor, rbf, unit_vectors,
                                              output_dim=output_dim,
                                              weights_initializer=weights_initializer,
                                              biases_initializer=biases_initializer)
                
                

                m = 0 if tensor_out.shape[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)
    
    return output_tensor_list
    

def self_interaction(input_tensor_list: Dict[int, List[torch.Tensor]], 
                    output_dim: int, 
                    weights_initializer=None, 
                    biases_initializer=None) -> Dict[int, List[torch.Tensor]]:
    """
    Self-interaction operation for tensor field networks.
    
    Args:
        input_tensor_list: Dictionary mapping tensor order to list of tensors
        output_dim: Dimension of output
        weights_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Dictionary mapping tensor order to list of tensors
    """
    output_tensor_list = {0: [], 1: []}
    
    for key in input_tensor_list:
        for i, tensor in enumerate(input_tensor_list[key]):
            if key == 0:
                tensor_out = self_interaction_layer_with_biases(
                    tensor, 
                    output_dim,
                    weights_initializer=weights_initializer,
                    biases_initializer=biases_initializer
                )
            else:
                tensor_out = self_interaction_layer_without_biases(
                    tensor, 
                    output_dim,
                    weights_initializer=weights_initializer,
                    biases_initializer=biases_initializer
                )
            # Determine output bucket based on last dimension
            m = 0 if tensor_out.shape[-1] == 1 else 1
            output_tensor_list[m].append(tensor_out)
    
    return output_tensor_list

def nonlinearity(input_tensor_list: Dict[int, List[torch.Tensor]], 
                nonlin: Callable = ssp, 
                biases_initializer=None) -> Dict[int, List[torch.Tensor]]:
    """
    Apply nonlinearity to tensor field networks.
    
    Args:
        input_tensor_list: Dictionary mapping tensor order to list of tensors
        nonlin: Nonlinearity function
        biases_initializer: Optional, unused here (kept for API consistency with TensorFlow)
        
    Returns:
        Dictionary mapping tensor order to list of tensors
    """
    output_tensor_list = {0: [], 1: []}

    for key in input_tensor_list:
        for i, tensor in enumerate(input_tensor_list[key]):
            # Call the provided rotation_equivariant_nonlinearity function.
            tensor_out = rotation_equivariant_nonlinearity(
                tensor,
                nonlin=nonlin,
                biases_initializer=biases_initializer
            )
            # Determine bucket: 0 if last dimension equals 1, else 1.
            m = 0 if tensor_out.shape[-1] == 1 else 1
            output_tensor_list[m].append(tensor_out)

    return output_tensor_list

    


def concatenation(input_tensor_list: Dict[int, List[torch.Tensor]]) -> Dict[int, List[torch.Tensor]]:
    """
    Concatenate tensors in tensor field networks.
    
    Args:
        input_tensor_list: Dictionary mapping tensor order to list of tensors
        
    Returns:
        Dictionary mapping tensor order to list of tensors
    """
    output_tensor_list = {}
    
    for l, tensors in input_tensor_list.items():
        if tensors:
            # Concatenate along channel dimension
            output_tensor_list[l] = [torch.cat(tensors, dim=1)]
    
    return output_tensor_list




