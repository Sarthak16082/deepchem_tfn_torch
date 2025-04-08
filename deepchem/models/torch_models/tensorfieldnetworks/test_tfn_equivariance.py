import unittest
import torch
import numpy as np
import math
import torch.nn.functional as F

# Import functions from layers and utils
from . import layers
from . import utils
from .tfn import TensorFieldNetworkModule

# --- Rotation Helper Functions ---

def random_rotation_matrix_torch(rng: np.random.RandomState, device='cpu', dtype=torch.float32) -> torch.Tensor:
    """Generates a random 3D rotation matrix using NumPy/SciPy and converts to Torch."""
    
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + 1e-8 
    theta = 2 * np.pi * rng.uniform(0.0, 1.0)
    import scipy.linalg
    rot_mat_np = scipy.linalg.expm(np.cross(np.eye(3), axis * theta)).astype(np.float32)
    return torch.tensor(rot_mat_np, dtype=dtype, device=device)

# Rotate L=1 features (shape [... , 3])
def rotate_L1_torch(tensor: torch.Tensor, rot_matrix: torch.Tensor) -> torch.Tensor:
    """Applies a 3x3 rotation matrix to the last dim of a tensor."""
    return torch.einsum('...j,kj->...k', tensor, rot_matrix)

# Rotate features with shape [N, C, 3]
def rotate_L1_NCD_torch(tensor: torch.Tensor, rot_matrix: torch.Tensor) -> torch.Tensor:
    """Applies a 3x3 rotation matrix to the last dim of a [N, C, 3] tensor."""
    return torch.einsum('ncj,kj->nck', tensor, rot_matrix)

# Rotate features with shape [N, N, C, 3]
def rotate_L1_NND_torch(tensor: torch.Tensor, rot_matrix: torch.Tensor) -> torch.Tensor:
    """Applies a 3x3 rotation matrix to the last dim of a [N, N, C, 3] tensor."""
    return torch.einsum('abcj,kj->abck', tensor, rot_matrix)

class TestTorchEquivariance(unittest.TestCase):

    def setUp(self):
        self.seed = 1234
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.RandomState(self.seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = utils.FLOAT_TYPE 
        # --- Parameters ---
        self.N = 5
        self.rbf_dim = 10
        self.channels_in = 4
        self.hidden_dim = 12
        self.channels_out = 4  

        # --- Geometry and Rotation ---
        self.geometry = torch.randn(self.N, 3, dtype=self.dtype, device=self.device)
        self.rot_matrix = random_rotation_matrix_torch(self.rng, device=self.device, dtype=self.dtype)
        self.rotated_geometry = torch.matmul(self.geometry, self.rot_matrix.T) # Apply rotation g' = g @ R^T

        # --- Derived Geometric Inputs ---
        self.rij = self.geometry.unsqueeze(1) - self.geometry.unsqueeze(0) # [N, N, 3]
        self.rotated_rij = self.rotated_geometry.unsqueeze(1) - self.rotated_geometry.unsqueeze(0)
        self.dij = utils.distance_matrix(self.geometry)
        self.rotated_dij = utils.distance_matrix(self.rotated_geometry)

        # --- RBF Features ---
        # Example: Gaussian basis functions (ensuring that these are invariant)
        self.rbf_widths = torch.abs(torch.randn(1, 1, self.rbf_dim, device=self.device, dtype=self.dtype)) * 0.5 + 0.1
        self.rbf_centers = torch.linspace(0.0, 5.0, self.rbf_dim, device=self.device, dtype=self.dtype).reshape(1, 1, -1)
        self.rbf_features = torch.exp(-self.rbf_widths * torch.square(self.dij.unsqueeze(-1) - self.rbf_centers))
        self.rotated_rbf_features = torch.exp(-self.rbf_widths * torch.square(self.rotated_dij.unsqueeze(-1) - self.rbf_centers))
        # Sanity check RBF invariance
        torch.testing.assert_close(self.rbf_features, self.rotated_rbf_features, rtol=1e-5, atol=1e-6)

        # --- Sample Input Features ([N, C_in, M]) ---
        self.input_L0 = torch.randn(self.N, self.channels_in, 1, dtype=self.dtype, device=self.device)
        self.rotated_input_L0 = self.input_L0 # Invariant

        self.input_L1 = torch.randn(self.N, self.channels_in, 3, dtype=self.dtype, device=self.device)
        self.rotated_input_L1 = rotate_L1_NCD_torch(self.input_L1, self.rot_matrix) # Rotate

    # --- Tests for Parameter-Free Functions ---

    def test_unit_vectors_equivariance(self):
        """Tests if unit_vectors transforms like an L=1 vector."""
        unit_vec_orig = layers.unit_vectors(self.rij)
        unit_vec_rotated_input = layers.unit_vectors(self.rotated_rij)
        expected_rotated_output = rotate_L1_torch(unit_vec_orig, self.rot_matrix)

        # Mask diagonal where rij is zero
        norm_rij = torch.norm(self.rij, dim=-1)
        epsilon_tensor = torch.tensor(utils.EPSILON, device=self.rij.device, dtype=self.rij.dtype)
        mask = norm_rij > epsilon_tensor

        # Compare only off-diagonal elements
        torch.testing.assert_close(
            unit_vec_rotated_input[mask], expected_rotated_output[mask],
            rtol=1e-5, atol=1e-6
        )

    def test_Y2_norm_invariance(self):
        """Tests if the norm of the Y_2 output is invariant under rotation."""
        Y2_orig = layers.Y_2(self.rij)
        Y2_rotated_input = layers.Y_2(self.rotated_rij)
        norm_Y2_orig = utils.norm_with_epsilon(Y2_orig, axis=-1)
        norm_Y2_rotated_input = utils.norm_with_epsilon(Y2_rotated_input, axis=-1)

        # Mask diagonal
        norm_rij = torch.norm(self.rij, dim=-1)
        epsilon_tensor = torch.tensor(utils.EPSILON, device=self.rij.device, dtype=self.rij.dtype)
        mask = norm_rij > epsilon_tensor

        torch.testing.assert_close(
            norm_Y2_rotated_input[mask], norm_Y2_orig[mask],
            rtol=1e-5, atol=1e-6
        )

    # --- Tests for functions in layers.py ---

    def test_R_invariance(self):
        """Tests layers.R invariance for invariant RBF inputs."""
        # Set the same seed before each call to ensure identical initialization
        torch.manual_seed(self.seed)
        output_orig = layers.R(self.rbf_features, F.relu, self.hidden_dim, self.channels_out)
        
        torch.manual_seed(self.seed)  # Reset seed for identical weight initialization
        output_rotated_input = layers.R(self.rotated_rbf_features, F.relu, self.hidden_dim, self.channels_out)
        
        # Output should be identical as input is invariant and weights are initialized the same
        torch.testing.assert_close(output_rotated_input, output_orig, rtol=1e-5, atol=1e-6)

    def test_F0_invariance(self):
        """Tests layers.F_0 invariance for invariant RBF inputs."""
        torch.manual_seed(self.seed)
        output_orig = layers.F_0(self.rbf_features, F.relu, self.hidden_dim, self.channels_out)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rotated_input = layers.F_0(self.rotated_rbf_features, F.relu, self.hidden_dim, self.channels_out)
        
        # Output should be identical [N, N, C, 1]
        torch.testing.assert_close(output_rotated_input, output_orig, rtol=1e-5, atol=1e-6)

    def test_F1_equivariance(self):
        """Tests layers.F_1 L=1 equivariance."""
        torch.manual_seed(self.seed)
        output_orig = layers.F_1(self.rbf_features, self.rij, F.relu, self.hidden_dim, self.channels_out)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rotated_input = layers.F_1(self.rotated_rbf_features, self.rotated_rij, F.relu, self.hidden_dim, self.channels_out)
        
        expected_rotated_output = rotate_L1_NND_torch(output_orig, self.rot_matrix)

        # Mask diagonal where rij is zero
        norm_rij = torch.norm(self.rij, dim=-1)
        epsilon_tensor = torch.tensor(utils.EPSILON, device=self.rij.device, dtype=self.rij.dtype)
        mask = norm_rij > epsilon_tensor
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1) # For [N, N, C, 3]

        torch.testing.assert_close(
            output_rotated_input * mask_expanded.float(), # Zero out diagonal
            expected_rotated_output * mask_expanded.float(),
            rtol=1e-5, atol=1e-6
        )

    def test_F2_norm_invariance(self):
        """Tests layers.F_2 L=2 norm invariance."""
        torch.manual_seed(self.seed)
        output_orig = layers.F_2(self.rbf_features, self.rij, F.relu, self.hidden_dim, self.channels_out)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rotated_input = layers.F_2(self.rotated_rbf_features, self.rotated_rij, F.relu, self.hidden_dim, self.channels_out)

        norm_orig = utils.norm_with_epsilon(output_orig, axis=-1)
        norm_rotated_input = utils.norm_with_epsilon(output_rotated_input, axis=-1)

        # Mask diagonal
        norm_rij = torch.norm(self.rij, dim=-1)
        epsilon_tensor = torch.tensor(utils.EPSILON, device=self.rij.device, dtype=self.rij.dtype)
        mask = norm_rij > epsilon_tensor

        torch.testing.assert_close(
            norm_rotated_input[mask], norm_orig[mask],
            rtol=1e-5, atol=1e-6
        )

    def test_filter0_equivariance(self):
        """Tests layers.filter_0: L_in x 0 -> L_in for L_in=0 and L_in=1."""

        # Case 1: L_in=0 -> L_out=0 (Invariance)
        torch.manual_seed(self.seed)
        output_orig_L0 = layers.filter_0(self.input_L0, self.rbf_features, F.relu, self.hidden_dim, self.channels_in)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rot_in_L0 = layers.filter_0(self.rotated_input_L0, self.rotated_rbf_features, F.relu, self.hidden_dim, self.channels_in)
        
        # Expected output is L=0, so invariant
        torch.testing.assert_close(output_rot_in_L0, output_orig_L0, rtol=1e-5, atol=1e-6, msg="filter_0 L_in=0")

        # Case 2: L_in=1 -> L_out=1 (Equivariance)
        torch.manual_seed(self.seed)
        output_orig_L1 = layers.filter_0(self.input_L1, self.rbf_features, F.relu, self.hidden_dim, self.channels_in)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rot_in_L1 = layers.filter_0(self.rotated_input_L1, self.rotated_rbf_features, F.relu, self.hidden_dim, self.channels_in)
        
        # Expected output is L=1, rotate original output
        expected_rot_out_L1 = rotate_L1_NCD_torch(output_orig_L1, self.rot_matrix)
        torch.testing.assert_close(output_rot_in_L1, expected_rot_out_L1, rtol=1e-5, atol=1e-6, msg="filter_0 L_in=1")
        
    def test_filter1_output0_equivariance(self):
        """Tests layers.filter_1_output_0: 1 x 1 -> 0."""
        torch.manual_seed(self.seed)
        output_orig = layers.filter_1_output_0(self.input_L1, self.rbf_features, self.rij, F.relu, self.hidden_dim, self.channels_in)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rot_in = layers.filter_1_output_0(self.rotated_input_L1, self.rotated_rbf_features, self.rotated_rij, F.relu, self.hidden_dim, self.channels_in)
        
        # Output should be invariant (L=0)
        torch.testing.assert_close(output_rot_in, output_orig, rtol=1e-5, atol=1e-6)
        
    def test_filter1_output1_equivariance(self):
        """Tests layers.filter_1_output_1: L x 1 -> 1 for L=0,1."""
        # Case 1: L_in=0 (Scalar) -> L_out=1 (Vector)
        torch.manual_seed(self.seed)
        output_orig_L0 = layers.filter_1_output_1(self.input_L0, self.rbf_features, self.rij, F.relu, self.hidden_dim, self.channels_in)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rot_in_L0 = layers.filter_1_output_1(self.rotated_input_L0, self.rotated_rbf_features, self.rotated_rij, F.relu, self.hidden_dim, self.channels_in)
        
        # Expected output is L=1, rotate original output
        expected_rot_out_L0 = rotate_L1_NCD_torch(output_orig_L0, self.rot_matrix)
        torch.testing.assert_close(output_rot_in_L0, expected_rot_out_L0, rtol=1e-5, atol=1e-6, msg="filter_1_output_1 L_in=0")
        
        # Case 2: L_in=1 (Vector) -> L_out=1 (Vector)
        torch.manual_seed(self.seed)
        output_orig_L1 = layers.filter_1_output_1(self.input_L1, self.rbf_features, self.rij, F.relu, self.hidden_dim, self.channels_in)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rot_in_L1 = layers.filter_1_output_1(self.rotated_input_L1, self.rotated_rbf_features, self.rotated_rij, F.relu, self.hidden_dim, self.channels_in)
        
        # Expected output is L=1, rotate original output
        expected_rot_out_L1 = rotate_L1_NCD_torch(output_orig_L1, self.rot_matrix)
        torch.testing.assert_close(output_rot_in_L1, expected_rot_out_L1, rtol=1e-5, atol=1e-6, msg="filter_1_output_1 L_in=1")
    
    def test_filter2_output2_equivariance(self):
        """Tests layers.filter_2_output_2: 0 x 2 -> 2."""
        # Setup L=2 shape tensor inputs
        self.input_L0_for_L2 = torch.randn(self.N, self.channels_in, 1, dtype=self.dtype, device=self.device)
        self.rotated_input_L0_for_L2 = self.input_L0_for_L2  # Invariant for L=0
        
        # L_in=0 -> L_out=2 (Outputs will have shape [N, C, 5])
        torch.manual_seed(self.seed)
        output_orig = layers.filter_2_output_2(self.input_L0_for_L2, self.rbf_features, self.rij, F.relu, self.hidden_dim, self.channels_in)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rot_in = layers.filter_2_output_2(self.rotated_input_L0_for_L2, self.rotated_rbf_features, self.rotated_rij, F.relu, self.hidden_dim, self.channels_in)
        
        # For L=2, we can check norm invariance (full equivariance is more complex with Wigner-D matrices)
        norm_orig = utils.norm_with_epsilon(output_orig, axis=-1)
        norm_rot_in = utils.norm_with_epsilon(output_rot_in, axis=-1)
        
        torch.testing.assert_close(norm_rot_in, norm_orig, rtol=1e-5, atol=1e-6)
        
    def test_self_interaction_layer_without_biases_equivariance(self):
        """Tests layers.self_interaction_layer_without_biases for both L=0 and L=1."""
        # For L=0 (invariant)
        torch.manual_seed(self.seed)
        output_orig_L0 = layers.self_interaction_layer_without_biases(self.input_L0, self.channels_out)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rot_in_L0 = layers.self_interaction_layer_without_biases(self.rotated_input_L0, self.channels_out)
        
        torch.testing.assert_close(output_rot_in_L0, output_orig_L0, rtol=1e-5, atol=1e-6, msg="self_interaction_layer_without_biases L=0")
        
        # For L=1 (equivariant)
        torch.manual_seed(self.seed)
        output_orig_L1 = layers.self_interaction_layer_without_biases(self.input_L1, self.channels_out)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rot_in_L1 = layers.self_interaction_layer_without_biases(self.rotated_input_L1, self.channels_out)
        
        expected_rot_out_L1 = rotate_L1_NCD_torch(output_orig_L1, self.rot_matrix)
        torch.testing.assert_close(output_rot_in_L1, expected_rot_out_L1, rtol=1e-5, atol=1e-6, msg="self_interaction_layer_without_biases L=1")

    def test_self_interaction_layer_with_biases_equivariance(self):
        """Tests layers.self_interaction_layer_with_biases for both L=0 and L=1."""
        # For L=0 (invariant)
        torch.manual_seed(self.seed)
        output_orig_L0 = layers.self_interaction_layer_with_biases(self.input_L0, self.channels_out)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rot_in_L0 = layers.self_interaction_layer_with_biases(self.rotated_input_L0, self.channels_out)
        
        torch.testing.assert_close(output_rot_in_L0, output_orig_L0, rtol=1e-5, atol=1e-6, msg="self_interaction_layer_with_biases L=0")
        
        # For L=1 (equivariant)
        torch.manual_seed(self.seed)
        output_orig_L1 = layers.self_interaction_layer_with_biases(self.input_L1, self.channels_out)
        
        torch.manual_seed(self.seed)  # Reset seed
        output_rot_in_L1 = layers.self_interaction_layer_with_biases(self.rotated_input_L1, self.channels_out)
        
        expected_rot_out_L1 = rotate_L1_NCD_torch(output_orig_L1, self.rot_matrix)
        torch.testing.assert_close(output_rot_in_L1, expected_rot_out_L1, rtol=1e-5, atol=1e-6, msg="self_interaction_layer_with_biases L=1")


    def test_tfn_module_equivariance(self):
        """Tests the equivariance of the entire TensorFieldNetworkModule."""
        # Set parameters for the model
        layer_dims = [8, 16, 8]  # Small network for testing
        num_atom_types = 4
        rbf_low = 0.0
        rbf_high = 5.0
        rbf_count = 10
        
        # Create a small, fixed seed model
        torch.manual_seed(self.seed)
        model = TensorFieldNetworkModule(
            layer_dims=layer_dims,
            num_atom_types=num_atom_types,
            rbf_low=rbf_low,
            rbf_high=rbf_high,
            rbf_count=rbf_count
        ).to(self.device)
        
        # Create sample input
        N = self.N  # Use the same number of points as other tests
        
        # Create coordinates and one-hot encodings
        coords = torch.randn(N, 3, dtype=self.dtype, device=self.device)
        one_hot = torch.zeros(N, num_atom_types, dtype=self.dtype, device=self.device)
        # Randomly assign atom types
        for i in range(N):
            one_hot[i, torch.randint(0, num_atom_types, (1,))] = 1.0
        
        # Create rotated inputs
        rotated_coords = torch.matmul(coords, self.rot_matrix.T)
        # One-hot encodings are invariant to rotation
        rotated_one_hot = one_hot.clone()
        
        # Verify inputs are correctly rotated
        torch.testing.assert_close(
            torch.matmul(coords, self.rot_matrix.T),
            rotated_coords,
            rtol=1e-5, atol=1e-6
        )
        
        # Forward pass on original inputs (set debug=False to avoid print statements)
        torch.manual_seed(self.seed)  # For any random operations during forward pass
        probability_scalars, missing_coordinates, atom_type_scalars = model(coords, one_hot, debug=False)
        
        # Forward pass on rotated inputs
        torch.manual_seed(self.seed)  # Same random seed for consistency
        rotated_probability_scalars, rotated_missing_coordinates, rotated_atom_type_scalars = model(rotated_coords, rotated_one_hot, debug=False)
        
        # Test 1: Probability scalars should be invariant (L=0 tensor)
        torch.testing.assert_close(
            rotated_probability_scalars,
            probability_scalars,
            rtol=1e-5, atol=1e-6,
            msg="Probability scalars (L=0) should be invariant under rotation"
        )
        
        # Test 2: Missing coordinates should be equivariant (L=1 tensor)
        # Apply rotation to original missing coordinates
        expected_rotated_missing_coordinates = rotate_L1_NCD_torch(missing_coordinates, self.rot_matrix)
        torch.testing.assert_close(
            rotated_missing_coordinates,
            expected_rotated_missing_coordinates,
            rtol=1e-5, atol=1e-6,
            msg="Missing coordinates (L=1) should transform equivariantly under rotation"
        )
        
        # Test 3: Atom type scalars should be invariant (L=0 tensor)
        torch.testing.assert_close(
            rotated_atom_type_scalars,
            atom_type_scalars,
            rtol=1e-5, atol=1e-6,
            msg="Atom type scalars (L=0) should be invariant under rotation"
        )