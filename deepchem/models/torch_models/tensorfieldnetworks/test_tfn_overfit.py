import numpy as np
import torch
import deepchem as dc
import torch.nn as nn
import torch.optim as optim
from deepchem.models.torch_models.tensorfieldnetworks.tfn import TensorFieldNetworkModel, TensorFieldNetworkModule
import random

def test_tfn_overfitting():
    # --- Set Seeds ---
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) 
    # --------------------------

    # Synthetic Data
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    one_hot = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32) # Matches num_atom_types=1
    target = torch.tensor([[0.5]], dtype=torch.float32) # Target shape [1, 1]

    # Model Initialization
    model = TensorFieldNetworkModule(
        layer_dims=[32, 32, 32, 1],
        num_atom_types=1,
        rbf_low=0.0,
        rbf_high=1.5,
        rbf_count=16
    )

    # --- Move model to GPU if available ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    coords = coords.to(device)
    one_hot = one_hot.to(device)
    target = target.to(device)
    # ------------------------------------

    # Use MSE loss
    loss_fn = nn.MSELoss()

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

    # Training Loop
    num_epochs = 6000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        _, _, atom_type_scalars = model(coords, one_hot, debug=False)

        # Aggregate over atoms
        output = torch.mean(atom_type_scalars.squeeze(-1), dim=0, keepdim=True)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.8f}")

    # Assert Overfitting
    final_loss = loss.item()
    print(f"Final Loss: {final_loss:.8f}")
    assert final_loss < 1e-4