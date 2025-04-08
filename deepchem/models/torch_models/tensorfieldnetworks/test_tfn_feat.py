import unittest
import numpy as np
import logging
from unittest import mock
import deepchem as dc
from deepchem.feat import MolecularFeaturizer
from deepchem.utils.typing import RDKitMol
from typing import List, Sequence, Tuple, Optional
from deepchem.models.torch_models.tensorfieldnetworks.tfn_feat import TFNFeaturizer

# Ensure RDKit is installed for molecule processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    logging.warning("RDKit is not installed. TFNFeaturizer requires RDKit.")

logger = logging.getLogger(__name__)

# --- Start of Unit Tests ---
class TestTFNFeaturizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass # No setup needed for all tests

    def test_featurize_ethanol_with_conformer(self):
        """Test featurizing ethanol (CCO) with a generated conformer."""
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)

        featurizer = TFNFeaturizer(atom_types=(6, 8), generate_conformers=False)
        # featurize now returns a list
        list_of_features = featurizer.featurize([mol])

        # Check list structure
        self.assertIsInstance(list_of_features, list)
        self.assertEqual(len(list_of_features), 1)

        # Extract features for the first molecule
        features = list_of_features[0]
        self.assertIsNotNone(features) # Check success
        self.assertIsInstance(features, list) # Check inner list
        self.assertEqual(len(features), 2)

        coords, one_hot = features
        self.assertIsInstance(coords, np.ndarray)
        self.assertEqual(coords.shape, (3, 3))
        self.assertEqual(coords.dtype, np.float32)
        self.assertIsInstance(one_hot, np.ndarray)
        self.assertEqual(one_hot.shape, (3, featurizer.num_atom_types))
        self.assertEqual(featurizer.num_atom_types, 3)
        self.assertEqual(one_hot.dtype, np.float32)
        np.testing.assert_array_equal(one_hot[0], [1., 0., 0.])
        np.testing.assert_array_equal(one_hot[1], [1., 0., 0.])
        np.testing.assert_array_equal(one_hot[2], [0., 1., 0.])


    def test_featurize_generate_conformer(self):
        """Test generating a conformer if none exists."""
        smiles = "CO"
        mol = Chem.MolFromSmiles(smiles)
        self.assertEqual(mol.GetNumConformers(), 0)

        featurizer = TFNFeaturizer(atom_types=(6, 8), generate_conformers=True)
        list_of_features = featurizer.featurize([mol])

        self.assertIsInstance(list_of_features, list)
        self.assertEqual(len(list_of_features), 1)

        features = list_of_features[0]
        self.assertIsNotNone(features)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 2)

        coords, one_hot = features
        self.assertEqual(coords.shape, (2, 3))
        self.assertEqual(one_hot.shape, (2, 3))
        self.assertEqual(coords.dtype, np.float32)
        self.assertEqual(one_hot.dtype, np.float32)


    def test_featurize_no_generate_conformer(self):
        """Test returning None if no conformer and generation is off."""
        smiles = "CO"
        mol = Chem.MolFromSmiles(smiles)
        self.assertEqual(mol.GetNumConformers(), 0)

        featurizer = TFNFeaturizer(atom_types=(6, 8), generate_conformers=False)
        list_of_features = featurizer.featurize([mol])

        self.assertIsInstance(list_of_features, list)
        self.assertEqual(len(list_of_features), 1)
        self.assertIsNone(list_of_features[0]) # Check the element is None


    def test_featurize_unknown_atom_type(self):
        """Test handling of atoms not in the specified atom_types."""
        smiles = "CS"
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)

        featurizer = TFNFeaturizer(atom_types=(1, 6), generate_conformers=False) # H, C only
        list_of_features = featurizer.featurize([mol])

        self.assertIsInstance(list_of_features, list)
        self.assertEqual(len(list_of_features), 1)

        features = list_of_features[0]
        self.assertIsNotNone(features)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 2)

        coords, one_hot = features
        self.assertEqual(coords.shape, (2, 3))
        self.assertEqual(one_hot.shape, (2, 3)) # H, C + unknown
        self.assertEqual(featurizer.num_atom_types, 3)

        np.testing.assert_array_equal(one_hot[0], [0., 1., 0.]) # Atom 0 is Carbon
        np.testing.assert_array_equal(one_hot[1], [0., 0., 1.]) # Atom 1 is Sulfur (unknown)

    def test_add_hydrogens_false(self):
        """Test default behavior (no explicit hydrogens added by featurizer)."""
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)

        featurizer = TFNFeaturizer(atom_types=(6, 8), add_hydrogens=False)
        list_of_features = featurizer.featurize([mol])

        self.assertIsInstance(list_of_features, list)
        self.assertEqual(len(list_of_features), 1)

        features = list_of_features[0]
        self.assertIsNotNone(features)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 2)

        coords, one_hot = features
        self.assertEqual(coords.shape, (3, 3)) # C, C, O
        self.assertEqual(one_hot.shape, (3, 3)) # C, O + unknown


    def test_add_hydrogens_true(self):
        """Test adding explicit hydrogens during featurization."""
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles) # No conformer, no Hs initially

        featurizer = TFNFeaturizer(atom_types=(1, 6, 8), add_hydrogens=True, generate_conformers=True)

        # featurize now returns a list of results
        list_of_features = featurizer.featurize([mol])

        # Check list structure
        self.assertIsInstance(list_of_features, list)
        self.assertEqual(len(list_of_features), 1)

        # Extract the features for the first molecule (should be a list [coords, one_hot])
        features = list_of_features[0]
        self.assertIsNotNone(features) # Check featurization succeeded
        self.assertIsInstance(features, list) # Verify it's the inner list
        self.assertEqual(len(features), 2)

        # Unpack and test coords and one_hot
        coords, one_hot = features
        self.assertIsInstance(coords, np.ndarray)
        self.assertIsInstance(one_hot, np.ndarray)

        self.assertEqual(coords.shape, (9, 3)) # C, C, O + 6 H
        self.assertEqual(one_hot.shape, (9, 4)) # H, C, O + unknown
        self.assertEqual(featurizer.num_atom_types, 4)

        num_H = np.sum(one_hot[:, 0])
        num_C = np.sum(one_hot[:, 1])
        num_O = np.sum(one_hot[:, 2])
        num_Unk = np.sum(one_hot[:, 3])
        self.assertEqual(num_H, 6)
        self.assertEqual(num_C, 2)
        self.assertEqual(num_O, 1)
        self.assertEqual(num_Unk, 0)

    def test_featurize_invalid_input(self):
        """Test featurizing None or non-mol object."""
        featurizer = TFNFeaturizer()
        # Featurizing None input
        list_of_features_none = featurizer.featurize([None])
        self.assertIsInstance(list_of_features_none, list)
        self.assertEqual(len(list_of_features_none), 1)
        self.assertIsNone(list_of_features_none[0])

        # Featurizing non-mol input (invalid SMILES)
        list_of_features_str = featurizer.featurize(["not_a_smiles"])
        self.assertIsInstance(list_of_features_str, list)
        self.assertEqual(len(list_of_features_str), 1)
        self.assertIsNone(list_of_features_str[0])

    @mock.patch('rdkit.Chem.AllChem.EmbedMolecule') # Target the function to mock
    def test_conformer_generation_failure(self, mock_embed): # Mock object is passed as argument
        """Test graceful handling when conformer generation fails by mocking."""
        # Configure the mock to simulate failure
        mock_embed.return_value = -1

        smiles = "CCO" # Use any valid SMILES, as embedding will be mocked anyway
        mol = Chem.MolFromSmiles(smiles)
        self.assertEqual(mol.GetNumConformers(), 0) # Ensure no conformer initially

        # We still need generate_conformers=True to reach the mocked function
        featurizer = TFNFeaturizer(generate_conformers=True)

        # Suppress potential warnings if the logger reports failure
        # logging.disable(logging.WARNING)
        # Run featurization - AllChem.EmbedMolecule will be the mock returning -1
        list_of_features = featurizer.featurize([mol])

        # logging.disable(logging.NOTSET) # Re-enable logging if disabled

        # Assertions: Expect the featurization to return None in the list
        self.assertIsInstance(list_of_features, list)
        self.assertEqual(len(list_of_features), 1)
        # THIS assertion should pass because the mock forced a failure (-1 return)
        self.assertIsNone(list_of_features[0])

        # Check that the mock was actually called
        mock_embed.assert_called_once()

    def test_zero_atom_molecule(self):
        """Test handling of molecule with zero atoms."""
        mol_empty = Chem.Mol() # An empty molecule object
        featurizer = TFNFeaturizer()

        # featurize returns a list
        list_of_features = featurizer.featurize([mol_empty])

        # Assert that the output is a list containing one element
        self.assertIsInstance(list_of_features, list)
        self.assertEqual(len(list_of_features), 1)

        # Assert that the element for the empty molecule is None
        self.assertIsNone(list_of_features[0])

    
