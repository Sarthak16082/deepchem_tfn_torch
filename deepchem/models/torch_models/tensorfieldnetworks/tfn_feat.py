import deepchem as dc
from deepchem.feat import MolecularFeaturizer
from deepchem.utils.typing import RDKitMol
import numpy as np
import logging
from typing import List, Sequence, Tuple, Optional

# Ensure RDKit is installed for molecule processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    logging.warning("RDKit is not installed. TFNFeaturizer requires RDKit.")

logger = logging.getLogger(__name__)

class TFNFeaturizer(MolecularFeaturizer):
    """
    Featurizer for the PyTorch Tensor Field Network (TFN) implementation.

    Generates the required inputs for the TFNModule's forward pass:
    1.  Atomic coordinates as a NumPy array.
    2.  One-hot encoded atom types as a NumPy array.

    Requires RDKit and expects molecules with 3D conformers. Can optionally
    attempt to generate a conformer if one is missing.

    Outputs a list containing [coordinates, one_hot_atom_types].
    """

    def __init__(self,
                 atom_types: Sequence[int] = (1, 6, 7, 8, 9), # Default: H, C, N, O, F
                 add_hydrogens: bool = False,
                 generate_conformers: bool = True,
                 max_attempts_confgen: int = 10):
        """
        Parameters
        ----------
        atom_types: Sequence[int], default (1, 6, 7, 8, 9)
            A sequence of atomic numbers to be included in the one-hot encoding.
            Atoms not in this list will be mapped to an "unknown" category
            if include_unknown is implicitly True (which it is here).
            The length of this sequence determines `num_atom_types`.
        add_hydrogens: bool, default False
            Whether to add explicit Hydrogens to the molecule before featurization.
            Set to True if your input data (e.g., SMILES) lacks hydrogens but
            they should be included in the model.
        generate_conformers: bool, default True
            If True, attempt to generate a 3D conformer using RDKit's ETKDG
            method if the input molecule doesn't have one.
        max_attempts_confgen: int, default 10
            Maximum number of attempts for conformer generation if needed.
        """
        if 'Chem' not in globals():
            raise ImportError("This featurizer requires RDKit to be installed.")

        self.atom_types = list(atom_types)
        # num_atom_types for the one-hot encoding includes the specified types +1 for unknown
        self.num_atom_types = len(self.atom_types) + 1
        self.add_hydrogens = add_hydrogens
        self.generate_conformers = generate_conformers
        self.max_attempts_confgen = max_attempts_confgen
        self._atom_map = {atom_num: i for i, atom_num in enumerate(self.atom_types)}




    def _featurize(self, datapoint: RDKitMol, **kwargs) -> Optional[List[np.ndarray]]:
        """
        Featurize a single RDKit Mol object.

        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
            RDKit Mol object.

        Returns
        -------
        Optional[List[np.ndarray]]
            A list containing [coordinates, one_hot_atom_types] NumPy arrays,
            or None if featurization fails (e.g., no conformer).
            - coordinates: shape (N, 3), dtype float32
            - one_hot_atom_types: shape (N, num_atom_types), dtype float32
        """
        if not isinstance(datapoint, Chem.Mol):
             # logger.warning("Input datapoint is not a valid RDKit Mol object.") # Noisy for tests
             return None # Cannot proceed if input isn't a molecule

        try:
            mol = Chem.Mol(datapoint) # Operate on a copy

            # Add Hydrogens if requested
            if self.add_hydrogens:
                mol = Chem.AddHs(mol, addCoords=True) # addCoords might be needed if generating conf later
                if mol is None: # Check if AddHs failed (can happen)
                     logger.warning("Failed to add hydrogens.")
                     return None # Cannot proceed

            # --- Conformer Check/Generation ---
            conformer = None
            try:
                conformer = mol.GetConformer()
            except ValueError: # No conformer exists
                 pass # Proceed to generation check

            if conformer is None:
                if self.generate_conformers:
                    #logger.info("No conformer found, attempting generation...") # Reduce noise in tests
                    # Add Hs temporarily for better conformer generation if not already added
                    mol_for_confgen = Chem.AddHs(mol, addCoords=True) if not self.add_hydrogens else mol
                    if mol_for_confgen is None:
                         logger.warning("Failed to add hydrogens for conformer generation.")
                         return None
                    # Use ETKDGv3 for potentially better results if available
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42
                    params.maxAttempts = self.max_attempts_confgen
                    conf_id = AllChem.EmbedMolecule(mol_for_confgen, params)
                    # Fallback to older ETKDG if v3 fails or isn't needed
                    # conf_id = AllChem.EmbedMolecule(mol_for_confgen, maxAttempts=self.max_attempts_confgen, randomSeed=42)

                    if conf_id == -1:
                        #logger.warning("Conformer generation failed.") # Reduce noise in tests
                        return None
                    try:
                        AllChem.UFFOptimizeMolecule(mol_for_confgen)
                    except Exception as opt_err: # Catch potential optimization errors
                         logger.warning(f"Conformer optimization failed: {opt_err}")
                         # Decide if you want to proceed with unoptimized conformer or fail
                         # return None # Option: Fail if optimization fails

                    # If hydrogens were added just for confgen, remove them now
                    if not self.add_hydrogens:
                        try:
                            mol = Chem.RemoveHs(mol_for_confgen) # Get the molecule with coords but no Hs
                            if mol is None: # Check if RemoveHs failed
                                 logger.warning("Failed to remove hydrogens after conformer generation.")
                                 return None
                        except Exception as e_rem: # Catch potential errors during RemoveHs
                             logger.warning(f"Error removing hydrogens after conformer generation: {e_rem}")
                             return None

                    else:
                        mol = mol_for_confgen # Keep the molecule with Hs and coords
                    try:
                       conformer = mol.GetConformer()
                    except ValueError:
                       #logger.error("Failed to retrieve conformer even after generation.") # Reduce noise in tests
                       return None
                else:
                    #logger.warning("Molecule has no 3D conformer and generate_conformers=False, skipping.") # Reduce noise
                    return None

            # --- Extract Coordinates ---
            try:
                 coords = conformer.GetPositions().astype(np.float32) # Shape (N, 3)
            except AttributeError: # If conformer is somehow still None or invalid
                 logger.warning("Invalid conformer object.")
                 return None

            num_atoms = mol.GetNumAtoms()
            if num_atoms == 0:
                 # logger.warning("Molecule has zero atoms after processing.") # Noisy
                 return None

            # --- Generate One-Hot Atom Types ---
            one_hot_types = np.zeros((num_atoms, self.num_atom_types), dtype=np.float32)
            unknown_type_index = len(self.atom_types) # Last index is for 'unknown'

            for i, atom in enumerate(mol.GetAtoms()):
                atom_num = atom.GetAtomicNum()
                type_index = self._atom_map.get(atom_num, unknown_type_index)
                one_hot_types[i, type_index] = 1.0

            return [coords, one_hot_types]

        except Exception as e:
            logger.warning(f"Failed to featurize datapoint: {e}", exc_info=True)
            return None
        
    def featurize(self, datapoints, log_every_n=1000, **kwargs) -> list:
        """Calculate features for molecules.

        Overrides the base class method to return a list of features
        instead of attempting (and failing) to cast to a NumPy array.
        Each element in the returned list corresponds to one input datapoint
        and is the list [coords, one_hot] returned by _featurize.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.rdchem import Mol
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        # Handle single datapoint case
        if isinstance(datapoints, str) or isinstance(datapoints, Mol):
            datapoints = [datapoints]
        else:
            datapoints = list(datapoints) # Ensure it's a list

        all_features: list = [] # Initialize list to store results
        for i, mol_input in enumerate(datapoints):
            if i % log_every_n == 0:
                logger.info("Featurizing datapoint %i" % i)

            feature = None # Default to None for this datapoint
            try:
                mol = mol_input # Assume it's already a Mol object or None/invalid
                if isinstance(mol_input, str):
                    # Simple SMILES parsing; add canonicalization if needed later
                    mol = Chem.MolFromSmiles(mol_input)
                    if mol is None:
                         raise ValueError(f"Failed to parse SMILES string: {mol_input}")

                # Pass **kwargs for potential future use
                if mol is not None: # Only call _featurize if we have a valid Mol object
                   feature = self._featurize(mol, **kwargs)

            except Exception as e:
                # Log warning and ensure feature remains None
                mol_id = Chem.MolToSmiles(mol) if isinstance(mol, Mol) else str(mol_input)
                logger.warning(
                    f"Failed to featurize datapoint {i}, {mol_id}. Appending None."
                )
                logger.debug(f"Exception: {e}", exc_info=True) # Log full error on debug level
                feature = None # Explicitly set feature to None on error

            finally:
                 all_features.append(feature) # Append the result (list or None)

        # Return the list of results directly
        return all_features

# Example Usage (similar to how you'd use it with DeepChem Loaders)
# featurizer = TFNFeaturizer(atom_types=[1, 6, 7, 8, 9], generate_conformers=True)
# smiles = 'CCO'
# mol = Chem.MolFromSmiles(smiles)
# features = featurizer.featurize(mol)
# if features:
#    print("Coordinates shape:", features[0].shape)
#    print("One-hot types shape:", features[1].shape)
#    print("Num atom types expected by featurizer:", featurizer.num_atom_types)