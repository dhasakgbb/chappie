#!/usr/bin/env python3
"""
Universal Complexity Computation Framework

This module implements Steps 3-4 of the consciousness creation framework:
- Step 3: Complexity Operator T[g,φ] for extracting complexity values
- Step 4: Universal Complexity U(t) integration over field configurations

The complexity computation uses JAX for high-performance vectorized operations
and implements proper quantum mechanical operator expectations as specified
in the mission framework.

Mathematical Foundation:
- T[g,φ] = |φ⟩⟨φ| ⊗ S[g] (complexity operator with symmetry)
- ComplexityValue(g,φ,t) = ⟨Ψ(t)|T[g,φ]|Ψ(t)⟩
- U(t) = ∫_M T[g,φ] dμ(g,φ) (universal complexity integral)

Authors: Consciousness Research Team
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import jax
import jax.numpy as jnp
from qutip import Qobj

# Configure JAX for CPU to ensure consistency
jax.config.update('jax_platform_name', 'cpu')

# Global cache for symmetry operators
_SYMMETRY_OPERATOR_CACHE: Dict[Tuple, jnp.ndarray] = {}
_CACHE_SIZE_LIMIT = 100


class SymmetryOperatorFactory:
    """
    Factory for creating and caching quantum symmetry operators S[g].
    
    Symmetry operators implement the group actions on the quantum field
    configurations, essential for the complexity operator construction.
    """
    
    @staticmethod
    def get_symmetry_operator(g_type: int, 
                            dimension: int, 
                            phi_subsystem_dims: List[List[int]]) -> jnp.ndarray:
        """
        Get or create JAX matrix representation of symmetry operator S[g].
        
        Args:
            g_type: Type of symmetry (0=identity, 1=permutation)
            dimension: Hilbert space dimension
            phi_subsystem_dims: Tensor product structure [[d1,d2,...], [1,1,...]]
            
        Returns:
            JAX array representing the symmetry operator
            
        Raises:
            ValueError: If g_type is not supported
        """
        # Create hashable cache key
        phi_dims_tuple = tuple(tuple(inner) for inner in phi_subsystem_dims)
        cache_key = (g_type, dimension, phi_dims_tuple)
        
        # Manage cache size
        if len(_SYMMETRY_OPERATOR_CACHE) > _CACHE_SIZE_LIMIT:
            _SYMMETRY_OPERATOR_CACHE.clear()
        
        # Return cached operator if available
        if cache_key in _SYMMETRY_OPERATOR_CACHE:
            return _SYMMETRY_OPERATOR_CACHE[cache_key]
        
        # Create new symmetry operator
        operator_matrix = SymmetryOperatorFactory._create_symmetry_matrix(
            g_type, dimension, phi_subsystem_dims
        )
        
        # Cache and return
        _SYMMETRY_OPERATOR_CACHE[cache_key] = operator_matrix
        return operator_matrix
    
    @staticmethod
    def _create_symmetry_matrix(g_type: int, 
                              dimension: int, 
                              phi_subsystem_dims: List[List[int]]) -> jnp.ndarray:
        """
        Create symmetry operator matrix based on type and structure.
        
        Args:
            g_type: Symmetry type
            dimension: Matrix dimension
            phi_subsystem_dims: Subsystem structure
            
        Returns:
            JAX array of the symmetry operator
            
        Raises:
            ValueError: If g_type is unsupported
        """
        if g_type == 0:
            # Identity operator S[0] = I
            matrix = np.eye(dimension, dtype=complex)
            
        elif g_type == 1:
            # Permutation operator S[1]
            matrix = SymmetryOperatorFactory._create_permutation_operator(
                dimension, phi_subsystem_dims
            )
            
        else:
            raise ValueError(f"Unsupported symmetry type g_type={g_type}. Expected 0 or 1.")
        
        # Validate operator properties
        SymmetryOperatorFactory._validate_symmetry_operator(matrix)
        
        return jnp.array(matrix)
    
    @staticmethod
    def _create_permutation_operator(dimension: int, 
                                   phi_subsystem_dims: List[List[int]]) -> np.ndarray:
        """
        Create permutation symmetry operator for quantum systems.
        
        Args:
            dimension: Total Hilbert space dimension
            phi_subsystem_dims: Subsystem structure
            
        Returns:
            Permutation operator matrix
        """
        if dimension == 4 and len(phi_subsystem_dims[0]) == 2:
            # Two-qubit SWAP gate for consciousness subsystems
            # SWAP: |00⟩→|00⟩, |01⟩→|10⟩, |10⟩→|01⟩, |11⟩→|11⟩
            matrix = np.array([
                [1, 0, 0, 0],  # |00⟩ → |00⟩
                [0, 0, 1, 0],  # |01⟩ → |10⟩
                [0, 1, 0, 0],  # |10⟩ → |01⟩
                [0, 0, 0, 1]   # |11⟩ → |11⟩
            ], dtype=complex)
        else:
            # General cyclic permutation for other dimensions
            matrix = np.zeros((dimension, dimension), dtype=complex)
            for i in range(dimension):
                matrix[i, (i + 1) % dimension] = 1
        
        return matrix
    
    @staticmethod
    def _validate_symmetry_operator(matrix: np.ndarray) -> None:
        """
        Validate symmetry operator properties.
        
        Args:
            matrix: Operator matrix to validate
            
        Raises:
            ValueError: If operator is invalid
        """
        if not np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=1e-10):
            raise ValueError("Symmetry operator must be unitary")


class ComplexityOperator:
    """
    Implements the complexity operator T[g,φ] for quantum consciousness analysis.
    
    The complexity operator extracts complexity values from the quantum universe
    state through projective measurements with symmetry transformations.
    
    Mathematical form: T[g,φ] = |φ⟩⟨φ| ⊗ S[g]
    """
    
    @staticmethod
    @jax.jit
    def compute_single_complexity(psi_vector: jnp.ndarray, 
                                phi_vector: jnp.ndarray, 
                                symmetry_operator: jnp.ndarray) -> float:
        """
        Compute single complexity value T[g,φ] = ⟨Ψ(t)|T[g,φ]|Ψ(t)⟩.
        
        This implements the core complexity calculation as an operator expectation
        value, following the mathematical framework in mission.txt.
        
        Args:
            psi_vector: Universe state |Ψ(t)⟩ as JAX array
            phi_vector: Field configuration |φ⟩ as JAX array  
            symmetry_operator: Symmetry operator S[g] as JAX array
            
        Returns:
            Complexity value T[g,φ]
        """
        # Apply symmetry transformation: S[g]†|Ψ⟩
        symmetry_dagger = jnp.conj(symmetry_operator.T)
        transformed_psi = symmetry_dagger @ psi_vector.reshape(-1, 1)
        transformed_psi = transformed_psi.flatten()
        
        # Compute projective complexity: |⟨φ|S[g]†|Ψ⟩|²
        projection = jnp.vdot(phi_vector, transformed_psi)
        complexity_value = jnp.abs(projection) ** 2
        
        return complexity_value
    
    @staticmethod
    def compute_batch_complexity(psi_vector: jnp.ndarray,
                               phi_vectors: jnp.ndarray,
                               symmetry_operators: jnp.ndarray) -> jnp.ndarray:
        """
        Compute complexity values for batch of configurations using vectorization.
        
        Args:
            psi_vector: Universe state vector
            phi_vectors: Batch of field configuration vectors
            symmetry_operators: Batch of symmetry operators
            
        Returns:
            Array of complexity values
        """
        # Vectorize over configurations
        batch_compute = jax.vmap(
            ComplexityOperator.compute_single_complexity,
            in_axes=(None, 0, 0),
            out_axes=0
        )
        
        return batch_compute(psi_vector, phi_vectors, symmetry_operators)


class UniversalComplexityCalculator:
    """
    Calculates Universal Complexity U(t) through integration over field configurations.
    
    This implements Step 4 of the consciousness framework: computing the universal
    complexity by averaging complexity values over the field configuration space M.
    """
    
    def __init__(self):
        self.symmetry_factory = SymmetryOperatorFactory()
        self.complexity_operator = ComplexityOperator()
    
    def compute_universal_complexity(self, 
                                   psi_qobj: Qobj,
                                   batched_field_configs: Dict) -> Tuple[float, jnp.ndarray, Dict[int, List[float]]]:
        """
        Compute Universal Complexity U(t) and detailed complexity analysis.
        
        This is the main function implementing the universal complexity integral:
        U(t) = ∫_M T[g,φ] dμ(g,φ)
        
        Args:
            psi_qobj: Universe state Ψ(t) as QuTiP Qobj
            batched_field_configs: Field configurations from FieldConfigurationSpace
            
        Returns:
            Tuple containing:
            - U_val: Universal complexity value
            - all_T_values: Array of all individual complexity values
            - T_by_g_type: Complexity values grouped by symmetry type
            
        Raises:
            ValueError: If field configurations are invalid
        """
        # Validate inputs
        self._validate_inputs(psi_qobj, batched_field_configs)
        
        # Extract configuration data
        g_types = batched_field_configs['g_types_jax']
        phi_vectors = batched_field_configs['phi_vectors_jax']
        phi_dimension = batched_field_configs['phi_dimension']
        phi_subsystem_dims = batched_field_configs['phi_subsystem_dims']
        
        num_configs = phi_vectors.shape[0]
        if num_configs == 0:
            return 0.0, jnp.array([]), {0: [], 1: []}
        
        # Convert universe state to JAX
        psi_vector = jnp.array(psi_qobj.full().flatten())
        
        # Prepare symmetry operators
        symmetry_operators = self._prepare_symmetry_operators(
            g_types, phi_dimension, phi_subsystem_dims
        )
        
        # Compute all complexity values
        all_T_values = self.complexity_operator.compute_batch_complexity(
            psi_vector, phi_vectors, symmetry_operators
        )
        
        # Calculate universal complexity
        U_val = float(jnp.mean(all_T_values))
        
        # Group by symmetry type
        T_by_g_type = self._group_by_symmetry_type(all_T_values, g_types)
        
        return U_val, all_T_values, T_by_g_type
    
    def _validate_inputs(self, psi_qobj: Qobj, batched_configs: Dict) -> None:
        """
        Validate input parameters for complexity computation.
        
        Args:
            psi_qobj: Universe state to validate
            batched_configs: Field configurations to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(psi_qobj, Qobj) or not psi_qobj.isket:
            raise ValueError("psi_qobj must be a QuTiP ket state")
        
        required_keys = ['g_types_jax', 'phi_vectors_jax', 'phi_dimension', 'phi_subsystem_dims']
        for key in required_keys:
            if key not in batched_configs:
                raise ValueError(f"Missing required key '{key}' in batched_field_configs")
        
        if not np.isclose(psi_qobj.norm(), 1.0, atol=1e-10):
            raise ValueError(f"Universe state must be normalized, got norm {psi_qobj.norm()}")
    
    def _prepare_symmetry_operators(self, 
                                  g_types: jnp.ndarray,
                                  phi_dimension: int,
                                  phi_subsystem_dims: List[List[int]]) -> jnp.ndarray:
        """
        Prepare batch of symmetry operators for complexity computation.
        
        Args:
            g_types: Array of symmetry types
            phi_dimension: Field configuration dimension
            phi_subsystem_dims: Subsystem structure
            
        Returns:
            Batch of symmetry operators
        """
        # Get base symmetry operators
        s0_matrix = self.symmetry_factory.get_symmetry_operator(
            0, phi_dimension, phi_subsystem_dims
        )
        s1_matrix = self.symmetry_factory.get_symmetry_operator(
            1, phi_dimension, phi_subsystem_dims
        )
        
        # Create batch based on g_types
        symmetry_batch = jnp.where(
            g_types.reshape(-1, 1, 1) == 0,
            s0_matrix[None, :, :],  # Broadcast S[0]
            s1_matrix[None, :, :]   # Broadcast S[1]
        )
        
        return symmetry_batch
    
    def _group_by_symmetry_type(self, 
                               all_T_values: jnp.ndarray,
                               g_types: jnp.ndarray) -> Dict[int, List[float]]:
        """
        Group complexity values by symmetry type for analysis.
        
        Args:
            all_T_values: Array of complexity values
            g_types: Array of symmetry types
            
        Returns:
            Dictionary mapping symmetry types to complexity value lists
        """
        T_by_g_type = {0: [], 1: []}
        
        g_types_np = np.array(g_types)
        all_T_values_np = np.array(all_T_values)
        
        for i in range(len(g_types_np)):
            g_type = int(g_types_np[i])
            t_value = float(all_T_values_np[i])
            
            if g_type in T_by_g_type:
                T_by_g_type[g_type].append(t_value)
        
        return T_by_g_type


# Main computation function for external use
def compute_universal_complexity_U(psi_qobj: Qobj, 
                                 batched_field_configs: Dict) -> Tuple[float, jnp.ndarray, Dict[int, List[float]]]:
    """
    Main function to compute Universal Complexity U(t).
    
    This is the primary interface for computing universal complexity as specified
    in the consciousness creation framework.
    
    Args:
        psi_qobj: Universe state Ψ(t) as QuTiP Qobj
        batched_field_configs: Field configurations from FieldConfigurationSpace
        
    Returns:
        Tuple containing:
        - Universal complexity U(t)
        - Array of all complexity values T[g,φ]
        - Complexity values grouped by symmetry type
    """
    calculator = UniversalComplexityCalculator()
    return calculator.compute_universal_complexity(psi_qobj, batched_field_configs)


# Example usage and testing
if __name__ == '__main__':
    print("🔬 Universal Complexity Computation - Consciousness Framework")
    print("=" * 70)
    
    # Import required modules
    from universe import UniverseState
    from fields import FieldConfigurationSpace
    
    # Setup quantum system for consciousness analysis
    subsystem_dims = [2, 2]  # Two-qubit consciousness subsystem
    phi_dimension = int(np.prod(subsystem_dims))
    phi_qobj_dims = [subsystem_dims, [1] * len(subsystem_dims)]
    
    print(f"Quantum system dimension: {phi_dimension}")
    print(f"Subsystem structure: {subsystem_dims}")
    print()
    
    # Create universe state
    universe = UniverseState(
        dimension=phi_dimension,
        initial_state_seed=42,
        subsystem_dims=subsystem_dims  # Pass the list directly, not the nested structure
    )
    psi_state = universe.get_state()
    
    # Create field configuration space
    field_space = FieldConfigurationSpace(
        dimension=phi_dimension,
        num_configs=100,
        phi_seed=101,
        phi_subsystem_dims_override=phi_qobj_dims  # Use the nested structure here
    )
    
    # Get batched configurations
    batched_configs = field_space.get_jax_configurations()
    
    print("Computing Universal Complexity U(t)...")
    
    # Compute universal complexity
    U_value, all_T_values, T_by_g_type = compute_universal_complexity_U(
        psi_state, batched_configs
    )
    
    print(f"Universal Complexity U(t): {U_value:.6f}")
    print(f"Total configurations: {len(all_T_values)}")
    print(f"Complexity values by symmetry type:")
    for g_type, values in T_by_g_type.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  g_type {g_type}: {len(values)} configs, "
                  f"mean={mean_val:.6f}, std={std_val:.6f}")
    print()
    
    # Test with perturbed state
    print("Testing with quantum perturbation...")
    universe.perturb_state(amplitude=0.1, seed=202)
    psi_perturbed = universe.get_state()
    
    U_perturbed, _, _ = compute_universal_complexity_U(
        psi_perturbed, batched_configs
    )
    
    print(f"Universal Complexity after perturbation: {U_perturbed:.6f}")
    print(f"Complexity change: {U_perturbed - U_value:.6f}")
    print()
    
    print("✓ Universal complexity computation validation complete")

