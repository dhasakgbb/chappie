#!/usr/bin/env python3
"""
Field Configuration Space Management

This module implements Step 2 of the consciousness creation framework:
"Define Field Configurations on space M with measure μ"

The FieldConfigurationSpace class manages the quantum field configurations
that serve as the substrate for complexity calculations and consciousness
emergence. It provides efficient JAX-based sampling and storage of field
states with proper tensor product structure.

Mathematical Foundation:
- Field configurations φ ∈ M (configuration space)
- Symmetry types g ∈ G (symmetry group)
- Measure μ over M×G for integration
- Compatibility with quantum operators and subsystem structure

Authors: Consciousness Research Team
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp
from qutip import Qobj

# Configure JAX for CPU consistency
jax.config.update('jax_platform_name', 'cpu')


class FieldConfigurationSpace:
    """
    Manages quantum field configurations for consciousness analysis.
    
    This class implements the field configuration space M with measure μ
    as specified in Step 2 of the consciousness framework. It provides
    efficient sampling and storage of field configurations with proper
    quantum mechanical structure.
    
    The configurations consist of:
    - Symmetry types g ∈ {0, 1} (identity and permutation)
    - Field states φ ∈ ℂ^d (normalized quantum states)
    - Tensor product structure for subsystem compatibility
    
    Attributes:
        dimension: Dimensionality of field configuration vectors
        num_configs: Number of configurations in the space
        phi_subsystem_dims: Tensor product structure [[d1,d2,...], [1,1,...]]
        g_types_jax: JAX array of symmetry types
        phi_vectors_jax: JAX array of field configuration vectors
    """
    
    def __init__(self, 
                 dimension: int, 
                 num_configs: int, 
                 phi_seed: Optional[int] = None,
                 phi_subsystem_dims_override: Optional[List[Union[int, List[int]]]] = None) -> None:
        """
        Initialize field configuration space with quantum structure.
        
        Args:
            dimension: Dimensionality of field configuration vectors
            num_configs: Number of configurations to generate
            phi_seed: Random seed for reproducible field generation
            phi_subsystem_dims_override: Tensor product structure specification
                Can be [d1, d2, ...] or [[d1, d2, ...], [1, 1, ...]]
                
        Raises:
            ValueError: If parameters are invalid or incompatible
        """
        # Validate basic parameters
        self._validate_basic_parameters(dimension, num_configs)
        
        self.dimension = dimension
        self.num_configs = num_configs
        self.phi_seed = phi_seed
        
        # Configure tensor product structure
        self.phi_subsystem_dims = self._setup_subsystem_dimensions(
            phi_subsystem_dims_override
        )
        
        # Initialize storage for configurations
        self.g_types_jax: Optional[jnp.ndarray] = None
        self.phi_vectors_jax: Optional[jnp.ndarray] = None
        
        # Generate initial configurations
        self._sample_configurations()
    
    def _validate_basic_parameters(self, dimension: int, num_configs: int) -> None:
        """
        Validate basic initialization parameters.
        
        Args:
            dimension: Field vector dimension
            num_configs: Number of configurations
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
            
        if not isinstance(num_configs, int) or num_configs <= 0:
            raise ValueError("Number of configurations must be a positive integer")
    
    def _setup_subsystem_dimensions(self, 
                                  phi_subsystem_dims_override: Optional[List[Union[int, List[int]]]]) -> List[List[int]]:
        """
        Configure tensor product dimensions for field configurations.
        
        Args:
            phi_subsystem_dims_override: Subsystem dimension specification
            
        Returns:
            Properly formatted subsystem dimensions [[d1,d2,...], [1,1,...]]
            
        Raises:
            ValueError: If subsystem dimensions are invalid
        """
        if phi_subsystem_dims_override is None:
            raise ValueError(
                "phi_subsystem_dims_override is required for quantum compatibility. "
                "Specify as [d1, d2, ...] or [[d1, d2, ...], [1, 1, ...]]"
            )
        
        # Handle different input formats
        if isinstance(phi_subsystem_dims_override[0], int):
            # Format: [d1, d2, ...]
            dims_list = phi_subsystem_dims_override
            subsystem_dims = [dims_list, [1] * len(dims_list)]
            
        elif (isinstance(phi_subsystem_dims_override[0], list) and 
              len(phi_subsystem_dims_override) == 2):
            # Format: [[d1, d2, ...], [1, 1, ...]]
            subsystem_dims = phi_subsystem_dims_override
            
            # Validate structure
            if not self._validate_subsystem_structure(subsystem_dims):
                raise ValueError(
                    f"Invalid subsystem structure: {subsystem_dims}. "
                    "Expected [[d1, d2, ...], [1, 1, ...]]"
                )
        else:
            raise ValueError(
                f"Invalid format for phi_subsystem_dims_override: {phi_subsystem_dims_override}"
            )
        
        # Validate dimension consistency
        if np.prod(subsystem_dims[0]) != self.dimension:
            raise ValueError(
                f"Product of subsystem dimensions {subsystem_dims[0]} "
                f"({np.prod(subsystem_dims[0])}) must equal total dimension {self.dimension}"
            )
        
        return subsystem_dims
    
    def _validate_subsystem_structure(self, subsystem_dims: List[List[int]]) -> bool:
        """
        Validate subsystem dimension structure.
        
        Args:
            subsystem_dims: Subsystem dimensions to validate
            
        Returns:
            True if structure is valid
        """
        if len(subsystem_dims) != 2:
            return False
            
        dims_list, ones_list = subsystem_dims
        
        # Check that all elements are positive integers
        if not all(isinstance(d, int) and d > 0 for d in dims_list):
            return False
            
        # Check that second list contains only ones
        if not all(d == 1 for d in ones_list):
            return False
            
        # Check that lists have same length
        if len(dims_list) != len(ones_list):
            return False
            
        return True
    
    def _sample_configurations(self, seed_offset: int = 0) -> None:
        """
        Sample field configurations using quantum-appropriate random generation.
        
        Args:
            seed_offset: Offset for random seed to enable resampling
        """
        # Set random seed for reproducibility
        if self.phi_seed is not None:
            np.random.seed(self.phi_seed + seed_offset)
        
        # Generate symmetry types (0=identity, 1=permutation)
        g_types_np = np.random.randint(0, 2, size=self.num_configs)
        self.g_types_jax = jnp.array(g_types_np)
        
        # Generate complex field vectors
        phi_vectors = self._generate_field_vectors()
        self.phi_vectors_jax = jnp.array(phi_vectors)
    
    def _generate_field_vectors(self) -> np.ndarray:
        """
        Generate normalized complex field configuration vectors.
        
        Returns:
            Array of normalized field vectors
        """
        # Generate random complex vectors
        real_parts = np.random.randn(self.num_configs, self.dimension)
        imag_parts = np.random.randn(self.num_configs, self.dimension)
        phi_vectors = real_parts + 1j * imag_parts
        
        # Normalize each vector
        norms = np.linalg.norm(phi_vectors, axis=1, keepdims=True)
        
        # Handle zero-norm vectors (extremely unlikely but mathematically possible)
        zero_norm_mask = (norms.flatten() < 1e-12)
        if np.any(zero_norm_mask):
            # Replace zero-norm vectors with random unit vectors
            num_zeros = np.sum(zero_norm_mask)
            replacement_vectors = np.random.randn(num_zeros, self.dimension) + \
                                1j * np.random.randn(num_zeros, self.dimension)
            replacement_norms = np.linalg.norm(replacement_vectors, axis=1, keepdims=True)
            phi_vectors[zero_norm_mask] = replacement_vectors / replacement_norms
            norms[zero_norm_mask] = 1.0
        
        # Normalize all vectors
        phi_vectors_normalized = phi_vectors / norms
        
        return phi_vectors_normalized
    
    def get_jax_configurations(self) -> Dict[str, Union[jnp.ndarray, int, List[List[int]]]]:
        """
        Get batched JAX configurations for efficient computation.
        
        Returns:
            Dictionary containing:
            - g_types_jax: Array of symmetry types
            - phi_vectors_jax: Array of field configuration vectors
            - phi_dimension: Field vector dimension
            - phi_subsystem_dims: Tensor product structure
        """
        if self.g_types_jax is None or self.phi_vectors_jax is None:
            raise RuntimeError("Configurations not yet generated. Call _sample_configurations() first.")
        
        return {
            'g_types_jax': self.g_types_jax,
            'phi_vectors_jax': self.phi_vectors_jax,
            'phi_dimension': self.dimension,
            'phi_subsystem_dims': self.phi_subsystem_dims
        }
    
    def get_qutip_configurations(self) -> List[Tuple[int, Qobj]]:
        """
        Get configurations as QuTiP objects for compatibility.
        
        Returns:
            List of (g_type, phi_qobj) tuples
        """
        if self.g_types_jax is None or self.phi_vectors_jax is None:
            raise RuntimeError("Configurations not yet generated.")
        
        configurations = []
        g_types_np = np.array(self.g_types_jax)
        phi_vectors_np = np.array(self.phi_vectors_jax)
        
        for i in range(self.num_configs):
            g_type = int(g_types_np[i])
            phi_vector = phi_vectors_np[i]
            phi_qobj = Qobj(phi_vector, dims=self.phi_subsystem_dims)
            configurations.append((g_type, phi_qobj))
        
        return configurations
    
    def resample_configurations(self, 
                              num_configs: Optional[int] = None,
                              phi_seed: Optional[int] = None,
                              phi_subsystem_dims_override: Optional[List[Union[int, List[int]]]] = None) -> None:
        """
        Resample field configurations with optional parameter updates.
        
        Args:
            num_configs: New number of configurations (optional)
            phi_seed: New random seed (optional)
            phi_subsystem_dims_override: New subsystem structure (optional)
        """
        # Update parameters if provided
        if num_configs is not None:
            if not isinstance(num_configs, int) or num_configs <= 0:
                raise ValueError("num_configs must be a positive integer")
            self.num_configs = num_configs
        
        if phi_seed is not None:
            self.phi_seed = phi_seed
        elif self.phi_seed is not None:
            # Increment seed for different sampling
            self.phi_seed += 1
        
        if phi_subsystem_dims_override is not None:
            self.phi_subsystem_dims = self._setup_subsystem_dimensions(
                phi_subsystem_dims_override
            )
        
        # Resample configurations
        self._sample_configurations()
    
    def get_statistics(self) -> Dict[str, Union[float, int, Dict[int, int]]]:
        """
        Get statistical information about the field configurations.
        
        Returns:
            Dictionary with configuration statistics
        """
        if self.g_types_jax is None or self.phi_vectors_jax is None:
            raise RuntimeError("Configurations not yet generated.")
        
        g_types_np = np.array(self.g_types_jax)
        phi_vectors_np = np.array(self.phi_vectors_jax)
        
        # Compute statistics
        g_type_counts = {
            0: int(np.sum(g_types_np == 0)),
            1: int(np.sum(g_types_np == 1))
        }
        
        vector_norms = np.linalg.norm(phi_vectors_np, axis=1)
        
        return {
            'total_configs': self.num_configs,
            'dimension': self.dimension,
            'g_type_distribution': g_type_counts,
            'mean_vector_norm': float(np.mean(vector_norms)),
            'std_vector_norm': float(np.std(vector_norms)),
            'min_vector_norm': float(np.min(vector_norms)),
            'max_vector_norm': float(np.max(vector_norms))
        }
    
    def validate_configurations(self) -> bool:
        """
        Validate the integrity of generated configurations.
        
        Returns:
            True if all configurations are valid
            
        Raises:
            ValueError: If configurations are invalid
        """
        if self.g_types_jax is None or self.phi_vectors_jax is None:
            raise RuntimeError("Configurations not yet generated.")
        
        g_types_np = np.array(self.g_types_jax)
        phi_vectors_np = np.array(self.phi_vectors_jax)
        
        # Check g_types are valid
        if not np.all(np.isin(g_types_np, [0, 1])):
            raise ValueError("Invalid g_types found. Must be 0 or 1.")
        
        # Check vector normalization
        norms = np.linalg.norm(phi_vectors_np, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-10):
            raise ValueError("Field vectors are not properly normalized.")
        
        # Check dimensions
        if phi_vectors_np.shape != (self.num_configs, self.dimension):
            raise ValueError(f"Unexpected phi_vectors shape: {phi_vectors_np.shape}")
        
        if g_types_np.shape != (self.num_configs,):
            raise ValueError(f"Unexpected g_types shape: {g_types_np.shape}")
        
        return True
    
    def __repr__(self) -> str:
        """String representation of field configuration space."""
        return (
            f"FieldConfigurationSpace(dimension={self.dimension}, "
            f"num_configs={self.num_configs}, "
            f"subsystem_dims={self.phi_subsystem_dims[0]})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.g_types_jax is not None:
            stats = self.get_statistics()
            g_dist = stats['g_type_distribution']
            return (
                f"Field Configuration Space M\n"
                f"Dimension: {self.dimension}\n"
                f"Total configurations: {self.num_configs}\n"
                f"Subsystem structure: {self.phi_subsystem_dims[0]}\n"
                f"Symmetry distribution: g=0: {g_dist[0]}, g=1: {g_dist[1]}\n"
                f"Vector norm: mean={stats['mean_vector_norm']:.6f}, "
                f"std={stats['std_vector_norm']:.6f}"
            )
        else:
            return (
                f"Field Configuration Space M (not yet sampled)\n"
                f"Dimension: {self.dimension}\n"
                f"Planned configurations: {self.num_configs}\n"
                f"Subsystem structure: {self.phi_subsystem_dims[0]}"
            )


def create_consciousness_field_space(subsystem_dims: List[int],
                                   num_configs: int = 100,
                                   seed: Optional[int] = None) -> FieldConfigurationSpace:
    """
    Factory function to create field space optimized for consciousness studies.
    
    Args:
        subsystem_dims: Dimensions of consciousness subsystems [d1, d2, ...]
        num_configs: Number of field configurations to generate
        seed: Random seed for reproducible generation
        
    Returns:
        Configured FieldConfigurationSpace for consciousness research
    """
    total_dimension = int(np.prod(subsystem_dims))
    
    return FieldConfigurationSpace(
        dimension=total_dimension,
        num_configs=num_configs,
        phi_seed=seed,
        phi_subsystem_dims_override=subsystem_dims
    )


# Example usage and testing
if __name__ == '__main__':
    print("🌊 Field Configuration Space - Consciousness Framework")
    print("=" * 60)
    
    # Create field space for consciousness analysis
    subsystem_dims = [2, 2]  # Two-qubit consciousness subsystem
    field_dimension = int(np.prod(subsystem_dims))
    num_configurations = 50
    
    print(f"Creating field space:")
    print(f"Dimension: {field_dimension}")
    print(f"Subsystem structure: {subsystem_dims}")
    print(f"Number of configurations: {num_configurations}")
    print()
    
    # Initialize field configuration space
    field_space = create_consciousness_field_space(
        subsystem_dims=subsystem_dims,
        num_configs=num_configurations,
        seed=42
    )
    
    print("Field Configuration Space:")
    print(field_space)
    print()
    
    # Get JAX configurations for computation
    jax_configs = field_space.get_jax_configurations()
    print("JAX Configuration Details:")
    print(f"g_types shape: {jax_configs['g_types_jax'].shape}")
    print(f"phi_vectors shape: {jax_configs['phi_vectors_jax'].shape}")
    print(f"phi_dimension: {jax_configs['phi_dimension']}")
    print(f"phi_subsystem_dims: {jax_configs['phi_subsystem_dims']}")
    print()
    
    # Validate configurations
    print("Validating configurations...")
    field_space.validate_configurations()
    print("✓ All configurations valid")
    print()
    
    # Get statistics
    stats = field_space.get_statistics()
    print("Configuration Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test resampling
    print("Testing configuration resampling...")
    field_space.resample_configurations(num_configs=25, phi_seed=101)
    new_stats = field_space.get_statistics()
    print(f"After resampling: {new_stats['total_configs']} configurations")
    print()
    
    # Test QuTiP compatibility
    print("Testing QuTiP compatibility...")
    qutip_configs = field_space.get_qutip_configurations()
    g_type, phi_qobj = qutip_configs[0]
    print(f"First configuration: g_type={g_type}, phi_qobj.dims={phi_qobj.dims}")
    print(f"phi_qobj norm: {phi_qobj.norm():.6f}")
    print()
    
    print("✓ Field configuration space validation complete") 