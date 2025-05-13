import numpy as np
import jax.numpy as jnp
from qutip import Qobj

# JAX setup for CPU, as default
import jax
jax.config.update('jax_platform_name', 'cpu')

class FieldConfigurationSpace:
    """
    Defines a space of field configurations (M) using JAX arrays primarily.
    Each configuration consists of a type 'g_type' and a state vector 'phi'.
    
    'g_type' is currently an integer (0 or 1) serving as a placeholder for richer 
    algebraic/symmetry properties of fields as envisioned in mission.txt. 
    It directly maps to the two distinct symmetry operator types (e.g., identity or 
    a permutation) handled in the JAX-based complexity.py module.
    
    The measure 'μ' over M is implicitly uniform due to random sampling of phi states.
    Future extensions (e.g., with NumPyro) may introduce more structured sampling 
    for g_type and phi, and allow for non-uniform measures.
    """
    def __init__(self, dimension: int, num_configs: int, phi_seed: int = None, 
                 phi_subsystem_dims_override: list[list[int]] = None):
        """
        Initializes the configuration space by sampling phi vectors.

        Args:
            dimension (int): The dimensionality of the phi vectors.
            num_configs (int): The number of (g_type, phi) configurations to generate.
            phi_seed (int, optional): Seed for NumPy's random number generator for phi sampling.
            phi_subsystem_dims_override (list[list[int]], optional): 
                QuTiP `dims` to be used for the `phi_qobj` objects. This is crucial for ensuring
                compatibility with UniverseState and symmetry operators in complexity.py.
                Example: [[dim1, dim2], [1, 1]] for a ket state.
        """
        self.dimension = dimension
        self.num_configs = num_configs
        self.phi_seed = phi_seed

        if phi_subsystem_dims_override is None:
            raise ValueError(
                "phi_subsystem_dims_override must be provided to FieldConfigurationSpace. "
                "It is crucial for ensuring compatibility with UniverseState and symmetry operators. "
                "Example: [[dim1, dim2], [1, 1]] for a ket state."
            )
        else:
            if isinstance(phi_subsystem_dims_override, list) and \
               len(phi_subsystem_dims_override) > 0:
                if isinstance(phi_subsystem_dims_override[0], int):
                    actual_dims_list = phi_subsystem_dims_override
                    self.phi_subsystem_dims = [actual_dims_list, [1] * len(actual_dims_list)]
                elif isinstance(phi_subsystem_dims_override[0], list) and \
                     len(phi_subsystem_dims_override) == 2 and \
                     isinstance(phi_subsystem_dims_override[1], list) and \
                     all(isinstance(d, int) for d in phi_subsystem_dims_override[0]) and \
                     all(d == 1 for d in phi_subsystem_dims_override[1]) and \
                     len(phi_subsystem_dims_override[0]) == len(phi_subsystem_dims_override[1]):
                    self.phi_subsystem_dims = phi_subsystem_dims_override
                else:
                    raise ValueError(
                        f"Invalid format for phi_subsystem_dims_override: {phi_subsystem_dims_override}. "
                        f"Expected [d1, d2, ...] or [[d1, d2, ...], [1, 1, ...]]."
                    )
            else:
                raise ValueError(
                    f"phi_subsystem_dims_override must be a list, got: {phi_subsystem_dims_override}"
                )

            if np.prod(self.phi_subsystem_dims[0]) != self.dimension:
                raise ValueError(
                    f"Product of subsystem dimensions {self.phi_subsystem_dims[0]} ({np.prod(self.phi_subsystem_dims[0])}) "
                    f"must equal the total dimension {self.dimension}."
                )
        
        # Batched JAX arrays
        self.g_types_jax: jnp.ndarray = None
        self.phi_vectors_jax: jnp.ndarray = None
        
        # Optional list of Qobj for compatibility or non-JAX parts (if any)
        # self.phi_qobj_list: list[Qobj] = None 

        self._sample_configurations(self.num_configs, self.phi_seed)

    def _sample_configurations(self, num_samples: int, seed_offset: int = 0):
        """Samples 'num_samples' configurations and stores them as batched JAX arrays."""
        if self.phi_seed is not None:
            # Use a JAX key for JAX random functions if we switch to them
            # For NumPy, seed normally.
            np.random.seed(self.phi_seed + (seed_offset if seed_offset is not None else 0))
        
        # Generate g_types (0 or 1)
        g_types_np = np.random.randint(0, 2, size=num_samples)
        self.g_types_jax = jnp.array(g_types_np)
        
        # Generate complex vectors for phi
        phi_real_np = np.random.randn(num_samples, self.dimension)
        phi_imag_np = np.random.randn(num_samples, self.dimension)
        phi_numpy_unnormalized = phi_real_np + 1j * phi_imag_np
        
        # Normalize each vector (row-wise)
        norms = np.linalg.norm(phi_numpy_unnormalized, axis=1, keepdims=True)
        # Avoid division by zero for zero-norm vectors (though unlikely with randn)
        norms[norms == 0] = 1.0 
        phi_numpy_normalized = phi_numpy_unnormalized / norms
        
        self.phi_vectors_jax = jnp.array(phi_numpy_normalized)

        # If Qobj list is needed:
        # self.phi_qobj_list = [Qobj(phi_numpy_normalized[i], dims=self.phi_subsystem_dims) for i in range(num_samples)]

    def get_jax_configurations(self) -> dict:
        """Returns the batched JAX configurations and necessary dimension info."""
        return {
            'g_types_jax': self.g_types_jax, 
            'phi_vectors_jax': self.phi_vectors_jax,
            'phi_dimension': self.dimension,
            'phi_subsystem_dims': self.phi_subsystem_dims # For S_g operator construction
        }

    def resample_configurations(self, num_configs: int = None, phi_seed: int = None, 
                                phi_subsystem_dims_override: list[list[int]] = None):
        """
        Re-samples the field configurations. Optionally updates number of configs and seed.

        Args:
            num_configs (int, optional): New number of configurations. Defaults to existing.
            phi_seed (int, optional): New seed for phi sampling. Defaults to existing or no seed.
            phi_subsystem_dims_override (list[list[int]], optional): 
                New QuTiP `dims` for phi_qobj objects. Defaults to previously set dims.
        """
        if num_configs is not None:
            self.num_configs = num_configs
        if phi_seed is not None: 
            self.phi_seed = phi_seed 
        elif phi_seed is None and self.phi_seed is not None:
             self.phi_seed += 1

        if phi_subsystem_dims_override is not None:
            # Re-validate if dims override changes
            if isinstance(phi_subsystem_dims_override, list) and \
               len(phi_subsystem_dims_override) > 0:
                if isinstance(phi_subsystem_dims_override[0], int):
                    actual_dims_list = phi_subsystem_dims_override
                    self.phi_subsystem_dims = [actual_dims_list, [1] * len(actual_dims_list)]
                elif isinstance(phi_subsystem_dims_override[0], list) and \
                     len(phi_subsystem_dims_override) == 2 and \
                     isinstance(phi_subsystem_dims_override[1], list) and \
                     all(isinstance(d, int) for d in phi_subsystem_dims_override[0]) and \
                     all(d == 1 for d in phi_subsystem_dims_override[1]) and \
                     len(phi_subsystem_dims_override[0]) == len(phi_subsystem_dims_override[1]):
                    self.phi_subsystem_dims = phi_subsystem_dims_override
                else:
                    raise ValueError(f"Invalid format for resample phi_subsystem_dims_override: {phi_subsystem_dims_override}.")
            else:
                raise ValueError(f"Resample phi_subsystem_dims_override must be a list, got: {phi_subsystem_dims_override}")

            if np.prod(self.phi_subsystem_dims[0]) != self.dimension:
                 raise ValueError(
                    f"Product of subsystem dimensions {self.phi_subsystem_dims[0]} for resample "
                    f"must equal the total dimension {self.dimension}."
                )
        
        print(f"Resampling Field Configurations: num_configs={self.num_configs}, new_seed_base={self.phi_seed}")
        self._sample_configurations(self.num_configs, seed_offset=0)

# Example usage:
if __name__ == '__main__':
    field_dim = 16
    n_configs = 10
    
    # Define QuTiP dims for the phi_qobj kets. Example: a 16-dim ket, could be [[16],[1]] or [[4,4],[1,1]] etc.
    # This needs to be compatible with how S[g] operators and Psi are structured in complexity.py and universe.py
    example_phi_qobj_dims = [[field_dim], [1]] # Simplest case: a single system ket

    print(f"Initializing FieldConfigurationSpace with {n_configs} configs of dimension {field_dim}.")
    print(f"phi_qobj instances will have dims: {example_phi_qobj_dims}")

    field_space = FieldConfigurationSpace(dimension=field_dim, 
                                          num_configs=n_configs, 
                                          phi_seed=42,
                                          phi_subsystem_dims_override=example_phi_qobj_dims)
    
    jax_configs = field_space.get_jax_configurations()
    print(f"\nGenerated JAX configurations:")
    print(f"  g_types_jax shape: {jax_configs['g_types_jax'].shape}")
    print(f"  phi_vectors_jax shape: {jax_configs['phi_vectors_jax'].shape}")
    print(f"  phi_dimension: {jax_configs['phi_dimension']}")
    print(f"  phi_subsystem_dims: {jax_configs['phi_subsystem_dims']}")
    print(f"  Example g_type[0]: {jax_configs['g_types_jax'][0]}")
    print(f"  Example phi_vector[0] norm: {jnp.linalg.norm(jax_configs['phi_vectors_jax'][0]):.4f}")

    field_space.resample_configurations(num_configs=5, phi_seed=43)
    jax_configs_resampled = field_space.get_jax_configurations()
    print(f"\nResampled to {jax_configs_resampled['g_types_jax'].shape[0]} configurations.")
    print(f"  Example g_type[0] after resample: {jax_configs_resampled['g_types_jax'][0]}") 