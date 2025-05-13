import numpy as np
import jax.numpy as jnp
from qutip import Qobj

# JAX setup for CPU, as default
import jax
jax.config.update('jax_platform_name', 'cpu')

class FieldConfigurationSpace:
    """
    Defines a space of field configurations (M) using QuTiP Qobjs and JAX arrays.
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
            # Default dims for a ket state: [[dimension], [1]]
            self.phi_subsystem_dims = [[dimension], [1]]
            print(f"""Warning: phi_subsystem_dims_override not provided to FieldConfigurationSpace. 
                  Defaulting phi_qobj dims to {self.phi_subsystem_dims}. 
                  Ensure this is compatible with UniverseState and T-operator expectations.""")
        else:
            # Ensure the override is formatted correctly for a ket.
            # It might come in as [d1, d2, ...] or already as [[d1, d2, ...], [1, 1, ...]]
            if isinstance(phi_subsystem_dims_override, list) and \
               len(phi_subsystem_dims_override) > 0:
                if isinstance(phi_subsystem_dims_override[0], int):
                    # Input is like [d1, d2, ...], convert to [[d1, d2, ...], [1, 1, ...]]
                    actual_dims_list = phi_subsystem_dims_override
                    self.phi_subsystem_dims = [actual_dims_list, [1] * len(actual_dims_list)]
                elif isinstance(phi_subsystem_dims_override[0], list) and \
                     len(phi_subsystem_dims_override) == 2 and \
                     isinstance(phi_subsystem_dims_override[1], list) and \
                     all(isinstance(d, int) for d in phi_subsystem_dims_override[0]) and \
                     all(d == 1 for d in phi_subsystem_dims_override[1]) and \
                     len(phi_subsystem_dims_override[0]) == len(phi_subsystem_dims_override[1]):
                    # Input is already correctly formatted like [[d1, d2], [1, 1]]
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

            # Validate that product of dims matches the overall dimension
            if np.prod(self.phi_subsystem_dims[0]) != self.dimension:
                raise ValueError(
                    f"Product of subsystem dimensions {self.phi_subsystem_dims[0]} ({np.prod(self.phi_subsystem_dims[0])}) "
                    f"must equal the total dimension {self.dimension}."
                )
        
        self.configurations = []
        self._sample_configurations(self.num_configs, self.phi_seed)

    def _generate_single_config(self) -> dict:
        """
        Generates a single configuration {g_type, phi_jax, phi_qobj}.
        g_type is sampled as 0 or 1.
        phi is a normalized random complex vector.
        """
        g_type = np.random.randint(0, 2) # g_type is 0 or 1, mapping to S[0] or S[1] in complexity.py
        
        # Generate complex vector components for phi
        phi_real = np.random.randn(self.dimension)
        phi_imag = np.random.randn(self.dimension)
        phi_numpy_unnormalized = phi_real + 1j * phi_imag
        norm = np.linalg.norm(phi_numpy_unnormalized)
        phi_numpy = phi_numpy_unnormalized / norm if norm != 0 else phi_numpy_unnormalized

        phi_jax = jnp.array(phi_numpy)
        phi_qobj = Qobj(phi_numpy, dims=self.phi_subsystem_dims) # Removed type='ket'
        
        return {'g_type': g_type, 'phi_jax': phi_jax, 'phi_qobj': phi_qobj}

    def _sample_configurations(self, num_samples: int, seed_offset: int = 0):
        """Samples 'num_samples' configurations and stores them."""
        if self.phi_seed is not None:
            np.random.seed(self.phi_seed + (seed_offset if seed_offset is not None else 0))
        
        self.configurations = [self._generate_single_config() for _ in range(num_samples)]

    def get_configurations(self) -> list[dict]:
        """Returns the list of all sampled configurations."""
        return self.configurations

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
        if phi_seed is not None: # Allow explicitly setting a new seed or removing it
            self.phi_seed = phi_seed 
        elif phi_seed is None and self.phi_seed is not None: # If new seed is None, but old one was set, increment old one if not explicitly None
             self.phi_seed += 1 # Increment seed to get new samples if not overridden

        if phi_subsystem_dims_override is not None:
            self.phi_subsystem_dims = phi_subsystem_dims_override
        
        print(f"Resampling Field Configurations: num_configs={self.num_configs}, new_seed_base={self.phi_seed}")
        self._sample_configurations(self.num_configs, seed_offset=0) # Seed already incorporates changes

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
    
    configurations = field_space.get_configurations()
    print(f"\nGenerated {len(configurations)} configurations.")
    for i, config in enumerate(configurations[:3]): # Print first 3
        print(f"Config {i}: g_type = {config['g_type']}, "
              f"phi_qobj norm = {config['phi_qobj'].norm():.4f}, "
              f"phi_qobj dims = {config['phi_qobj'].dims}, "
              f"phi_jax shape = {config['phi_jax'].shape}")

    field_space.resample_configurations(num_configs=5, phi_seed=43) # Seed will be 43
    configurations_resampled = field_space.get_configurations()
    print(f"\nResampled to {len(configurations_resampled)} configurations.")
    for i, config in enumerate(configurations_resampled[:2]): # Print first 2 of resampled
        print(f"Config {i}: g_type = {config['g_type']}, phi_qobj norm = {config['phi_qobj'].norm():.4f}") 