import jax
import jax.numpy as jnp
import numpy as np
import qutip # For Qobj type hinting and potential operations
# from functools import partial # Not strictly needed if using @jax.jit directly
from qutip import Qobj

# JAX setup for CPU, as default, to ensure consistency if GPU is available but not intended for use here.
jax.config.update('jax_platform_name', 'cpu')

_qutip_operator_cache = {} # Global cache for QuTiP S[g] operators

# --- Define Symmetry Operators (as QuTiP Qobjs, converted to JAX matrices later) ---
def get_qutip_symmetry_operator(g_type: int, dimension: int, phi_subsystem_dims: list[list[int]]) -> Qobj:
    """
    Retrieves or creates a QuTiP symmetry operator S[g] based on g_type.
    These operators act on phi_qobj vectors.
    The `phi_subsystem_dims` are crucial for defining the operator's own dimensions correctly.
    
    Args:
        g_type (int): The type of symmetry operator (0 for identity, 1 for a permutation).
        dimension (int): The dimension of the Hilbert space phi_qobj lives in.
        phi_subsystem_dims (list[list[int]]): The `dims` attribute of the phi_qobj 
                                               (e.g., [[d1, d2], [1, 1]] for a ket).
                                               S[g] will have dims [phi_subsystem_dims[0], phi_subsystem_dims[0]].
    Returns:
        Qobj: The QuTiP symmetry operator S[g].
    """
    cache_key = (g_type, dimension, tuple(map(tuple, phi_subsystem_dims)))
    if cache_key in _qutip_operator_cache:
        return _qutip_operator_cache[cache_key]

    operator_dims = [phi_subsystem_dims[0], phi_subsystem_dims[0]] # S[g] acts on kets, so it's an operator

    if g_type == 0: # Identity operator S[0]
        s_matrix = np.eye(dimension, dtype=complex)
    elif g_type == 1: # Permutation operator S[1] (e.g., swaps halves of the vector)
        if dimension % 2 != 0:
            print(f"Warning: Permutation S[1] expects even dimension, got {dimension}. Using identity.")
            s_matrix = np.eye(dimension, dtype=complex)
        else:
            s_matrix = np.zeros((dimension, dimension), dtype=complex)
            half_d = dimension // 2
            # Example: Swaps first half with second half
            for i in range(half_d):
                s_matrix[i, half_d + i] = 1
                s_matrix[half_d + i, i] = 1
            # Could also implement other permutations based on g_type if expanded
    else:
        raise ValueError(f"Unknown g_type: {g_type} for symmetry operator S[g]. Expected 0 or 1.")
    
    s_qobj = Qobj(s_matrix, dims=operator_dims)
    _qutip_operator_cache[cache_key] = s_qobj
    return s_qobj

# --- JAX-based Complexity Computation ---

@jax.jit
def _compute_single_complexity_value_jax(psi_vector_jax: jnp.ndarray, 
                                         phi_vector_jax: jnp.ndarray, 
                                         s_g_matrix_jax: jnp.ndarray) -> float:
    """
    Computes T[g,φ] = |<Ψ|S[g]φ>|² for a single Ψ, φ, and S[g].
    All inputs are JAX arrays. S[g] is its JAX matrix representation.
    Assumes psi_vector_jax and (S[g] @ phi_vector_jax) are in the same space.
    """
    # Ensure phi_vector_jax is a column vector for matmul: S[g] @ φ
    effective_phi_vector_jax = s_g_matrix_jax @ phi_vector_jax.reshape(-1, 1)
    effective_phi_vector_jax = effective_phi_vector_jax.flatten()

    # Inner product: <Ψ|effective_φ> = Ψ† ⋅ effective_φ
    projection_scalar_jax = jnp.vdot(psi_vector_jax, effective_phi_vector_jax)
    complexity_val = jnp.abs(projection_scalar_jax)**2
    return complexity_val

# Vectorize over configurations (phi_vector_jax and s_g_matrix_jax)
_compute_batch_complexity_vmap = jax.vmap(
    _compute_single_complexity_value_jax, in_axes=(None, 0, 0), out_axes=0
) # psi_vector_jax is broadcasted, phi_vectors and s_g_matrices are batched

def compute_universal_complexity_U(psi_qobj: Qobj, 
                                 field_configurations_list: list[dict], 
                                 qutip_ops_cache: dict, # This argument is currently unused, get_qutip_symmetry_operator uses a global _qutip_operator_cache
                                 phi_subsystem_dims_for_S_operators: list[list[int]]) -> tuple[float, list[float], dict]:
    """
    Computes the Universal Complexity U(t) by averaging T[g,φ] values over all configurations.
    Also returns all individual T values and T values grouped by g_type.

    Args:
        psi_qobj (Qobj): The universe state Ψ.
        field_configurations_list (list[dict]): A list of configurations from FieldConfigurationSpace.
                                              Each dict: {'g_type', 'phi_jax', 'phi_qobj'}.
        qutip_ops_cache (dict): Cache that *could* be used by get_qutip_symmetry_operator if passed down.
                                Currently, get_qutip_symmetry_operator uses a global _qutip_operator_cache.
        phi_subsystem_dims_for_S_operators (list[list[int]]): 
            The QuTiP `dims` of the phi_qobj states, passed to S[g] operator generation.

    Returns:
        tuple[float, list[float], dict]: 
            - U_val (float): The Universal Complexity (mean of all T values).
            - all_T_values (list[float]): List of all computed T[g,φ] values.
            - T_values_by_g_type (dict): Dict mapping g_type to a list of T values for that type.
    """
    all_T_values = []
    T_values_by_g_type = {0: [], 1: []} # Initialize for expected g_types

    if not field_configurations_list:
        return 0.0, [], T_values_by_g_type

    psi_jax = jnp.array(psi_qobj.full().flatten()) # Convert psi_qobj to JAX array once

    for config in field_configurations_list:
        g = config['g_type']
        phi_jax_current = config['phi_jax'] # Already a JAX array
        
        # Get the QuTiP S[g] operator. 
        # get_qutip_symmetry_operator uses the global _qutip_operator_cache.
        # It needs the dimension of phi_qobj, not psi_qobj, if they can differ.
        # Assuming phi_qobj from FieldConfigurationSpace has .dims and .shape[0] (dimension)
        phi_qobj_current = config['phi_qobj']
        s_g_qutip_op = get_qutip_symmetry_operator(g, phi_qobj_current.shape[0], phi_qobj_current.dims)
        s_g_matrix_jax = jnp.array(s_g_qutip_op.full()) # Convert S[g] to JAX matrix

        T_val = _compute_single_complexity_value_jax(
            psi_jax, phi_jax_current, s_g_matrix_jax
        )
        
        all_T_values.append(float(T_val)) # Store as Python float
        if g not in T_values_by_g_type:
            T_values_by_g_type[g] = []
        T_values_by_g_type[g].append(float(T_val))
    
    if not all_T_values:
        U_val = 0.0
    else:
        U_val = float(np.mean(np.array(all_T_values))) # Use numpy.mean for list of floats

    return U_val, all_T_values, T_values_by_g_type

# Example Usage:
if __name__ == '__main__':
    from universe import UniverseState
    from fields import FieldConfigurationSpace

    # Define dimensions for phi space (e.g., 2 qubits S1xS2)
    s1_d, s2_d = 2, 2 
    phi_subsystem_dims = [s1_d, s2_d]
    phi_dim = np.prod(phi_subsystem_dims)

    # For this example, Ψ(t) will live in the same space as φ for direct inner product.
    # If Ψ were in a larger space (e.g., S x E), projection would be needed before this specific T calculation.
    psi_dim = phi_dim 
    psi_subsystem_dims = phi_subsystem_dims # Ψ is structured like φ

    print(f"Setting up example: Ψ and φ in {psi_dim}-dim space, structured as {psi_subsystem_dims}.")

    uni = UniverseState(dimension=psi_dim, initial_state_seed=1, subsystem_dims=psi_subsystem_dims)
    psi_qobj_state = uni.get_state()

    g_types = ["identity", "half_swap", "phase_flip_S1"]
    field_space = FieldConfigurationSpace(dimension=phi_dim, 
                                          num_configs=5, 
                                          available_g_types=g_types, 
                                          phi_seed=2)
    configs = field_space.get_configurations()
    op_cache = {}

    U, all_T, T_by_g = compute_universal_complexity_U(psi_qobj_state, configs, op_cache, phi_subsystem_dims)

    print(f"\nUniversal Complexity U(t): {U:.4f}")
    print(f"All T[g,φ] values: {[round(t, 4) for t in all_T]}")
    print("T values by g_type:")
    for g_key, t_vals_list in T_by_g.items():
        print(f"  {g_key}: {[round(t, 4) for t in t_vals_list]}")
    
    print("\nCached QuTiP S[g] operators (Keys):")
    for cache_key_tuple in op_cache.keys():
        print(f"--- Key: {cache_key_tuple} ---")

    # Test with perturbation and resample
    print("\n--- Perturbing state and resampling configurations ---")
    uni.perturb_state(amplitude=0.5, seed=3)
    psi_qobj_state_perturbed = uni.get_state()
    
    field_space.resample_configurations(num_configs=3, phi_seed=4)
    configs_new = field_space.get_configurations()

    U_new, all_T_new, T_by_g_new = compute_universal_complexity_U(psi_qobj_state_perturbed, configs_new, op_cache, phi_subsystem_dims)
    print(f"\nAfter perturbation and resample:")
    print(f"Universal Complexity U(t): {U_new:.4f}")
    print(f"All T[g,φ] values: {[round(t, 4) for t in all_T_new]}")
    print("T values by g_type:")
    for g_key_new, t_vals_list_new in T_by_g_new.items():
        print(f"  {g_key_new}: {[round(t, 4) for t in t_vals_list_new]}")

