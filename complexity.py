import jax
import jax.numpy as jnp
import numpy as np
import qutip # For Qobj type hinting and potential operations
# from functools import partial # Not strictly needed if using @jax.jit directly
from qutip import Qobj

# JAX setup for CPU, as default, to ensure consistency if GPU is available but not intended for use here.
jax.config.update('jax_platform_name', 'cpu')

_jax_operator_cache = {} # Global cache for JAX S[g] operator matrices

# --- Define Symmetry Operators (as JAX matrices, derived from QuTiP Qobjs) ---
def get_jax_symmetry_operator(g_type: int, dimension: int, phi_subsystem_dims: list[list[int]]) -> jnp.ndarray:
    """
    Retrieves or creates a JAX matrix representation of a symmetry operator S[g].
    Internally, it may use QuTiP to define the operator structure first.
    The `phi_subsystem_dims` are crucial for defining the operator's own dimensions correctly.
    
    Args:
        g_type (int): The type of symmetry operator (0 for identity, 1 for a permutation).
        dimension (int): The dimension of the Hilbert space phi_qobj lives in.
        phi_subsystem_dims (list[list[int]]): The `dims` attribute of the phi_qobj 
                                               (e.g., [[d1, d2], [1, 1]] for a ket).
                                               S[g] will have dims [phi_subsystem_dims[0], phi_subsystem_dims[0]].
    Returns:
        jnp.ndarray: The JAX matrix representation of the symmetry operator S[g].
    """
    # Convert list of lists to tuple of tuples for cache key to ensure hashability
    phi_subsystem_dims_tuple = tuple(tuple(inner_list) for inner_list in phi_subsystem_dims)
    cache_key = (g_type, dimension, phi_subsystem_dims_tuple)
    
    if cache_key in _jax_operator_cache:
        return _jax_operator_cache[cache_key]

    operator_dims = [phi_subsystem_dims[0], phi_subsystem_dims[0]] # S[g] acts on kets, so it's an operator

    if g_type == 0: # Identity operator S[0]
        s_matrix_np = np.eye(dimension, dtype=complex)
    elif g_type == 1: # Permutation operator S[1] (e.g., swaps halves of the vector)
        if dimension % 2 != 0:
            print(f"Warning: Permutation S[1] expects even dimension, got {dimension}. Using identity.")
            s_matrix_np = np.eye(dimension, dtype=complex)
        else:
            s_matrix_np = np.zeros((dimension, dimension), dtype=complex)
            half_d = dimension // 2
            # Example: Swaps first half with second half
            for i in range(half_d):
                s_matrix_np[i, half_d + i] = 1
                s_matrix_np[half_d + i, i] = 1
            # Could also implement other permutations based on g_type if expanded
    else:
        raise ValueError(f"Unknown g_type: {g_type} for symmetry operator S[g]. Expected 0 or 1.")
    
    # Create Qobj temporarily to ensure correct structure if needed, then convert to JAX array.
    s_qobj = Qobj(s_matrix_np, dims=operator_dims)
    s_jax_matrix = jnp.array(s_qobj.full()) # Convert to JAX array
    
    _jax_operator_cache[cache_key] = s_jax_matrix
    return s_jax_matrix

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
                                 batched_field_configs: dict, 
                                 # phi_subsystem_dims_for_S_operators is now part of batched_field_configs
                                 ) -> tuple[float, jnp.ndarray, dict]: # all_T_values will be JAX array initially
    """
    Computes the Universal Complexity U(t) by averaging T[g,φ] values over all configurations.
    Uses batched JAX operations via vmap for efficiency.
    Also returns all individual T values (as JAX array) and T values grouped by g_type.

    Args:
        psi_qobj (Qobj): The universe state Ψ.
        batched_field_configs (dict): A dictionary from FieldConfigurationSpace.get_jax_configurations(),
                                      containing {'g_types_jax', 'phi_vectors_jax', 
                                                 'phi_dimension', 'phi_subsystem_dims'}.
    Returns:
        tuple[float, jnp.ndarray, dict]: 
            - U_val (float): The Universal Complexity (mean of all T values).
            - all_T_values_jax (jnp.ndarray): JAX array of all computed T[g,φ] values.
            - T_values_by_g_type (dict): Dict mapping g_type (int) to a list of Python float T values.
    """
    g_types_jax = batched_field_configs['g_types_jax']
    phi_vectors_jax = batched_field_configs['phi_vectors_jax']
    phi_dimension = batched_field_configs['phi_dimension']
    phi_s_dims = batched_field_configs['phi_subsystem_dims'] # Used for S[g] construction

    num_configs = phi_vectors_jax.shape[0]
    if num_configs == 0:
        return 0.0, jnp.array([]), {0: [], 1: []}

    psi_jax = jnp.array(psi_qobj.full().flatten()) # Convert psi_qobj to JAX array once

    # Get S[0] and S[1] matrices
    s0_jax = get_jax_symmetry_operator(0, phi_dimension, phi_s_dims)
    s1_jax = get_jax_symmetry_operator(1, phi_dimension, phi_s_dims)

    # Construct the batch of S[g] matrices based on g_types_jax
    # g_types_jax is (num_configs,). We need to make it (num_configs, 1, 1) for jnp.where broadcasting with matrices.
    s_g_matrix_batch_jax = jnp.where(
        g_types_jax.reshape(-1, 1, 1) == 0, # Condition, broadcasted
        s0_jax[None, :, :], # If true, use S0 (broadcasted to batch)
        s1_jax[None, :, :]  # If false, use S1 (broadcasted to batch)
    )
    
    # Validate shapes for vmap (already done by previous checks in old loop, good to keep in mind)
    # psi_jax shape: (D,)
    # phi_vectors_jax shape: (M, D)
    # s_g_matrix_batch_jax shape: (M, D, D)
    # where M is num_configs, D is phi_dimension.

    all_T_values_jax = _compute_batch_complexity_vmap(
        psi_jax, phi_vectors_jax, s_g_matrix_batch_jax
    )

    U_val = float(jnp.mean(all_T_values_jax))

    # Group T_values by g_type (convert JAX array to Python floats for dict)
    T_values_by_g_type = {0: [], 1: []}
    g_types_np = np.array(g_types_jax) # Convert g_types to NumPy array for easier iteration/masking
    all_T_values_np = np.array(all_T_values_jax)
    
    for i in range(num_configs):
        g = int(g_types_np[i])
        t_val = float(all_T_values_np[i])
        if g in T_values_by_g_type:
            T_values_by_g_type[g].append(t_val)
        # else: # Should not happen if g_types are only 0 or 1
        #     T_values_by_g_type[g] = [t_val]

    return U_val, all_T_values_jax, T_values_by_g_type

# Example Usage:
if __name__ == '__main__':
    from universe import UniverseState
    from fields import FieldConfigurationSpace # Uses the updated FieldConfigurationSpace

    s1_d, s2_d = 2, 2 
    phi_actual_subsystems = [s1_d, s2_d]
    phi_qobj_dims_struct = [phi_actual_subsystems, [1]*len(phi_actual_subsystems)]
    phi_dim_calc = np.prod(phi_actual_subsystems)

    psi_dim_calc = phi_dim_calc 
    psi_qobj_dims_struct = phi_qobj_dims_struct

    print(f"Setting up example: Ψ and φ in {psi_dim_calc}-dim space.")

    uni = UniverseState(dimension=psi_dim_calc, initial_state_seed=1, subsystem_dims=psi_qobj_dims_struct)
    psi_qobj_state = uni.get_state()

    field_space_inst = FieldConfigurationSpace(dimension=phi_dim_calc, 
                                               num_configs=500, # Test with more configs
                                               phi_seed=42,
                                               phi_subsystem_dims_override=psi_qobj_dims_struct)
    
    batched_configs = field_space_inst.get_jax_configurations()
    
    print(f"Prepared batched JAX configurations for {batched_configs['phi_vectors_jax'].shape[0]} samples.")

    U, all_T_jax, T_by_g = compute_universal_complexity_U(psi_qobj_state, batched_configs)

    print(f"\nUniversal Complexity U(t): {U:.4f}")
    # print(f"All T[g,φ] values (JAX array sample): {all_T_jax[:5]}") # Print first 5 JAX values
    print(f"Number of T values: {all_T_jax.shape[0]}")
    print("T values by g_type:")
    for g_key, t_vals_list in T_by_g.items():
        print(f"  g={g_key} (count {len(t_vals_list)}): example first 5: {[round(t, 4) for t in t_vals_list[:5]]}")
    
    # ... (rest of example can be adapted if needed) ...

