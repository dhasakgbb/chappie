# Comprehensive mathematical verification test
import numpy as np
import qutip
import traceback

print('=== COMPREHENSIVE MATHEMATICAL VERIFICATION ===')
print()

# Test 1: Universe State quantum mechanics
print('1. UNIVERSE STATE QUANTUM MECHANICS TEST')
try:
    from universe import UniverseState
    uni = UniverseState(dimension=4, initial_state_seed=42, subsystem_dims=[2, 2])
    psi = uni.get_state()
    
    # Check normalization
    norm = psi.norm()
    print(f'   ✓ State normalization: {norm:.10f} (should be 1.0)')
    assert abs(norm - 1.0) < 1e-10, f'State not normalized: {norm}'
    
    # Check density matrix trace
    rho = uni.get_density_matrix()
    trace = rho.tr()
    print(f'   ✓ Density matrix trace: {trace:.10f} (should be 1.0)')
    assert abs(trace - 1.0) < 1e-10, f'Density matrix trace wrong: {trace}'
    
    # Check partial trace
    rho_sub = uni.get_subsystem_density_matrix(0)
    trace_sub = rho_sub.tr()
    print(f'   ✓ Subsystem trace: {trace_sub:.10f} (should be 1.0)')
    assert abs(trace_sub - 1.0) < 1e-10, f'Subsystem trace wrong: {trace_sub}'
    
    print('   ✅ Universe state quantum mechanics: CORRECT')
    
except Exception as e:
    print(f'   ❌ Universe state test failed: {e}')
    traceback.print_exc()

print()

# Test 2: Complexity operator mathematics
print('2. COMPLEXITY OPERATOR MATHEMATICS TEST')
try:
    from complexity import get_jax_symmetry_operator, _compute_single_complexity_value_jax
    from fields import FieldConfigurationSpace
    import jax.numpy as jnp
    
    # Test symmetry operators
    dimension = 4
    phi_dims = [[2, 2], [1, 1]]
    
    S0 = get_jax_symmetry_operator(0, dimension, phi_dims)  # Identity
    S1 = get_jax_symmetry_operator(1, dimension, phi_dims)  # SWAP for 2-qubit
    
    # Check S0 is identity
    expected_identity = jnp.eye(dimension)
    identity_check = jnp.allclose(S0, expected_identity)
    print(f'   ✓ S[0] is identity: {identity_check}')
    assert identity_check, 'S[0] is not identity matrix'
    
    # Check S1 is proper SWAP gate for 2-qubit system
    expected_swap = jnp.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0], 
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    swap_check = jnp.allclose(S1, expected_swap)
    print(f'   ✓ S[1] is proper SWAP gate: {swap_check}')
    assert swap_check, 'S[1] is not proper SWAP gate'
    
    # Check unitarity
    S0_unitary = jnp.allclose(S0 @ jnp.conj(S0.T), jnp.eye(dimension))
    S1_unitary = jnp.allclose(S1 @ jnp.conj(S1.T), jnp.eye(dimension))
    print(f'   ✓ S[0] unitary: {S0_unitary}')
    print(f'   ✓ S[1] unitary: {S1_unitary}')
    assert S0_unitary and S1_unitary, 'Symmetry operators not unitary'
    
    # Test complexity calculation
    psi_test = jnp.array([1, 0, 0, 0], dtype=complex)  # |00⟩
    phi_test = jnp.array([0, 1, 0, 0], dtype=complex)  # |01⟩
    
    T_val = _compute_single_complexity_value_jax(psi_test, phi_test, S0)
    print(f'   ✓ T[0,φ] value: {T_val:.6f} (should be real, non-negative)')
    assert jnp.isreal(T_val) and T_val >= 0, f'Invalid complexity value: {T_val}'
    
    print('   ✅ Complexity operator mathematics: CORRECT')
    
except Exception as e:
    print(f'   ❌ Complexity operator test failed: {e}')
    traceback.print_exc()

print()

# Test 3: Mutual information calculation
print('3. MUTUAL INFORMATION MATHEMATICS TEST')
try:
    # Create a two-qubit entangled state
    from qutip import bell_state, entropy_vn, ptrace
    
    # Bell state |00⟩ + |11⟩
    bell = bell_state('00') + bell_state('11')
    bell = bell.unit()
    rho_bell = qutip.ket2dm(bell)
    rho_bell.dims = [[2, 2], [2, 2]]
    
    # Calculate mutual information I(A:B) = S(A) + S(B) - S(AB)
    rho_A = rho_bell.ptrace(0)  # First qubit
    rho_B = rho_bell.ptrace(1)  # Second qubit
    
    S_A = entropy_vn(rho_A)
    S_B = entropy_vn(rho_B) 
    S_AB = entropy_vn(rho_bell)
    
    mutual_info = S_A + S_B - S_AB
    
    print(f'   ✓ S(A) = {S_A:.6f}')
    print(f'   ✓ S(B) = {S_B:.6f}')
    print(f'   ✓ S(AB) = {S_AB:.6f}')
    print(f'   ✓ I(A:B) = {mutual_info:.6f} (should be > 0 for entangled state)')
    
    # For maximally entangled Bell state, I(A:B) should be 2 (in bits)
    expected_mutual_info = 2.0  # log2(4) for 2-qubit maximally entangled
    assert mutual_info > 1.5, f'Mutual information too low: {mutual_info}'
    
    print('   ✅ Mutual information mathematics: CORRECT')
    
except Exception as e:
    print(f'   ❌ Mutual information test failed: {e}')
    traceback.print_exc()

print()

# Test 4: PyPhi integration
print('4. PYPHI INTEGRATION TEST')
try:
    import pyphi
    
    # Simple 2-node network
    tpm = [
        [0, 0],  # 00 -> 00
        [0, 1],  # 01 -> 01  
        [1, 0],  # 10 -> 10
        [1, 1]   # 11 -> 11
    ]
    network = pyphi.Network(tpm)
    state = (0, 1)
    
    subsystem = pyphi.Subsystem(network, state, tuple(range(len(state))))
    sia = pyphi.compute.sia(subsystem)
    
    phi_value = float(sia.phi)
    print(f'   ✓ PyPhi Φ calculation: {phi_value:.6f}')
    print(f'   ✓ Number of concepts: {len(sia.ces)}')
    
    assert phi_value >= 0, f'Negative phi value: {phi_value}'
    assert isinstance(sia.ces, (list, tuple)), 'Invalid concept structure'
    
    print('   ✅ PyPhi integration: CORRECT')
    
except Exception as e:
    print(f'   ❌ PyPhi integration test failed: {e}')
    traceback.print_exc()

print()

# Test 5: Category theory limit calculation  
print('5. CATEGORY THEORY LIMIT CALCULATION TEST')
try:
    from category import CategoricalStructure
    
    # Create test configurations
    configs = [
        {'g_type': 0, 'phi': np.random.randn(4)},
        {'g_type': 1, 'phi': np.random.randn(4)},
        {'g_type': 0, 'phi': np.random.randn(4)},
        {'g_type': 1, 'phi': np.random.randn(4)}
    ]
    
    complexity_vals = [0.1, 0.8, 0.2, 0.9]
    
    cat_struct = CategoricalStructure(configs, complexity_vals)
    f_structure = cat_struct.compute_F_structure_enhanced()
    
    # Check limit cone properties
    limit_cone = f_structure['categorical_limit']['limit_cone']
    apex = limit_cone['apex']
    coherence = limit_cone['coherence']
    universality = limit_cone['universality']
    
    print(f'   ✓ Categorical limit apex: {apex:.6f}')
    print(f'   ✓ Limit cone coherence: {coherence:.6f}')
    print(f'   ✓ Universal property measure: {universality:.6f}')
    
    assert 0 <= coherence <= 1, f'Invalid coherence: {coherence}'
    assert 0 <= universality <= 1, f'Invalid universality: {universality}'
    assert apex >= 0, f'Invalid apex: {apex}'
    
    print('   ✅ Category theory limit calculation: CORRECT')
    
except Exception as e:
    print(f'   ❌ Category theory test failed: {e}')
    traceback.print_exc()

print()

# Test 6: Consciousness calculation mathematics
print('6. CONSCIOUSNESS CALCULATION MATHEMATICS TEST')
try:
    from consciousness import ConsciousAgent, Subsystem, IntegratedInformationCalculator
    from universe import UniverseState
    
    # Create a simple universe and conscious agent
    universe = UniverseState(dimension=4, initial_state_seed=42, subsystem_dims=[2, 2])
    agent = ConsciousAgent(universe, [2, 2], 0)
    
    # Test phi calculation
    phi_val = agent.iic.compute_phi(agent.sub)
    print(f'   ✓ Φ calculation: {phi_val:.6f} (should be 0 ≤ Φ ≤ 1)')
    assert 0 <= phi_val <= 1, f'Invalid phi value: {phi_val}'
    
    # Test consciousness level calculation
    agent.perform_deep_introspection(0.5, 1)
    consciousness_level = agent.consciousness_level
    print(f'   ✓ Consciousness level: {consciousness_level:.6f}')
    assert consciousness_level >= 0, f'Invalid consciousness level: {consciousness_level}'
    
    print('   ✅ Consciousness calculation mathematics: CORRECT')
    
except Exception as e:
    print(f'   ❌ Consciousness calculation test failed: {e}')
    traceback.print_exc()

print()

# Test 7: Field configuration mathematics
print('7. FIELD CONFIGURATION MATHEMATICS TEST')
try:
    from fields import FieldConfigurationSpace
    
    # Create field configuration space
    field_space = FieldConfigurationSpace(
        dimension=4,
        num_configs=10,
        phi_seed=42,
        phi_subsystem_dims_override=[[2, 2], [1, 1]]
    )
    
    jax_configs = field_space.get_jax_configurations()
    phi_vectors = jax_configs['phi_vectors_jax']
    
    # Check normalization of all phi vectors
    import jax.numpy as jnp
    norms = jnp.linalg.norm(phi_vectors, axis=1)
    all_normalized = jnp.allclose(norms, 1.0)
    print(f'   ✓ All φ vectors normalized: {all_normalized}')
    assert all_normalized, 'Some phi vectors not normalized'
    
    # Check g_types are valid
    g_types = jax_configs['g_types_jax']
    valid_g_types = jnp.all((g_types == 0) | (g_types == 1))
    print(f'   ✓ All g_types valid (0 or 1): {valid_g_types}')
    assert valid_g_types, 'Invalid g_types found'
    
    print('   ✅ Field configuration mathematics: CORRECT')
    
except Exception as e:
    print(f'   ❌ Field configuration test failed: {e}')
    traceback.print_exc()

print()

# Test 8: Universal complexity integration
print('8. UNIVERSAL COMPLEXITY INTEGRATION TEST')
try:
    from complexity import compute_universal_complexity_U
    from universe import UniverseState
    from fields import FieldConfigurationSpace
    
    # Create test universe and field space
    universe = UniverseState(dimension=4, initial_state_seed=42, subsystem_dims=[2, 2])
    field_space = FieldConfigurationSpace(
        dimension=4,
        num_configs=50,
        phi_seed=42,
        phi_subsystem_dims_override=[[2, 2], [1, 1]]
    )
    
    batched_configs = field_space.get_jax_configurations()
    U_val, all_T_vals, T_by_g = compute_universal_complexity_U(universe.get_state(), batched_configs)
    
    print(f'   ✓ Universal complexity U: {U_val:.6f}')
    print(f'   ✓ Number of T values: {len(all_T_vals)}')
    print(f'   ✓ T values by g_type: {[(g, len(vals)) for g, vals in T_by_g.items()]}')
    
    # Check U is valid
    assert U_val >= 0, f'Invalid U value: {U_val}'
    assert len(all_T_vals) == 50, f'Wrong number of T values: {len(all_T_vals)}'
    
    # Check all T values are valid
    import jax.numpy as jnp
    all_T_valid = jnp.all(all_T_vals >= 0) and jnp.all(jnp.isfinite(all_T_vals))
    print(f'   ✓ All T values valid: {all_T_valid}')
    assert all_T_valid, 'Some T values invalid'
    
    print('   ✅ Universal complexity integration: CORRECT')
    
except Exception as e:
    print(f'   ❌ Universal complexity test failed: {e}')
    traceback.print_exc()

print()
print('=== MATHEMATICAL VERIFICATION COMPLETE ===') 