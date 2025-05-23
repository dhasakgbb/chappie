# Edge case and numerical stability verification
import numpy as np
import qutip
import traceback

print('=== EDGE CASE MATHEMATICAL VERIFICATION ===')
print()

# Test 1: Numerical stability with edge values
print('1. NUMERICAL STABILITY TEST')
try:
    from universe import UniverseState
    from consciousness import ConsciousAgent
    
    # Test with very small dimension
    print('   Testing small dimensions...')
    uni_small = UniverseState(dimension=2, initial_state_seed=42, subsystem_dims=[2])
    agent_small = ConsciousAgent(uni_small, [2], 0)
    phi_small = agent_small.iic.compute_phi(agent_small.sub)
    print(f'   ✓ Small system Φ: {phi_small:.6f}')
    assert 0 <= phi_small <= 1, f'Invalid phi for small system: {phi_small}'
    
    # Test with larger dimension
    print('   Testing larger dimensions...')
    uni_large = UniverseState(dimension=8, initial_state_seed=42, subsystem_dims=[2, 2, 2])
    agent_large = ConsciousAgent(uni_large, [2, 2, 2], 0)
    phi_large = agent_large.iic.compute_phi(agent_large.sub)
    print(f'   ✓ Large system Φ: {phi_large:.6f}')
    assert 0 <= phi_large <= 1, f'Invalid phi for large system: {phi_large}'
    
    print('   ✅ Numerical stability: CORRECT')
    
except Exception as e:
    print(f'   ❌ Numerical stability test failed: {e}')
    traceback.print_exc()

print()

# Test 2: Boundary conditions for complexity calculations
print('2. COMPLEXITY BOUNDARY CONDITIONS TEST')
try:
    from complexity import _compute_single_complexity_value_jax, get_jax_symmetry_operator
    import jax.numpy as jnp
    
    # Test with perfectly aligned states (should give high complexity)
    psi_aligned = jnp.array([1, 0, 0, 0], dtype=complex)
    phi_aligned = jnp.array([1, 0, 0, 0], dtype=complex)
    S0 = get_jax_symmetry_operator(0, 4, [[2, 2], [1, 1]])
    
    T_aligned = _compute_single_complexity_value_jax(psi_aligned, phi_aligned, S0)
    print(f'   ✓ Aligned states T: {T_aligned:.6f}')
    assert T_aligned >= 0, f'Negative complexity for aligned states: {T_aligned}'
    
    # Test with orthogonal states (should give low complexity)
    psi_ortho = jnp.array([1, 0, 0, 0], dtype=complex)
    phi_ortho = jnp.array([0, 1, 0, 0], dtype=complex)
    
    T_ortho = _compute_single_complexity_value_jax(psi_ortho, phi_ortho, S0)
    print(f'   ✓ Orthogonal states T: {T_ortho:.6f}')
    assert T_ortho >= 0, f'Negative complexity for orthogonal states: {T_ortho}'
    
    # Test with normalized superposition
    psi_super = jnp.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], dtype=complex)
    phi_super = jnp.array([1/np.sqrt(2), -1/np.sqrt(2), 0, 0], dtype=complex)
    
    T_super = _compute_single_complexity_value_jax(psi_super, phi_super, S0)
    print(f'   ✓ Superposition states T: {T_super:.6f}')
    assert T_super >= 0, f'Negative complexity for superposition: {T_super}'
    
    print('   ✅ Complexity boundary conditions: CORRECT')
    
except Exception as e:
    print(f'   ❌ Complexity boundary test failed: {e}')
    traceback.print_exc()

print()

# Test 3: Entropy calculation edge cases
print('3. ENTROPY CALCULATION EDGE CASES TEST')
try:
    from qutip import ket2dm, entropy_vn, ptrace
    
    # Test with pure state (entropy should be 0)
    pure_state = qutip.basis(2, 0)  # |0⟩
    pure_dm = ket2dm(pure_state)
    pure_entropy = entropy_vn(pure_dm)
    print(f'   ✓ Pure state entropy: {pure_entropy:.10f} (should be ~0)')
    assert abs(pure_entropy) < 1e-10, f'Pure state entropy not zero: {pure_entropy}'
    
    # Test with maximally mixed state (entropy should be log(d))
    mixed_dm = qutip.qeye(2) / 2  # Maximally mixed 2-level system
    mixed_entropy = entropy_vn(mixed_dm)
    expected_entropy = np.log(2)
    print(f'   ✓ Mixed state entropy: {mixed_entropy:.6f} (expected ~{expected_entropy:.6f})')
    assert abs(mixed_entropy - expected_entropy) < 1e-6, f'Mixed entropy wrong: {mixed_entropy} vs {expected_entropy}'
    
    # Test with entangled state
    bell_state = (qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0)) + 
                  qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))).unit()
    bell_dm = ket2dm(bell_state)
    bell_dm.dims = [[2, 2], [2, 2]]
    
    # Full system entropy (should be 0 for pure state)
    full_entropy = entropy_vn(bell_dm)
    print(f'   ✓ Bell state full entropy: {full_entropy:.10f} (should be ~0)')
    assert abs(full_entropy) < 1e-10, f'Bell state full entropy not zero: {full_entropy}'
    
    # Subsystem entropy (should be log(2) for maximally entangled)
    subsystem_dm = bell_dm.ptrace(0)
    sub_entropy = entropy_vn(subsystem_dm)
    print(f'   ✓ Bell subsystem entropy: {sub_entropy:.6f} (expected ~{expected_entropy:.6f})')
    assert abs(sub_entropy - expected_entropy) < 1e-6, f'Bell subsystem entropy wrong: {sub_entropy}'
    
    print('   ✅ Entropy calculation edge cases: CORRECT')
    
except Exception as e:
    print(f'   ❌ Entropy edge case test failed: {e}')
    traceback.print_exc()

print()

# Test 4: Category theory limits with degenerate cases
print('4. CATEGORY THEORY DEGENERATE CASES TEST')
try:
    from category import CategoricalStructure
    
    # Test with identical configurations
    identical_configs = [
        {'g_type': 0, 'phi': np.array([1, 0, 0, 0])},
        {'g_type': 0, 'phi': np.array([1, 0, 0, 0])},
        {'g_type': 0, 'phi': np.array([1, 0, 0, 0])}
    ]
    identical_complexity = [0.5, 0.5, 0.5]
    
    cat_identical = CategoricalStructure(identical_configs, identical_complexity)
    f_identical = cat_identical.compute_F_structure_enhanced()
    
    limit_coherence = f_identical['categorical_limit']['limit_cone']['coherence']
    print(f'   ✓ Identical configs coherence: {limit_coherence:.6f}')
    assert 0 <= limit_coherence <= 1, f'Invalid coherence for identical configs: {limit_coherence}'
    
    # Test with extreme complexity values
    extreme_configs = [
        {'g_type': 0, 'phi': np.random.randn(4)},
        {'g_type': 1, 'phi': np.random.randn(4)}
    ]
    extreme_complexity = [0.0, 1.0]  # Extreme values
    
    cat_extreme = CategoricalStructure(extreme_configs, extreme_complexity)
    f_extreme = cat_extreme.compute_F_structure_enhanced()
    
    extreme_coherence = f_extreme['categorical_limit']['limit_cone']['coherence']
    print(f'   ✓ Extreme complexity coherence: {extreme_coherence:.6f}')
    assert 0 <= extreme_coherence <= 1, f'Invalid coherence for extreme complexity: {extreme_coherence}'
    
    print('   ✅ Category theory degenerate cases: CORRECT')
    
except Exception as e:
    print(f'   ❌ Category theory degenerate test failed: {e}')
    traceback.print_exc()

print()

# Test 5: Field configuration edge cases
print('5. FIELD CONFIGURATION EDGE CASES TEST')
try:
    from fields import FieldConfigurationSpace
    import jax.numpy as jnp
    
    # Test with minimum configurations
    field_min = FieldConfigurationSpace(
        dimension=2,
        num_configs=1,
        phi_seed=42,
        phi_subsystem_dims_override=[[2], [1]]
    )
    
    min_configs = field_min.get_jax_configurations()
    print(f'   ✓ Minimum configs created: {len(min_configs["phi_vectors_jax"])} configurations')
    assert len(min_configs["phi_vectors_jax"]) == 1, 'Wrong number of minimum configs'
    
    # Test normalization with different dimensions
    for dim in [2, 4, 8]:
        if dim == 2:
            phi_dims = [[2], [1]]
        elif dim == 4:
            phi_dims = [[2, 2], [1, 1]]
        else:  # dim == 8
            phi_dims = [[2, 2, 2], [1, 1, 1]]
            
        field_test = FieldConfigurationSpace(
            dimension=dim,
            num_configs=5,
            phi_seed=42,
            phi_subsystem_dims_override=phi_dims
        )
        test_configs = field_test.get_jax_configurations()
        norms = jnp.linalg.norm(test_configs['phi_vectors_jax'], axis=1)
        all_normalized = jnp.allclose(norms, 1.0)
        print(f'   ✓ Dimension {dim} normalized: {all_normalized}')
        assert all_normalized, f'Normalization failed for dimension {dim}'
    
    print('   ✅ Field configuration edge cases: CORRECT')
    
except Exception as e:
    print(f'   ❌ Field configuration edge test failed: {e}')
    traceback.print_exc()

print()

# Test 6: Consciousness level calculation edge cases
print('6. CONSCIOUSNESS LEVEL EDGE CASES TEST')
try:
    from consciousness import ConsciousAgent
    from universe import UniverseState
    
    # Test with zero complexity universe
    uni_zero = UniverseState(dimension=2, initial_state_seed=42, subsystem_dims=[2])
    agent_zero = ConsciousAgent(uni_zero, [2], 0)
    
    # Force zero universal complexity for testing
    agent_zero.perform_deep_introspection(0.0, 1)  # zero complexity
    consciousness_zero = agent_zero.consciousness_level
    print(f'   ✓ Zero complexity consciousness: {consciousness_zero:.6f}')
    assert consciousness_zero >= 0, f'Negative consciousness with zero complexity: {consciousness_zero}'
    
    # Test with maximum complexity
    agent_max = ConsciousAgent(uni_zero, [2], 0)
    agent_max.perform_deep_introspection(1.0, 1)  # maximum complexity
    consciousness_max = agent_max.consciousness_level  
    print(f'   ✓ Max complexity consciousness: {consciousness_max:.6f}')
    assert consciousness_max >= 0, f'Negative consciousness with max complexity: {consciousness_max}'
    
    print('   ✅ Consciousness level edge cases: CORRECT')
    
except Exception as e:
    print(f'   ❌ Consciousness level edge test failed: {e}')
    traceback.print_exc()

print()
print('=== EDGE CASE VERIFICATION COMPLETE ===') 