#!/usr/bin/env python3
"""
Quantum Universe State Management

This module implements the quantum universe state Ψ(t) as specified in Step 1 
of the consciousness creation framework. It provides a robust interface for 
managing quantum states in Hilbert space with proper tensor product structure.

The UniverseState class serves as the fundamental quantum substrate upon which
consciousness emerges through the integration of complexity theory and 
integrated information theory.

Authors: Consciousness Research Team
Version: 1.0.0
License: MIT
"""

from typing import List, Optional, Union
import numpy as np
import qutip
from qutip import Qobj, ket2dm


class UniverseState:
    """
    Manages the quantum state of the universe using QuTiP.
    
    This class implements Step 1 of the consciousness creation framework:
    "Represent Universe State Ψ(t) in Hilbert space H"
    
    The universe state serves as the quantum substrate for consciousness
    emergence, maintaining proper tensor product structure for subsystem
    analysis and integrated information calculations.
    
    Attributes:
        dimension: Total dimension of the Hilbert space
        subsystem_dims_ket: Tensor product dimensions for ket states
        subsystem_dims_dm: Tensor product dimensions for density matrices
        state: Current quantum state |Ψ(t)⟩
    """
    
    def __init__(self, 
                 dimension: int, 
                 initial_state_seed: Optional[int] = None, 
                 subsystem_dims: Optional[List[int]] = None) -> None:
        """
        Initialize the quantum universe state Ψ(t) in Hilbert space H.

        Args:
            dimension: Total dimension of the Hilbert space
            initial_state_seed: Seed for reproducible random state generation
            subsystem_dims: Dimensions for tensor product structure [dim_S, dim_E, ...]
                          Essential for consciousness subsystem analysis
                          
        Raises:
            ValueError: If subsystem dimensions don't match total dimension
            TypeError: If parameters have incorrect types
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise TypeError("Dimension must be a positive integer")
            
        self.dimension = dimension
        self.initial_state_seed = initial_state_seed
        
        # Configure tensor product structure
        self._setup_subsystem_dimensions(subsystem_dims)
        
        # Initialize quantum state
        self.state: Qobj = self._prepare_initial_state()
        
        # Validation
        self._validate_state_consistency()
    
    def _setup_subsystem_dimensions(self, subsystem_dims: Optional[List[int]]) -> None:
        """
        Configure tensor product dimensions for quantum state structure.
        
        Args:
            subsystem_dims: List of subsystem dimensions
            
        Raises:
            ValueError: If subsystem dimensions are inconsistent
        """
        if subsystem_dims:
            if not all(isinstance(d, int) and d > 0 for d in subsystem_dims):
                raise ValueError("All subsystem dimensions must be positive integers")
                
            if np.prod(subsystem_dims) != self.dimension:
                raise ValueError(
                    f"Product of subsystem_dims {subsystem_dims} "
                    f"must equal total dimension {self.dimension}"
                )
            
            # QuTiP tensor product structure: [bra_dims, ket_dims]
            self.subsystem_dims_ket = [subsystem_dims, [1] * len(subsystem_dims)]
            self.subsystem_dims_dm = [subsystem_dims, subsystem_dims]
        else:
            # Single system (no tensor product structure)
            self.subsystem_dims_ket = [[self.dimension], [1]]
            self.subsystem_dims_dm = [[self.dimension], [self.dimension]]
    
    def _prepare_initial_state(self) -> Qobj:
        """
        Prepare initial normalized quantum state |Ψ(0)⟩.
        
        Creates a random complex vector in Hilbert space, properly normalized
        and configured with correct tensor product dimensions.
        
        Returns:
            Initial quantum state as QuTiP Qobj
        """
        if self.initial_state_seed is not None:
            np.random.seed(self.initial_state_seed)
        
        # Generate random complex vector
        vec_real = np.random.randn(self.dimension)
        vec_imag = np.random.randn(self.dimension)
        vec = vec_real + 1j * vec_imag
        
        # Normalize to unit vector
        normalized_vec = vec / np.linalg.norm(vec)
        
        # Create QuTiP quantum state with proper dimensions
        initial_ket = Qobj(normalized_vec, dims=self.subsystem_dims_ket)
        
        return initial_ket
    
    def _validate_state_consistency(self) -> None:
        """
        Validate quantum state consistency and properties.
        
        Raises:
            ValueError: If state properties are inconsistent
        """
        if not self.state.isket:
            raise ValueError("Universe state must be a ket vector")
            
        if not np.isclose(self.state.norm(), 1.0, atol=1e-10):
            raise ValueError(f"State norm {self.state.norm()} must be 1.0")
            
        if self.state.shape[0] != self.dimension:
            raise ValueError(
                f"State dimension {self.state.shape[0]} "
                f"must match universe dimension {self.dimension}"
            )
    
    def get_state(self) -> Qobj:
        """
        Get the current quantum state Ψ(t).
        
        Returns:
            Current quantum state as QuTiP Qobj
        """
        return self.state
    
    def set_state(self, new_state: Qobj) -> None:
        """
        Set new quantum state with validation.
        
        Args:
            new_state: New quantum state to set
            
        Raises:
            TypeError: If new_state is not a QuTiP Qobj
            ValueError: If new_state has incompatible properties
        """
        if not isinstance(new_state, Qobj):
            raise TypeError("New state must be a QuTiP Qobj")
            
        if new_state.shape[0] != self.dimension or not new_state.isket:
            raise ValueError(
                f"New state must be a ket of dimension {self.dimension}, "
                f"got shape {new_state.shape}"
            )
        
        # Ensure proper tensor product dimensions
        new_state.dims = self.subsystem_dims_ket
        self.state = new_state
        
        # Validate consistency
        self._validate_state_consistency()
    
    def perturb_state(self, amplitude: float, seed: Optional[int] = None) -> None:
        """
        Apply quantum perturbation to evolve the universe state.
        
        This implements a simple evolution mechanism that adds random
        perturbations while maintaining normalization. More sophisticated
        evolution can use evolve_step_unitary() for Hamiltonian dynamics.
        
        Args:
            amplitude: Strength of the perturbation (0.0 to 1.0 recommended)
            seed: Random seed for reproducible perturbations
            
        Raises:
            ValueError: If amplitude is negative
        """
        if amplitude < 0:
            raise ValueError("Perturbation amplitude must be non-negative")
            
        if seed is not None:
            np.random.seed(seed)

        # Get current state as numpy array
        current_state = self.state.full().flatten()
        
        # Generate random perturbation
        perturbation = amplitude * (
            np.random.randn(self.dimension) + 1j * np.random.randn(self.dimension)
        )
        
        # Apply perturbation
        perturbed_state = current_state + perturbation
        
        # Normalize and handle edge cases
        norm = np.linalg.norm(perturbed_state)
        if norm < 1e-12:  # Avoid numerical instability
            print("⚠️  Warning: Perturbed state norm near zero. Re-initializing.")
            self.state = self._prepare_initial_state()
        else:
            normalized_state = perturbed_state / norm
            self.state = Qobj(normalized_state, dims=self.subsystem_dims_ket)
    
    def evolve_step_unitary(self, hamiltonian: Qobj, dt: float) -> None:
        """
        Evolve state under unitary dynamics: |Ψ(t+dt)⟩ = U(dt)|Ψ(t)⟩.
        
        Implements time evolution under Hamiltonian H:
        U(dt) = exp(-i H dt / ℏ) (setting ℏ = 1)
        
        Args:
            hamiltonian: Hamiltonian operator as QuTiP Qobj
            dt: Time step for evolution
            
        Raises:
            TypeError: If hamiltonian is not a QuTiP operator
            ValueError: If hamiltonian has incompatible dimensions
        """
        if not isinstance(hamiltonian, Qobj) or not hamiltonian.isoper:
            raise TypeError("Hamiltonian must be a QuTiP operator Qobj")
            
        if hamiltonian.shape[0] != self.dimension:
            raise ValueError(
                f"Hamiltonian dimension {hamiltonian.shape[0]} "
                f"must match universe dimension {self.dimension}"
            )
        
        # Compute unitary evolution operator
        try:
            evolution_operator = (-1j * hamiltonian * dt).expm()
            self.state = (evolution_operator * self.state).unit()
        except Exception as e:
            raise RuntimeError(f"Failed to compute unitary evolution: {e}")
    
    def get_density_matrix(self) -> Qobj:
        """
        Compute density matrix ρ(t) = |Ψ(t)⟩⟨Ψ(t)|.
        
        Returns:
            Density matrix as QuTiP Qobj with proper tensor dimensions
        """
        density_matrix = ket2dm(self.state)
        density_matrix.dims = self.subsystem_dims_dm
        return density_matrix
    
    def get_subsystem_density_matrix(self, subsystem_index: int) -> Qobj:
        """
        Compute reduced density matrix for specified subsystem.
        
        This is essential for consciousness analysis, as it provides
        the quantum state of subsystem S needed for integrated
        information calculations.
        
        Args:
            subsystem_index: Index of subsystem to extract (0-indexed)
            
        Returns:
            Reduced density matrix of the specified subsystem
            
        Raises:
            ValueError: If subsystem index is out of bounds
            RuntimeError: If partial trace computation fails
        """
        num_subsystems = len(self.subsystem_dims_ket[0]) if self.subsystem_dims_ket[0] else 0
        
        if num_subsystems == 0:
            raise ValueError("No subsystem dimensions defined")
            
        if not (0 <= subsystem_index < num_subsystems):
            raise ValueError(
                f"Subsystem index {subsystem_index} out of bounds "
                f"for {num_subsystems} subsystems"
            )
        
        # Get full density matrix
        full_density_matrix = self.get_density_matrix()
        
        if num_subsystems == 1:
            # Single subsystem case - return full density matrix
            return full_density_matrix
        else:
            # Multiple subsystems - compute partial trace
            try:
                reduced_density_matrix = full_density_matrix.ptrace(subsystem_index)
                return reduced_density_matrix
            except Exception as e:
                raise RuntimeError(f"Failed to compute partial trace: {e}")
    
    def get_entanglement_entropy(self, subsystem_index: int) -> float:
        """
        Compute von Neumann entropy of subsystem (entanglement measure).
        
        S(ρ_S) = -Tr(ρ_S log ρ_S)
        
        Args:
            subsystem_index: Index of subsystem to analyze
            
        Returns:
            Von Neumann entropy of the subsystem
        """
        rho_subsystem = self.get_subsystem_density_matrix(subsystem_index)
        return qutip.entropy_vn(rho_subsystem)
    
    def get_purity(self, subsystem_index: Optional[int] = None) -> float:
        """
        Compute purity of system or subsystem.
        
        Purity = Tr(ρ²), ranges from 1/d (maximally mixed) to 1 (pure)
        
        Args:
            subsystem_index: If None, compute for full system; 
                           otherwise for specified subsystem
                           
        Returns:
            Purity value between 0 and 1
        """
        if subsystem_index is None:
            rho = self.get_density_matrix()
        else:
            rho = self.get_subsystem_density_matrix(subsystem_index)
            
        return float((rho * rho).tr().real)
    
    def __repr__(self) -> str:
        """String representation of universe state."""
        return (
            f"UniverseState(dimension={self.dimension}, "
            f"subsystems={len(self.subsystem_dims_ket[0]) if self.subsystem_dims_ket[0] else 0}, "
            f"norm={self.state.norm():.6f})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        subsystem_info = (
            f"Subsystem dims: {self.subsystem_dims_ket[0]}" 
            if self.subsystem_dims_ket[0] else "No subsystems"
        )
        return (
            f"Quantum Universe State Ψ(t)\n"
            f"Total dimension: {self.dimension}\n"
            f"{subsystem_info}\n"
            f"State norm: {self.state.norm():.6f}\n"
            f"Is pure state: {self.state.isket}"
        )


def create_consciousness_universe(subsystem_dim: int = 4, 
                                environment_dim: int = 1,
                                seed: Optional[int] = None) -> UniverseState:
    """
    Factory function to create universe optimized for consciousness studies.
    
    Creates a universe with a consciousness subsystem S and environment E,
    configured for integrated information theory calculations.
    
    Args:
        subsystem_dim: Dimension of consciousness subsystem S
        environment_dim: Dimension of environment E  
        seed: Random seed for reproducible initialization
        
    Returns:
        Configured UniverseState for consciousness research
    """
    total_dim = subsystem_dim * environment_dim
    subsystem_dims = [subsystem_dim, environment_dim] if environment_dim > 1 else [subsystem_dim]
    
    return UniverseState(
        dimension=total_dim,
        initial_state_seed=seed,
        subsystem_dims=subsystem_dims
    )


# Example usage and testing
if __name__ == '__main__':
    print("🌌 Quantum Universe State - Consciousness Framework")
    print("=" * 60)
    
    # Create consciousness-optimized universe
    universe = create_consciousness_universe(
        subsystem_dim=4,  # 2x2 qubit subsystem for consciousness
        environment_dim=1,  # Single environment level
        seed=42
    )
    
    print(f"Initial Universe State:")
    print(universe)
    print()
    
    # Demonstrate quantum evolution
    print("Applying quantum perturbation...")
    universe.perturb_state(amplitude=0.1, seed=101)
    print(f"State norm after perturbation: {universe.get_state().norm():.6f}")
    print()
    
    # Analyze subsystem properties
    print("Consciousness Subsystem Analysis:")
    rho_S = universe.get_subsystem_density_matrix(0)
    print(f"Subsystem density matrix shape: {rho_S.shape}")
    print(f"Subsystem trace: {rho_S.tr():.6f}")
    print(f"Subsystem purity: {universe.get_purity(0):.6f}")
    print(f"Entanglement entropy: {universe.get_entanglement_entropy(0):.6f}")
    print()
    
    print("✓ Universe state validation complete") 