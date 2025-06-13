"""
Universe State Representation Module
Implements the quantum state |Ψ(t)⟩ in Hilbert space H
"""
import numpy as np
from typing import List, Tuple, Optional
import scipy.linalg as la
from dataclasses import dataclass


@dataclass
class UniverseState:
    """Represents the universal quantum state |Ψ(t)⟩"""
    amplitudes: np.ndarray
    dimension: int
    time: float
    
    def __post_init__(self):
        # Normalize the state
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
    
    def evolve(self, hamiltonian: np.ndarray, dt: float) -> 'UniverseState':
        """Evolve the universe state by time dt using the Hamiltonian"""
        # U(t) = exp(-iHt/ℏ), setting ℏ = 1
        evolution_operator = la.expm(-1j * hamiltonian * dt)
        new_amplitudes = evolution_operator @ self.amplitudes
        return UniverseState(new_amplitudes, self.dimension, self.time + dt)
    
    def reduced_density_matrix(self, subsystem_indices: List[int]) -> np.ndarray:
        """Compute reduced density matrix for subsystem S"""
        # Create full density matrix
        rho_full = np.outer(self.amplitudes, np.conj(self.amplitudes))
        
        # For simplicity, extract submatrix corresponding to subsystem
        # In a real implementation, this would require proper tensor operations
        subsystem_size = len(subsystem_indices)
        max_index = min(max(subsystem_indices) + 1, self.dimension)
        
        # Extract submatrix
        rho_reduced = rho_full[:max_index, :max_index]
        
        # If subsystem is smaller, further reduce
        if subsystem_size < max_index:
            rho_reduced = rho_reduced[:subsystem_size, :subsystem_size]
        
        # Ensure the matrix is properly normalized
        trace = np.trace(rho_reduced)
        if trace > 0:
            rho_reduced = rho_reduced / trace
        
        return rho_reduced
    
    def expectation_value(self, operator: np.ndarray) -> complex:
        """Compute expectation value ⟨Ψ|O|Ψ⟩"""
        return np.conj(self.amplitudes) @ operator @ self.amplitudes
    
    def entropy(self) -> float:
        """Compute von Neumann entropy of the state"""
        rho = np.outer(self.amplitudes, np.conj(self.amplitudes))
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        return -np.sum(eigenvals * np.log2(eigenvals)).real


class HilbertSpace:
    """Represents the Hilbert space H"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.basis_states = self._generate_basis()
    
    def _generate_basis(self) -> List[np.ndarray]:
        """Generate orthonormal basis states"""
        basis = []
        for i in range(self.dimension):
            state = np.zeros(self.dimension, dtype=complex)
            state[i] = 1.0
            basis.append(state)
        return basis
    
    def random_state(self, time: float = 0.0) -> UniverseState:
        """Generate a random normalized state"""
        amplitudes = np.random.complex128(self.dimension)
        return UniverseState(amplitudes, self.dimension, time)
    
    def coherent_superposition(self, coefficients: List[complex], time: float = 0.0) -> UniverseState:
        """Create a coherent superposition of basis states"""
        if len(coefficients) != self.dimension:
            raise ValueError("Number of coefficients must match dimension")
        
        amplitudes = np.array(coefficients, dtype=complex)
        return UniverseState(amplitudes, self.dimension, time)


def create_universe(dimension: int = 64, initial_time: float = 0.0) -> Tuple[HilbertSpace, UniverseState]:
    """Factory function to create a universe with initial state"""
    hilbert_space = HilbertSpace(dimension)
    
    # Create initial state with some structure (not completely random)
    # This represents the "genesis" state
    coefficients = []
    for i in range(dimension):
        # Create patterns that might lead to emergent complexity
        phase = 2 * np.pi * i / dimension
        amplitude = np.exp(1j * phase) / np.sqrt(dimension)
        # Add some structured randomness
        amplitude *= (1 + 0.1 * np.random.randn())
        coefficients.append(amplitude)
    
    initial_state = hilbert_space.coherent_superposition(coefficients, initial_time)
    return hilbert_space, initial_state
