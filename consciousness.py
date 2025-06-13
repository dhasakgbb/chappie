"""
Consciousness Module
Implements consciousness calculation C(S,t) = f(I) for subsystem S
Using Integrated Information Theory (IIT) principles
"""
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass
from itertools import combinations, chain
import scipy.linalg as la
from universe import UniverseState
from fields import FieldConfiguration


@dataclass
class IntegratedInformation:
    """Represents integrated information Φ for a subsystem"""
    phi: float  # Integrated information value
    mip: Set[int]  # Minimum information partition
    subsystem: Set[int]  # The subsystem nodes
    mechanisms: List[Dict]  # Contributing mechanisms
    
    def __str__(self):
        return f"Φ = {self.phi:.6f} for subsystem {self.subsystem}"


@dataclass
class ConsciousnessState:
    """Represents consciousness C(S,t) for a subsystem at time t"""
    integrated_info: IntegratedInformation
    consciousness_value: float
    subsystem_indices: Set[int]
    timestamp: float
    field_influence: Dict[str, float]
    
    def __str__(self):
        return f"C(S,t) = {self.consciousness_value:.6f} at t = {self.timestamp:.3f}"


class ConsciousnessCalculator:
    """Implements consciousness calculation based on IIT"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.consciousness_function = self._define_consciousness_function()
    
    def _define_consciousness_function(self) -> callable:
        """Define function f such that C(S,t) = f(I)"""
        def f(integrated_info: float, field_influence: float = 1.0) -> float:
            # Consciousness as a nonlinear function of integrated information
            # Includes field influence and threshold effects
            
            # Threshold below which consciousness is negligible
            threshold = 0.01
            
            if integrated_info < threshold:
                return 0.0
            
            # Sigmoid-like function with field modulation
            base_consciousness = integrated_info**2 / (1 + integrated_info**2)
            
            # Field influence modulation
            field_modulation = 1.0 + 0.5 * np.tanh(field_influence - 1.0)
            
            # Final consciousness value
            return base_consciousness * field_modulation
        
        return f
    
    def compute_consciousness(self, universe_state: UniverseState, 
                            subsystem_indices: Set[int],
                            field_config: Optional[FieldConfiguration] = None) -> ConsciousnessState:
        """Compute consciousness C(S,t) for subsystem S"""
        
        # Step 1: Compute reduced density matrix
        subsystem_list = list(subsystem_indices)
        rho_s = universe_state.reduced_density_matrix(subsystem_list)
        
        # Step 2: Calculate integrated information
        integrated_info = self._calculate_integrated_information(rho_s, subsystem_indices)
        
        # Step 3: Calculate field influence
        field_influence = self._calculate_field_influence(field_config) if field_config else {}
        field_influence_value = sum(field_influence.values()) / len(field_influence) if field_influence else 1.0
        
        # Step 4: Apply consciousness function
        consciousness_value = self.consciousness_function(integrated_info.phi, field_influence_value)
        
        return ConsciousnessState(
            integrated_info=integrated_info,
            consciousness_value=consciousness_value,
            subsystem_indices=subsystem_indices,
            timestamp=universe_state.time,
            field_influence=field_influence
        )
    
    def _calculate_integrated_information(self, rho_s: np.ndarray, 
                                        subsystem_indices: Set[int]) -> IntegratedInformation:
        """Calculate integrated information Φ using IIT principles"""
        
        # For computational tractability, we use a simplified version of Φ
        # Real IIT calculations are extremely complex
        
        mechanisms = []
        subsystem_list = list(subsystem_indices)
        n_elements = len(subsystem_list)
        
        if n_elements < 2:
            return IntegratedInformation(0.0, set(), subsystem_indices, [])
        
        # Find all possible partitions
        all_partitions = self._generate_partitions(subsystem_indices)
        
        phi_values = []
        
        for partition in all_partitions:
            phi_partition = self._calculate_phi_partition(rho_s, partition, subsystem_list)
            phi_values.append((phi_partition, partition))
        
        # Find minimum information partition (MIP)
        if phi_values:
            min_phi, mip = min(phi_values, key=lambda x: x[0])
            phi = min_phi
        else:
            phi = 0.0
            mip = set()
        
        return IntegratedInformation(
            phi=phi,
            mip=mip,
            subsystem=subsystem_indices,
            mechanisms=mechanisms
        )
    
    def _generate_partitions(self, subsystem: Set[int]) -> List[Tuple[Set[int], Set[int]]]:
        """Generate all possible bipartitions of the subsystem"""
        if len(subsystem) < 2:
            return []
        
        partitions = []
        subsystem_list = list(subsystem)
        
        # Generate all non-trivial bipartitions
        for i in range(1, 2**(len(subsystem_list) - 1)):
            part1 = set()
            part2 = set()
            
            for j, element in enumerate(subsystem_list):
                if i & (1 << j):
                    part1.add(element)
                else:
                    part2.add(element)
            
            if part1 and part2:  # Non-trivial partition
                partitions.append((part1, part2))
        
        return partitions
    
    def _calculate_phi_partition(self, rho_s: np.ndarray, 
                               partition: Tuple[Set[int], Set[int]],
                               subsystem_list: List[int]) -> float:
        """Calculate Φ for a specific partition"""
        part1, part2 = partition
        
        # Calculate mutual information between parts
        # This is a simplified measure - real IIT uses more sophisticated metrics
        
        # Compute marginal densities
        part1_indices = [subsystem_list.index(i) for i in part1 if i in subsystem_list]
        part2_indices = [subsystem_list.index(i) for i in part2 if i in subsystem_list]
        
        if not part1_indices or not part2_indices:
            return 0.0
        
        # Simplified mutual information calculation
        try:
            # Trace out part2 to get part1 marginal
            rho1 = np.trace(rho_s.reshape((len(part1_indices), -1, len(part1_indices), -1)), 
                           axis=(1, 3))
            
            # Trace out part1 to get part2 marginal  
            rho2 = np.trace(rho_s.reshape((-1, len(part2_indices), -1, len(part2_indices))), 
                           axis=(0, 2))
            
            # Calculate entropies
            S1 = self._von_neumann_entropy(rho1)
            S2 = self._von_neumann_entropy(rho2)
            S12 = self._von_neumann_entropy(rho_s)
            
            # Mutual information
            mutual_info = S1 + S2 - S12
            return max(0.0, mutual_info)  # Φ is non-negative
            
        except:
            # Fallback calculation
            return np.trace(rho_s @ rho_s).real
    
    def _von_neumann_entropy(self, rho: np.ndarray) -> float:
        """Calculate von Neumann entropy of density matrix"""
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        
        if len(eigenvals) == 0:
            return 0.0
        
        return -np.sum(eigenvals * np.log2(eigenvals)).real
    
    def _calculate_field_influence(self, field_config: FieldConfiguration) -> Dict[str, float]:
        """Calculate how field configuration influences consciousness"""
        influence = {}
        
        # Scalar field influence
        scalar_variance = np.var(field_config.scalar_field)
        influence['scalar_field'] = scalar_variance
        
        # Gauge field influence
        gauge_magnitude = np.mean(np.abs(field_config.gauge_field))
        influence['gauge_field'] = gauge_magnitude
        
        # Symmetry influence
        symmetry_count = len(field_config.symmetries)
        influence['symmetries'] = symmetry_count / 10.0  # Normalized
        
        # Energy influence
        energy_influence = 1.0 / (1.0 + field_config.energy)
        influence['energy'] = energy_influence
        
        return influence
    
    def consciousness_dynamics(self, universe_state: UniverseState,
                             subsystem_indices: Set[int],
                             hamiltonian: np.ndarray,
                             dt: float,
                             field_config: Optional[FieldConfiguration] = None) -> Tuple[ConsciousnessState, ConsciousnessState]:
        """Analyze consciousness dynamics over time step dt"""
        
        # Current consciousness
        current_consciousness = self.compute_consciousness(universe_state, subsystem_indices, field_config)
        
        # Evolve state
        evolved_state = universe_state.evolve(hamiltonian, dt)
        
        # Future consciousness
        future_consciousness = self.compute_consciousness(evolved_state, subsystem_indices, field_config)
        
        return current_consciousness, future_consciousness
    
    def find_maximal_consciousness_subsystems(self, universe_state: UniverseState,
                                            max_subsystem_size: int = 8,
                                            field_config: Optional[FieldConfiguration] = None) -> List[ConsciousnessState]:
        """Find subsystems with maximal consciousness"""
        consciousness_states = []
        
        # Try different subsystem sizes
        for size in range(2, min(max_subsystem_size + 1, self.dimension + 1)):
            # Try random subsystems of this size
            for _ in range(min(20, int(np.math.comb(self.dimension, size)))):  # Limit combinations
                indices = np.random.choice(self.dimension, size, replace=False)
                subsystem = set(indices)
                
                consciousness_state = self.compute_consciousness(universe_state, subsystem, field_config)
                consciousness_states.append(consciousness_state)
        
        # Sort by consciousness value and return top candidates
        consciousness_states.sort(key=lambda x: x.consciousness_value, reverse=True)
        return consciousness_states[:10]  # Return top 10


class ConsciousnessEvolution:
    """Tracks consciousness evolution over time"""
    
    def __init__(self, calculator: ConsciousnessCalculator):
        self.calculator = calculator
        self.history = []
    
    def track_consciousness(self, universe_state: UniverseState,
                          subsystem_indices: Set[int],
                          field_config: Optional[FieldConfiguration] = None):
        """Track consciousness state over time"""
        consciousness_state = self.calculator.compute_consciousness(
            universe_state, subsystem_indices, field_config
        )
        
        self.history.append(consciousness_state)
        return consciousness_state
    
    def analyze_consciousness_trajectory(self) -> Dict[str, Any]:
        """Analyze consciousness evolution trajectory"""
        if len(self.history) < 2:
            return {'error': 'Insufficient history'}
        
        times = [state.timestamp for state in self.history]
        consciousness_values = [state.consciousness_value for state in self.history]
        phi_values = [state.integrated_info.phi for state in self.history]
        
        return {
            'consciousness_trend': np.polyfit(times, consciousness_values, 1)[0],
            'phi_trend': np.polyfit(times, phi_values, 1)[0],
            'peak_consciousness': max(consciousness_values),
            'average_consciousness': np.mean(consciousness_values),
            'consciousness_stability': 1.0 / (1.0 + np.var(consciousness_values))
        }
