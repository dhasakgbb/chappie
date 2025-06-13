"""
Complexity Operator Module
Implements operator T acting on Ψ(t) to extract complexity
"""
import numpy as np
from typing import List, Callable, Tuple, Dict, Any
from dataclasses import dataclass
import scipy.linalg as la
from universe import UniverseState
from fields import FieldConfiguration


@dataclass
class ComplexityMeasure:
    """Result of complexity calculation"""
    value: float
    components: Dict[str, float]
    field_config: FieldConfiguration
    timestamp: float


class ComplexityOperator:
    """The operator T that extracts complexity from universe states"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.operators = self._initialize_operators()
    
    def _initialize_operators(self) -> Dict[str, np.ndarray]:
        """Initialize various complexity operators"""
        operators = {}
        
        # Information theoretic complexity operator
        # Based on quantum Fisher information
        operators['fisher'] = self._fisher_information_operator()
        
        # Entanglement complexity operator
        operators['entanglement'] = self._entanglement_operator()
        
        # Symmetry breaking operator
        operators['symmetry'] = self._symmetry_breaking_operator()
        
        # Topological complexity operator
        operators['topology'] = self._topological_operator()
        
        # Dynamical complexity operator
        operators['dynamics'] = self._dynamical_complexity_operator()
        
        return operators
    
    def _fisher_information_operator(self) -> np.ndarray:
        """Quantum Fisher information operator"""
        # Simplified Fisher information metric
        operator = np.zeros((self.dimension, self.dimension), dtype=complex)
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    operator[i, j] = i + 1  # Frequency-like weights
                else:
                    operator[i, j] = 1.0 / (abs(i - j) + 1)  # Coupling terms
        return operator
    
    def _entanglement_operator(self) -> np.ndarray:
        """Operator measuring entanglement complexity"""
        # Create operator that measures off-diagonal correlations
        operator = np.random.randn(self.dimension, self.dimension)
        operator = (operator + operator.T) / 2  # Make Hermitian
        return operator
    
    def _symmetry_breaking_operator(self) -> np.ndarray:
        """Operator measuring symmetry breaking"""
        # Operator that breaks various symmetries
        operator = np.zeros((self.dimension, self.dimension), dtype=complex)
        for i in range(self.dimension):
            for j in range(self.dimension):
                if (i + j) % 2 == 1:  # Break parity
                    operator[i, j] = 1.0
        operator = (operator + operator.conj().T) / 2
        return operator
    
    def _topological_operator(self) -> np.ndarray:
        """Operator measuring topological complexity"""
        # Simple topological operator based on winding numbers
        operator = np.zeros((self.dimension, self.dimension), dtype=complex)
        for i in range(self.dimension):
            for j in range(self.dimension):
                phase = 2 * np.pi * (i * j) / self.dimension
                operator[i, j] = np.exp(1j * phase)
        return operator
    
    def _dynamical_complexity_operator(self) -> np.ndarray:
        """Operator measuring dynamical complexity"""
        # Based on Lyapunov-like instability
        operator = np.zeros((self.dimension, self.dimension), dtype=complex)
        for i in range(self.dimension - 1):
            operator[i, i + 1] = 1.0  # Forward coupling
            operator[i + 1, i] = -1.0  # Backward coupling
        return operator
    
    def apply(self, universe_state: UniverseState, field_config: FieldConfiguration) -> ComplexityMeasure:
        """Apply complexity operator T[g,φ] to universe state"""
        components = {}
        
        # Modify operators based on field configuration
        modified_operators = self._modify_operators_by_fields(field_config)
        
        # Calculate complexity components
        for name, operator in modified_operators.items():
            expectation = universe_state.expectation_value(operator)
            components[name] = float(expectation.real)
          # Combine components into total complexity
        total_complexity = self._combine_complexity_components(components, field_config)
        
        return ComplexityMeasure(
            value=total_complexity,
            components=components,
            field_config=field_config,
            timestamp=universe_state.time
        )
    
    def _modify_operators_by_fields(self, field_config: FieldConfiguration) -> Dict[str, np.ndarray]:
        """Modify operators based on field configuration"""
        modified = {}
        
        for name, base_operator in self.operators.items():
            # Scale by field energy
            energy_factor = 1.0 / (1.0 + field_config.energy)
            
            # Modify by symmetries
            symmetry_factor = 1.0 + 0.1 * len(field_config.symmetries)
            
            # Field-dependent modifications - ensure dimension compatibility
            field_dim = min(len(field_config.scalar_field), self.dimension)
            field_influence = np.zeros((self.dimension, self.dimension))
            
            # Create field influence matrix with proper dimensions
            for i in range(field_dim):
                for j in range(field_dim):
                    if i < len(field_config.scalar_field) and j < len(field_config.scalar_field):
                        field_influence[i, j] = field_config.scalar_field[i] * field_config.scalar_field[j]
            
            # Combine influences
            modified_op = base_operator * energy_factor * symmetry_factor
            modified_op += 0.01 * field_influence  # Small field coupling
            
            modified[name] = modified_op
        
        return modified
    
    def _combine_complexity_components(self, components: Dict[str, float], 
                                     field_config: FieldConfiguration) -> float:
        """Combine different complexity measures into total complexity"""
        weights = {
            'fisher': 0.3,
            'entanglement': 0.25,
            'symmetry': 0.2,
            'topology': 0.15,
            'dynamics': 0.1
        }
        
        total = 0.0
        for name, value in components.items():
            weight = weights.get(name, 0.1)
            total += weight * abs(value)
        
        # Field-dependent scaling
        field_scale = 1.0 + 0.1 * len(field_config.symmetries)
        energy_scale = 1.0 / (1.0 + field_config.energy**2)
        
        return total * field_scale * energy_scale


class UniversalComplexity:
    """Computes U(t) = ∫ ComplexityValue(g,φ,t) dμ(g,φ)"""
    
    def __init__(self, complexity_operator: ComplexityOperator):
        self.complexity_operator = complexity_operator
        self.history = []
    
    def compute(self, universe_state: UniverseState, configuration_space) -> Tuple[float, List[ComplexityMeasure]]:
        """Compute universal complexity by integrating over configuration space"""
        measures = []
        
        # Compute complexity for each field configuration
        for config in configuration_space.configurations:
            measure = self.complexity_operator.apply(universe_state, config)
            measures.append(measure)
        
        # Integrate using the configuration space measure
        def complexity_function(config: FieldConfiguration) -> float:
            # Find corresponding measure
            for measure in measures:
                if measure.field_config is config:
                    return measure.value
            return 0.0
        
        universal_complexity = configuration_space.integrate_over_space(complexity_function)
        
        # Store in history
        self.history.append({
            'time': universe_state.time,
            'universal_complexity': universal_complexity,
            'measures': measures
        })
        
        return universal_complexity, measures
    
    def complexity_gradient(self, universe_state: UniverseState, configuration_space) -> np.ndarray:
        """Compute gradient of complexity with respect to universe state"""
        epsilon = 1e-6
        gradient = np.zeros_like(universe_state.amplitudes)
        
        base_complexity, _ = self.compute(universe_state, configuration_space)
        
        for i in range(len(universe_state.amplitudes)):
            # Perturb state
            perturbed_amplitudes = universe_state.amplitudes.copy()
            perturbed_amplitudes[i] += epsilon
            perturbed_state = UniverseState(perturbed_amplitudes, universe_state.dimension, universe_state.time)
            
            # Compute perturbed complexity
            perturbed_complexity, _ = self.compute(perturbed_state, configuration_space)
            
            # Numerical gradient
            gradient[i] = (perturbed_complexity - base_complexity) / epsilon
        
        return gradient
    
    def analyze_complexity_evolution(self) -> Dict[str, Any]:
        """Analyze how complexity evolves over time"""
        if len(self.history) < 2:
            return {'error': 'Insufficient history for analysis'}
        
        times = [entry['time'] for entry in self.history]
        complexities = [entry['universal_complexity'] for entry in self.history]
        
        # Compute time derivatives
        complexity_derivative = np.gradient(complexities, times)
        
        # Find critical points
        critical_points = []
        for i in range(1, len(complexity_derivative) - 1):
            if (complexity_derivative[i-1] * complexity_derivative[i+1] < 0):
                critical_points.append(times[i])
        
        return {
            'complexity_growth_rate': complexity_derivative[-1],
            'average_complexity': np.mean(complexities),
            'complexity_variance': np.var(complexities),
            'critical_points': critical_points,
            'total_complexity_change': complexities[-1] - complexities[0]
        }
