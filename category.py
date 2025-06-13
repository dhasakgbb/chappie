"""
Category Theory Module
Implements reflective abstract algebra and categorical framework
"""
import numpy as np
from typing import Any, Dict, List, Callable, Optional, Set, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from fields import FieldConfiguration
from complexity import ComplexityMeasure


class CategoryObject(ABC):
    """Abstract base class for objects in category C"""
    
    def __init__(self, name: str, data: Any):
        self.name = name
        self.data = data
    
    @abstractmethod
    def identity_morphism(self) -> 'Morphism':
        """Return identity morphism for this object"""
        pass
    
    def __str__(self):
        return f"Object({self.name})"


class ConfigurationObject(CategoryObject):
    """Object representing a field configuration in category C"""
    
    def __init__(self, name: str, config: FieldConfiguration):
        super().__init__(name, config)
        self.config = config
    
    def identity_morphism(self) -> 'Morphism':
        return IdentityMorphism(self)


class ComplexityObject(CategoryObject):
    """Object representing complexity values in category C"""
    
    def __init__(self, name: str, complexity_measure: ComplexityMeasure):
        super().__init__(name, complexity_measure)
        self.complexity_measure = complexity_measure
    
    def identity_morphism(self) -> 'Morphism':
        return IdentityMorphism(self)


class Morphism(ABC):
    """Abstract base class for morphisms in category C"""
    
    def __init__(self, source: CategoryObject, target: CategoryObject, name: str):
        self.source = source
        self.target = target
        self.name = name
    
    @abstractmethod
    def apply(self, obj: CategoryObject) -> CategoryObject:
        """Apply morphism to object"""
        pass
    
    def compose(self, other: 'Morphism') -> 'CompositeMorphism':
        """Compose with another morphism"""
        if self.target != other.source:
            raise ValueError("Cannot compose morphisms: target/source mismatch")
        return CompositeMorphism(other.source, self.target, [other, self])
    
    def __str__(self):
        return f"{self.name}: {self.source.name} → {self.target.name}"


class IdentityMorphism(Morphism):
    """Identity morphism for any object"""
    
    def __init__(self, obj: CategoryObject):
        super().__init__(obj, obj, f"id_{obj.name}")
    
    def apply(self, obj: CategoryObject) -> CategoryObject:
        if obj != self.source:
            raise ValueError("Identity morphism applied to wrong object")
        return obj


class CompositeMorphism(Morphism):
    """Composition of multiple morphisms"""
    
    def __init__(self, source: CategoryObject, target: CategoryObject, morphisms: List[Morphism]):
        super().__init__(source, target, "composite")
        self.morphisms = morphisms
    
    def apply(self, obj: CategoryObject) -> CategoryObject:
        result = obj
        for morphism in self.morphisms:
            result = morphism.apply(result)
        return result


class ComplexityFunctor:
    """Functor F: C → Set mapping objects to complexity values"""
    
    def __init__(self, name: str = "ComplexityFunctor"):
        self.name = name
        self.object_mapping = {}
        self.morphism_mapping = {}
    
    def map_object(self, obj: CategoryObject) -> Set[float]:
        """Map object to set of complexity values"""
        if isinstance(obj, ConfigurationObject):
            # Extract complexity-related values from configuration
            complexity_set = {
                obj.config.energy,
                len(obj.config.symmetries),
                np.mean(np.abs(obj.config.scalar_field)),
                np.mean(np.abs(obj.config.gauge_field))
            }
            self.object_mapping[obj.name] = complexity_set
            return complexity_set
        
        elif isinstance(obj, ComplexityObject):
            # Extract values from complexity measure
            complexity_set = {obj.complexity_measure.value}
            complexity_set.update(obj.complexity_measure.components.values())
            self.object_mapping[obj.name] = complexity_set
            return complexity_set
        
        else:
            # Default mapping
            default_set = {0.0}
            self.object_mapping[obj.name] = default_set
            return default_set
    
    def map_morphism(self, morphism: Morphism) -> Callable[[Set[float]], Set[float]]:
        """Map morphism to function between sets"""
        def set_function(input_set: Set[float]) -> Set[float]:
            # Transform the set based on morphism type
            if isinstance(morphism, IdentityMorphism):
                return input_set
            elif isinstance(morphism, CompositeMorphism):
                result = input_set
                for m in morphism.morphisms:
                    mapped_func = self.map_morphism(m)
                    result = mapped_func(result)
                return result
            else:
                # Apply some transformation
                return {x * 1.1 + 0.01 for x in input_set}  # Example transformation
        
        self.morphism_mapping[morphism.name] = set_function
        return set_function


class UniversalStructure:
    """Represents the universal structure F emerging as a limit"""
    
    def __init__(self):
        self.objects = []
        self.morphisms = []
        self.functor = ComplexityFunctor()
        self.limit_structure = None
    
    def add_object(self, obj: CategoryObject):
        """Add object to the category"""
        self.objects.append(obj)
    
    def add_morphism(self, morphism: Morphism):
        """Add morphism to the category"""
        self.morphisms.append(morphism)
    
    def compute_limit(self) -> Dict[str, Any]:
        """Compute limit (or inverse limit) to find universal structure"""
        if not self.objects:
            return {'error': 'No objects in category'}
        
        # Map all objects through functor
        mapped_objects = {}
        for obj in self.objects:
            mapped_objects[obj.name] = self.functor.map_object(obj)
        
        # Find common structure (intersection-like operation)
        if mapped_objects:
            # Find patterns across all mapped objects
            all_values = []
            for obj_values in mapped_objects.values():
                all_values.extend(list(obj_values))
            
            # Statistical analysis of the limit structure
            mean_value = np.mean(all_values)
            std_value = np.std(all_values)
            
            # The limit structure captures universal properties
            limit_structure = {
                'universal_complexity_mean': mean_value,
                'universal_complexity_std': std_value,
                'object_count': len(self.objects),
                'morphism_count': len(self.morphisms),
                'emergent_patterns': self._find_emergent_patterns(mapped_objects),
                'categorical_invariants': self._compute_categorical_invariants()
            }
            
            self.limit_structure = limit_structure
            return limit_structure
        
        return {'error': 'No mapped objects'}
    
    def _find_emergent_patterns(self, mapped_objects: Dict[str, Set[float]]) -> List[Dict[str, Any]]:
        """Find emergent patterns in the mapped objects"""
        patterns = []
        
        # Pattern 1: Clustering of complexity values
        all_values = []
        for values in mapped_objects.values():
            all_values.extend(list(values))
        
        if len(all_values) > 1:
            # Simple clustering analysis
            sorted_values = sorted(all_values)
            gaps = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values)-1)]
            
            if gaps:
                patterns.append({
                    'type': 'value_clustering',
                    'largest_gap': max(gaps),
                    'gap_variance': np.var(gaps)
                })
        
        # Pattern 2: Symmetry in object relationships
        symmetry_score = self._compute_symmetry_score(mapped_objects)
        patterns.append({
            'type': 'structural_symmetry',
            'symmetry_score': symmetry_score
        })
        
        return patterns
    
    def _compute_symmetry_score(self, mapped_objects: Dict[str, Set[float]]) -> float:
        """Compute symmetry score for the categorical structure"""
        if len(mapped_objects) < 2:
            return 0.0
        
        # Compare pairwise similarities
        object_names = list(mapped_objects.keys())
        similarities = []
        
        for i in range(len(object_names)):
            for j in range(i+1, len(object_names)):
                obj1_values = mapped_objects[object_names[i]]
                obj2_values = mapped_objects[object_names[j]]
                
                # Jaccard similarity for sets
                intersection = len(obj1_values.intersection(obj2_values))
                union = len(obj1_values.union(obj2_values))
                
                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_categorical_invariants(self) -> Dict[str, float]:
        """Compute categorical invariants"""
        invariants = {}
        
        # Euler characteristic analog
        invariants['euler_characteristic'] = len(self.objects) - len(self.morphisms)
        
        # Connectivity measure
        if self.objects:
            connectivity = len(self.morphisms) / len(self.objects)
            invariants['connectivity'] = connectivity
        
        # Compositional complexity
        composite_count = sum(1 for m in self.morphisms if isinstance(m, CompositeMorphism))
        if self.morphisms:
            invariants['compositional_ratio'] = composite_count / len(self.morphisms)
        
        return invariants
    
    def evolve_structure(self, time_step: float) -> 'UniversalStructure':
        """Evolve the categorical structure over time"""
        new_structure = UniversalStructure()
        
        # Evolve objects
        for obj in self.objects:
            if isinstance(obj, ConfigurationObject):
                # Create evolved configuration
                evolved_config = self._evolve_configuration(obj.config, time_step)
                evolved_obj = ConfigurationObject(f"{obj.name}_evolved", evolved_config)
                new_structure.add_object(evolved_obj)
            else:
                new_structure.add_object(obj)
        
        # Evolve morphisms (create new connections)
        for morphism in self.morphisms:
            new_structure.add_morphism(morphism)
        
        # Add evolutionary morphisms
        if len(new_structure.objects) > 1:
            for i in range(len(new_structure.objects) - 1):
                evolution_morphism = EvolutionMorphism(
                    new_structure.objects[i], 
                    new_structure.objects[i + 1]
                )
                new_structure.add_morphism(evolution_morphism)
        
        return new_structure
    
    def _evolve_configuration(self, config: FieldConfiguration, time_step: float) -> FieldConfiguration:
        """Evolve a field configuration"""
        # Simple evolution: add noise and adjust energy
        new_scalar = config.scalar_field + 0.01 * time_step * np.random.randn(len(config.scalar_field))
        new_gauge = config.gauge_field + 0.01 * time_step * (np.random.randn(len(config.gauge_field)) + 1j * np.random.randn(len(config.gauge_field)))
        
        # Recalculate energy
        new_energy = config.energy * (1 + 0.05 * time_step * np.random.randn())
        
        return FieldConfiguration(new_gauge, new_scalar, config.symmetries.copy(), new_energy)


class EvolutionMorphism(Morphism):
    """Morphism representing evolution between configurations"""
    
    def __init__(self, source: CategoryObject, target: CategoryObject):
        super().__init__(source, target, f"evolve_{source.name}_to_{target.name}")
    
    def apply(self, obj: CategoryObject) -> CategoryObject:
        if obj != self.source:
            raise ValueError("Evolution morphism applied to wrong object")
        return self.target


def create_categorical_framework(configurations: List[FieldConfiguration], 
                               complexity_measures: List[ComplexityMeasure]) -> UniversalStructure:
    """Create categorical framework from configurations and complexity measures"""
    structure = UniversalStructure()
    
    # Create configuration objects
    config_objects = []
    for i, config in enumerate(configurations):
        obj = ConfigurationObject(f"config_{i}", config)
        structure.add_object(obj)
        config_objects.append(obj)
    
    # Create complexity objects
    complexity_objects = []
    for i, measure in enumerate(complexity_measures):
        obj = ComplexityObject(f"complexity_{i}", measure)
        structure.add_object(obj)
        complexity_objects.append(obj)
    
    # Create morphisms between configurations
    for i, obj1 in enumerate(config_objects):
        for j, obj2 in enumerate(config_objects):
            if i != j:
                morphism = ConfigurationMorphism(obj1, obj2, f"morph_{i}_to_{j}")
                structure.add_morphism(morphism)
    
    return structure


class ConfigurationMorphism(Morphism):
    """Morphism between field configurations"""
    
    def __init__(self, source: ConfigurationObject, target: ConfigurationObject, name: str):
        super().__init__(source, target, name)
    
    def apply(self, obj: CategoryObject) -> CategoryObject:
        if obj != self.source:
            raise ValueError("Configuration morphism applied to wrong object")
        return self.target
