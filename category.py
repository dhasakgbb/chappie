#!/usr/bin/env python3
"""
Categorical Structure and Functor Analysis

This module implements Step 6 of the consciousness creation framework:
"Reflective Abstract Algebra - Category Theory Implementation"

The CategoricalStructure class implements the functor F: C → Set that maps
field configurations to complexity values, and computes categorical limits
to discover universal structures underlying consciousness emergence.

Mathematical Foundation:
- Category C with objects as field configurations (g,φ)
- Morphisms as symmetry transformations and field operations
- Functor F: C → Set mapping objects to complexity values T[g,φ]
- Categorical limit F(C) revealing universal consciousness structure

Authors: Consciousness Research Team
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import networkx as nx
from dataclasses import dataclass


@dataclass
class CategoryObject:
    """
    Represents an object in category C (field configuration).
    
    Attributes:
        node_id: Unique identifier for the object
        g_type: Symmetry type (0=identity, 1=permutation)
        complexity: Complexity value T[g,φ] (image under functor F)
        phi_vector: Field configuration vector φ
        stability: Measure of stability under morphisms
    """
    node_id: int
    g_type: int
    complexity: float
    phi_vector: Optional[np.ndarray] = None
    stability: Optional[float] = None


@dataclass
class CategoryMorphism:
    """
    Represents a morphism in category C.
    
    Attributes:
        source: Source object ID
        target: Target object ID
        morphism_type: Type of transformation
        transformation: Description of the transformation
        weight: Strength/importance of the morphism
    """
    source: int
    target: int
    morphism_type: str
    transformation: str
    weight: float = 1.0


@dataclass
class CategoricalLimit:
    """
    Represents the categorical limit structure F(C).
    
    Attributes:
        apex_object: The universal object at the apex of the limit cone
        limit_cone: Collection of morphisms forming the cone
        universal_properties: Properties that characterize universality
        fixed_points: Stable configurations under functor F
    """
    apex_object: Optional[CategoryObject]
    limit_cone: List[CategoryMorphism]
    universal_properties: Dict[str, Any]
    fixed_points: List[CategoryObject]


class CategoricalStructure:
    """
    Implements categorical analysis of consciousness field configurations.
    
    This class represents the category C where:
    - Objects are field configurations (g,φ) 
    - Morphisms are symmetry transformations
    - Functor F maps objects to complexity values T[g,φ]
    
    The categorical limit F(C) reveals universal structures underlying
    consciousness emergence through abstract algebraic analysis.
    
    Attributes:
        graph: NetworkX directed graph representing the category
        objects: Dictionary of category objects
        morphisms: List of category morphisms
        symmetry_groups: Grouping of objects by symmetry type
        complexity_distribution: Distribution of complexity values
    """
    
    def __init__(self, 
                 configurations: List[Dict[str, Any]], 
                 complexity_values: List[float]) -> None:
        """
        Initialize categorical structure from field configurations.
        
        Args:
            configurations: List of field configuration dictionaries
                          Each contains {'g_type': int, 'phi': np.ndarray}
            complexity_values: Corresponding complexity values T[g,φ]
            
        Raises:
            ValueError: If configurations and complexity values don't match
        """
        self._validate_inputs(configurations, complexity_values)
        
        # Core categorical structure
        self.graph = nx.DiGraph()
        self.objects: Dict[int, CategoryObject] = {}
        self.morphisms: List[CategoryMorphism] = []
        
        # Configuration data
        self.configurations = configurations
        self.complexity_values = complexity_values
        
        # Analytical structures
        self.symmetry_groups: Dict[int, List[int]] = {}
        self.complexity_distribution: Dict[float, int] = {}
        self.morphism_count = 0
        
        # Build categorical structure
        self._build_category()
    
    def _validate_inputs(self, 
                        configurations: List[Dict[str, Any]], 
                        complexity_values: List[float]) -> None:
        """
        Validate input parameters for categorical construction.
        
        Args:
            configurations: Field configurations to validate
            complexity_values: Complexity values to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        if len(configurations) != len(complexity_values):
            raise ValueError(
                f"Number of configurations ({len(configurations)}) must match "
                f"number of complexity values ({len(complexity_values)})"
            )
        
        if not configurations:
            raise ValueError("At least one configuration is required")
        
        # Validate configuration structure
        for i, config in enumerate(configurations):
            if not isinstance(config, dict):
                raise ValueError(f"Configuration {i} must be a dictionary")
            
            if 'g_type' not in config:
                raise ValueError(f"Configuration {i} missing 'g_type' field")
            
            if not isinstance(config['g_type'], (int, np.integer)) or config['g_type'] not in [0, 1]:
                raise ValueError(f"Configuration {i} g_type must be 0 or 1")
        
        # Validate complexity values
        if not all(isinstance(val, (int, float)) for val in complexity_values):
            raise ValueError("All complexity values must be numeric")
    
    def _build_category(self) -> None:
        """
        Build the categorical structure from configurations.
        
        Creates objects, morphisms, and analyzes the functor F: C → Set
        mapping field configurations to complexity values.
        """
        # Create category objects
        self._create_objects()
        
        # Create morphisms between objects
        self._create_morphisms()
        
        # Analyze symmetry structure
        self._analyze_symmetry_groups()
        
        # Analyze complexity distribution
        self._analyze_complexity_distribution()
    
    def _create_objects(self) -> None:
        """Create category objects from field configurations."""
        for i, (config, complexity) in enumerate(zip(self.configurations, self.complexity_values)):
            # Extract configuration data
            g_type = config['g_type']
            phi_vector = config.get('phi', None)
            
            # Create category object
            obj = CategoryObject(
                node_id=i,
                g_type=g_type,
                complexity=complexity,
                phi_vector=phi_vector
            )
            
            # Store object
            self.objects[i] = obj
            
            # Add to graph with enhanced attributes
            self.graph.add_node(
                i,
                g_type=g_type,
                complexity=complexity,
                functor_image=complexity,
                symmetry_class=f"g_{g_type}",
                object_type="field_configuration"
            )
    
    def _create_morphisms(self) -> None:
        """
        Create morphisms representing transformations between objects.
        
        Morphisms encode the categorical structure of symmetry operations
        and field transformations that preserve or relate complexity values.
        """
        # Intra-group morphisms (within same symmetry class)
        self._create_intra_group_morphisms()
        
        # Inter-group morphisms (between different symmetry classes)
        self._create_inter_group_morphisms()
        
        # Complexity-based morphisms
        self._create_complexity_morphisms()
    
    def _create_intra_group_morphisms(self) -> None:
        """Create morphisms within the same symmetry group."""
        # Group objects by symmetry type
        symmetry_groups = {}
        for obj_id, obj in self.objects.items():
            g_type = obj.g_type
            if g_type not in symmetry_groups:
                symmetry_groups[g_type] = []
            symmetry_groups[g_type].append(obj_id)
        
        # Create morphisms within each group
        for g_type, obj_ids in symmetry_groups.items():
            for i, source_id in enumerate(obj_ids):
                for j, target_id in enumerate(obj_ids):
                    if i != j:
                        # Calculate morphism weight based on complexity similarity
                        source_complexity = self.objects[source_id].complexity
                        target_complexity = self.objects[target_id].complexity
                        weight = 1.0 / (1.0 + abs(source_complexity - target_complexity))
                        
                        # Create morphism
                        morphism = CategoryMorphism(
                            source=source_id,
                            target=target_id,
                            morphism_type=f"intra_symmetry_g{g_type}",
                            transformation="symmetry_preserving",
                            weight=weight
                        )
                        
                        self.morphisms.append(morphism)
                        self.graph.add_edge(
                            source_id, target_id,
                            morphism_type=morphism.morphism_type,
                            transformation=morphism.transformation,
                            weight=weight
                        )
                        self.morphism_count += 1
    
    def _create_inter_group_morphisms(self) -> None:
        """Create morphisms between different symmetry groups."""
        g_types = list(set(obj.g_type for obj in self.objects.values()))
        
        for g_type1 in g_types:
            for g_type2 in g_types:
                if g_type1 != g_type2:
                    # Find representative objects from each group
                    group1_objs = [obj_id for obj_id, obj in self.objects.items() if obj.g_type == g_type1]
                    group2_objs = [obj_id for obj_id, obj in self.objects.items() if obj.g_type == g_type2]
                    
                    if group1_objs and group2_objs:
                        # Create morphism between group representatives
                        source_id = group1_objs[0]
                        target_id = group2_objs[0]
                        
                        # Weight based on complexity relationship
                        source_complexity = self.objects[source_id].complexity
                        target_complexity = self.objects[target_id].complexity
                        weight = 0.5 / (1.0 + abs(source_complexity - target_complexity))
                        
                        morphism = CategoryMorphism(
                            source=source_id,
                            target=target_id,
                            morphism_type=f"inter_symmetry_g{g_type1}_g{g_type2}",
                            transformation="symmetry_breaking",
                            weight=weight
                        )
                        
                        self.morphisms.append(morphism)
                        self.graph.add_edge(
                            source_id, target_id,
                            morphism_type=morphism.morphism_type,
                            transformation=morphism.transformation,
                            weight=weight
                        )
                        self.morphism_count += 1
    
    def _create_complexity_morphisms(self) -> None:
        """Create morphisms based on complexity value relationships."""
        # Sort objects by complexity
        sorted_objects = sorted(self.objects.items(), key=lambda x: x[1].complexity)
        
        # Create morphisms between objects with similar complexity
        complexity_threshold = 0.1
        
        for i, (obj_id1, obj1) in enumerate(sorted_objects):
            for j, (obj_id2, obj2) in enumerate(sorted_objects[i+1:], i+1):
                complexity_diff = abs(obj1.complexity - obj2.complexity)
                
                if complexity_diff < complexity_threshold:
                    weight = 1.0 - (complexity_diff / complexity_threshold)
                    
                    morphism = CategoryMorphism(
                        source=obj_id1,
                        target=obj_id2,
                        morphism_type="complexity_similarity",
                        transformation="complexity_preserving",
                        weight=weight
                    )
                    
                    self.morphisms.append(morphism)
                    self.graph.add_edge(
                        obj_id1, obj_id2,
                        morphism_type=morphism.morphism_type,
                        transformation=morphism.transformation,
                        weight=weight
                    )
                    self.morphism_count += 1
    
    def _analyze_symmetry_groups(self) -> None:
        """Analyze symmetry group structure of the category."""
        for obj_id, obj in self.objects.items():
            g_type = obj.g_type
            if g_type not in self.symmetry_groups:
                self.symmetry_groups[g_type] = []
            self.symmetry_groups[g_type].append(obj_id)
    
    def _analyze_complexity_distribution(self) -> None:
        """Analyze distribution of complexity values across the category."""
        for obj in self.objects.values():
            # Discretize complexity for distribution analysis
            complexity_bin = round(obj.complexity, 2)
            if complexity_bin not in self.complexity_distribution:
                self.complexity_distribution[complexity_bin] = 0
            self.complexity_distribution[complexity_bin] += 1
    
    def compute_categorical_limit(self) -> CategoricalLimit:
        """
        Compute the categorical limit F(C) of the functor F: C → Set.
        
        This implements the core categorical analysis to find universal
        structures underlying consciousness emergence.
        
        Returns:
            CategoricalLimit object containing the limit structure
        """
        # Find fixed points of the functor F
        fixed_points = self._find_functor_fixed_points()
        
        # Compute limit cone structure
        limit_cone = self._compute_limit_cone()
        
        # Find apex object (universal object)
        apex_object = self._find_apex_object(fixed_points)
        
        # Analyze universal properties
        universal_properties = self._analyze_universal_properties()
        
        return CategoricalLimit(
            apex_object=apex_object,
            limit_cone=limit_cone,
            universal_properties=universal_properties,
            fixed_points=fixed_points
        )
    
    def _find_functor_fixed_points(self) -> List[CategoryObject]:
        """
        Find fixed points of functor F where F(x) exhibits stability.
        
        Returns:
            List of objects that are stable under the functor
        """
        fixed_points = []
        
        for obj_id, obj in self.objects.items():
            # Calculate stability based on morphism structure
            stability = self._calculate_object_stability(obj_id)
            
            # Update object with stability measure
            obj.stability = stability
            
            # Consider object a fixed point if highly stable
            if stability > 0.7:  # High stability threshold
                fixed_points.append(obj)
        
        return fixed_points
    
    def _calculate_object_stability(self, obj_id: int) -> float:
        """
        Calculate stability of an object under morphisms.
        
        Args:
            obj_id: ID of object to analyze
            
        Returns:
            Stability measure between 0 and 1
        """
        obj_complexity = self.objects[obj_id].complexity
        
        # Get complexities of connected objects
        connected_complexities = []
        for neighbor in self.graph.neighbors(obj_id):
            neighbor_complexity = self.objects[neighbor].complexity
            connected_complexities.append(neighbor_complexity)
        
        if not connected_complexities:
            return 1.0  # Isolated objects are maximally stable
        
        # Calculate stability as inverse of complexity variance
        mean_complexity = np.mean(connected_complexities)
        std_complexity = np.std(connected_complexities)
        
        if std_complexity < 1e-10:
            return 1.0  # Perfect stability
        
        stability = 1.0 / (1.0 + std_complexity / (abs(mean_complexity) + 1e-6))
        return min(stability, 1.0)
    
    def _compute_limit_cone(self) -> List[CategoryMorphism]:
        """
        Compute the limit cone structure over F(C).
        
        Returns:
            List of morphisms forming the limit cone
        """
        # Find the most central object as potential apex
        centrality = nx.degree_centrality(self.graph)
        if not centrality:
            return []
        
        apex_candidate = max(centrality, key=centrality.get)
        
        # Create cone morphisms from apex to all other objects
        cone_morphisms = []
        for obj_id in self.objects:
            if obj_id != apex_candidate:
                morphism = CategoryMorphism(
                    source=apex_candidate,
                    target=obj_id,
                    morphism_type="limit_cone_projection",
                    transformation="universal_projection",
                    weight=centrality[apex_candidate]
                )
                cone_morphisms.append(morphism)
        
        return cone_morphisms
    
    def _find_apex_object(self, fixed_points: List[CategoryObject]) -> Optional[CategoryObject]:
        """
        Find the apex object of the categorical limit.
        
        Args:
            fixed_points: List of fixed point objects
            
        Returns:
            The apex object, or None if not found
        """
        if not fixed_points:
            return None
        
        # Choose the fixed point with highest stability and centrality
        best_score = -1
        apex_object = None
        
        for obj in fixed_points:
            centrality = nx.degree_centrality(self.graph).get(obj.node_id, 0)
            score = obj.stability * centrality
            
            if score > best_score:
                best_score = score
                apex_object = obj
        
        return apex_object
    
    def _analyze_universal_properties(self) -> Dict[str, Any]:
        """
        Analyze universal properties of the categorical structure.
        
        Returns:
            Dictionary of universal properties
        """
        return {
            "symmetry_groups": len(self.symmetry_groups),
            "morphism_density": self.morphism_count / max(1, len(self.objects)**2),
            "complexity_coherence": self._measure_complexity_coherence(),
            "categorical_dimension": self._estimate_categorical_dimension(),
            "connectivity": nx.is_weakly_connected(self.graph),
            "clustering_coefficient": nx.average_clustering(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else 0,
            "diameter": nx.diameter(self.graph) if nx.is_weakly_connected(self.graph) else float('inf')
        }
    
    def _measure_complexity_coherence(self) -> float:
        """
        Measure coherence of complexity values across the category.
        
        Returns:
            Coherence measure between 0 and 1
        """
        if len(self.complexity_values) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        mean_complexity = np.mean(self.complexity_values)
        std_complexity = np.std(self.complexity_values)
        
        if mean_complexity < 1e-10:
            return 1.0
        
        cv = std_complexity / mean_complexity
        coherence = 1.0 / (1.0 + cv)
        
        return coherence
    
    def _estimate_categorical_dimension(self) -> int:
        """
        Estimate the categorical dimension of the structure.
        
        Returns:
            Estimated dimension
        """
        # Use number of symmetry groups as a proxy for dimension
        return len(self.symmetry_groups)
    
    def get_functor_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of the functor F: C → Set.
        
        Returns:
            Dictionary containing functor analysis results
        """
        categorical_limit = self.compute_categorical_limit()
        
        return {
            "functor_name": "F: C → Set",
            "domain_category": {
                "objects": len(self.objects),
                "morphisms": len(self.morphisms),
                "symmetry_groups": self.symmetry_groups
            },
            "codomain_set": {
                "complexity_values": self.complexity_values,
                "value_range": (min(self.complexity_values), max(self.complexity_values)),
                "distribution": dict(self.complexity_distribution)
            },
            "categorical_limit": {
                "apex_object": categorical_limit.apex_object,
                "fixed_points": len(categorical_limit.fixed_points),
                "limit_cone_size": len(categorical_limit.limit_cone),
                "universal_properties": categorical_limit.universal_properties
            },
            "graph_properties": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "connected_components": nx.number_weakly_connected_components(self.graph),
                "is_connected": nx.is_weakly_connected(self.graph)
            }
        }
    
    def __repr__(self) -> str:
        """String representation of categorical structure."""
        return (
            f"CategoricalStructure(objects={len(self.objects)}, "
            f"morphisms={len(self.morphisms)}, "
            f"symmetry_groups={len(self.symmetry_groups)})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Category C with Functor F: C → Set\n"
            f"Objects: {len(self.objects)} field configurations\n"
            f"Morphisms: {len(self.morphisms)} transformations\n"
            f"Symmetry groups: {len(self.symmetry_groups)}\n"
            f"Complexity range: [{min(self.complexity_values):.3f}, {max(self.complexity_values):.3f}]"
        )


# Example usage and testing
if __name__ == '__main__':
    print("🔗 Categorical Structure Analysis - Consciousness Framework")
    print("=" * 65)
    
    # Create sample field configurations for testing
    np.random.seed(42)
    num_configs = 20
    
    configurations = []
    complexity_values = []
    
    for i in range(num_configs):
        g_type = np.random.choice([0, 1])
        phi_vector = np.random.randn(4) + 1j * np.random.randn(4)
        phi_vector = phi_vector / np.linalg.norm(phi_vector)
        
        config = {
            'g_type': g_type,
            'phi': phi_vector
        }
        
        # Generate complexity value with some structure
        complexity = 0.5 + 0.3 * g_type + 0.2 * np.random.randn()
        complexity = max(0, complexity)  # Ensure non-negative
        
        configurations.append(config)
        complexity_values.append(complexity)
    
    print(f"Created {num_configs} field configurations")
    print(f"Complexity range: [{min(complexity_values):.3f}, {max(complexity_values):.3f}]")
    print()
    
    # Create categorical structure
    print("Building categorical structure...")
    categorical_structure = CategoricalStructure(configurations, complexity_values)
    
    print("Categorical Structure:")
    print(categorical_structure)
    print()
    
    # Compute categorical limit
    print("Computing categorical limit F(C)...")
    categorical_limit = categorical_structure.compute_categorical_limit()
    
    print(f"Fixed points found: {len(categorical_limit.fixed_points)}")
    print(f"Limit cone size: {len(categorical_limit.limit_cone)}")
    
    if categorical_limit.apex_object:
        apex = categorical_limit.apex_object
        print(f"Apex object: node_{apex.node_id} (g_type={apex.g_type}, "
              f"complexity={apex.complexity:.3f}, stability={apex.stability:.3f})")
    print()
    
    # Get comprehensive functor analysis
    print("Functor F: C → Set Analysis:")
    functor_analysis = categorical_structure.get_functor_analysis()
    
    for key, value in functor_analysis.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    print()
    
    print("✓ Categorical structure analysis complete") 