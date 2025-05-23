import networkx as nx
import numpy as np
# from fields import FieldConfigurationSpace # If needed for explicit config types
# from complexity import get_qutip_symmetry_operator # If needed to define actions

class CategoricalStructure:
    """
    Represents the categorical structure F: C -> Set (Step 6).
    Enhanced implementation aligned with mission.txt requirements.
    
    Objects in category C are field configurations (g, φ).
    Morphisms represent symmetry operations and field transformations.
    The functor F maps these objects to complexity values T[g,φ].
    
    Mission Alignment: "Represent each local configuration as an object in category C.
    Define a functor F mapping these objects to complexity values.
    Compute inverse limit (or suitable limit) F(C) to find universal structure F."
    """
    def __init__(self, configurations: list[dict], complexity_values: list[float]):
        """
        Args:
            configurations (list[dict]): List of {'g_type': str, 'phi': np.ndarray}.
                                         Each phi is a node.
            complexity_values (list[float]): List of T[g,φ] values corresponding to each configuration.
                                            These are the values F maps to.
        """
        if len(configurations) != len(complexity_values):
            raise ValueError("Number of configurations must match number of complexity values.")

        self.graph = nx.DiGraph() # Directed graph
        self.configurations = configurations
        self.complexity_values = complexity_values
        self.node_to_config_idx = {}
        
        # Enhanced categorical structure tracking
        self.morphism_count = 0
        self.symmetry_groups = {}
        self.complexity_distribution = {}

        self._build_enhanced_graph()

    def _build_enhanced_graph(self):
        """
        Enhanced graph building with categorical structure tracking.
        Implements proper functor F: C -> Set mapping.
        """
        for i, config in enumerate(self.configurations):
            # Use index as node ID for simplicity, store actual config data as attributes
            node_id = i 
            self.node_to_config_idx[node_id] = i
            
            # Enhanced node attributes for categorical analysis
            complexity_val = self.complexity_values[i]
            g_type = config['g_type']
            
            self.graph.add_node(node_id, 
                                g_type=g_type, 
                                complexity=complexity_val,
                                functor_image=complexity_val,  # F(object) = complexity value
                                symmetry_class=f"g_{g_type}")
            
            # Track symmetry groups and complexity distribution
            if g_type not in self.symmetry_groups:
                self.symmetry_groups[g_type] = []
            self.symmetry_groups[g_type].append(node_id)
            
            # Discretize complexity for distribution analysis
            complexity_bin = round(complexity_val, 2)
            if complexity_bin not in self.complexity_distribution:
                self.complexity_distribution[complexity_bin] = 0
            self.complexity_distribution[complexity_bin] += 1
        
        # Add morphisms based on symmetry relationships
        self._add_symmetry_morphisms()

    def _add_symmetry_morphisms(self):
        """
        Add morphisms representing symmetry transformations between configurations.
        Enhanced implementation of categorical morphisms.
        """
        # Add morphisms within symmetry groups (identity-like transformations)
        for g_type, node_list in self.symmetry_groups.items():
            for i, node1 in enumerate(node_list):
                for j, node2 in enumerate(node_list):
                    if i != j:
                        # Morphism representing transformation within same symmetry class
                        self.graph.add_edge(node1, node2, 
                                          morphism_type=f"symmetry_g{g_type}",
                                          transformation="intra_group")
                        self.morphism_count += 1
        
        # Add morphisms between different symmetry groups
        for g_type1, nodes1 in self.symmetry_groups.items():
            for g_type2, nodes2 in self.symmetry_groups.items():
                if g_type1 != g_type2:
                    # Sample morphism between groups
                    if nodes1 and nodes2:
                        self.graph.add_edge(nodes1[0], nodes2[0],
                                          morphism_type=f"inter_g{g_type1}_g{g_type2}",
                                          transformation="inter_group")
                        self.morphism_count += 1

    def compute_F_structure_enhanced(self) -> dict:
        """
        Enhanced computation of the categorical limit F_structure.
        
        Mission Alignment: "Compute inverse limit (or suitable limit) F(C) to find universal structure F"
        
        This implements a sophisticated approximation to the categorical limit
        by analyzing the functor F's behavior across the category structure.
        """
        if not self.configurations:
            return {"name": "F_structure_enhanced", "type": "empty_category", 
                    "categorical_limit": None, "universal_structure": None}

        # 1. Analyze functor F's fixed points and stable structures
        fixed_points = self._find_functor_fixed_points()
        
        # 2. Compute limit cone over F(C)
        limit_cone = self._compute_limit_cone()
        
        # 3. Analyze universal property emergence
        universal_properties = self._analyze_universal_properties()
        
        # 4. Category-theoretic invariants
        categorical_invariants = self._compute_categorical_invariants()
        
        # 5. Enhanced F-structure representing the categorical limit
        f_structure_enhanced = {
            "name": "F_structure_categorical_limit",
            "type": "enhanced_categorical_analysis",
            
            # Core categorical limit structure
            "categorical_limit": {
                "fixed_points": fixed_points,
                "limit_cone": limit_cone,
                "universal_properties": universal_properties
            },
            
            # Universal structure emergence
            "universal_structure": {
                "symmetry_groups": len(self.symmetry_groups),
                "morphism_density": self.morphism_count / max(1, len(self.configurations)**2),
                "complexity_coherence": self._measure_complexity_coherence(),
                "categorical_dimension": self._estimate_categorical_dimension()
            },
            
            # Mathematical invariants
            "categorical_invariants": categorical_invariants,
            
            # Complexity distribution over the category
            "complexity_landscape": {
                "distribution": dict(self.complexity_distribution),
                "symmetry_complexity_map": self._map_symmetry_to_complexity(),
                "functor_image_statistics": self._analyze_functor_image()
            },
            
            # Graph-theoretic properties
            "graph_properties": {
                "num_objects": self.graph.number_of_nodes(),
                "num_morphisms": self.graph.number_of_edges(),
                "connected_components": nx.number_weakly_connected_components(self.graph),
                "clustering_coefficient": nx.average_clustering(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else 0
            }
        }
        
        return f_structure_enhanced

    def _find_functor_fixed_points(self) -> list:
        """Find configurations where F exhibits stable behavior (fixed points)"""
        fixed_points = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            complexity = node_data['complexity']
            
            # A configuration is a "fixed point" if its complexity is stable
            # under morphisms (represented by similar complexity in connected nodes)
            connected_complexities = []
            for neighbor in self.graph.neighbors(node_id):
                neighbor_complexity = self.graph.nodes[neighbor]['complexity']
                connected_complexities.append(neighbor_complexity)
            
            if connected_complexities:
                stability = 1.0 - np.std(connected_complexities) / (np.mean(connected_complexities) + 1e-6)
                if stability > 0.8:  # High stability threshold
                    fixed_points.append({
                        'node_id': node_id,
                        'complexity': complexity,
                        'g_type': node_data['g_type'],
                        'stability': stability
                    })
        
        return fixed_points

    def _compute_limit_cone(self) -> dict:
        """
        Compute approximation to the categorical limit cone over F(C).
        
        In category theory, a limit cone consists of:
        1. An apex object
        2. Projection morphisms to each object in the diagram  
        3. Universal property: unique factorization through the apex
        
        This computes an approximation by finding the "most universal" complexity value.
        """
        all_complexities = [data['complexity'] for _, data in self.graph.nodes(data=True)]
        
        if not all_complexities:
            return {"apex": 0, "projections": [], "coherence": 0, "universality": 0}
        
        # Find the complexity value that minimizes variance to all others
        # This approximates the "universal" object in the categorical sense
        min_variance = float('inf')
        best_apex = 0
        
        for candidate_apex in all_complexities:
            variance = np.var([abs(candidate_apex - comp) for comp in all_complexities])
            if variance < min_variance:
                min_variance = variance
                best_apex = candidate_apex
        
        # Projections from apex to each object (categorical morphisms)
        projections = [abs(best_apex - comp) for comp in all_complexities]
        
        # Coherence: how well the universal property holds
        # (lower projection variance = better universal property)
        coherence = 1.0 / (1.0 + np.std(projections)) if projections else 0.0
        
        # Universality measure: how many objects are "close" to the apex
        close_threshold = np.std(all_complexities) * 0.5 if len(all_complexities) > 1 else 0.1
        universal_count = sum(1 for proj in projections if proj <= close_threshold)
        universality = universal_count / len(projections) if projections else 0.0
        
        return {
            "apex": best_apex,
            "projections": projections,
            "coherence": coherence,
            "universality": universality,
            "projection_variance": np.var(projections) if projections else 0.0
        }

    def _analyze_universal_properties(self) -> dict:
        """Analyze emergence of universal properties in the categorical structure"""
        properties = {}
        
        # Universal property: uniqueness of morphisms to/from certain objects
        if self.graph.number_of_nodes() > 0:
            in_degrees = dict(self.graph.in_degree())
            out_degrees = dict(self.graph.out_degree())
            
            # Find "universal" objects (high connectivity)
            max_in_degree = max(in_degrees.values()) if in_degrees else 0
            max_out_degree = max(out_degrees.values()) if out_degrees else 0
            
            universal_objects = []
            for node_id in self.graph.nodes():
                if (in_degrees.get(node_id, 0) > 0.7 * max_in_degree or 
                    out_degrees.get(node_id, 0) > 0.7 * max_out_degree):
                    universal_objects.append(node_id)
            
            properties["universal_objects"] = universal_objects
            properties["universality_measure"] = len(universal_objects) / max(1, self.graph.number_of_nodes())
        
        return properties

    def _compute_categorical_invariants(self) -> dict:
        """Compute category-theoretic invariants"""
        invariants = {}
        
        if self.graph.number_of_nodes() > 0:
            # Euler characteristic approximation
            V = self.graph.number_of_nodes()  # Objects
            E = self.graph.number_of_edges()  # Morphisms
            invariants["euler_characteristic"] = V - E
            
            # Homology-like measures
            cycles = list(nx.simple_cycles(self.graph))
            invariants["cycle_count"] = len(cycles)
            invariants["acyclicity"] = len(cycles) == 0
            
            # Categorical dimension (rough approximation)
            invariants["categorical_dimension"] = self._estimate_categorical_dimension()
        
        return invariants

    def _estimate_categorical_dimension(self) -> int:
        """Estimate the 'dimension' of the category based on morphism chains"""
        if self.graph.number_of_nodes() == 0:
            return 0
            
        # Approximate dimension as the longest path length
        try:
            longest_path = max(nx.dag_longest_path_length(self.graph) 
                             if nx.is_directed_acyclic_graph(self.graph) else 1,
                             1)
        except:
            longest_path = 1
            
        return min(longest_path, 5)  # Cap at reasonable dimension

    def _measure_complexity_coherence(self) -> float:
        """Measure how coherently complexity is distributed across symmetry groups"""
        if not self.symmetry_groups:
            return 0.0
            
        group_complexities = {}
        for g_type, nodes in self.symmetry_groups.items():
            complexities = [self.graph.nodes[node]['complexity'] for node in nodes]
            if complexities:
                group_complexities[g_type] = np.mean(complexities)
        
        if len(group_complexities) < 2:
            return 1.0
            
        # Coherence is inversely related to variance between groups
        group_means = list(group_complexities.values())
        coherence = 1.0 / (1.0 + np.var(group_means))
        return coherence

    def _map_symmetry_to_complexity(self) -> dict:
        """Map each symmetry type to its characteristic complexity"""
        symmetry_complexity_map = {}
        
        for g_type, nodes in self.symmetry_groups.items():
            complexities = [self.graph.nodes[node]['complexity'] for node in nodes]
            if complexities:
                symmetry_complexity_map[f"g_type_{g_type}"] = {
                    "mean": np.mean(complexities),
                    "std": np.std(complexities),
                    "count": len(complexities)
                }
        
        return symmetry_complexity_map

    def _analyze_functor_image(self) -> dict:
        """Analyze the image of functor F: C -> Set"""
        all_complexities = [data['complexity'] for _, data in self.graph.nodes(data=True)]
        
        if not all_complexities:
            return {"range": 0, "density": 0, "coverage": 0}
        
        return {
            "range": max(all_complexities) - min(all_complexities),
            "mean": np.mean(all_complexities),
            "density": len(set(np.round(all_complexities, 3))) / len(all_complexities),
            "coverage": len(set(np.round(all_complexities, 2))),
            "entropy": -sum(p * np.log(p + 1e-10) for p in np.histogram(all_complexities, bins=10)[0]/len(all_complexities) if p > 0)
        }

    # Keep the original method for backward compatibility
    def compute_F_structure_proxy(self) -> dict:
        """Original method - now delegates to enhanced version"""
        enhanced_result = self.compute_F_structure_enhanced()
        
        # Extract compatible format for backward compatibility
        simplified_result = {
            "name": "F_structure_proxy_v3_enhanced",
            "type": "enhanced_categorical_with_legacy_compat",
            "overall_mean_complexity": enhanced_result["complexity_landscape"]["functor_image_statistics"]["mean"],
            "overall_variance_complexity": enhanced_result["complexity_landscape"]["functor_image_statistics"].get("variance", 0),
            "stats_by_g_type": enhanced_result["complexity_landscape"]["symmetry_complexity_map"],
            "num_graph_nodes": enhanced_result["graph_properties"]["num_objects"],
            "num_graph_edges": enhanced_result["graph_properties"]["num_morphisms"],
            "num_weakly_connected_components": enhanced_result["graph_properties"]["connected_components"],
            "categorical_limit_info": enhanced_result["categorical_limit"],
            "universal_structure_measures": enhanced_result["universal_structure"]
        }
        
        return simplified_result

# Example Usage:
if __name__ == '__main__':
    # Sample data (mimicking output from other modules)
    sample_configs = [
        {'g_type': "identity", 'phi': np.array([1,0,0,0], dtype=complex)}, 
        {'g_type': "half_swap", 'phi': np.array([0,0,1,0], dtype=complex)},
        {'g_type': "identity", 'phi': np.array([0,1,0,0], dtype=complex)},
        {'g_type': "phase_flip_S1", 'phi': np.array([1,0,0,0], dtype=complex)*1j},
        {'g_type': "half_swap", 'phi': np.array([0,0,0,1], dtype=complex)}
    ]
    # Corresponding T[g,φ] values (these would come from complexity.py)
    sample_T_values = [0.8, 0.5, 0.7, 0.9, 0.4]

    cat_struct = CategoricalStructure(configurations=sample_configs, complexity_values=sample_T_values)
    
    # At this point, graph has nodes but no edges from add_morphism_edges (it's a placeholder)
    print(f"Graph has {cat_struct.graph.number_of_nodes()} nodes and {cat_struct.graph.number_of_edges()} edges.")

    f_structure_result = cat_struct.compute_F_structure_proxy()
    print("\nF_structure Proxy Results:")
    for key, value in f_structure_result.items():
        if key == "stats_by_g_type":
            print(f"  {key}:")
            for g, stats in value.items():
                print(f"    {g}: {stats}")
        else:
            print(f"  {key}: {value}")

    # To make it more interesting, one would need a proper `s_operator_func` and `phi_subsystem_dims`
    # to call `cat_struct.add_morphism_edges(...)` if that method was fully implemented.
    # For example, if using get_qutip_symmetry_operator from complexity.py:
    # from complexity import get_qutip_symmetry_operator, PHI_SUBSYSTEM_DIMS_EXAMPLE # (assuming it's defined)
    # cat_struct.add_morphism_edges(get_qutip_symmetry_operator) # This would need phi_subsystem_dims too.
    # f_structure_result_with_edges = cat_struct.compute_F_structure_proxy()
    # print("\nF_structure Proxy Results (after attempting to add edges):")
    # ... (print again) 