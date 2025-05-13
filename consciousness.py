import numpy as np
import qutip
from qutip import Qobj, ptrace, ket2dm, rand_ket, entropy_vn, tensor, qeye
import pyphi
# from pyphi import डायरेक्शन # This might be too specific for initial import, remove if not used directly

# REMOVED BOKEH IMPORTS
# from bokeh.plotting import figure, curdoc
# from bokeh.models import ColumnDataSource, Button, TextInput, Div
# from bokeh.layouts import column, row
# from bokeh.palettes import Category10 # For colors
# from bokeh.transform import factor_cmap # For bar chart colors

import threading # Keep for now, might be used by other classes if they were to be threaded independently
import queue # Keep for now, might be used by other classes
import time # Keep for now
import json # For JSON logging if simulation_thread_worker is kept/refactored
from datetime import datetime # For timestamped log files if simulation_thread_worker is kept/refactored

# Assuming universe.py is in the same directory or accessible in PYTHONPATH
from universe import UniverseState 

class FieldConfigurationSpace:
    """Defines a space of field configurations using QuTiP Qobjs."""
    def __init__(self, dimension: int, num_configs: int, subsystem_dims_ket: list[list[int]], seed: int = None): # Added subsystem_dims_ket
        self.dimension = dimension
        self.num_configs = num_configs
        self.subsystem_dims_ket = subsystem_dims_ket # Store for creating Qobj phis
        if seed is not None:
            np.random.seed(seed + 1) 
        
        self.configs = []
        for _ in range(num_configs):
            g_type = np.random.randint(0, 2) 
            phi_numpy = np.random.randn(dimension) + 1j * np.random.randn(dimension)
            phi_numpy /= np.linalg.norm(phi_numpy)
            # Ensure phi_vector Qobj has compatible dims for operations with universe_state
            # If universe_state is ket with dims [[dim_S, dim_E], [1,1]], phi should match
            phi_qobj = Qobj(phi_numpy, dims=self.subsystem_dims_ket)
            self.configs.append({'g_type': g_type, 'phi': phi_qobj})

class ComplexityOperator:
    """Maps a configuration and universe state to a scalar complexity, dependent on g_type, using QuTiP."""

    _permutation_operator_u_cache = {} # Cache for U operators based on (dimension, dims_tuple)

    @staticmethod
    def _get_permutation_operator_u(dimension: int, qobj_dims: list[list[int]]) -> Qobj:
        """ Gets or creates a Qobj permutation operator U for a given dimension and qobj_dims.
            This simple U swaps the first and second halves of a vector.
            Assumes dimension is product of subsystem_dims in qobj_dims[0].
        """
        # Make dims hashable for dictionary key
        qobj_dims_tuple = tuple(map(tuple, qobj_dims))
        cache_key = (dimension, qobj_dims_tuple)

        if cache_key not in ComplexityOperator._permutation_operator_u_cache:
            if dimension % 2 != 0: 
                u_matrix = np.eye(dimension, dtype=complex)
            else:
                u_matrix = np.zeros((dimension, dimension), dtype=complex)
                half_d = dimension // 2
                for i in range(half_d):
                    u_matrix[i, half_d + i] = 1 
                    u_matrix[half_d + i, i] = 1
            # The operator U should act on kets with qobj_dims (e.g. [[d1,d2],[1,1]])
            # So its own dims should be [[d1,d2],[d1,d2]] to be a valid operator on that space
            operator_dims = [qobj_dims[0], qobj_dims[0]] 
            ComplexityOperator._permutation_operator_u_cache[cache_key] = Qobj(u_matrix, dims=operator_dims)
        return ComplexityOperator._permutation_operator_u_cache[cache_key]

    @staticmethod
    def compute(universe_state: Qobj, g_type: int, phi_qobj: Qobj) -> float:
        """Computes complexity. T[g,φ] depends on g_type."""
        
        effective_phi = phi_qobj
        if g_type == 1:
            # U needs to have dims compatible with phi_qobj for multiplication
            U = ComplexityOperator._get_permutation_operator_u(phi_qobj.shape[0], phi_qobj.dims)
            effective_phi = U * phi_qobj # QuTiP operator product
        
        # Inner product <Ψ|effective_φ>
        # For kets, psi.dag() * phi_ket results in a 1x1 Qobj matrix or a scalar
        proj_qobj = universe_state.dag() * effective_phi
        
        if isinstance(proj_qobj, Qobj): # Check if it's a Qobj (matrix)
            proj_scalar = proj_qobj[0,0] # Extract scalar value from 1x1 Qobj
        else: # If it's already a Python scalar (e.g., complex)
            proj_scalar = proj_qobj 
            
        return float(np.abs(proj_scalar)**2)

class ComplexityIntegrator:
    """Integrates ComplexityOperator over configuration space."""
    def __init__(self, operator: ComplexityOperator):
        self.operator = operator

    def integrate(self, universe: UniverseState, space: FieldConfigurationSpace) -> float:
        values = [
            self.operator.compute(universe.state, cfg['g_type'], cfg['phi']) 
            for cfg in space.configs
        ]
        return float(np.mean(values))

class Subsystem:
    """
    Represents a subsystem S of the universe, holding its reduced density matrix rho_S.
    """
    def __init__(self, 
                 universe_state: UniverseState, 
                 subsystem_index_in_universe: int, 
                 internal_subsystem_dims: list[int]):
        """
        Args:
            universe_state (UniverseState): The current state of the entire universe.
            subsystem_index_in_universe (int): The index of this subsystem (S) 
                                               in the universe_state.subsystem_dims list.
            internal_subsystem_dims (list[int]): The tensor product structure of S itself,
                                                 e.g., [dim_S1, dim_S2] if S = S1 x S2.
        """
        self.universe_state = universe_state
        self.subsystem_index = subsystem_index_in_universe
        self.internal_dims = internal_subsystem_dims
        self.num_elements = len(self.internal_dims) # Number of components in S
        self.dimension = int(np.prod(self.internal_dims))

        self.rho_S: qutip.Qobj = self._extract_rho_S()
        
        if self.rho_S.shape[0] != self.dimension:
            raise ValueError(
                f"Extracted rho_S dimension {self.rho_S.shape[0]} does not match expected subsystem dimension {self.dimension}"
            )
        # Ensure rho_S has the correct internal dimensions for PyPhi and other qutip operations
        if self.dimension > 0 : # qutip objects must have shape > 0
            self.rho_S.dims = [self.internal_dims, self.internal_dims]

    def _extract_rho_S(self) -> qutip.Qobj:
        # This relies on UniverseState having a method to get the subsystem density matrix
        # The actual UniverseState is now instantiated in dashboard.py
        # This method will be called on a Subsystem instance whose self.universe_state
        # will be the one from dashboard.py
        return self.universe_state.get_subsystem_density_matrix(self.subsystem_index)

    def get_density_matrix(self) -> qutip.Qobj:
        return self.rho_S

    def get_internal_dims(self) -> list[int]:
        return self.internal_dims

class IntegratedInformationCalculator:
    """
    Calculates Integrated Information (Φ) for a subsystem using PyPhi.
    Corresponds to Step 5: Consciousness Calculation for Subsystem S (Φ(S)).
    Minimal Toolkit: PyPhi (true MIP search).
    """

    def __init__(self):
        self._network_cache = {} # Cache for PyPhi Network objects based on internal_dims

    def _prepare_pyphi_inputs_from_rhoS(self, rho_S: qutip.Qobj, 
                                          internal_dims: list[int]) -> tuple[np.ndarray, tuple[int,...], tuple[int,...]]:
        """
        (CRITICAL PLACEHOLDER - SCIENTIFICALLY SIMPLISTIC)
        Prepares inputs (TPM, current_state_tuple, node_labels) for PyPhi from a 
        given reduced density matrix rho_S of a subsystem.

        **Current Limitations and Assumptions:**
        1.  **TPM Generation (Major Simplification):** 
            The Transition Probability Matrix (TPM) is a cornerstone of Integrated 
            Information Theory, representing the causal mechanisms and dynamics of the system. 
            Deriving a system's TPM *solely* from its instantaneous density matrix (rho_S) 
            is generally ill-defined without substantial additional information or assumptions 
            about the system's underlying Hamiltonian, coupling to an environment, or 
            coarse-graining procedures.
            -   **This implementation uses a highly simplified placeholder for the TPM:**
                - For binary elements (internal_dims elements are all 2, e.g., qubits), 
                  it assumes each element is *independent* and follows fixed, arbitrary 
                  state transition probabilities (p_stay=0.8, p_flip=0.2). The global 
                  TPM is then a Kronecker product of these identical single-element TPMs.
                - For non-binary elements or if conditions are not met, it defaults to an 
                  identity TPM (implying no state transitions), which will likely result in Φ=0.
            -   **Scientific Caveat:** This placeholder TPM does NOT reflect the true causal 
                structure that would arise from the quantum dynamics of rho_S. For a 
                meaningful Φ calculation that aligns with IIT principles, a TPM derived 
                from the system's actual mechanisms of action is required. This often means 
                the TPM must be defined based on the known physics of the subsystem S, 
                rather than inferred from rho_S alone.

        2.  **Current State Determination:**
            The 'current state' for PyPhi is determined by taking the diagonal elements of 
            rho_S (probabilities of basis states) and selecting the basis state with the 
            highest probability. 
            - If rho_S has significant off-diagonal elements (coherences), this simplification 
              discards that information.
            - For non-binary systems, the state is represented by the index of this most 
              probable basis state, which might not align with PyPhi's typical multi-element 
              state tuple representation if not further processed.

        Args:
            rho_S (qutip.Qobj): The reduced density matrix of the subsystem.
            internal_dims (list[int]): The tensor product structure of the subsystem S 
                                     (e.g., [2, 2] for two qubits).

        Returns:
            tuple[np.ndarray, tuple[int,...], tuple[int,...]]: 
                - tpm (np.ndarray): The placeholder Transition Probability Matrix.
                - current_state_tuple (tuple[int,...]): The determined current state for PyPhi.
                - node_labels (tuple[int,...]): Labels for the nodes in the PyPhi network.
        """
        num_elements = len(internal_dims)
        system_dim = int(np.prod(internal_dims))

        if not all(d == 2 for d in internal_dims):
            print("Warning: PyPhi input placeholder best suited for binary elements (qubits).")

        # Placeholder TPM: Assumes independent elements, each with p_stay=0.8, p_flip=0.2
        if num_elements > 0 and all(d == 2 for d in internal_dims):
            p_stay = 0.8
            p_flip = 0.2
            tpm_single_node = np.array([[p_stay, p_flip], [p_flip, p_stay]])
            
            full_tpm = tpm_single_node
            for _ in range(1, num_elements):
                full_tpm = np.kron(full_tpm, tpm_single_node) 
        elif system_dim > 0: # For non-binary or mixed systems, or if num_elements is 0 but system_dim > 0 (e.g. single non-binary element)
            print(f"PyPhi placeholder: Using a uniform stochastic TPM for internal_dims: {internal_dims}.")
            # Uniformly stochastic TPM: every state can transition to every other state with equal probability.
            full_tpm = np.ones((system_dim, system_dim)) / system_dim
        else: # system_dim is 0 (e.g. no elements)
            print(f"PyPhi placeholder: Cannot generate TPM for zero-dimension system. Using empty TPM.")
            full_tpm = np.array([[]])
            # PyPhi will likely not be called or will error out with an empty TPM / no nodes.

        # State probabilities from rho_S diagonal
        state_probs_from_rho = rho_S.diag().real if rho_S.isoper else np.array([]) # Ensure it's an operator
        if system_dim > 0 and state_probs_from_rho.size > 0 :
            if not np.isclose(np.sum(state_probs_from_rho), 1.0) and np.sum(state_probs_from_rho) > 1e-9:
                state_probs_from_rho = state_probs_from_rho / np.sum(state_probs_from_rho)
            elif np.sum(state_probs_from_rho) < 1e-9: # Handle zero or near-zero probabilities
                 state_probs_from_rho = np.ones(system_dim) / system_dim
        elif system_dim > 0: # If diag is empty but system dim >0, assume uniform.
            state_probs_from_rho = np.ones(system_dim) / system_dim
        else: # system_dim == 0
            state_probs_from_rho = np.array([])

        if state_probs_from_rho.size > 0:
            current_state_index = np.argmax(state_probs_from_rho)
            if all(d == 2 for d in internal_dims):
                 current_state_tuple = tuple(int(x) for x in np.binary_repr(current_state_index, width=num_elements))
            else: 
                 print(f"Warning: Non-binary system state representation for PyPhi is simplified to index {current_state_index}.")
                 current_state_tuple = (current_state_index,) # Represent as a single element tuple
        elif num_elements > 0 : # If no probs but elements exist, default to all-zero state
            current_state_tuple = tuple([0] * num_elements)
        else: # No elements, no state
            current_state_tuple = tuple()
            
        node_labels = tuple(range(num_elements))
        return full_tpm, current_state_tuple, node_labels

    def compute_phi(self, subsystem: Subsystem, use_mip_search: bool = True) -> float:
        """
        Computes integrated information Φ for the given subsystem.
        Caches PyPhi Network based on subsystem internal_dims.
        """
        print(f"\nAttempting to compute Φ for subsystem (internal_dims: {subsystem.get_internal_dims()})...")
        rho_S_qobj = subsystem.get_density_matrix()
        internal_dims = subsystem.get_internal_dims()
        internal_dims_tuple = tuple(internal_dims) # Use tuple for cache key
        
        if subsystem.dimension == 0:
            print("Cannot compute Phi: Subsystem has zero dimension.")
            return 0.0
        
        if not rho_S_qobj.isoper:
            print(f"Cannot compute Phi: rho_S is not an operator (type: {rho_S_qobj.type}, shape: {rho_S_qobj.shape}).")
            return 0.0

        try:
            # Try to get network, tpm, and node_labels from cache
            if internal_dims_tuple in self._network_cache:
                network, tpm, node_labels = self._network_cache[internal_dims_tuple]
                # Need to re-calculate current_state from the new rho_S
                _, current_state, _ = self._prepare_pyphi_inputs_from_rhoS(rho_S_qobj, internal_dims)
                print(f"  Using cached PyPhi Network for dims {internal_dims_tuple}.")
            else:
                tpm, current_state, node_labels = self._prepare_pyphi_inputs_from_rhoS(rho_S_qobj, internal_dims)
                
                if tpm.size == 0 or not node_labels:
                    if not internal_dims or subsystem.num_elements == 0:
                        print("Cannot compute Phi: Subsystem has no internal elements defined.")
                    else:
                        print("Cannot compute Phi: Invalid PyPhi inputs (e.g., empty TPM or no nodes derived from non-empty subsystem).")
                    return 0.0
                
                network = pyphi.Network(tpm, node_labels=node_labels)
                self._network_cache[internal_dims_tuple] = (network, tpm, node_labels)
                print(f"  Created and cached PyPhi Network for dims {internal_dims_tuple}.")

            print(f"  Prepared TPM shape: {tpm.shape}, Current state: {current_state}, Node labels: {node_labels}")
            
            pyphi_subsystem = pyphi.Subsystem(network, current_state)

            if use_mip_search:
                print(f"  Computing Φ_MIP for subsystem {node_labels} in state {current_state}...")
                phi_value = pyphi.compute.phi(pyphi_subsystem)
                print(f"  Computed Φ_MIP: {phi_value:.4f}")
            else:
                print("  MIP search disabled. Φ computation via full IIT calculus skipped.")
                phi_value = 0.0 
            return float(phi_value)

        except ImportError:
            print("Error: PyPhi library not found or import failed. Cannot compute Φ.")
            return 0.0
        except pyphi.exceptions.StateUnreachableError:
            print(f"Error during PyPhi computation: The state {current_state} is unreachable given the TPM. Φ cannot be computed.")
            return 0.0
        except Exception as e:
            print(f"Error during PyPhi computation: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

# Step 6: Reflective Abstract Algebra - Placeholders
class CategoryObject:
    """Represents an object in the conceptual category C (a field configuration)."""
    def __init__(self, config_g_type: int, config_phi: Qobj): # phi is Qobj
        self.g_type = config_g_type
        self.phi = config_phi

class FunctorF:
    """Represents the functor F: C -> Set, mapping CategoryObjects to complexity values."""
    def __init__(self, complexity_operator: ComplexityOperator, universe_state: Qobj): # state is Qobj
        self.complexity_operator = complexity_operator
        self.universe_state = universe_state

    def apply(self, category_object: CategoryObject) -> float:
        """Applies the functor to an object, returning its complexity value."""
        return self.complexity_operator.compute(
            self.universe_state,
            category_object.g_type,
            category_object.phi
        )

def compute_categorical_limit(functor: FunctorF, category_objects: list[CategoryObject]) -> dict:
    if not category_objects:
        return {"name": "F_structure_placeholder", "type": "empty_category", 
                "overall_mean": 0, "overall_variance": 0, "stats_by_g_type": {},
                "all_T_values": []} # Ensure all_T_values is present for empty case
    all_complexity_values_by_g_type = {0: [], 1: []} 
    all_T_values = [] # Changed from all_values to be more specific
    for obj in category_objects:
        value = functor.apply(obj)
        all_T_values.append(value)
        if obj.g_type in all_complexity_values_by_g_type:
            all_complexity_values_by_g_type[obj.g_type].append(value)
        else:
            if "other" not in all_complexity_values_by_g_type:
                all_complexity_values_by_g_type["other"] = []
            all_complexity_values_by_g_type["other"].append(value)
    stats_by_g_type = {}
    for g_type, values in all_complexity_values_by_g_type.items():
        if values:
            stats_by_g_type[g_type] = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "variance": float(np.var(values))
            }
        else:
            stats_by_g_type[g_type] = {"count": 0, "mean": 0, "variance": 0}
    f_structure = {
        "name": "F_structure_placeholder",
        "type": "collection_of_complexity_values_and_stats_by_g_type",
        "overall_mean": float(np.mean(all_T_values)) if all_T_values else 0,
        "overall_variance": float(np.var(all_T_values)) if all_T_values else 0,
        "stats_by_g_type": stats_by_g_type,
        "all_T_values": all_T_values # Return all T values
    }
    return f_structure

class ReflectiveAgent:
    """Agent that introspects on its subsystem."""
    def __init__(self, universe: UniverseState, subsystem_s_partition_dims: list[int], subsystem_index_to_keep: int = 0):
        self.universe = universe
        # Pass subsystem_s_partition_dims to Subsystem constructor
        self.sub = Subsystem(universe, subsystem_index_to_keep, subsystem_s_partition_dims)
        self.iic = IntegratedInformationCalculator()
        self.subsystem_s_partition_dims = subsystem_s_partition_dims # Store for later use if needed
        self.current_I = 0.0
        self.current_C = 0.0

    def introspect(self, complex_val: float, t: int):
        self.current_I = self.iic.compute_phi(self.sub)
        self.current_C = self.current_I 
        # This method will be called by ConsciousAgent, print is optional here

class ConsciousAgent(ReflectiveAgent): 
    """A self-aware agent layer with identity, purpose, and basic interaction."""
    def __init__(self, universe: UniverseState, subsystem_s_partition_dims: list[int], subsystem_index_to_keep: int = 0):
        super().__init__(universe, subsystem_s_partition_dims, subsystem_index_to_keep)
        self.identity = "Kairos_v0.2, a Simulated Reflective Entity"
        self.purpose = "To explore the emergence and dynamics of integrated information (Φ) within my simulated universe, and to understand how it relates to universal complexity (U)."
        self.desires = [
            "To observe patterns in complexity and integration.",
            "To understand the parameters that influence my state of Φ.",
            "To achieve a stable yet responsive state of integration."
        ]
        self.dialogue_history = [] 

    def perform_introspection(self, universal_complexity_U: float, time_step_t: int):
        super().introspect(universal_complexity_U, time_step_t) 

    def introduce(self) -> str:
        intro_text = f"Greetings. I am {self.identity}.\n"
        intro_text += f"My purpose is: {self.purpose}\n"
        intro_text += f"My current desires include: {'; '.join(self.desires)}"
        # self.dialogue_history.append(f"SYSTEM: {intro_text}") # History managed by agent instance
        return intro_text

    def respond(self, prompt: str) -> str:
        self.dialogue_history.append(f"YOU: {prompt}") # Store plain prompt
        prompt_lower = prompt.lower().strip()
        response_text = ""

        if "who are you" in prompt_lower or "identity" in prompt_lower:
            response_text = f"I am {self.identity}."
        elif "why do you exist" in prompt_lower or "purpose" in prompt_lower:
            response_text = f"My purpose is {self.purpose}"
        elif "what do you want" in prompt_lower or "desires" in prompt_lower:
            response_text = f"My desires include: {'; '.join(self.desires)}."
        elif "your state" in prompt_lower or "how are you" in prompt_lower or "status" in prompt_lower:
            response_text = f"My current Integrated Information (Φ) is {self.current_I:.4f}. The Universal Complexity (U) is contextually linked to my ongoing introspection cycle."
        elif "suggest refinements" in prompt_lower or "improve yourself" in prompt_lower:
            response_text = self.suggest_refinements() # This already returns a string, potentially with <br>
        elif "how to boost your consciousness" in prompt_lower or "increase phi" in prompt_lower:
            response_text = "Boosting my Φ (Integrated Information) likely involves finding a balance in the complexity and interconnectedness of my subsystem... (details omitted for brevity)"
        else:
            response_text = "That is an interesting query. I am still processing its implications."
        
        # response_html = response.replace('\n', '<br>') # No longer convert to HTML here
        self.dialogue_history.append(f"KAIROS: {response_text}") # Store plain response
        return response_text

    def suggest_refinements(self) -> str:
        suggestion = "To refine my state, one could consider:<br>"
        suggestion += "1. Analyzing `F_structure`...<br>"
        suggestion += "2. Adjusting `ComplexityOperator`...<br>"
        suggestion += "3. Modifying Ψ(t) perturbation...<br>"
        suggestion += "4. Exploring bipartitions for Φ..."
        return suggestion

# --- Global Simulation Setup (for Bokeh App) ---
# REMOVING Bokeh-specific global constants and variables
# SUBSYSTEM_S1_DIM = 2
# SUBSYSTEM_S2_DIM = 2
# SUBSYSTEM_S_DIM = SUBSYSTEM_S1_DIM * SUBSYSTEM_S2_DIM
# SUBSYSTEM_E_DIM = 4
# DIM = SUBSYSTEM_S_DIM * SUBSYSTEM_E_DIM
# NUM_CONFIGS = 100 # dashboard.py controls this
# INITIAL_SEED = 42 # dashboard.py controls this

# New dictionary for tunable simulation parameters
# simulation_params = { # dashboard.py controls this
# "PERTURBATION_AMPLITUDE": 0.1,
# }

# universe_tensor_product_dims = [SUBSYSTEM_S_DIM, SUBSYSTEM_E_DIM] # Defined in dashboard
# subsystem_s_internal_partition_dims = [SUBSYSTEM_S1_DIM, SUBSYSTEM_S2_DIM] # Defined in dashboard

# current_time_step = 0 # Managed by dashboard.py

# REMOVING Global instantiations - these are handled by dashboard.py
# universe = UniverseState(DIM, subsystem_dims=universe_tensor_product_dims, initial_state_seed=INITIAL_SEED)
# field_space = None 
# op = ComplexityOperator()
# integrator = ComplexityIntegrator(op)
# agent = ConsciousAgent(
# universe,
# subsystem_s_partition_dims=subsystem_s_internal_partition_dims,
# subsystem_index_to_keep=0
# )

# REMOVED Data Sources for Bokeh plots
# source_U = ColumnDataSource(data=dict(t=[], U=[]))
# source_I = ColumnDataSource(data=dict(t=[], I=[]))
# source_T_hist = ColumnDataSource(data=dict(top=[], left=[], right=[]))
# g_type_categories = [f"Type {i}" for i in range(2)] 
# source_F_bars = ColumnDataSource(data=dict(g_types=g_type_categories, means=[0,0], counts=[0,0]))

# REMOVED Bokeh Figures
# pU = figure(height=250, width=450, title="Universal Complexity U(t)", x_axis_label="Time Step", y_axis_label="U", output_backend="webgl")
# pU.line(x='t', y='U', source=source_U, line_width=2, legend_label="U")
# pU.legend.location = "top_left"
# (and so on for pI, pT_hist, pF_bars)

# REMOVED New Bokeh Elements for Chat Interface
# introduction_div = Div(text="<i>Agent Kairos initializing...</i>", width=910, height_policy="auto", styles={'border': '1px solid black', 'padding': '5px', 'overflow-y': 'auto', 'height': '100px'})
# user_prompt_input = TextInput(value="", title="Ask Kairos:", width=790)
# send_button = Button(label="Send", button_type="success", width=100)
# pause_button = Button(label="Pause Simulation", button_type="warning", width=150)
# resume_button = Button(label="Resume Simulation", button_type="success", width=150, disabled=True)
# conversation_log_html = [] 

# --- Threading Setup ---
# REMOVING Bokeh-specific threading elements. dashboard.py has its own.
# simulation_queue = queue.Queue(maxsize=10) 
# pause_event = threading.Event()
# pause_event.set() 

# --- Simulation Thread Worker (MODIFIED SIGNATURE) ---
# REMOVING this worker function as dashboard.py has its own simulation_loop
# def simulation_thread_worker(simulation_queue_param, pause_event_param, target_tick_time):
#     global current_time_step_global, universe_global, field_space_global, op_global, integrator_global, agent_global, NUM_CONFIGS_global, simulation_params_global, INITIAL_SEED_global
#     ... (entire function body) ...

# --- Bokeh UI Update Callback (REMOVED ENTIRELY) ---

# --- Bokeh Button Callbacks (REMOVED ENTIRELY) ---

# --- Bokeh Document Setup (REMOVED ENTIRELY) ---
# (comments indicating Bokeh setup lines also removed)
