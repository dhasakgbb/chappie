import qutip
import numpy as np

class UniverseState:
    """
    Manages the quantum state of the universe using QuTiP.
    Corresponds to Step 1: Represent Universe State Ψ(t).
    Minimal Toolkit: QuTiP (Qobj, mesolve/sesolve for dynamics).
    """
    def __init__(self, dimension: int, initial_state_seed: int = None, subsystem_dims: list[int] = None):
        """
        Initializes Ψ(t) in Hilbert space H.

        Args:
            dimension (int): Total dimension of the Hilbert space.
            initial_state_seed (int, optional): Seed for random state generation. Defaults to None.
            subsystem_dims (list[int], optional): 
                Dimensions for tensor product structure, e.g., [dim_S, dim_E]. 
                This is crucial for defining the state's dims attribute for QuTiP operations
                like ptrace. If dimension is a product of subsystem_dims, these will be used.
                The ket dims will be [[d1, d2, ...], [1, 1, ...]].
        """
        self.dimension = dimension
        self.initial_state_seed = initial_state_seed

        if subsystem_dims:
            if np.prod(subsystem_dims) != dimension:
                raise ValueError(
                    f"Product of subsystem_dims {subsystem_dims} must equal dimension {dimension}"
                )
            self.subsystem_dims_ket = [subsystem_dims, [1] * len(subsystem_dims)]
            self.subsystem_dims_dm = [subsystem_dims, subsystem_dims]
        else:
            # If no subsystem_dims, treat as a single system
            self.subsystem_dims_ket = [[dimension], [1]]
            self.subsystem_dims_dm = [[dimension], [dimension]]
            
        self.state: qutip.Qobj = self._prepare_initial_state()

    def _prepare_initial_state(self) -> qutip.Qobj:
        """
        Prepares an initial random normalized ket state.
        """
        if self.initial_state_seed is not None:
            np.random.seed(self.initial_state_seed)
        
        # Create a random complex vector
        vec_real = np.random.randn(self.dimension)
        vec_imag = np.random.randn(self.dimension)
        vec = vec_real + 1j * vec_imag
        normalized_vec = vec / np.linalg.norm(vec)
        
        initial_ket = qutip.Qobj(normalized_vec, dims=self.subsystem_dims_ket)
        return initial_ket

    def get_state(self) -> qutip.Qobj:
        """Returns the current quantum state Ψ(t)."""
        return self.state

    def set_state(self, new_state: qutip.Qobj):
        """
        Sets the current quantum state. Ensures new_state has compatible dims.
        """
        if not isinstance(new_state, qutip.Qobj):
            raise TypeError("New state must be a QuTiP Qobj.")
        if new_state.shape[0] != self.dimension or not new_state.isket:
            raise ValueError(
                f"New state must be a ket of dimension {self.dimension}, got shape {new_state.shape}"
            )
        # It's also good practice to ensure the dims attribute matches
        new_state.dims = self.subsystem_dims_ket
        self.state = new_state

    def perturb_state(self, amplitude: float, seed: int = None):
        """
        Applies a random perturbation to the current state and re-normalizes.
        This is a simple evolution mechanism for now.
        
        Args:
            amplitude (float): Strength of the perturbation.
            seed (int, optional): Seed for this specific perturbation.
        """
        if seed is not None:
            np.random.seed(seed)

        current_np_state = self.state.full().flatten() # Get as NumPy array
        perturbation_np = amplitude * (
            np.random.randn(self.dimension) + 1j * np.random.randn(self.dimension)
        )
        perturbed_np_state = current_np_state + perturbation_np
        
        # Normalize
        norm = np.linalg.norm(perturbed_np_state)
        if norm < 1e-9: # Avoid division by zero if state collapses to zero
            print("Warning: Perturbed state norm is close to zero. Re-initializing to random state.")
            self.state = self._prepare_initial_state() # Fallback
        else:
            perturbed_normalized_np = perturbed_np_state / norm
            self.state = qutip.Qobj(perturbed_normalized_np, dims=self.subsystem_dims_ket)

    def evolve_step_unitary(self, H_operator: qutip.Qobj, dt: float):
        """
        Evolves the state by one step under a unitary evolution: U = exp(-i H dt).
        Requires H_operator to be a Qobj.
        (Placeholder for sesolve/mesolve if complex dynamics are needed)
        """
        if not isinstance(H_operator, qutip.Qobj) or not H_operator.isoper:
            raise ValueError("H_operator must be a QuTiP operator Qobj.")
        
        # Simple unitary evolution U = (-1j * H_operator * dt).expm()
        # This is a basic implementation. For more complex time-dependent H or Lindblad, use qutip.sesolve or qutip.mesolve
        U = (-1j * H_operator * dt).expm()
        self.state = (U * self.state).unit() # Apply and re-normalize

    def get_density_matrix(self) -> qutip.Qobj:
        """Returns the density matrix ρ(t) = |Ψ(t)><Ψ(t)|."""
        dm = qutip.ket2dm(self.state)
        dm.dims = self.subsystem_dims_dm # Ensure correct tensor product dims
        return dm

    def get_subsystem_density_matrix(self, k: int) -> qutip.Qobj:
        """
        Computes the reduced density matrix for the k-th subsystem by tracing out others.
        Requires self.state.dims to be set correctly for a tensor product structure.
        
        Args:
            k (int): The index of the subsystem to keep (0-indexed).
        
        Returns:
            qutip.Qobj: The reduced density matrix of the k-th subsystem.
        """
        # self.subsystem_dims_ket[0] holds the list of dimensions of tensor components, e.g., [dim_S, dim_E]
        num_components = len(self.subsystem_dims_ket[0]) if self.subsystem_dims_ket and self.subsystem_dims_ket[0] else 0

        if num_components == 0:
            raise ValueError("UniverseState has no subsystem dimensions defined (subsystem_dims_ket[0] is empty or None).")
        
        if k < 0 or k >= num_components:
            raise ValueError(f"Subsystem index {k} out of bounds for {num_components} defined subsystem(s).")

        rho_universe = self.get_density_matrix() # This sets rho_universe.dims to self.subsystem_dims_dm

        if num_components == 1:
            # If there's only one component defined for the universe (e.g., dims are [[d1],[1]]),
            # and k must be 0 (already checked by bounds), then the "subsystem" is the entire universe.
            # No ptrace is needed or meaningful. rho_universe is already the state of this single component.
            return rho_universe 
        else:
            # Multiple components, proceed with ptrace.
            # rho_universe.dims is already set by get_density_matrix() to self.subsystem_dims_dm,
            # which is in the correct format for ptrace (e.g., [[d1, d2], [d1, d2]]).
            rho_S_traced = rho_universe.ptrace(k)
            return rho_S_traced

# Example usage (for testing or if run directly)
if __name__ == '__main__':
    # Example: A 2-qubit system (S) and a 2-level environment (E)
    # Total dimension = 2 * 2 * 2 = 8
    # Subsystem S is composed of two qubits [S1, S2] with dims [2,2]
    # Environment E has dim [2]
    # For UniverseState, subsystem_dims should represent the top-level tensor structure,
    # e.g., H_S x H_E. So, dim_S = 2*2 = 4.
    dim_S1 = 2
    dim_S2 = 2
    dim_S_total = dim_S1 * dim_S2 # Dimension of the main subsystem we might be interested in
    dim_E = 2 # Dimension of an "environment"
    
    total_dim = dim_S_total * dim_E # Total dimension for the universe state

    # UniverseState sees H_S_total and H_E as its primary tensor components
    uni_state = UniverseState(dimension=total_dim, 
                              initial_state_seed=42, 
                              subsystem_dims=[dim_S_total, dim_E])
    
    print("Initial Universe State Ψ(t):")
    print(uni_state.get_state())
    print(f"State dimensions: {uni_state.get_state().dims}")
    print(f"Is ket: {uni_state.get_state().isket}")
    print(f"Norm: {uni_state.get_state().norm()}")

    uni_state.perturb_state(amplitude=0.1, seed=101)
    print("\nPerturbed Universe State Ψ(t):")
    print(uni_state.get_state())
    print(f"Norm after perturbation: {uni_state.get_state().norm()}")

    # Get density matrix of the first subsystem (dim_S_total)
    rho_S_total = uni_state.get_subsystem_density_matrix(0)
    print("\nReduced Density Matrix for S_total (ptrace over E):")
    print(rho_S_total)
    print(f"rho_S_total dimensions: {rho_S_total.dims}")
    print(f"rho_S_total trace: {rho_S_total.tr()}")

    # If S_total itself is a tensor product, e.g., S1 x S2, and we want rho_S1 from rho_S_total
    # We need to ensure rho_S_total has the correct internal dims for further ptrace
    # rho_S_total currently has dims [[dim_S_total], [dim_S_total]] from the perspective of the ptrace
    # To trace out S2 from S_total = S1 x S2, we need rho_S_total to have dims [[dim_S1, dim_S2], [dim_S1, dim_S2]]
    
    # This part demonstrates further processing if needed, but UniverseState focuses on Ψ(t)
    # and its top-level subsystems. The new 'consciousness.py' will handle detailed subsystem IIT.
    if dim_S_total == dim_S1 * dim_S2:
        rho_S_total_internal_dims = [[dim_S1, dim_S2], [dim_S1, dim_S2]]
        rho_S_total.dims = rho_S_total_internal_dims # Re-assign dims for internal structure of S_total
        
        rho_S1 = rho_S_total.ptrace(0) # Trace out S2 to get S1
        print("\nReduced Density Matrix for S1 (ptrace over S2 from rho_S_total):")
        print(rho_S1)
        print(f"rho_S1 dimensions: {rho_S1.dims}")
        print(f"rho_S1 trace: {rho_S1.tr()}")
        
    # Example of unitary evolution (simple Hamiltonian: SigmaZ on first part of S_total, identity on rest)
    # Assuming S_total is the first system in universe's [dim_S_total, dim_E] structure.
    # H_S_total component: e.g. sigmaz() on S1, identity on S2 (if S_total = S1 x S2)
    # If dim_S_total = 4, dim_S1=2, dim_S2=2:
    if dim_S_total == 4 and dim_S1 == 2 and dim_S2 == 2 :
        H_S1_local = qutip.sigmaz()
        H_S_total_internal = qutip.tensor(H_S1_local, qutip.qeye(dim_S2))
        
        # Full Hamiltonian for the universe: H_S_total_internal on S_total, Identity on E
        H_universe = qutip.tensor(H_S_total_internal, qutip.qeye(dim_E))
        H_universe.dims = uni_state.subsystem_dims_dm # Operator dims [[dStot,dE],[dStot,dE]]
        
        print("\nApplying simple unitary evolution...")
        print(f"Hamiltonian H_universe dims: {H_universe.dims}")
        uni_state.evolve_step_unitary(H_universe, dt=0.1)
        print("State after evolution:")
        print(uni_state.get_state())
        print(f"Norm: {uni_state.get_state().norm()}") 