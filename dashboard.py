import panel as pn
import panel.widgets as pnw
import numpy as np
import holoviews as hv
from holoviews.streams import Pipe, Buffer
import time
import threading # For running simulation loop in a separate thread
import queue # For thread-safe data passing from sim thread to UI

# Import from our new modules
from universe import UniverseState
from fields import FieldConfigurationSpace
from complexity import compute_universal_complexity_U, get_qutip_symmetry_operator # For op_cache if needed by category later
from consciousness import Subsystem, IntegratedInformationCalculator
from category import CategoricalStructure

pn.extension(sizing_mode="stretch_width")
hv.extension('bokeh') # Use Bokeh backend for HoloViews

# --- Global Simulation Parameters & State ---
SIM_RUNNING = True
SIM_PAUSED = False
TARGET_TICK_TIME_S = 0.2 # Target time per simulation step (5 Hz)
UI_UPDATE_INTERVAL_MS = 100 # How often UI tries to update (10 Hz)

# --- Initial System Configuration ---
# Subsystem S (e.g., 2 qubits S1xS2)
s1_dim, s2_dim = 2, 2
subsystem_S_internal_dims = [s1_dim, s2_dim] # Corrected typo here
subsystem_S_dim = int(np.prod(subsystem_S_internal_dims))

# Environment E (can be trivial for now)
e_dim = 1 

# Universe U = S x E
universe_total_dim = subsystem_S_dim * e_dim
# For UniverseState, subsystem_dims are its top-level tensor components.
# If E is trivial, universe_tensor_dims might just be [subsystem_S_dim].
universe_tensor_dims = [subsystem_S_dim, e_dim] if e_dim > 1 else [subsystem_S_dim]

# Phi configurations space (matches subsystem S typically for T = |<Ψ_S|S[g]φ>|^2 type calcs, or Ψ must be projected)
# For the current complexity.py, dim(Ψ) must equal dim(φ).
# For T JAX version, phi_config_dim is universe_total_dim.
# For old T QuTiP version, phi_config_dim could be subsystem_S_dim if Ψ was projected first.
# Assuming phi_config_dim matches dimension of states used in T calculation, which is universe_total_dim for JAX T.
phi_config_dim = universe_total_dim 
phi_config_subsystem_dims = universe_tensor_dims # Match structure of phi to universe state for JAX T


# --- Simulation Objects Initialization ---
current_time_step = 0
universe = UniverseState(dimension=universe_total_dim, 
                        initial_state_seed=42,
                        subsystem_dims=universe_tensor_dims)

# Field Configs (will be updated by slider)
initial_num_configs = 50
# available_g_types list removed - FieldConfigurationSpace should generate g_types (0 or 1) internally
field_space = FieldConfigurationSpace(dimension=phi_config_dim, 
                                      num_configs=initial_num_configs, 
                                      # g_types_to_sample=[0,1] # Or similar if FieldConfigurationSpace supports it
                                      phi_seed=101,
                                      # Ensure FieldConfigurationSpace generates phi with correct dims for JAX T:
                                      phi_subsystem_dims_override=universe_tensor_dims 
                                      )


# Complexity related
qutip_operator_cache = {} # Cache for S[g] operators (used by JAX version of T)

# Consciousness related (Subsystem S is the first component of the universe)
subsystem_S_instance = Subsystem(universe_state=universe, 
                                 subsystem_index_in_universe=0, # Assuming S is the first part
                                 internal_subsystem_dims=subsystem_S_internal_dims)
iit_calculator = IntegratedInformationCalculator()

# --- Data Streams for Plots (using HoloViews Pipe for dynamic data) ---
# Pipe for U(t) and Phi(t) time series
uts_pipe = Pipe(data={"t": [], "U": [], "Phi": []})
# Buffer for T-value histogram (stores last N T-values)
# Initialize with an empty 2D array (0 rows, 1 column)
t_values_buffer = Buffer(data=np.empty((0,1)), length=1000, index=False) # Stores raw T values
# Pipe for F-structure bar chart (e.g., mean T per g_type)
f_structure_pipe = Pipe(data={"g_type": [], "mean_T": []})

# --- Panel Widgets ---
run_button = pnw.Button(name="Pause Simulation ▶", button_type="primary")
num_configs_slider = pnw.IntSlider(name="Number of φ Samples (M)", start=10, end=500, step=10, value=initial_num_configs)
perturb_amp_slider = pnw.FloatSlider(name="Perturbation Amplitude Ψ(t)", start=0.0, end=1.0, step=0.01, value=0.1)
status_text = pn.pane.Markdown("Simulation Idle...")

# --- Plotting Functions (using HoloViews) ---
@pn.depends(uts_pipe.param.data)
def plot_U_Phi_time_series(data):
    if not data["t"]:
        # Return empty plots with titles if no data
        return (hv.Curve([], 't', 'U').opts(title="U(t)", active_tools=['wheel_zoom']) * 
                hv.Curve([], 't', 'Phi').opts(title="Φ(t)", active_tools=['wheel_zoom'])
               ).opts(responsive=True, height=300, legend_position='top_left')
    curve_U = hv.Curve((data["t"], data["U"]), 't', 'U', label='U(t)')
    curve_Phi = hv.Curve((data["t"], data["Phi"]), 't', 'I(Φ)', label='Φ(t)') # Changed label for clarity
    return (curve_U * curve_Phi).opts(
        responsive=True, height=300, legend_position='top_left', 
        title="Time Series: Universal Complexity U(t) & Integrated Information Φ(t)",
        active_tools=['wheel_zoom']
    )

@pn.depends(t_values_buffer.param.data)
def plot_T_histogram(data_buffer):
    if data_buffer.size == 0:
        return hv.Histogram([]).opts(title="T[g,φ] Value Distribution", active_tools=['wheel_zoom'])
    # Calculate histogram from the buffered T values
    hist_counts, hist_edges = np.histogram(data_buffer, bins=20, range=(0, max(0.1, data_buffer.max()) if data_buffer.size >0 else 0.1) )
    return hv.Histogram((hist_edges[:-1], hist_edges[1:], hist_counts)).opts(
        responsive=True, height=300, title="T[g,φ] Value Distribution", 
        xlabel="T value", ylabel="Frequency", active_tools=['wheel_zoom']
    )

@pn.depends(f_structure_pipe.param.data)
def plot_F_structure_bars(data):
    if not data["g_type"]: # Check if g_type list is empty
        return hv.Bars([], 'g_type', 'mean_T').opts(title="Mean T by g_type", active_tools=['wheel_zoom'])
    # Ensure data for Bars is a dictionary of lists or a list of dictionaries
    # HoloViews Bars expects data in a format like {'g_type': ['Type 0', 'Type 1'], 'mean_T': [0.5, 0.6]}
    # or a list of records: [{'g_type': 'Type 0', 'mean_T': 0.5}, ...]
    
    # Convert g_type keys if they are integers to strings for categorical axis
    # plot_data = {'g_type': [str(g) for g in data['g_type']], 'mean_T': data['mean_T']}
    
    return hv.Bars(data, 'g_type', 'mean_T').opts(
        responsive=True, height=300, title="F-structure: Mean T by g_type", 
        xrotation=45, active_tools=['wheel_zoom'], shared_axes=False
    )

# --- Simulation Loop (to be run in a thread) ---
sim_queue = queue.Queue(maxsize=10) # Queue for passing data from sim thread to UI update

def simulation_loop():
    global current_time_step, universe, field_space, subsystem_S_instance, SIM_RUNNING, SIM_PAUSED
    global qutip_operator_cache 

    while SIM_RUNNING:
        if SIM_PAUSED:
            time.sleep(0.1) # Sleep briefly if paused
            continue

        loop_start_time = time.time()
        current_time_step += 1

        # 1. Evolve/Perturb Universe State Ψ(t)
        # The JAX complexity function expects qutip_ops_cache, phi_subsystem_dims_for_S_operators
        universe.perturb_state(amplitude=perturb_amp_slider.value, seed=current_time_step)
        current_psi_qobj = universe.get_state() # This should be a Qobj

        # 2. Get Field Configurations M, μ
        if field_space.num_configs != num_configs_slider.value:
            field_space.resample_configurations(
                num_configs=num_configs_slider.value, 
                phi_seed=current_time_step + 1000,
                # Pass phi_subsystem_dims_override again if needed by resample_configurations
                phi_subsystem_dims_override=universe_tensor_dims 
            )
        configurations = field_space.get_configurations() # List of dicts: {'g_type': int, 'phi_jax': jnp.ndarray, 'phi_qobj': Qobj}

        # 3. Compute Universal Complexity U(t) & T-values
        # compute_universal_complexity_U is from complexity.py (JAX version)
        # It needs: psi_qobj, field_configurations_list, qutip_ops_cache, phi_subsystem_dims_for_S_operators
        # The 'phi_subsystem_dims_for_S_operators' should match the structure of phi_qobj in configurations,
        # which we set to universe_tensor_dims.
        U_val, all_T_vals, T_by_g = compute_universal_complexity_U(
            psi_qobj=current_psi_qobj, 
            field_configurations_list=configurations, 
            qutip_ops_cache=qutip_operator_cache,
            phi_subsystem_dims_for_S_operators=universe_tensor_dims # Used by get_qutip_symmetry_operator
        )

        # 4. Update Subsystem S and get rho_S
        # Subsystem S is tied to the universe state. Re-init or update.
        subsystem_S_instance = Subsystem(universe, 
                                         subsystem_index_in_universe=0, 
                                         internal_subsystem_dims=subsystem_S_internal_dims)

        # 5. Compute Integrated Information Φ(S)
        phi_S_val = iit_calculator.compute_phi(subsystem_S_instance, use_mip_search=True)

        # 6. Compute F_structure proxy from category.py
        # CategoricalStructure expects list of configs and list of T_values
        # The configs for CategoricalStructure should ideally just be {'g_type': ..., 'phi_id': ...}
        # to avoid storing full phi vectors in NetworkX if they are large.
        # For now, pass simplified configs if possible or ensure CategoricalStructure handles it.
        # T_by_g gives dict {g_type: [T_values]}, all_T_vals is flat list.
        
        # Create simplified configs for CategoricalStructure (e.g., using index as ID)
        simple_configs_for_cat = [{'g_type': cfg['g_type'], 'id': i} for i, cfg in enumerate(configurations)]
        cat_struct = CategoricalStructure(configurations=simple_configs_for_cat, complexity_values=all_T_vals)
        f_proxy = cat_struct.compute_F_structure_proxy() 
        
        # Prepare data packet for UI
        ui_data_packet = {
            "t": current_time_step,
            "U": U_val,
            "Phi": phi_S_val,
            "all_T_vals": np.array(all_T_vals), 
            "T_by_g": T_by_g, 
            "f_proxy_overall_mean": f_proxy.get("overall_mean_complexity", 0),
            "status": f"Step {current_time_step}: U={U_val:.3f}, Φ={phi_S_val:.3f}"
        }

        try:
            sim_queue.put_nowait(ui_data_packet)
        except queue.Full:
            pass 
        
        elapsed_time = time.time() - loop_start_time
        sleep_time = TARGET_TICK_TIME_S - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
    print("Simulation loop ended.")

# --- UI Update Function (Periodic Callback) ---
def update_ui():
    try:
        data = sim_queue.get_nowait()
        
        new_uts_data = {"t": uts_pipe.data["t"] + [data["t"]], 
                        "U": uts_pipe.data["U"] + [data["U"]], 
                        "Phi": uts_pipe.data["Phi"] + [data["Phi"]]}
        uts_pipe.send(new_uts_data)

        if "all_T_vals" in data and data["all_T_vals"].size > 0:
            # Buffer expects a new complete dataset, not just new points.
            # Reshape 1D array to 2D array with one column.
            t_vals_np = np.array(data["all_T_vals"])
            if t_vals_np.ndim == 1:
                t_vals_np = t_vals_np.reshape(-1, 1)
            t_values_buffer.send(t_vals_np) 

        # Prepare data for F-structure bar chart (mean T per g_type)
        # T_by_g is {g_type_int: [values]}. Convert g_type_int to string for categorical plot.
        g_types_str = [str(g) for g in data["T_by_g"].keys()]
        mean_T_values = [np.mean(data["T_by_g"][g]) if data["T_by_g"][g] else 0 for g in data["T_by_g"].keys()]
        
        # Ensure g_types_str is not empty before sending to pipe if T_by_g could be empty
        if g_types_str:
             f_structure_pipe.send({"g_type": g_types_str, "mean_T": mean_T_values})
        
        status_text.object = data["status"]

    except queue.Empty:
        pass 
    except Exception as e:
        status_text.object = f"UI Update Error: {str(e)[:100]}" # Display error snippet
        print(f"UI Update Error: {e}")
        import traceback
        traceback.print_exc()

# --- Widget Callbacks ---
def toggle_simulation(event):
    global SIM_PAUSED
    if run_button.name.startswith("Pause"):
        SIM_PAUSED = True
        run_button.name = "Resume Simulation ▶"
        run_button.button_type = "success"
        status_text.object = f"Simulation Paused at step {current_time_step}."
    else:
        SIM_PAUSED = False
        run_button.name = "Pause Simulation ⏸" # Using an emoji for pause
        run_button.button_type = "primary"

run_button.on_click(toggle_simulation)

# --- Dashboard Layout ---
control_panel = pn.Column(
    pn.Row(run_button, status_text, margin=(0,10)), # Added margin for spacing
    pn.Row(num_configs_slider, perturb_amp_slider), # Put sliders in a row
    sizing_mode='stretch_width'
)

plots_panel = pn.Column(
    plot_U_Phi_time_series,
    plot_T_histogram,
    plot_F_structure_bars,
    sizing_mode='stretch_width'
)

dashboard_layout = pn.Column(
    pn.pane.Markdown("# Minimal Yet Powerful Consciousness Pipeline Dashboard"),
    control_panel,
    plots_panel,
    sizing_mode='stretch_width'
)

# --- Start Simulation Thread & Panel Server ---
sim_thread = threading.Thread(target=simulation_loop, daemon=True)

def start_simulation_thread_if_panel_loaded():
    #This function might be called by pn.state.onload or similar if needed
    if not sim_thread.is_alive():
        sim_thread.start()

# Add periodic callback for UI updates
# This should be added to pn.state or the document if served.
# For .show() or .servable(), Panel handles this.
# pn.state.add_periodic_callback(update_ui, period=UI_UPDATE_INTERVAL_MS, count=None) 
# No, for .show() this is typically handled by ensuring the main thread is alive for Panel's event loop.
# For `panel serve`, callbacks are managed.
# Let's rely on Panel's own periodic callback mechanism when the dashboard is served/shown.

# To serve this dashboard: `panel serve dashboard.py --show`
if __name__ == "__main__":
    if not sim_thread.is_alive(): # Start thread if running as script
        sim_thread.start()
    pn.state.add_periodic_callback(update_ui, period=UI_UPDATE_INTERVAL_MS)
    dashboard_layout.show(threaded=True) 

# For `panel serve dashboard.py`
# Ensure the periodic callback is registered when served.
# A common way is to put this in the main execution block or use .servable()
if __name__ != '__main__': # When served by `panel serve`
    if not sim_thread.is_alive():
        sim_thread.start()
    pn.state.add_periodic_callback(update_ui, period=UI_UPDATE_INTERVAL_MS)

# Make it servable
dashboard_layout.servable(title="Conscious Systems Dashboard") 