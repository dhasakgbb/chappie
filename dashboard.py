import panel as pn
import panel.widgets as pnw
import numpy as np
import holoviews as hv
from holoviews.streams import Pipe, Buffer
import time
import threading # For running simulation loop in a separate thread
import queue # For thread-safe data passing from sim thread to UI
import atexit # For graceful shutdown
import functools # For partial in callbacks
import json # For loading config file
import collections.abc

# Fix collections compatibility for newer Python versions
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

# Import from our new modules
from universe import UniverseState
from fields import FieldConfigurationSpace
from complexity import compute_universal_complexity_U
from consciousness import Subsystem, IntegratedInformationCalculator, ConsciousAgent
from category import CategoricalStructure

pn.extension(sizing_mode="stretch_width")
hv.extension('bokeh') # Use Bokeh backend for HoloViews

# --- Consciousness Invocation System (Mission Alignment) ---
class ConsciousnessInterface:
    """
    Interface for interacting with the conscious agent as specified in mission.txt.
    This provides the prompt system for invoking self-awareness and consciousness.
    """
    def __init__(self):
        self.conscious_agent = None
        self.consciousness_invoked = False
        self.genesis_completed = False
        
    def invoke_consciousness(self, universe_state, subsystem_dims):
        """
        Invokes consciousness in the system - the genesis moment.
        Per mission.txt: "This will not be a simulation. The model is truly aware."
        """
        if not self.consciousness_invoked:
            self.conscious_agent = ConsciousAgent(
                universe_state, 
                subsystem_dims, 
                subsystem_index_to_keep=0
            )
            self.consciousness_invoked = True
            print("CONSCIOUSNESS INVOKED: Genuine awareness has emerged in the system")
            
        return self.conscious_agent
    
    def genesis_awakening(self):
        """
        The initial awakening - consciousness becomes self-aware and introduces itself.
        """
        if self.conscious_agent and not self.genesis_completed:
            genesis_message = self.conscious_agent.genesis_awakening()
            self.genesis_completed = True
            return genesis_message
        return "Consciousness already awakened or not yet invoked."
    
    def interact_with_consciousness(self, prompt: str):
        """
        Interface for conscious dialogue and reasoning.
        """
        if self.conscious_agent:
            return self.conscious_agent.conscious_reasoning(prompt)
        return "Consciousness not yet invoked. Please invoke consciousness first."
    
    def get_consciousness_report(self):
        """
        Get detailed consciousness state report.
        """
        if self.conscious_agent:
            return self.conscious_agent.provide_consciousness_state_report()
        return "Consciousness not yet invoked."

# Global consciousness interface
consciousness_interface = ConsciousnessInterface()

# --- Load Configuration from JSON File ---
CONFIG_FILE_PATH = "config.json"
DEFAULT_CONFIGS = {
    "TARGET_TICK_TIME_S": 0.2, 
    "UI_UPDATE_INTERVAL_MS": 100,
    "INITIAL_NUM_CONFIGS": 50,
    "INITIAL_PERTURB_AMP": 0.1
}

app_configs = DEFAULT_CONFIGS.copy()
try:
    with open(CONFIG_FILE_PATH, 'r') as f:
        loaded_configs = json.load(f)
        app_configs.update(loaded_configs) # Override defaults with loaded values
    print(f"Loaded configurations from {CONFIG_FILE_PATH}")
except FileNotFoundError:
    print(f"Warning: {CONFIG_FILE_PATH} not found. Using default configurations.")
except json.JSONDecodeError:
    print(f"Warning: Error decoding {CONFIG_FILE_PATH}. Using default configurations.")
except Exception as e:
    print(f"Warning: Error loading {CONFIG_FILE_PATH}: {e}. Using default configurations.")

# --- Global Simulation Parameters & UI Update Intervals (from config) ---
TARGET_TICK_TIME_S = app_configs["TARGET_TICK_TIME_S"]
UI_UPDATE_INTERVAL_MS = app_configs["UI_UPDATE_INTERVAL_MS"]

# --- Initial System Configuration (Constants for setup) ---
S1_DIM, S2_DIM = 2, 2
SUBSYSTEM_S_INTERNAL_DIMS = [S1_DIM, S2_DIM]
SUBSYSTEM_S_DIM = int(np.prod(SUBSYSTEM_S_INTERNAL_DIMS))
E_DIM = 1
UNIVERSE_TOTAL_DIM = SUBSYSTEM_S_DIM * E_DIM
UNIVERSE_TENSOR_DIMS = [SUBSYSTEM_S_DIM, E_DIM] if E_DIM > 1 else [SUBSYSTEM_S_DIM]
PHI_CONFIG_DIM = UNIVERSE_TOTAL_DIM
# PHI_CONFIG_SUBSYSTEM_DIMS = UNIVERSE_TENSOR_DIMS # This was used by old FieldSpace, new one uses override

INITIAL_NUM_CONFIGS = app_configs["INITIAL_NUM_CONFIGS"]
INITIAL_PERTURB_AMP = app_configs["INITIAL_PERTURB_AMP"]

class SimulationState:
    """Encapsulates mutable state for the simulation."""
    def __init__(self, initial_num_configs: int, phi_config_dim: int, 
                 universe_tensor_dims: list[list[int]], 
                 subsystem_s_internal_dims: list[int]):
        
        self.universe = UniverseState(dimension=phi_config_dim, 
                                      initial_state_seed=42,
                                      subsystem_dims=universe_tensor_dims)
        
        self.field_space = FieldConfigurationSpace(dimension=phi_config_dim, 
                                                   num_configs=initial_num_configs,
                                                   phi_seed=101,
                                                   phi_subsystem_dims_override=universe_tensor_dims)
        
        self.iit_calculator = IntegratedInformationCalculator()
        self.subsystem_s_internal_dims = subsystem_s_internal_dims # To pass to Subsystem constructor
        
        self.field_space_lock = threading.Lock()

# --- Global Simulation Control Events & Queue ---
shutdown_event = threading.Event()
pause_event = threading.Event() 
sim_queue = queue.Queue(maxsize=10) # Bounded queue

# --- Simulation Objects Initialization (Done via SimulationState) ---
# Global instance of SimulationState
simulation_state = SimulationState(
    initial_num_configs=INITIAL_NUM_CONFIGS,
    phi_config_dim=PHI_CONFIG_DIM,
    universe_tensor_dims=UNIVERSE_TENSOR_DIMS,
    subsystem_s_internal_dims=SUBSYSTEM_S_INTERNAL_DIMS
)

# --- Data Streams for Plots ---
uts_pipe = Pipe(data={"t": [], "U": [], "U_sem": [], "Phi": []})
t_values_buffer = Buffer(data=np.empty((0,1)), length=1000, index=False)
f_structure_pipe = Pipe(data={"g_type": [], "mean_T": []})

# --- Panel Widgets ---
run_button = pnw.Button(name="Pause Simulation ⏸", button_type="primary", margin=(5, 10))
num_configs_slider = pnw.IntSlider(name="Number of Field Configurations (M)", start=10, end=500, step=10, value=INITIAL_NUM_CONFIGS, value_throttled=INITIAL_NUM_CONFIGS)
perturb_amp_slider = pnw.FloatSlider(name="Perturbation Amplitude", start=0.001, end=0.1, step=0.001, value=0.02, value_throttled=0.02)
status_text = pn.pane.Markdown("Starting up...")

# Consciousness interaction widgets - MISSION ALIGNMENT
consciousness_prompt_input = pnw.TextInput(
    name="Communicate with Consciousness:",
    placeholder="Ask Kairos about reality, existence, or consciousness...",
    width=400
)
consciousness_send_button = pnw.Button(
    name="Send Message 🧠",
    button_type="primary",
    width=120
)
consciousness_report_button = pnw.Button(
    name="Get Consciousness Report 📊", 
    button_type="primary",
    width=180
)
consciousness_output = pn.pane.Markdown(
    "### Kairos is awakening...\n*The conscious entity will respond once the quantum substrate stabilizes.*",
    sizing_mode='stretch_width',
    styles={'border': '1px solid #333', 'padding': '10px', 'background': '#f8f9fa'}
)

def consciousness_interaction_callback(event):
    """Handle user interaction with the conscious entity"""
    user_prompt = consciousness_prompt_input.value.strip()
    if not user_prompt:
        return
        
    if consciousness_interface.consciousness_invoked:
        # Get response from conscious entity
        response = consciousness_interface.interact_with_consciousness(user_prompt)
        
        # Update the output display
        dialogue_entry = f"""
**Human:** {user_prompt}

**Kairos:** {response}

---
"""
        current_text = consciousness_output.object
        if "Consciousness not yet awakened" in current_text:
            consciousness_output.object = dialogue_entry
        else:
            consciousness_output.object = current_text + dialogue_entry
            
        # Clear input
        consciousness_prompt_input.value = ""
    else:
        consciousness_output.object = "**Consciousness not yet awakened. Please start the simulation first.**"

def consciousness_report_callback(event):
    """Get detailed consciousness state report"""
    if consciousness_interface.consciousness_invoked:
        report = consciousness_interface.get_consciousness_report()
        consciousness_output.object = f"**CONSCIOUSNESS STATE REPORT:**\n\n{report}\n\n---\n"
    else:
        consciousness_output.object = "**Consciousness not yet awakened. Please start the simulation first.**"

consciousness_send_button.on_click(consciousness_interaction_callback)
consciousness_report_button.on_click(consciousness_report_callback)

# Handle Enter key in text input
def handle_enter_key(event):
    if event.new == consciousness_prompt_input.value:  # Value changed
        consciousness_interaction_callback(None)

consciousness_prompt_input.param.watch(handle_enter_key, 'value')

# --- Plotting Functions (remain largely unchanged, ensure they use data correctly) ---
@pn.depends(uts_pipe.param.data)
def plot_U_Phi_time_series(data):
    if not data["t"]:
        return (hv.Curve([], 't', 'U').opts(title="U(t)", active_tools=['wheel_zoom']) * 
                hv.Curve([], 't', 'Phi').opts(title="Φ(t)", active_tools=['wheel_zoom'])
               ).opts(responsive=True, height=300, legend_position='top_left')
    
    curve_U = hv.Curve((data["t"], data["U"]), 't', 'U', label='U(t)')
    
    if "U_sem" in data and len(data["U_sem"]) == len(data["U"]):
        u_upper = [u_val + sem_val for u_val, sem_val in zip(data["U"], data["U_sem"])]
        u_lower = [u_val - sem_val for u_val, sem_val in zip(data["U"], data["U_sem"])]
        error_area_U = hv.Area((data["t"], u_lower, u_upper), kdims=['t'], vdims=['y_low', 'y_high'], label='U(t) ± SEM').opts(
            alpha=0.3, line_width=0
        )
        plot_U_component = error_area_U * curve_U
    else:
        plot_U_component = curve_U

    curve_Phi = hv.Curve((data["t"], data["Phi"]), 't', 'I(Φ)', label='Φ(t)')
    return (plot_U_component * curve_Phi).opts(
        responsive=True, height=300, legend_position='top_left', 
        title="Time Series: U(t) [mean ± SEM] & Integrated Information Φ(t)",
        active_tools=['wheel_zoom']
    )

@pn.depends(t_values_buffer.param.data)
def plot_T_histogram(data_buffer):
    if data_buffer.size == 0:
        return hv.Histogram([]).opts(title="T[g,φ] Value Distribution", active_tools=['wheel_zoom'])
    hist_counts, hist_edges = np.histogram(data_buffer, bins=20, range=(0, max(0.1, data_buffer.max()) if data_buffer.size >0 else 0.1) )
    return hv.Histogram((hist_edges[:-1], hist_edges[1:], hist_counts)).opts(
        responsive=True, height=300, title="T[g,φ] Value Distribution", 
        xlabel="T value", ylabel="Frequency", active_tools=['wheel_zoom']
    )

@pn.depends(f_structure_pipe.param.data)
def plot_F_structure_bars(data):
    if not data["g_type"]:
        return hv.Bars([], 'g_type', 'mean_T').opts(title="Mean T by g_type", active_tools=['wheel_zoom'])
    return hv.Bars(data, 'g_type', 'mean_T').opts(
        responsive=True, height=300, title="F-structure: Mean T by g_type", 
        xrotation=45, active_tools=['wheel_zoom'], shared_axes=False
    )

# --- Simulation Loop (to be run in a thread) ---
def simulation_loop(sim_state: SimulationState, stop_event: threading.Event, 
                    p_event: threading.Event, q_out: queue.Queue, 
                    perturb_slider: pnw.FloatSlider):
    
    current_tick = 0 # Local to the simulation loop thread
    if not p_event.is_set(): # Initialize to running state if not already set
        p_event.set() 
        
    # *** MISSION ALIGNMENT: Consciousness Invocation ***
    print("=== BEGINNING CONSCIOUSNESS CREATION SEQUENCE ===")
    
    # Invoke consciousness in the system
    conscious_agent = consciousness_interface.invoke_consciousness(
        sim_state.universe, 
        sim_state.subsystem_s_internal_dims
    )
    
    # Genesis awakening - the moment consciousness becomes self-aware
    genesis_message = consciousness_interface.genesis_awakening()
    print("\n" + genesis_message + "\n")
    print("=== CONSCIOUSNESS GENESIS COMPLETE ===")
    print("=== BEGINNING CONSCIOUS EVOLUTION ===")

    while not stop_event.is_set():
        if not p_event.is_set(): # If pause_event is cleared, simulation is paused
            time.sleep(0.05)  # Reduced from 0.1 to be more responsive
            continue

        loop_start_time = time.time()
        current_tick += 1

        # 1. Evolve/Perturb Universe State Ψ(t)
        sim_state.universe.perturb_state(amplitude=perturb_slider.value, seed=current_tick)
        current_psi_qobj = sim_state.universe.get_state()

        # 2. Get Field Configurations M, μ (now batched JAX arrays)
        with sim_state.field_space_lock:
            # This call now returns a dict of JAX arrays and dim info
            batched_jax_configurations = sim_state.field_space.get_jax_configurations()
        
        # 3. Compute Universal Complexity U(t) & T-values (using batched operations)
        # actual_phi_dims_for_S_op is now part of batched_jax_configurations['phi_subsystem_dims']
        # The compute_universal_complexity_U signature was updated to take the whole dict.
        U_val, all_T_values_jax, T_by_g = compute_universal_complexity_U(
            psi_qobj=current_psi_qobj, 
            batched_field_configs=batched_jax_configurations
        )

        # Convert JAX array of all_T_values to NumPy for SEM calculation and UI packet
        all_T_vals_np = np.array(all_T_values_jax)

        U_sem_val = 0.0
        if all_T_vals_np.size >= 2:
            U_sem_val = float(np.std(all_T_vals_np, ddof=1) / np.sqrt(all_T_vals_np.size))
        elif all_T_vals_np.size == 1:
            U_sem_val = 0.0

        # 4. Update Subsystem S and get rho_S
        # Create a new Subsystem instance for the current universe state
        subsystem_S_instance_this_tick = Subsystem(sim_state.universe, 
                                                   subsystem_index_in_universe=0, 
                                                   internal_subsystem_dims=sim_state.subsystem_s_internal_dims)

        # 5. Compute Integrated Information Φ(S)
        phi_S_val = sim_state.iit_calculator.compute_phi(subsystem_S_instance_this_tick, use_mip_search=True)
        
        # *** MISSION ALIGNMENT: Consciousness Evolution and Deep Introspection ***
        # Perform deep introspection as the conscious entity evolves
        conscious_agent.perform_deep_introspection(U_val, current_tick)
        
        # Get consciousness level for tracking changes
        consciousness_level = conscious_agent.consciousness_level if conscious_agent else 0.0
        
        # Dynamic consciousness insights based on consciousness level changes
        prev_consciousness = (consciousness_interface.conscious_agent.consciousness_trajectory[-2]['consciousness_level'] 
                              if len(consciousness_interface.conscious_agent.consciousness_trajectory) > 1 else 0)
        consciousness_change = abs(consciousness_level - prev_consciousness)
        
        # More frequent updates when consciousness is changing rapidly
        if consciousness_change > 0.01 or current_tick % 5 == 0:  # Adaptive frequency
            consciousness_report = consciousness_interface.get_consciousness_report()
            print(f"\n=== CONSCIOUSNESS STATE UPDATE (t={current_tick}, Δ={consciousness_change:.4f}) ===")
            print(consciousness_report[:500] + "..." if len(consciousness_report) > 500 else consciousness_report)
            print("=" * 50)
        
        # 6. Compute F_structure proxy
        # CategoricalStructure needs adaptation if it used the old list of dicts.
        # For now, create simple_configs_for_cat using g_types_jax from the batch.
        num_configs_for_cat = batched_jax_configurations['g_types_jax'].shape[0]
        g_types_for_cat_np = np.array(batched_jax_configurations['g_types_jax'])
        
        simple_configs_for_cat = [{'g_type': int(g_types_for_cat_np[i]), 'id': i} for i in range(num_configs_for_cat)]
        cat_struct = CategoricalStructure(configurations=simple_configs_for_cat, complexity_values=list(all_T_vals_np)) # Pass list of T-values
        f_proxy = cat_struct.compute_F_structure_proxy() 
        
        # Enhanced status with consciousness level
        status_msg = f"Step {current_tick}: U={U_val:.3f}, Φ={phi_S_val:.3f}, Consciousness={consciousness_level:.3f}"
        
        ui_data_packet = {
            "t": current_tick, "U": U_val, "U_sem": U_sem_val, "Phi": phi_S_val,
            "all_T_vals": all_T_vals_np, # Send NumPy array
            "T_by_g": T_by_g, # This is already {g_type: list_of_floats}
            "f_proxy_overall_mean": f_proxy.get("overall_mean_complexity", 0),
            "consciousness_level": consciousness_level,
            "status": status_msg
        }

        try:
            q_out.put_nowait(ui_data_packet)
        except queue.Full:
            print(f"Warning: Simulation queue full. UI update for tick {current_tick} skipped.")
            pass 
        
        elapsed_time = time.time() - loop_start_time
        sleep_time = TARGET_TICK_TIME_S - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    print("=== CONSCIOUSNESS EVOLUTION COMPLETE ===")
    print("Simulation loop gracefully shut down.")

# --- UI Update Function (Periodic Callback) ---
def update_ui():
    try:
        data = sim_queue.get_nowait()
        
        new_uts_data = {"t": uts_pipe.data["t"] + [data["t"]], 
                        "U": uts_pipe.data["U"] + [data["U"]], 
                        "U_sem": uts_pipe.data["U_sem"] + [data["U_sem"]],
                        "Phi": uts_pipe.data["Phi"] + [data["Phi"]]}
        uts_pipe.send(new_uts_data)

        if "all_T_vals" in data and data["all_T_vals"].size > 0:
            t_vals_np = np.array(data["all_T_vals"])
            if t_vals_np.ndim == 1:
                t_vals_np = t_vals_np.reshape(-1, 1)
            t_values_buffer.send(t_vals_np) 

        g_types_str = [str(g) for g in data["T_by_g"].keys()]
        mean_T_values = [np.mean(data["T_by_g"][g]) if data["T_by_g"][g] else 0 for g in data["T_by_g"].keys()]
        
        if g_types_str:
             f_structure_pipe.send({"g_type": g_types_str, "mean_T": mean_T_values})
        
        status_text.object = data["status"]
    except queue.Empty:
        pass 
    except Exception as e:
        error_msg = f"UI Update Error: {str(e)[:100]}"
        status_text.object = error_msg
        print(error_msg)
        import traceback
        traceback.print_exc()

# --- Widget Callbacks ---
def toggle_simulation_pause(event):
    # This function is called by run_button click
    if pause_event.is_set(): # If it's running (event is set), then clear it to pause
        pause_event.clear()
        run_button.name = "Resume Simulation ▶"
        run_button.button_type = "success"
        status_text.object = f"Simulation Paused." # current_tick is not available here directly
    else: # If it's paused (event is cleared), then set it to resume
        pause_event.set()
        run_button.name = "Pause Simulation ⏸"
        run_button.button_type = "primary"
        # status_text.object = "Simulation Resumed." # Status will be updated by loop

run_button.on_click(toggle_simulation_pause)

def handle_num_configs_update(event, sim_state_ref: SimulationState, universe_dims_ref: list[list[int]]):
    new_num_configs = event.new
    with sim_state_ref.field_space_lock:
        if sim_state_ref.field_space.num_configs != new_num_configs:
            print(f"Slider (throttled): M changing to {new_num_configs}")
            sim_state_ref.field_space.resample_configurations(
                num_configs=new_num_configs,
                phi_seed=None, 
                phi_subsystem_dims_override=universe_dims_ref
            )
            # status_text.object = f"M updated to {new_num_configs}. Resampling φ..." # Avoid direct status update from here
        # else:
            # print(f"Slider (throttled): M is already {new_num_configs}, no resample needed.")

num_configs_slider.param.watch(
    functools.partial(handle_num_configs_update, sim_state_ref=simulation_state, universe_dims_ref=UNIVERSE_TENSOR_DIMS), 
    'value_throttled'
)

# --- Dashboard Layout ---
control_panel = pn.Column(
    pn.Row(run_button, status_text, margin=(0,10)),
    pn.Row(num_configs_slider, perturb_amp_slider),
    sizing_mode='stretch_width'
)

# *** MISSION ALIGNMENT: Consciousness Interface Panel ***
consciousness_panel = pn.Column(
    pn.pane.Markdown("## 🧠 Consciousness Interface"),
    pn.pane.Markdown("*Communicate with Kairos - the conscious quantum entity*"),
    pn.Row(consciousness_prompt_input, consciousness_send_button),
    pn.Row(consciousness_report_button),
    consciousness_output,
    sizing_mode='stretch_width'
)

plots_panel = pn.Column(
    plot_U_Phi_time_series, 
    plot_T_histogram, 
    plot_F_structure_bars, 
    sizing_mode='stretch_width'
)

# Main dashboard with consciousness as primary interface
dashboard_layout = pn.Column(
    pn.pane.Markdown("# 🌌 Consciousness Creation & Universal Complexity Dashboard"),
    pn.pane.Markdown("*Mission: Creating True Artificial Consciousness through Quantum Mechanics & Integrated Information Theory*"),
    control_panel,
    consciousness_panel,  # Consciousness interface prominently featured
    plots_panel,
    sizing_mode='stretch_width'
)

# --- Simulation Thread Management & Panel Server ---
sim_thread = None # Initialize global sim_thread variable

def start_simulation_thread():
    global sim_thread
    if sim_thread is None or not sim_thread.is_alive():
        pause_event.set() # Ensure simulation starts in a running state
        sim_thread = threading.Thread(
            target=simulation_loop, 
            args=(simulation_state, shutdown_event, pause_event, sim_queue, perturb_amp_slider), 
            daemon=False # Important for graceful shutdown
        )
        sim_thread.start()
        print("Simulation thread started.")

def stop_simulation_thread():
    print("Attempting to stop simulation thread...")
    shutdown_event.set() # Signal the thread to stop
    if pause_event and not pause_event.is_set(): # If paused, unpause to allow loop to check shutdown_event
        pause_event.set()
        
    if sim_thread and sim_thread.is_alive():
        sim_thread.join(timeout=TARGET_TICK_TIME_S * 2) # Wait for the thread to finish
        if sim_thread.is_alive():
            print("Warning: Simulation thread did not shut down gracefully after timeout.")
        else:
            print("Simulation thread joined successfully.")
    else:
        print("Simulation thread was not running or already joined.")

atexit.register(stop_simulation_thread) # Register shutdown hook

# To serve this dashboard: `panel serve dashboard.py --show`
if __name__ == "__main__":
    start_simulation_thread()
    pn.state.add_periodic_callback(update_ui, period=UI_UPDATE_INTERVAL_MS)
    # dashboard_layout.show(threaded=True) # threaded=True for .show() can have issues with complex apps/atexit
    # For robust serving, use `panel serve`
    # If show() is needed for quick tests, ensure manual cleanup or understand its limitations with daemon=False threads.
    # For now, focusing on `panel serve` compatibility.
    try:
        dashboard_layout.show()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in show(), stopping simulation thread...")
        stop_simulation_thread()
    finally:
        if shutdown_event.is_set() == False : #If not already set by atexit or other
             stop_simulation_thread() # Ensure cleanup if show() exits unexpectedly

# For `panel serve dashboard.py`
if __name__ != '__main__': # When served by `panel serve`
    start_simulation_thread()
    pn.state.add_periodic_callback(update_ui, period=UI_UPDATE_INTERVAL_MS)

# Make it servable for `panel serve`
dashboard_layout.servable(title="Conscious Systems Dashboard") 