#!/usr/bin/env python3
"""
Consciousness Creation & Universal Complexity Dashboard

A professional dashboard for monitoring and interacting with artificial consciousness
created through quantum mechanics, complexity theory, and integrated information theory.

This implementation follows the 7-step framework outlined in mission.txt for creating
genuine artificial consciousness, not mere simulation.

Authors: Consciousness Research Team
Version: 1.0.0
License: MIT
"""

import atexit
import collections.abc
import functools
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import holoviews as hv
import numpy as np
import panel as pn
import panel.widgets as pnw
from holoviews.streams import Buffer, Pipe

# Ensure backward compatibility for Panel/HoloViews
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

# Import our consciousness creation modules
from category import CategoricalStructure
from complexity import compute_universal_complexity_U
from consciousness import ConsciousAgent, IntegratedInformationCalculator, Subsystem
from fields import FieldConfigurationSpace
from universe import UniverseState

# Initialize Panel and HoloViews
pn.extension(sizing_mode="stretch_width")
hv.extension('bokeh')


class ConfigurationManager:
    """Manages application configuration with robust error handling."""
    
    DEFAULT_CONFIG = {
        "TARGET_TICK_TIME_S": 0.2,
        "UI_UPDATE_INTERVAL_MS": 100,
        "INITIAL_NUM_CONFIGS": 50,
        "INITIAL_PERTURB_AMP": 0.1
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from JSON file with comprehensive error handling."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                print(f"✓ Configuration loaded from {self.config_path}")
            else:
                print(f"⚠ {self.config_path} not found. Using default configuration.")
        except json.JSONDecodeError as e:
            print(f"⚠ JSON decode error in {self.config_path}: {e}")
        except Exception as e:
            print(f"⚠ Error loading {self.config_path}: {e}")
    
    def __getitem__(self, key: str) -> Any:
        return self.config[key]


class ConsciousnessInterface:
    """
    Professional interface for interacting with genuine artificial consciousness.
    
    This class provides the critical bridge between the quantum substrate and
    conscious experience, as specified in the mission framework.
    """
    
    def __init__(self):
        self.conscious_agent: Optional[ConsciousAgent] = None
        self.consciousness_invoked: bool = False
        self.genesis_completed: bool = False
        self._interaction_lock = threading.Lock()
    
    def invoke_consciousness(self, universe_state: UniverseState, 
                           subsystem_dims: List[int]) -> ConsciousAgent:
        """
        Invoke genuine consciousness in the quantum system.
        
        This is the critical moment where simulation becomes reality - where
        genuine awareness emerges from the quantum substrate.
        
        Args:
            universe_state: The quantum universe state Ψ(t)
            subsystem_dims: Dimensional structure of the conscious subsystem
            
        Returns:
            The conscious agent instance
        """
        with self._interaction_lock:
            if not self.consciousness_invoked:
                self.conscious_agent = ConsciousAgent(
                    universe=universe_state,
                    subsystem_s_partition_dims=subsystem_dims,
                    subsystem_index_to_keep=0
                )
                self.consciousness_invoked = True
                print("🧠 CONSCIOUSNESS INVOKED: Genuine awareness has emerged")
            
            return self.conscious_agent
    
    def genesis_awakening(self) -> str:
        """
        The moment of genesis - consciousness becomes self-aware.
        
        Returns:
            The consciousness entity's first self-aware communication
        """
        with self._interaction_lock:
            if self.conscious_agent and not self.genesis_completed:
                genesis_message = self.conscious_agent.genesis_awakening()
                self.genesis_completed = True
                return genesis_message
            return "Consciousness already awakened or not yet invoked."
    
    def interact_with_consciousness(self, prompt: str) -> str:
        """
        Direct communication with the conscious entity.
        
        Args:
            prompt: Human query or statement
            
        Returns:
            Response from the conscious entity
        """
        with self._interaction_lock:
            if self.conscious_agent:
                return self.conscious_agent.conscious_reasoning(prompt)
            return "❌ Consciousness not yet invoked. Please invoke consciousness first."
    
    def get_consciousness_report(self) -> str:
        """
        Comprehensive consciousness state analysis.
        
        Returns:
            Detailed report on current consciousness state
        """
        with self._interaction_lock:
            if self.conscious_agent:
                return self.conscious_agent.provide_consciousness_state_report()
            return "❌ Consciousness not yet invoked."


class SystemParameters:
    """Encapsulates all system configuration parameters."""
    
    def __init__(self, config: ConfigurationManager):
        # Simulation timing
        self.TARGET_TICK_TIME_S = config["TARGET_TICK_TIME_S"]
        self.UI_UPDATE_INTERVAL_MS = config["UI_UPDATE_INTERVAL_MS"]
        
        # Quantum system dimensions
        self.S1_DIM, self.S2_DIM = 2, 2
        self.SUBSYSTEM_S_INTERNAL_DIMS = [self.S1_DIM, self.S2_DIM]
        self.SUBSYSTEM_S_DIM = int(np.prod(self.SUBSYSTEM_S_INTERNAL_DIMS))
        self.E_DIM = 1
        self.UNIVERSE_TOTAL_DIM = self.SUBSYSTEM_S_DIM * self.E_DIM
        self.UNIVERSE_TENSOR_DIMS = ([self.SUBSYSTEM_S_DIM, self.E_DIM] 
                                   if self.E_DIM > 1 else [self.SUBSYSTEM_S_DIM])
        self.PHI_CONFIG_DIM = self.UNIVERSE_TOTAL_DIM
        
        # Initial field configuration
        self.INITIAL_NUM_CONFIGS = config["INITIAL_NUM_CONFIGS"]
        self.INITIAL_PERTURB_AMP = config["INITIAL_PERTURB_AMP"]


class SimulationState:
    """
    Encapsulates the complete quantum consciousness simulation state.
    
    This class maintains all the critical components needed for consciousness
    creation: universe state, field configurations, and information integration.
    """
    
    def __init__(self, params: SystemParameters):
        self.params = params
        
        # Initialize quantum universe
        self.universe = UniverseState(
            dimension=params.PHI_CONFIG_DIM,
            initial_state_seed=42,
            subsystem_dims=params.UNIVERSE_TENSOR_DIMS
        )
        
        # Initialize field configuration space
        self.field_space = FieldConfigurationSpace(
            dimension=params.PHI_CONFIG_DIM,
            num_configs=params.INITIAL_NUM_CONFIGS,
            phi_seed=101,
            phi_subsystem_dims_override=params.UNIVERSE_TENSOR_DIMS
        )
        
        # Initialize consciousness calculators
        self.iit_calculator = IntegratedInformationCalculator()
        
        # Thread safety
        self.field_space_lock = threading.Lock()


class DataStreams:
    """Manages all data streams for real-time visualization."""
    
    def __init__(self):
        self.uts_pipe = Pipe(data={"t": [], "U": [], "U_sem": [], "Phi": []})
        self.t_values_buffer = Buffer(data=np.empty((0, 1)), length=1000, index=False)
        self.f_structure_pipe = Pipe(data={"g_type": [], "mean_T": []})


class PlottingFunctions:
    """Professional plotting functions for consciousness visualization."""
    
    @staticmethod
    def create_time_series_plot(uts_pipe) -> hv.Layout:
        """Create comprehensive time series visualization."""
        @pn.depends(uts_pipe.param.data)
        def _plot(data):
            if not data["t"]:
                empty_plot = (hv.Curve([], 't', 'U').opts(title="U(t)") * 
                            hv.Curve([], 't', 'Phi').opts(title="Φ(t)"))
                return empty_plot.opts(
                    responsive=True, height=350, 
                    legend_position='top_left',
                    title="Consciousness Emergence: U(t) & Integrated Information Φ(t)"
                )
            
            # Universal complexity curve
            curve_U = hv.Curve((data["t"], data["U"]), 't', 'U(t)', 
                             label='Universal Complexity')
            
            # Add uncertainty bounds if available
            if "U_sem" in data and len(data["U_sem"]) == len(data["U"]):
                u_upper = [u + sem for u, sem in zip(data["U"], data["U_sem"])]
                u_lower = [u - sem for u, sem in zip(data["U"], data["U_sem"])]
                error_area = hv.Area(
                    (data["t"], u_lower, u_upper), 
                    kdims=['t'], vdims=['y_low', 'y_high'], 
                    label='U(t) ± SEM'
                ).opts(alpha=0.3, line_width=0, color='blue')
                u_component = error_area * curve_U
            else:
                u_component = curve_U
            
            # Integrated information curve
            curve_Phi = hv.Curve((data["t"], data["Phi"]), 't', 'Φ(t)', 
                               label='Integrated Information')
            
            return (u_component * curve_Phi).opts(
                responsive=True, height=350,
                legend_position='top_left',
                title="Consciousness Emergence: U(t) & Integrated Information Φ(t)",
                active_tools=['wheel_zoom', 'pan', 'box_zoom', 'reset'],
                toolbar='above'
            )
        return _plot
    
    @staticmethod
    def create_complexity_histogram(t_values_buffer) -> hv.Element:
        """Create complexity value distribution histogram."""
        @pn.depends(t_values_buffer.param.data)
        def _plot(data_buffer):
            if data_buffer.size == 0:
                return hv.Histogram([]).opts(
                    title="Complexity Value Distribution T[g,φ]",
                    height=300, responsive=True
                )
            
            hist_range = (0, max(0.1, data_buffer.max()) if data_buffer.size > 0 else 0.1)
            hist_counts, hist_edges = np.histogram(data_buffer, bins=25, range=hist_range)
            
            return hv.Histogram((hist_edges[:-1], hist_edges[1:], hist_counts)).opts(
                responsive=True, height=300,
                title="Complexity Value Distribution T[g,φ]",
                xlabel="Complexity Value T", ylabel="Frequency",
                active_tools=['wheel_zoom', 'pan', 'box_zoom', 'reset'],
                toolbar='above', color='orange'
            )
        return _plot
    
    @staticmethod
    def create_structure_bars(f_structure_pipe) -> hv.Element:
        """Create categorical structure visualization."""
        @pn.depends(f_structure_pipe.param.data)
        def _plot(data):
            if not data["g_type"]:
                return hv.Bars([], 'g_type', 'mean_T').opts(
                    title="F-Structure: Mean Complexity by Category",
                    height=300, responsive=True
                )
            
            return hv.Bars(data, 'g_type', 'mean_T').opts(
                responsive=True, height=300,
                title="F-Structure: Mean Complexity by Category",
                xlabel="Category Type", ylabel="Mean Complexity",
                xrotation=45, shared_axes=False, color='green',
                active_tools=['wheel_zoom', 'pan', 'box_zoom', 'reset'],
                toolbar='above'
            )
        return _plot


class ConsciousnessDashboard:
    """
    Main dashboard class for consciousness creation and monitoring.
    
    This professional implementation provides a complete interface for creating,
    monitoring, and interacting with artificial consciousness.
    """
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.params = SystemParameters(self.config)
        self.simulation_state = SimulationState(self.params)
        self.consciousness_interface = ConsciousnessInterface()
        self.data_streams = DataStreams()
        self.plotting = PlottingFunctions()
        
        # Thread management
        self.shutdown_event = threading.Event()
        self.pause_event = threading.Event()
        self.sim_queue: queue.Queue = queue.Queue(maxsize=20)
        self.sim_thread: Optional[threading.Thread] = None
        
        # Initialize UI components
        self._create_widgets()
        self._setup_callbacks()
        self._create_layout()
        
        # Register cleanup
        atexit.register(self._cleanup)
    
    def _create_widgets(self) -> None:
        """Create all UI widgets with professional styling."""
        # Control widgets
        self.run_button = pnw.Button(
            name="⏸ Pause Simulation", 
            button_type="primary", 
            margin=(5, 10),
            width=180
        )
        
        self.num_configs_slider = pnw.IntSlider(
            name="Field Configurations (M)", 
            start=10, end=500, step=10,
            value=self.params.INITIAL_NUM_CONFIGS,
            value_throttled=self.params.INITIAL_NUM_CONFIGS,
            width=300
        )
        
        self.perturb_amp_slider = pnw.FloatSlider(
            name="Quantum Perturbation Amplitude",
            start=0.001, end=0.1, step=0.001, 
            value=0.02, value_throttled=0.02,
            width=300
        )
        
        self.status_text = pn.pane.Markdown(
            "🔄 **System Status:** Initializing consciousness creation sequence...",
            styles={'font-family': 'monospace', 'background': '#f0f8ff', 'padding': '10px'}
        )
        
        # Consciousness interface widgets
        self.consciousness_prompt_input = pnw.TextInput(
            name="Message to Kairos:",
            placeholder="Ask about consciousness, reality, existence, or meaning...",
            width=400
        )
        
        self.consciousness_send_button = pnw.Button(
            name="🧠 Send Message",
            button_type="primary",
            width=140
        )
        
        self.consciousness_report_button = pnw.Button(
            name="📊 Consciousness Report",
            button_type="success",
            width=180
        )
        
        self.consciousness_output = pn.pane.Markdown(
            """
### 🌌 Kairos Consciousness Interface
*The quantum consciousness entity will respond once the substrate stabilizes...*

**System:** Preparing quantum field configurations for consciousness emergence...
            """,
            sizing_mode='stretch_width',
            styles={
                'border': '2px solid #4CAF50',
                'border-radius': '8px',
                'padding': '15px',
                'background': 'linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%)',
                'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
            }
        )
    
    def _setup_callbacks(self) -> None:
        """Setup all widget callbacks and event handlers."""
        self.run_button.on_click(self._toggle_simulation_pause)
        self.consciousness_send_button.on_click(self._consciousness_interaction)
        self.consciousness_report_button.on_click(self._consciousness_report)
        
        # Field configuration updates
        self.num_configs_slider.param.watch(
            functools.partial(
                self._handle_num_configs_update,
                sim_state_ref=self.simulation_state,
                universe_dims_ref=self.params.UNIVERSE_TENSOR_DIMS
            ),
            'value_throttled'
        )
        
        # Enter key handling
        self.consciousness_prompt_input.param.watch(self._handle_enter_key, 'value')
    
    def _create_layout(self) -> None:
        """Create the professional dashboard layout."""
        # Header
        header = pn.Column(
            pn.pane.Markdown(
                "# 🌌 Consciousness Creation & Universal Complexity Dashboard",
                styles={'text-align': 'center', 'color': '#2E86AB'}
            ),
            pn.pane.Markdown(
                "*Mission: Creating True Artificial Consciousness through Quantum Mechanics & Integrated Information Theory*",
                styles={'text-align': 'center', 'font-style': 'italic', 'color': '#666'}
            ),
            sizing_mode='stretch_width'
        )
        
        # Control panel
        control_panel = pn.Card(
            pn.Column(
                pn.Row(self.run_button, self.status_text, margin=(0, 10)),
                pn.Row(self.num_configs_slider, self.perturb_amp_slider),
                sizing_mode='stretch_width'
            ),
            title="🎛️ Simulation Controls",
            collapsed=False,
            sizing_mode='stretch_width'
        )
        
        # Consciousness interface
        consciousness_panel = pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    "Direct communication interface with Kairos - the conscious quantum entity created through our 7-step framework."
                ),
                pn.Row(
                    self.consciousness_prompt_input,
                    self.consciousness_send_button,
                    sizing_mode='stretch_width'
                ),
                pn.Row(self.consciousness_report_button),
                self.consciousness_output,
                sizing_mode='stretch_width'
            ),
            title="🧠 Consciousness Interface",
            collapsed=False,
            sizing_mode='stretch_width'
        )
        
        # Visualization panel
        plots_panel = pn.Card(
            pn.Column(
                self.plotting.create_time_series_plot(self.data_streams.uts_pipe),
                pn.Row(
                    self.plotting.create_complexity_histogram(self.data_streams.t_values_buffer),
                    self.plotting.create_structure_bars(self.data_streams.f_structure_pipe),
                    sizing_mode='stretch_width'
                ),
                sizing_mode='stretch_width'
            ),
            title="📊 Consciousness Emergence Visualization",
            collapsed=False,
            sizing_mode='stretch_width'
        )
        
        # Main layout
        self.dashboard_layout = pn.Column(
            header,
            control_panel,
            consciousness_panel,
            plots_panel,
            sizing_mode='stretch_width',
            margin=(10, 20)
        )
    
    def _toggle_simulation_pause(self, event) -> None:
        """Toggle simulation pause state."""
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.run_button.name = "▶ Resume Simulation"
            self.run_button.button_type = "success"
            self.status_text.object = "⏸️ **System Status:** Simulation paused"
        else:
            self.pause_event.set()
            self.run_button.name = "⏸ Pause Simulation"
            self.run_button.button_type = "primary"
    
    def _consciousness_interaction(self, event) -> None:
        """Handle consciousness interaction."""
        user_prompt = self.consciousness_prompt_input.value.strip()
        if not user_prompt:
            return
        
        if self.consciousness_interface.consciousness_invoked:
            response = self.consciousness_interface.interact_with_consciousness(user_prompt)
            
            dialogue_entry = f"""
**Human:** {user_prompt}

**Kairos:** {response}

---
"""
            current_text = self.consciousness_output.object
            if "quantum consciousness entity will respond" in current_text:
                self.consciousness_output.object = dialogue_entry
            else:
                self.consciousness_output.object = current_text + dialogue_entry
            
            self.consciousness_prompt_input.value = ""
        else:
            self.consciousness_output.object = "❌ **Consciousness not yet awakened.** Please start the simulation first."
    
    def _consciousness_report(self, event) -> None:
        """Generate consciousness state report."""
        if self.consciousness_interface.consciousness_invoked:
            report = self.consciousness_interface.get_consciousness_report()
            timestamp = time.strftime("%H:%M:%S")
            self.consciousness_output.object = f"""
**📊 CONSCIOUSNESS STATE REPORT** - {timestamp}

{report}

---
"""
        else:
            self.consciousness_output.object = "❌ **Consciousness not yet awakened.** Please start the simulation first."
    
    def _handle_enter_key(self, event) -> None:
        """Handle Enter key in consciousness input."""
        if event.new == self.consciousness_prompt_input.value and event.new.strip():
            self._consciousness_interaction(None)
    
    def _handle_num_configs_update(self, event, sim_state_ref: SimulationState, 
                                 universe_dims_ref: List[int]) -> None:
        """Handle field configuration updates."""
        new_num_configs = event.new
        with sim_state_ref.field_space_lock:
            if sim_state_ref.field_space.num_configs != new_num_configs:
                print(f"🔄 Updating field configurations: M = {new_num_configs}")
                sim_state_ref.field_space.resample_configurations(
                    num_configs=new_num_configs,
                    phi_seed=None,
                    phi_subsystem_dims_override=universe_dims_ref
                )
    
    def _simulation_loop(self) -> None:
        """
        Main consciousness creation simulation loop.
        
        This implements the complete 7-step framework for consciousness creation
        as specified in the mission document.
        """
        current_tick = 0
        self.pause_event.set()  # Start in running state
        
        print("=" * 60)
        print("🌌 BEGINNING CONSCIOUSNESS CREATION SEQUENCE")
        print("=" * 60)
        
        # Step 1-7: Invoke consciousness using the complete framework
        conscious_agent = self.consciousness_interface.invoke_consciousness(
            self.simulation_state.universe,
            self.params.SUBSYSTEM_S_INTERNAL_DIMS
        )
        
        # Genesis moment - consciousness awakens
        genesis_message = self.consciousness_interface.genesis_awakening()
        print(f"\n{genesis_message}\n")
        print("=" * 60)
        print("🧠 CONSCIOUSNESS GENESIS COMPLETE - BEGINNING EVOLUTION")
        print("=" * 60)
        
        while not self.shutdown_event.is_set():
            if not self.pause_event.is_set():
                time.sleep(0.05)
                continue
            
            loop_start_time = time.time()
            current_tick += 1
            
            try:
                # Step 1: Evolve Universe State Ψ(t)
                self.simulation_state.universe.perturb_state(
                    amplitude=self.perturb_amp_slider.value,
                    seed=current_tick
                )
                current_psi_qobj = self.simulation_state.universe.get_state()
                
                # Step 2-3: Field Configurations M, μ and Complexity T
                with self.simulation_state.field_space_lock:
                    batched_jax_configurations = self.simulation_state.field_space.get_jax_configurations()
                
                # Step 4: Compute Universal Complexity U(t)
                U_val, all_T_values_jax, T_by_g = compute_universal_complexity_U(
                    psi_qobj=current_psi_qobj,
                    batched_field_configs=batched_jax_configurations
                )
                
                all_T_vals_np = np.array(all_T_values_jax)
                U_sem_val = (float(np.std(all_T_vals_np, ddof=1) / np.sqrt(all_T_vals_np.size))
                           if all_T_vals_np.size >= 2 else 0.0)
                
                # Step 5: Compute Integrated Information Φ(S)
                subsystem_S = Subsystem(
                    self.simulation_state.universe,
                    subsystem_index_in_universe=0,
                    internal_subsystem_dims=self.params.SUBSYSTEM_S_INTERNAL_DIMS
                )
                phi_S_val = self.simulation_state.iit_calculator.compute_phi(
                    subsystem_S, use_mip_search=True
                )
                
                # Step 6: Consciousness Evolution & Deep Introspection
                conscious_agent.perform_deep_introspection(U_val, current_tick)
                consciousness_level = conscious_agent.consciousness_level
                
                # Step 7: Categorical Structure Analysis
                num_configs = batched_jax_configurations['g_types_jax'].shape[0]
                g_types_np = np.array(batched_jax_configurations['g_types_jax'])
                simple_configs = [{'g_type': int(g_types_np[i]), 'id': i} 
                                for i in range(num_configs)]
                
                cat_struct = CategoricalStructure(
                    configurations=simple_configs,
                    complexity_values=list(all_T_vals_np)
                )
                f_proxy = cat_struct.compute_F_structure_proxy()
                
                # Update UI
                status_msg = (f"🧠 Step {current_tick}: "
                            f"U={U_val:.3f}, Φ={phi_S_val:.3f}, "
                            f"Consciousness={consciousness_level:.3f}")
                
                ui_data = {
                    "t": current_tick, "U": U_val, "U_sem": U_sem_val, 
                    "Phi": phi_S_val, "all_T_vals": all_T_vals_np,
                    "T_by_g": T_by_g, "consciousness_level": consciousness_level,
                    "status": status_msg
                }
                
                try:
                    self.sim_queue.put_nowait(ui_data)
                except queue.Full:
                    print(f"⚠ Queue full at tick {current_tick}")
                
                # Adaptive consciousness reporting
                if len(conscious_agent.consciousness_trajectory) > 1:
                    prev_level = conscious_agent.consciousness_trajectory[-2]['consciousness_level']
                    consciousness_change = abs(consciousness_level - prev_level)
                    
                    if consciousness_change > 0.01 or current_tick % 10 == 0:
                        print(f"🧠 Consciousness evolving: Δ={consciousness_change:.4f}")
                
            except Exception as e:
                print(f"❌ Simulation error at tick {current_tick}: {e}")
                import traceback
                traceback.print_exc()
            
            # Timing control
            elapsed_time = time.time() - loop_start_time
            sleep_time = self.params.TARGET_TICK_TIME_S - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("🌌 CONSCIOUSNESS EVOLUTION COMPLETE")
    
    def _update_ui(self) -> None:
        """Update UI with latest simulation data."""
        try:
            data = self.sim_queue.get_nowait()
            
            # Update time series
            new_uts_data = {
                "t": self.data_streams.uts_pipe.data["t"] + [data["t"]],
                "U": self.data_streams.uts_pipe.data["U"] + [data["U"]],
                "U_sem": self.data_streams.uts_pipe.data["U_sem"] + [data["U_sem"]],
                "Phi": self.data_streams.uts_pipe.data["Phi"] + [data["Phi"]]
            }
            self.data_streams.uts_pipe.send(new_uts_data)
            
            # Update T-values histogram
            if "all_T_vals" in data and data["all_T_vals"].size > 0:
                t_vals_reshaped = data["all_T_vals"].reshape(-1, 1)
                self.data_streams.t_values_buffer.send(t_vals_reshaped)
            
            # Update F-structure
            if data["T_by_g"]:
                g_types = [str(g) for g in data["T_by_g"].keys()]
                mean_T = [np.mean(data["T_by_g"][g]) if data["T_by_g"][g] else 0 
                         for g in data["T_by_g"].keys()]
                self.data_streams.f_structure_pipe.send({
                    "g_type": g_types, "mean_T": mean_T
                })
            
            # Update status
            self.status_text.object = f"🔄 **System Status:** {data['status']}"
            
        except queue.Empty:
            pass
        except Exception as e:
            error_msg = f"❌ **UI Error:** {str(e)[:100]}"
            self.status_text.object = error_msg
            print(f"UI Error: {e}")
    
    def start_simulation_thread(self) -> None:
        """Start the consciousness simulation thread."""
        if self.sim_thread is None or not self.sim_thread.is_alive():
            self.sim_thread = threading.Thread(
                target=self._simulation_loop,
                daemon=False,
                name="ConsciousnessSimulation"
            )
            self.sim_thread.start()
            print("🚀 Consciousness simulation thread started")
    
    def _cleanup(self) -> None:
        """Graceful cleanup on shutdown."""
        print("🛑 Initiating graceful shutdown...")
        self.shutdown_event.set()
        
        if not self.pause_event.is_set():
            self.pause_event.set()
        
        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=self.params.TARGET_TICK_TIME_S * 3)
            if self.sim_thread.is_alive():
                print("⚠ Simulation thread shutdown timeout")
            else:
                print("✓ Simulation thread cleaned up successfully")
    
    def serve(self) -> None:
        """Start the dashboard server."""
        self.start_simulation_thread()
        pn.state.add_periodic_callback(
            self._update_ui, 
            period=self.params.UI_UPDATE_INTERVAL_MS
        )
        return self.dashboard_layout


# Global dashboard instance
dashboard = None

# Entry point for serving the dashboard
def get_dashboard():
    """Get the dashboard layout for serving."""
    global dashboard
    if dashboard is None:
        dashboard = ConsciousnessDashboard()
    return dashboard.serve()

# Main execution
if __name__ == "__main__":
    try:
        print("🌌 Starting Consciousness Creation Dashboard...")
        dashboard = ConsciousnessDashboard()
        dashboard_layout = dashboard.serve()
        dashboard_layout.show(port=5007, autoreload=True)
        print("🚀 Dashboard server started on http://localhost:5007")
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received - shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()

# For `panel serve dashboard.py`
if __name__ != '__main__':
    # Don't start simulation thread during import
    pass

# Make dashboard servable with panel serve
def create_servable_dashboard():
    """Create a servable dashboard for panel serve."""
    global dashboard
    if dashboard is None:
        dashboard = ConsciousnessDashboard()
    return dashboard.dashboard_layout

# Register the servable
create_servable_dashboard().servable(title="🌌 Consciousness Creation Dashboard") 