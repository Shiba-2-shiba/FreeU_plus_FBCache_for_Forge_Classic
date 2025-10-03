# /scripts/integrate_utils.py
#
# FBCache and FreeU integration script, controlled by a single patch.
#
# --- v8 Final ---
# - Refactored the patch logic to no longer replicate the UNet's forward pass.
# - Instead of manually processing UNet blocks, the patch now calls the original
#   `apply_model` function and then applies modifications to its inputs/outputs.
# - FreeU is now applied by patching `transformer_options` before the call.
# - FBCache is now applied by capturing and comparing the `model_output` after the call.
# - This approach removes all direct dependencies on internal UNet helper functions
#   (e.g., `timestep_embedding`), resolving the `ModuleNotFoundError` permanently
#   and increasing robustness against future Forge updates.

import torch
import gradio as gr
import traceback
import weakref
import datetime
import sys
import os
from functools import wraps

# --- WebUI/Forge Module Imports ---
from modules import scripts, shared, script_callbacks

# --- Path Resolution and Core Module Imports ---
try:
    script_path = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(script_path)
    extension_root = os.path.dirname(scripts_dir)
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    from modules.freeu_core import apply_freeu_scaling
    from modules.fb_cache_core import FBCacheState, are_two_tensors_similar
    print("[IntegratedUtils] Info: Core modules imported successfully.")
except (ImportError, ValueError, NameError) as e:
    print(f"\n[IntegratedUtils] Error: Could not import core modules. ({e})")
    def apply_freeu_scaling(h, hsp, *args, **kwargs): return h, hsp
    class FBCacheState: pass
    def are_two_tensors_similar(*args, **kwargs): return False

# --- Main Script Class ---
class IntegratedUtilsScript(scripts.Script):
    _instance = None

    def __init__(self):
        super().__init__()
        if IntegratedUtilsScript._instance is None:
            IntegratedUtilsScript._instance = self

        self.active_fb_state_object = None
        self.is_debug_logging_enabled = False
        self.fb_params_runtime = {}
        self.freeu_params_runtime = {}

        self.original_k_model = None
        self.original_apply_model_method = None

        self.log_info("Script instance initialized.")

    def title(self): return "FBCache + FreeU (Integrated)"
    def show(self, is_img2img): return scripts.AlwaysVisible

    # --- Logging Methods ---
    def _log_prefix(self): return f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} IntegratedUtils]"
    def log_info(self, message: str): print(f"{self._log_prefix()} Info: {message}")
    def log_debug(self, message: str):
        if self.is_debug_logging_enabled: print(f"{self._log_prefix()} Debug: {message}")
    def log_error(self, message: str): print(f"{self._log_prefix()} ERROR: {message}\n{traceback.format_exc()}")

    def ui(self, is_img2img):
        # UI Definition
        ui_components = []
        with gr.Accordion(self.title(), open=False):
            with gr.Row():
                ui_components.append(gr.Checkbox(label="Enable Debug Logging (Console)", value=False))
            with gr.Tabs():
                with gr.TabItem("First Block Cache"):
                    with gr.Tabs():
                        with gr.TabItem("First Pass"):
                            fb_enabled_first = gr.Checkbox(label="Enable for First Pass", value=False)
                            fb_threshold_first = gr.Slider(label="Similarity Threshold", minimum=0.001, maximum=0.5, step=0.001, value=0.3)
                            fb_blocks_first = gr.Slider(label="UNet Initial Blocks to Cache (Ignored, for compatibility)", minimum=1, maximum=4, step=1, value=3)
                            fb_start_first = gr.Slider(label="Start At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.3)
                            fb_end_first = gr.Slider(label="End At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.99)
                            fb_max_hits_first = gr.Number(label="Max Consecutive Hits (-1 for unlimited)", value=-1, precision=0)
                            ui_components.extend([fb_enabled_first, fb_threshold_first, fb_blocks_first, fb_start_first, fb_end_first, fb_max_hits_first])
                        with gr.TabItem("Hires Fix Pass"):
                            fb_enabled_hires = gr.Checkbox(label="Enable for Hires Fix Pass", value=False)
                            fb_use_first_settings = gr.Checkbox(label="Use First Pass settings", value=True)
                            with gr.Group(visible=False) as hires_specific_settings:
                                fb_threshold_hires = gr.Slider(label="Similarity Threshold (Hires)", minimum=0.001, maximum=0.5, step=0.001, value=0.3)
                                fb_blocks_hires = gr.Slider(label="UNet Initial Blocks (Hires, Ignored)", minimum=1, maximum=4, step=1, value=3)
                                fb_start_hires = gr.Slider(label="Start At % of Steps (Hires)", minimum=0.0, maximum=1.0, step=0.01, value=0.3)
                                fb_end_hires = gr.Slider(label="End At % of Steps (Hires)", minimum=0.0, maximum=1.0, step=0.01, value=0.99)
                                fb_max_hits_hires = gr.Number(label="Max Consecutive Hits (Hires)", value=-1, precision=0)
                            ui_components.extend([fb_enabled_hires, fb_use_first_settings, fb_threshold_hires, fb_blocks_hires, fb_start_hires, fb_end_hires, fb_max_hits_hires])
                    fb_use_first_settings.change(fn=lambda use_first: gr.update(visible=not use_first), inputs=[fb_use_first_settings], outputs=[hires_specific_settings])
                with gr.TabItem("FreeU"):
                    freeu_enabled = gr.Checkbox(label="Enable FreeU", value=False)
                    freeu_b1 = gr.Slider(label="Backbone 1 (b1)", minimum=0, maximum=2, step=0.01, value=1.3)
                    freeu_b2 = gr.Slider(label="Backbone 2 (b2)", minimum=0, maximum=2, step=0.01, value=1.4)
                    freeu_s1 = gr.Slider(label="Skip 1 (s1)", minimum=0, maximum=4, step=0.01, value=1.2)
                    freeu_s2 = gr.Slider(label="Skip 2 (s2)", minimum=0, maximum=4, step=0.01, value=0.7)
                    freeu_start_at = gr.Slider(label="Start At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.01)
                    freeu_stop_at = gr.Slider(label="Stop At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.2)
                    ui_components.extend([freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_start_at, freeu_stop_at])
        return ui_components

    def process(self, p, enable_debug_logging, fb_enabled_first, fb_threshold_first, fb_blocks_first, fb_start_first, fb_end_first, fb_max_hits_first, fb_enabled_hires, fb_use_first_settings, fb_threshold_hires, fb_blocks_hires, fb_start_hires, fb_end_hires, fb_max_hits_hires, freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_start_at, freeu_stop_at):
        self.is_debug_logging_enabled = enable_debug_logging
        self.log_info("New generation process started. Parsing UI parameters.")
        self.fb_params_runtime['first'] = {'enabled': fb_enabled_first, 'threshold': fb_threshold_first, 'blocks': fb_blocks_first, 'start': fb_start_first, 'end': fb_end_first, 'max_hits': fb_max_hits_first}
        self.fb_params_runtime['hires'] = self.fb_params_runtime['first'].copy() if fb_use_first_settings else {'enabled': fb_enabled_hires, 'threshold': fb_threshold_hires, 'blocks': fb_blocks_hires, 'start': fb_start_hires, 'end': fb_end_hires, 'max_hits': fb_max_hits_hires}
        if fb_use_first_settings: self.fb_params_runtime['hires']['enabled'] = fb_enabled_hires
        self.freeu_params_runtime = {'enabled': freeu_enabled, 'b1': freeu_b1, 'b2': freeu_b2, 's1': freeu_s1, 's2': freeu_s2, 'start_at': freeu_start_at, 'stop_at': freeu_stop_at}
        if self.fb_params_runtime['first']['enabled'] or (self.fb_params_runtime['hires']['enabled'] and getattr(p, 'enable_hr', False)): p.extra_generation_params["Integrated FBCache"] = "Enabled"
        if freeu_enabled: p.extra_generation_params["Integrated FreeU"] = "Enabled"

    def process_before_every_sampling(self, p, *args, **kwargs):
        pass_type = "hires" if getattr(p, 'is_hr_pass', False) else "first"
        fb_params_for_pass = self.fb_params_runtime.get(pass_type, {})
        is_fb_enabled = fb_params_for_pass.get('enabled', False)
        is_freeu_enabled = self.freeu_params_runtime.get('enabled', False)
        
        if not is_fb_enabled and not is_freeu_enabled:
            self._restore_original_k_model(p)
            return

        self.log_info(f"[{pass_type} pass] Applying patches. FBCache: {is_fb_enabled}, FreeU: {is_freeu_enabled}")
        
        k_model_instance = p.sd_model.forge_objects.unet.model
        
        if self.original_apply_model_method is None:
            self.original_k_model = k_model_instance
            self.original_apply_model_method = k_model_instance.apply_model

        self.runtime_params_for_patch = self._prepare_runtime_params(p, pass_type)

        if is_fb_enabled:
            if self.active_fb_state_object is None:
                self.log_info("Initializing FBCache state for this generation.")
                model_dtype = k_model_instance.computation_dtype
                self.active_fb_state_object = FBCacheState(weakref.ref(k_model_instance.diffusion_model), model_dtype, self.is_debug_logging_enabled)
            self.active_fb_state_object.check_and_clear_if_critical_params_changed(pass_type, self.runtime_params_for_patch['fb_cache_params'])
        
        script_instance = self
        original_apply_model = self.original_apply_model_method

        @wraps(original_apply_model)
        def patched_apply_model_wrapper(x, t, **kwargs):
            return patched_k_model_apply_logic(k_model_instance, x, t,
                                               script_instance=script_instance,
                                               original_apply_callable=original_apply_model,
                                               **kwargs)

        k_model_instance.apply_model = patched_apply_model_wrapper

    def _prepare_runtime_params(self, p, pass_type):
        fb_params_for_pass = self.fb_params_runtime.get(pass_type, {})
        runtime_params = {'fb_cache_params': {}, 'freeu_params': self.freeu_params_runtime}
        if fb_params_for_pass.get('enabled', False):
            total_steps = p.hr_second_pass_steps if pass_type == "hires" and hasattr(p, 'hr_second_pass_steps') and p.hr_second_pass_steps > 0 else p.steps
            runtime_params['fb_cache_params'] = {'enabled': True, 'current_pass_type': pass_type, 'threshold': fb_params_for_pass.get('threshold'), 'num_initial_blocks': fb_params_for_pass.get('blocks'), 'start_step': int(fb_params_for_pass.get('start', 0.0) * total_steps), 'end_step': int(fb_params_for_pass.get('end', 1.0) * total_steps), 'max_hits': int(fb_params_for_pass.get('max_hits', -1))}
        if self.freeu_params_runtime.get('enabled', False):
            diffusion_model = p.sd_model.forge_objects.unet.model.diffusion_model
            model_channels = getattr(diffusion_model, 'model_channels', 320)
            runtime_params['freeu_params']['scale_dict'] = {model_channels * 4: (self.freeu_params_runtime['b1'], self.freeu_params_runtime['s1']), model_channels * 2: (self.freeu_params_runtime['b2'], self.freeu_params_runtime['s2'])}
            runtime_params['freeu_params']['on_cpu_devices_ref'] = {}
        return runtime_params

    def _restore_original_k_model(self, p):
        if self.original_k_model is not None and hasattr(p, 'sd_model'):
            if hasattr(self.original_k_model, 'apply_model'):
                self.original_k_model.apply_model = self.original_apply_model_method
                self.log_debug("Restored original K-Model apply_model method.")
        self.original_k_model = None
        self.original_apply_model_method = None

    def postprocess(self, p, processed, *args):
        self.log_info("Restoring original K-Model state after generation.")
        self._restore_original_k_model(p)
        if self.active_fb_state_object:
            summary = self.active_fb_state_object.get_hit_rate_summary()
            if summary and "not loaded" not in summary and "0 calls" not in summary: self.log_info(f"Final FBCache Stats: {summary}")
            else: self.log_info("FBCache: No cache activity was recorded.")
            self.active_fb_state_object = None
        self.log_debug("Postprocess finished.")
        return processed
        
    def on_script_unloaded(self):
        self.log_info("Script unloading, attempting to restore original K-Model state.")
        if shared.p: self._restore_original_k_model(shared.p)
        if IntegratedUtilsScript._instance == self: IntegratedUtilsScript._instance = None

# --- FreeU Injection Logic ---
def freeu_patch(h, hsp, scale_dict, on_cpu_devices_ref):
    return apply_freeu_scaling(h, hsp, scale_dict, on_cpu_devices_ref)

# --- Main patch logic function ---
def patched_k_model_apply_logic(k_model_instance, x, t, *, script_instance, original_apply_callable, **kwargs):
    params = script_instance.runtime_params_for_patch
    fb_params = params.get('fb_cache_params', {})
    freeu_params = params.get('freeu_params', {})
    is_fb_enabled = fb_params.get('enabled', False)
    is_freeu_enabled = freeu_params.get('enabled', False) and freeu_params.get('start_at', 1.0) < freeu_params.get('stop_at', 0.0)

    # --- FreeU Application (before original call) ---
    if is_freeu_enabled:
        total_steps = shared.state.sampling_steps if hasattr(shared.state, 'sampling_steps') and shared.state.sampling_steps > 0 else 1
        progress = (shared.state.sampling_step / (total_steps - 1)) if total_steps > 1 else 1.0
        
        if freeu_params.get('start_at', 0.0) <= progress <= freeu_params.get('stop_at', 1.0):
            if 'transformer_options' not in kwargs: kwargs['transformer_options'] = {}
            
            # This is how FreeU is implemented in ComfyUI
            kwargs['transformer_options']['freeu_patch'] = {
                'h_patch': lambda h, hsp: freeu_patch(h, hsp, freeu_params['scale_dict'], freeu_params['on_cpu_devices_ref'])
            }
            script_instance.log_debug(f"FreeU patch injected at step {shared.state.sampling_step}.")

    # --- FBCache Pre-computation & Check ---
    is_fbcache_active_for_step = False
    if is_fb_enabled:
        current_step_index = shared.state.sampling_step if hasattr(shared.state, 'sampling_step') else 0
        is_fbcache_active_for_step = (fb_params.get('start_step', 0) <= current_step_index < fb_params.get('end_step', float('inf')))
        
        if is_fbcache_active_for_step:
            fb_state = script_instance.active_fb_state_object
            bs, pt = x.shape[0], fb_params.get('current_pass_type')
            fb_state.record_call(bs, pt)
            script_instance.log_debug(f"FBCache Check: BS {bs} ({pt}), Step {current_step_index}. Active: True.")
            
            if fb_state.get_key(bs, pt) is not None:
                max_hits = fb_params.get('max_hits', -1)
                consecutive_hits = fb_state.get_consecutive_hits(bs, pt)
                
                # We use the denoised output `x` as the key for similarity check
                if max_hits < 0 or consecutive_hits < max_hits:
                    if are_two_tensors_similar(fb_state.get_key(bs, pt), x, fb_params.get('threshold', 0.1), bs, pt, script_instance.is_debug_logging_enabled) and fb_state.get_residual(bs, pt) is not None:
                        script_instance.log_debug(f"FBCache: HIT on BS {bs} ({pt}). Applying cached residual.")
                        cached_residual = fb_state.get_residual(bs, pt)
                        fb_state.increment_consecutive_hits(bs, pt)
                        # The "residual" is the final denoised output from the previous step.
                        return cached_residual.to(x.device, dtype=x.dtype)

    # --- Original Model Call ---
    model_output = original_apply_callable(x=x, t=t, **kwargs)

    # --- FBCache Post-computation & Storage ---
    if is_fbcache_active_for_step:
        bs, pt = x.shape[0], fb_params.get('current_pass_type')
        script_instance.log_debug(f"FBCache: MISS on BS {bs} ({pt}). Storing new key and residual.")
        fb_state.reset_consecutive_hits(bs, pt)
        fb_state.store_key(x, bs, pt) # Store input `x` as key
        fb_state.store_residual(model_output, bs, pt) # Store `model_output` as the result to cache

    return model_output

script_callbacks.on_script_unloaded(lambda: IntegratedUtilsScript._instance.on_script_unloaded() if IntegratedUtilsScript._instance else None)

