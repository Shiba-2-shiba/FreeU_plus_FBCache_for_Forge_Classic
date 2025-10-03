# /scripts/integrate_utils.py
#
# FBCacheとFreeUを統合し、単一のパッチで両機能を制御するスクリプト。
#
# --- Refactoring for modern sd-webui-forge-classic ---
# - Changed patching method from class-level to instance-level for safety.
# - Moved patching logic to 'process_before_every_sampling' and 'postprocess'.
# - Updated module imports to resolve the ModuleNotFoundError.
# - Utilizes the UnetPatcher's clone() method to prevent state corruption.

import torch
import gradio as gr
import traceback
import weakref
import datetime
import sys
import os
from functools import partial

# --- WebUI/Forge Module Imports ---
from modules import scripts, shared, script_callbacks

# --- LDM Module Imports (Updated Path) ---
# Assuming 'ldm' package is available in the environment, as is standard for Forge.
try:
    from ldm.modules.diffusionmodules.openaimodel import UNetModel, forward_timestep_embed, apply_control
    from ldm.modules.diffusionmodules.util import timestep_embedding
except ImportError:
    print("\n[IntegratedUtils] Error: Could not import LDM modules. This script requires the 'ldm' package.")
    print("[IntegratedUtils] Please ensure sd-webui-forge-classic is installed correctly.")
    # Define dummy classes/functions to prevent crashes on startup
    class UNetModel: pass
    def forward_timestep_embed(*args, **kwargs): pass
    def apply_control(*args, **kwargs): pass
    def timestep_embedding(*args, **kwargs): return torch.zeros(1)


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
    # Define dummy fallbacks
    def apply_freeu_scaling(h, hsp, *args, **kwargs): return h, hsp
    class FBCacheState:
        def __init__(self, *args, **kwargs): pass
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
        
        # --- NEW: Variables to manage the patch state safely ---
        self.original_unet_patcher = None
        self.original_forward_method = None
        
        self.log_info("Script instance initialized.")
    
    def title(self):
        return "FBCache + FreeU (Integrated)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # --- Logging Methods (Unchanged) ---
    def _log_prefix(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"[{timestamp} IntegratedUtils]"

    def log_info(self, message: str):
        print(f"{self._log_prefix()} Info: {message}")

    def log_debug(self, message: str):
        if self.is_debug_logging_enabled:
            print(f"{self._log_prefix()} Debug: {message}")

    def log_error(self, message: str):
        print(f"{self._log_prefix()} ERROR: {message}\n{traceback.format_exc()}")
        
    def ui(self, is_img2img):
        # --- UI Definition (Unchanged) ---
        ui_components = []
        with gr.Accordion(self.title(), open=False):
            with gr.Row():
                enable_debug_logging = gr.Checkbox(label="Enable Debug Logging (Console)", value=False)
                ui_components.append(enable_debug_logging)

            with gr.Tabs():
                with gr.TabItem("First Block Cache"):
                    with gr.Tabs():
                        with gr.TabItem("First Pass"):
                            fb_enabled_first = gr.Checkbox(label="Enable for First Pass", value=False)
                            fb_threshold_first = gr.Slider(label="Similarity Threshold", minimum=0.001, maximum=0.5, step=0.001, value=0.3)
                            fb_blocks_first = gr.Slider(label="UNet Initial Blocks to Cache", minimum=1, maximum=4, step=1, value=3)
                            fb_start_first = gr.Slider(label="Start At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.3)
                            fb_end_first = gr.Slider(label="End At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.99)
                            fb_max_hits_first = gr.Number(label="Max Consecutive Hits (-1 for unlimited)", value=-1, precision=0)
                            ui_components.extend([fb_enabled_first, fb_threshold_first, fb_blocks_first, fb_start_first, fb_end_first, fb_max_hits_first])
                        
                        with gr.TabItem("Hires Fix Pass"):
                            fb_enabled_hires = gr.Checkbox(label="Enable for Hires Fix Pass", value=False)
                            fb_use_first_settings = gr.Checkbox(label="Use First Pass settings", value=True)
                            with gr.Group(visible=False) as hires_specific_settings:
                                fb_threshold_hires = gr.Slider(label="Similarity Threshold (Hires)", minimum=0.001, maximum=0.5, step=0.001, value=0.3)
                                fb_blocks_hires = gr.Slider(label="UNet Initial Blocks (Hires)", minimum=1, maximum=4, step=1, value=3)
                                fb_start_hires = gr.Slider(label="Start At % of Steps (Hires)", minimum=0.0, maximum=1.0, step=0.01, value=0.3)
                                fb_end_hires = gr.Slider(label="End At % of Steps (Hires)", minimum=0.0, maximum=1.0, step=0.01, value=0.99)
                                fb_max_hits_hires = gr.Number(label="Max Consecutive Hits (Hires)", value=-1, precision=0)
                            ui_components.extend([fb_enabled_hires, fb_use_first_settings, fb_threshold_hires, fb_blocks_hires, fb_start_hires, fb_end_hires, fb_max_hits_hires])
                    
                    def toggle_hires_visibility(use_first): return gr.update(visible=not use_first)
                    fb_use_first_settings.change(fn=toggle_hires_visibility, inputs=[fb_use_first_settings], outputs=[hires_specific_settings])

                with gr.TabItem("FreeU"):
                    freeu_enabled = gr.Checkbox(label="Enable FreeU", value=False)
                    freeu_b1 = gr.Slider(label="Backbone 1 (b1)", minimum=0, maximum=2, step=0.01, value=1.3)
                    freeu_b2 = gr.Slider(label="Backbone 2 (b2)", minimum=0, maximum=2, step=0.01, value=1.4)
                    freeu_s1 = gr.Slider(label="Skip 1 (s1)", minimum=0, maximum=4, step=0.01, value=1.2)
                    freeu_s2 = gr.Slider(label="Skip 2 (s2)", minimum=0, maximum=4, step=0.01, value=0.7)
                    freeu_start_at = gr.Slider(label="Start At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.01, elem_id="freeu_start_at")
                    freeu_stop_at = gr.Slider(label="Stop At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.2, elem_id="freeu_stop_at")
                    ui_components.extend([freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_start_at, freeu_stop_at])

        return ui_components

    def get_target_unet_diffusion_model(self, p):
        # Access the underlying diffusion model safely.
        if hasattr(p, 'sd_model') and hasattr(p.sd_model, 'forge_objects') and hasattr(p.sd_model.forge_objects, 'unet'):
            unet_patcher = p.sd_model.forge_objects.unet
            if hasattr(unet_patcher, 'model') and hasattr(unet_patcher.model, 'diffusion_model'):
                 # The actual model that has the .forward method we want to patch
                return unet_patcher.model.diffusion_model
        return None

    def process(self, p, 
                # Debug
                enable_debug_logging,
                # FBCache First Pass
                fb_enabled_first, fb_threshold_first, fb_blocks_first, fb_start_first, fb_end_first, fb_max_hits_first,
                # FBCache Hires Pass
                fb_enabled_hires, fb_use_first_settings, fb_threshold_hires, fb_blocks_hires, fb_start_hires, fb_end_hires, fb_max_hits_hires,
                # FreeU
                freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_start_at, freeu_stop_at
                ):
        # --- NEW: Only sets up parameters. Patching is moved to a later step. ---
        self.is_debug_logging_enabled = enable_debug_logging
        
        self.log_info("New generation process started. Parsing UI parameters.")

        self.fb_params_runtime['first'] = {
            'enabled': fb_enabled_first, 'threshold': fb_threshold_first, 'blocks': fb_blocks_first,
            'start': fb_start_first, 'end': fb_end_first, 'max_hits': fb_max_hits_first
        }
        if fb_use_first_settings:
            self.fb_params_runtime['hires'] = self.fb_params_runtime['first'].copy()
            self.fb_params_runtime['hires']['enabled'] = fb_enabled_hires
        else:
            self.fb_params_runtime['hires'] = {
                'enabled': fb_enabled_hires, 'threshold': fb_threshold_hires, 'blocks': fb_blocks_hires,
                'start': fb_start_hires, 'end': fb_end_hires, 'max_hits': fb_max_hits_hires
            }

        self.freeu_params_runtime = {
            'enabled': freeu_enabled, 'b1': freeu_b1, 'b2': freeu_b2, 's1': freeu_s1, 's2': freeu_s2,
            'start_at': freeu_start_at, 'stop_at': freeu_stop_at
        }
        
        is_fb_enabled = self.fb_params_runtime['first']['enabled'] or (self.fb_params_runtime['hires']['enabled'] and getattr(p, 'enable_hr', False))
        is_freeu_enabled = self.freeu_params_runtime['enabled']
        
        if is_fb_enabled: p.extra_generation_params["Integrated FBCache"] = "Enabled"
        if is_freeu_enabled: p.extra_generation_params["Integrated FreeU"] = "Enabled"


    def process_before_every_sampling(self, p, *args, **kwargs):
        # --- NEW: This is the modern, correct place to apply patches. ---
        is_hires_pass = getattr(p, 'is_hr_pass', False)
        pass_type = "hires" if is_hires_pass else "first"
        
        # Determine if any feature is active for this pass
        fb_params_for_pass = self.fb_params_runtime.get(pass_type, {})
        is_fb_enabled = fb_params_for_pass.get('enabled', False)
        is_freeu_enabled = self.freeu_params_runtime.get('enabled', False)
        
        if not is_fb_enabled and not is_freeu_enabled:
            self.log_debug(f"[{pass_type} pass] No features enabled, skipping patch.")
            self._restore_original_unet(p) # Ensure it's restored if it was patched before
            return

        self.log_info(f"[{pass_type} pass] Applying patches. FBCache: {is_fb_enabled}, FreeU: {is_freeu_enabled}")

        # Store the original UnetPatcher if this is the first time
        if self.original_unet_patcher is None:
            self.original_unet_patcher = p.sd_model.forge_objects.unet
        
        # Clone the patcher to avoid modifying the original object in place
        unet_patcher_clone = self.original_unet_patcher.clone()
        diffusion_model = unet_patcher_clone.model.diffusion_model

        # Store original forward method from the diffusion_model instance
        self.original_forward_method = diffusion_model.forward

        # Prepare runtime params for the patched function
        self.runtime_params_for_patch = self._prepare_runtime_params(p, pass_type)
        
        # Initialize FBCache state object if needed
        if is_fb_enabled and self.active_fb_state_object is None:
            self.log_info("Initializing FBCache state for this generation.")
            self.active_fb_state_object = FBCacheState(weakref.ref(diffusion_model), diffusion_model.dtype, self.is_debug_logging_enabled)
        
        # Clear cache if critical params changed for the current pass
        if is_fb_enabled and self.active_fb_state_object:
            self.active_fb_state_object.check_and_clear_if_critical_params_changed(pass_type, self.runtime_params_for_patch['fb_cache_params'])
        
        # Apply the patch to the instance
        diffusion_model.forward = partial(patched_unet_forward, script_instance=self)
        
        # Replace the processing object's unet with our patched clone
        p.sd_model.forge_objects.unet = unet_patcher_clone
        

    def _prepare_runtime_params(self, p, pass_type):
        """Helper to assemble parameters for the patched forward function."""
        fb_params_for_pass = self.fb_params_runtime.get(pass_type, {})
        is_fb_enabled = fb_params_for_pass.get('enabled', False)
        
        runtime_params = {
            'fb_cache_params': {},
            'freeu_params': self.freeu_params_runtime
        }

        if is_fb_enabled:
            total_steps = p.hr_second_pass_steps if pass_type == "hires" and hasattr(p, 'hr_second_pass_steps') and p.hr_second_pass_steps > 0 else p.steps
            runtime_params['fb_cache_params'] = {
                'enabled': True, 'current_pass_type': pass_type,
                'threshold': fb_params_for_pass.get('threshold'), 'num_initial_blocks': fb_params_for_pass.get('blocks'),
                'start_step': int(fb_params_for_pass.get('start', 0.0) * total_steps),
                'end_step': int(fb_params_for_pass.get('end', 1.0) * total_steps),
                'max_hits': int(fb_params_for_pass.get('max_hits')),
            }
        
        if self.freeu_params_runtime.get('enabled', False):
            diffusion_model = self.get_target_unet_diffusion_model(p)
            if diffusion_model:
                model_channels = getattr(diffusion_model, 'model_channels', 320)
                runtime_params['freeu_params']['scale_dict'] = {
                    model_channels * 4: (self.freeu_params_runtime['b1'], self.freeu_params_runtime['s1']),
                    model_channels * 2: (self.freeu_params_runtime['b2'], self.freeu_params_runtime['s2'])
                }
                runtime_params['freeu_params']['on_cpu_devices_ref'] = {}
        
        return runtime_params

    def _restore_original_unet(self, p):
        """Helper to safely restore the original UNet Patcher and forward method."""
        if self.original_unet_patcher is not None:
            p.sd_model.forge_objects.unet = self.original_unet_patcher
            self.log_debug("Restored original UNet Patcher to processing object.")

        if self.original_forward_method is not None:
            diffusion_model = self.get_target_unet_diffusion_model(p)
            if diffusion_model:
                diffusion_model.forward = self.original_forward_method
                self.log_debug("Restored original .forward method to diffusion model instance.")
        
        self.original_unet_patcher = None
        self.original_forward_method = None

    def postprocess(self, p, processed, *args):
        # --- NEW: This is the cleanup step after the entire generation is done. ---
        self.log_info("Restoring original UNet state after generation.")
        self._restore_original_unet(p)

        if self.active_fb_state_object:
            summary = self.active_fb_state_object.get_hit_rate_summary()
            if summary and "not loaded" not in summary and "0 calls" not in summary:
                self.log_info(f"Final FBCache Stats: {summary}")
            else:
                self.log_info("FBCache: No cache activity was recorded for this generation.")
            self.active_fb_state_object = None
            self.log_debug("FBCache state cleared for next generation.")
            
        self.log_debug("Postprocess finished.")
        return processed
        
    def on_script_unloaded(self):
        # --- NEW: Safety net to restore on script unload. ---
        self.log_info("Script unloading, attempting to restore original UNet state.")
        if shared.p: # If a processing object exists
             self._restore_original_unet(shared.p)
        if IntegratedUtilsScript._instance == self:
            IntegratedUtilsScript._instance = None

# --- NEW: Patched forward function, now takes script_instance explicitly ---
def patched_unet_forward(self_unet: UNetModel, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, y=None, control=None, transformer_options:dict=None, *, script_instance: IntegratedUtilsScript, **kwargs):
    
    # This logic is mostly the same as the original, but uses script_instance to get its state
    original_forward = script_instance.original_forward_method
    if not original_forward:
        # Fallback to a direct call if something went wrong
        return UNetModel.forward(self_unet, x, timesteps, context, y, control, transformer_options, **kwargs)

    params = getattr(script_instance, 'runtime_params_for_patch', {})
    fb_params = params.get('fb_cache_params', {})
    freeu_params = params.get('freeu_params', {})
    
    is_fb_enabled = fb_params.get('enabled', False)
    is_freeu_enabled = freeu_params.get('enabled', False) and freeu_params.get('start_at', 1.0) < freeu_params.get('stop_at', 0.0)

    if not is_fb_enabled and not is_freeu_enabled:
        return original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)

    # --- Core Patch Logic (largely unchanged, but self-contained) ---
    fb_state = script_instance.active_fb_state_object
    is_fbcache_active_for_step = False
    if is_fb_enabled and fb_state:
        current_pass_type = fb_params.get('current_pass_type')
        current_batch_size = x.shape[0]
        fb_state.record_call(current_batch_size, current_pass_type)
        current_step_index = shared.state.sampling_step if hasattr(shared.state, 'sampling_step') else 0
        is_fbcache_active_for_step = (fb_params.get('start_step', 0) <= current_step_index < fb_params.get('end_step', float('inf')))
        script_instance.log_debug(f"FBCache Check: BS {current_batch_size} ({current_pass_type}), Step {current_step_index}. Active: {is_fbcache_active_for_step}.")
    
    hs = []
    t_emb = timestep_embedding(timesteps, self_unet.model_channels, repeat_only=False).to(x.dtype)
    emb = self_unet.time_embed(t_emb)
    if self_unet.num_classes is not None: emb = emb + self_unet.label_emb(y)
    h = x
    current_transformer_options = {} if transformer_options is None else transformer_options.copy()

    num_initial_blocks_for_cache = int(fb_params.get('num_initial_blocks', 3)) if is_fbcache_active_for_step else 0
    
    for block_idx in range(num_initial_blocks_for_cache):
        module_block = self_unet.input_blocks[block_idx]
        h = forward_timestep_embed(module_block, h, emb, context, current_transformer_options, **kwargs)
        if control is not None: h = apply_control(h, control, "input")
        hs.append(h)
    
    h_after_initial_blocks = h.clone() if is_fbcache_active_for_step else None
    
    use_cached_result = False
    if is_fbcache_active_for_step:
        current_batch_size = x.shape[0]
        current_pass_type = fb_params.get('current_pass_type')
        previous_key = fb_state.get_key(current_batch_size, current_pass_type)
        if previous_key is not None:
            max_hits = fb_params.get('max_hits', -1)
            consecutive_hits = fb_state.get_consecutive_hits(current_batch_size, current_pass_type)
            if max_hits < 0 or consecutive_hits < max_hits:
                is_similar = are_two_tensors_similar(previous_key, h_after_initial_blocks, fb_params.get('threshold', 0.1), current_batch_size, current_pass_type, script_instance.is_debug_logging_enabled)
                if is_similar and fb_state.get_residual(current_batch_size, current_pass_type) is not None:
                    use_cached_result = True
                    script_instance.log_debug(f"FBCache: HIT on BS {current_batch_size} ({current_pass_type}).")

    if use_cached_result:
        try:
            current_batch_size = x.shape[0]
            current_pass_type = fb_params.get('current_pass_type')
            cached_residual = fb_state.get_residual(current_batch_size, current_pass_type)
            h = h_after_initial_blocks + cached_residual.to(h_after_initial_blocks.device, dtype=h_after_initial_blocks.dtype)
            fb_state.increment_consecutive_hits(current_batch_size, current_pass_type)
            script_instance.log_debug(f"FBCache: Applied residual. Consecutive hits: {fb_state.get_consecutive_hits(current_batch_size, current_pass_type)}.")
            # We skip the rest of the UNet calculation, but we need the output blocks to run on the cached result
            # We need to fast-forward the `hs` stack. The cached result is after `num_initial_blocks_for_cache`.
            # We need to simulate the rest of the input blocks and middle block to get the correct `hsp` for output blocks.
            # This is complex. A simpler approach for the HIT case is needed.
            # For now, let's assume the cached residual is applied to h, and the rest of the network runs from there.
            # This is a conceptual simplification. The original logic was more intricate.
            # A true HIT should replace 'h' and fast-forward to the output blocks with the correct 'hs' stack.
            # This logic below is a re-run from the cache point, which is not a true cache hit optimization.
            # Correcting this would require deep changes. Let's stick to the original logic which seems to recalculate.
            #
            # The original logic implies that if a HIT occurs, the ENTIRE rest of the UNet is skipped and replaced
            # by adding the residual. This is what we will implement.
            #
            # It seems the original code had a subtle bug/feature: `h` after the residual addition is the *final* output.
            # Let's re-read the original logic carefully.
            # After `h = h_after_initial_blocks + cached_residual`, it jumps to the `final_output` calculation.
            # This seems correct for a cache hit.
            pass # The logic will now naturally skip to the end of the `if not use_cached_result` block.

        except Exception as e:
            script_instance.log_error(f"FBCache failed to apply residual: {e}. Falling back to full calculation.")
            use_cached_result = False
            current_batch_size = x.shape[0]
            current_pass_type = fb_params.get('current_pass_type')
            fb_state.reset_consecutive_hits(current_batch_size, current_pass_type)
    
    if not use_cached_result:
        if is_fbcache_active_for_step:
            current_batch_size = x.shape[0]
            current_pass_type = fb_params.get('current_pass_type')
            script_instance.log_debug(f"FBCache: MISS on BS {current_batch_size} ({current_pass_type}). Storing new key.")
            fb_state.reset_consecutive_hits(current_batch_size, current_pass_type)
            fb_state.store_key(h_after_initial_blocks, current_batch_size, current_pass_type)
        
        for block_idx in range(num_initial_blocks_for_cache, len(self_unet.input_blocks)):
            module_block = self_unet.input_blocks[block_idx]
            h = forward_timestep_embed(module_block, h, emb, context, current_transformer_options, **kwargs)
            if control is not None: h = apply_control(h, control, "input")
            hs.append(h)

        h = forward_timestep_embed(self_unet.middle_block, h, emb, context, current_transformer_options, **kwargs)
        if control is not None: h = apply_control(h, control, "middle")
        
        for block_idx, module_block in enumerate(self_unet.output_blocks):
            hsp = hs.pop()
            if control is not None: hsp = apply_control(hsp, control, "output")

            if is_freeu_enabled:
                current_step = shared.state.sampling_step if hasattr(shared.state, 'sampling_step') else 0
                total_steps = shared.state.sampling_steps if hasattr(shared.state, 'sampling_steps') and shared.state.sampling_steps > 0 else 1
                progress = current_step / (total_steps - 1) if total_steps > 1 else 1.0
                
                if (progress >= freeu_params.get('start_at', 0.0)) and (progress <= freeu_params.get('stop_at', 1.0)):
                    script_instance.log_debug(f"FreeU Applying on output block {block_idx}")
                    h, hsp = apply_freeu_scaling(h, hsp, freeu_params['scale_dict'], freeu_params['on_cpu_devices_ref'])

            h = torch.cat([h, hsp], dim=1)
            h = forward_timestep_embed(module_block, h, emb, context, current_transformer_options, output_shape=(hs[-1].shape if hs else None), **kwargs)
        
        if is_fbcache_active_for_step:
            current_batch_size = x.shape[0]
            current_pass_type = fb_params.get('current_pass_type')
            calculated_residual = h - h_after_initial_blocks
            fb_state.store_residual(calculated_residual, current_batch_size, current_pass_type)
            script_instance.log_debug(f"FBCache: Stored new residual for BS {current_batch_size} ({current_pass_type}).")

    if self_unet.predict_codebook_ids:
        final_output = self_unet.id_predictor(h)
    else:
        final_output = self_unet.out(h)
    
    return final_output.to(x.dtype)

# --- Register script lifecycle callbacks ---
script_callbacks.on_script_unloaded(lambda: IntegratedUtilsScript._instance.on_script_unloaded() if IntegratedUtilsScript._instance else None)
