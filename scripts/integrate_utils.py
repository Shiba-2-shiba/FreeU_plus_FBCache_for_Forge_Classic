# /scripts/integrate_utils.py
#
# FBCacheとFreeUを統合し、単一のパッチで両機能を制御するスクリプト。
#
# 改修履歴:
# - v3.4 (Global Patch Version):
#   - パッチが他の拡張機能によって上書きされる問題を解決するため、グローバル・モンキーパッチ方式に変更。
#     - UNetModelクラスのforwardメソッドを直接書き換えることで、実行順序に依存せず機能することを保証。
#   - processでパッチを適用し、postprocessで確実に元のメソッドに戻すようにライフサイクルを管理。
#   - 実行時のパラメータをスクリプトのクラスインスタンス経由でパッチ関数に渡すように変更。

import torch
import gradio as gr
import traceback
import weakref
import datetime
import sys
import os
from functools import wraps

# --- WebUI/Forgeのモジュールインポート ---
from modules import scripts, shared, script_callbacks
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel, forward_timestep_embed, apply_control
from ldm_patched.ldm.modules.diffusionmodules.util import timestep_embedding
from ldm_patched.modules import model_management

# --- パス解決とコアモジュールのインポート ---
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
    print("[IntegratedUtils] Please ensure 'freeu_core.py' and 'fb_cache_core.py' exist in the 'modules' subdirectory.")
    print("[IntegratedUtils] Script will not function correctly.\n")
    def apply_freeu_scaling(h, hsp, *args, **kwargs): return h, hsp
    class FBCacheState:
        def __init__(self, *args, **kwargs): pass
        def record_call(self, *args, **kwargs): pass
        def get_consecutive_hits(self, *args, **kwargs): return 0
        def get_key(self, *args, **kwargs): return None
        def get_residual(self, *args, **kwargs): return None
        def reset_consecutive_hits(self, *args, **kwargs): pass
        def store_key(self, *args, **kwargs): pass
        def increment_consecutive_hits(self, *args, **kwargs): pass
        def store_residual(self, *args, **kwargs): pass
        def check_and_clear_if_critical_params_changed(self, *args, **kwargs): pass
        def get_hit_rate_summary(self, *args, **kwargs): return "Core modules not loaded."
        def clear_all_data(self, *args, **kwargs): pass
    def are_two_tensors_similar(*args, **kwargs): return False

# --- メインのスクリプトクラス ---
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
        self.original_forward = None
        self.log_info("Script instance initialized.")
    
    def title(self):
        return "FBCache + FreeU (Integrated)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

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
        # UI部分は変更なし
        ui_components = []
        with gr.Accordion(self.title(), open=False):
            with gr.Row():
                enable_debug_logging = gr.Checkbox(label="Enable Debug Logging (Console)", value=False)
                ui_components.append(enable_debug_logging)

            with gr.Tabs():
                with gr.TabItem("First Block Cache"):
                    fb_enabled_first = gr.Checkbox(label="Enable for First Pass", value=False)
                    fb_threshold_first = gr.Slider(label="Similarity Threshold", minimum=0.001, maximum=0.5, step=0.001, value=0.3)
                    fb_blocks_first = gr.Slider(label="UNet Initial Blocks to Cache", minimum=1, maximum=4, step=1, value=3)
                    fb_start_first = gr.Slider(label="Start At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.3)
                    fb_end_first = gr.Slider(label="End At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.99)
                    fb_max_hits_first = gr.Number(label="Max Consecutive Hits (-1 for unlimited)", value=-1, precision=0)
                    ui_components.extend([fb_enabled_first, fb_threshold_first, fb_blocks_first, fb_start_first, fb_end_first, fb_max_hits_first])
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

    def get_target_unet_model(self):
        if not (shared.sd_model and hasattr(shared.sd_model, 'forge_objects') and hasattr(shared.sd_model.forge_objects, 'unet')):
             return None
        unet_candidate = shared.sd_model.forge_objects.unet
        q = [(unet_candidate, 0)]; visited_ids = {id(unet_candidate)}
        while q:
            current_obj, depth = q.pop(0)
            if depth > 10: continue
            if isinstance(current_obj, UNetModel): return current_obj
            for attr_name in ['model', 'diffusion_model', 'wrapped', 'patcher', '_model', 'unet']:
                if hasattr(current_obj, attr_name):
                    inner_obj = getattr(current_obj, attr_name)
                    if inner_obj is not None and id(inner_obj) not in visited_ids:
                        visited_ids.add(id(inner_obj)); q.append((inner_obj, depth + 1))
        return None

    def process(self, p, enable_debug_logging,
                fb_enabled_first, fb_threshold_first, fb_blocks_first, fb_start_first, fb_end_first, fb_max_hits_first,
                freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_start_at, freeu_stop_at
                ):
        self.is_debug_logging_enabled = enable_debug_logging
        self.active_fb_state_object = None

        # Hires Fixは現在サポート外としてUIを単純化
        fb_params = {
            'enabled': fb_enabled_first, 'threshold': fb_threshold_first, 'blocks': fb_blocks_first,
            'start': fb_start_first, 'end': fb_end_first, 'max_hits': fb_max_hits_first
        }
        self.fb_params_runtime = fb_params

        freeu_params = {
            'enabled': freeu_enabled, 'b1': freeu_b1, 'b2': freeu_b2, 's1': freeu_s1, 's2': freeu_s2,
            'start_at': freeu_start_at, 'stop_at': freeu_stop_at
        }
        self.freeu_params_runtime = freeu_params
        
        is_fb_enabled = fb_params['enabled']
        is_freeu_enabled = freeu_params['enabled'] and freeu_params['start_at'] < freeu_params['stop_at']
        
        if not is_fb_enabled and not is_freeu_enabled:
            self.log_debug("Both features disabled, skipping patch.")
            return

        self.log_info(f"Applying global patch. FBCache: {is_fb_enabled}, FreeU: {is_freeu_enabled}")
        
        try:
            # ★★★ 修正点: グローバルパッチを適用 ★★★
            self.original_forward = UNetModel.forward
            UNetModel.forward = patched_unet_forward
        except Exception as e:
            self.log_error(f"Failed to apply global UNet patch: {e}")

        p.extra_generation_params["Integrated FBCache Enabled"] = is_fb_enabled
        p.extra_generation_params["Integrated FreeU Enabled"] = is_freeu_enabled

    def process_before_every_sampling(self, p, *args, **kwargs):
        if self.original_forward is None: return

        # ★★★ 修正点: ランタイムパラメータをインスタンスに設定 ★★★
        pass_type = "hires" if getattr(p, 'is_hr_pass', False) else "first"
        
        # FBCache
        fb_enabled = self.fb_params_runtime.get('enabled', False)
        if fb_enabled:
            total_steps = p.hr_second_pass_steps if pass_type == "hires" and p.hr_second_pass_steps > 0 else p.steps
            self.fb_params_runtime['current_pass_type'] = pass_type
            self.fb_params_runtime['start_step'] = int(self.fb_params_runtime.get('start', 0.0) * total_steps)
            self.fb_params_runtime['end_step'] = int(self.fb_params_runtime.get('end', 1.0) * total_steps)

            if self.active_fb_state_object is None:
                unet_model = self.get_target_unet_model()
                if unet_model:
                    self.log_info("Initializing FBCache state for the first time in this generation.")
                    self.active_fb_state_object = FBCacheState(weakref.ref(unet_model), unet_model.dtype, self.is_debug_logging_enabled)
            
            if self.active_fb_state_object:
                self.active_fb_state_object.check_and_clear_if_critical_params_changed(pass_type, self.fb_params_runtime)
        
        # FreeU
        if self.freeu_params_runtime.get('enabled', False):
            unet_model = self.get_target_unet_model()
            if unet_model:
                model_channels = getattr(unet_model, 'model_channels', 320)
                self.freeu_params_runtime['scale_dict'] = {
                    model_channels * 4: (self.freeu_params_runtime['b1'], self.freeu_params_runtime['s1']),
                    model_channels * 2: (self.freeu_params_runtime['b2'], self.freeu_params_runtime['s2'])
                }
                self.freeu_params_runtime['on_cpu_devices_ref'] = {}
        
        self.log_debug(f"Runtime params updated for {pass_type} pass.")


    def postprocess(self, p, processed, *args):
        # ★★★ 修正点: グローバルパッチを確実に戻す ★★★
        if self.original_forward is not None:
            self.log_info("Restoring original UNet forward method.")
            UNetModel.forward = self.original_forward
            self.original_forward = None

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
        self.log_info("Script unloading, restoring original forward method if patched.")
        if self.original_forward is not None:
            UNetModel.forward = self.original_forward
            self.original_forward = None
        if IntegratedUtilsScript._instance == self:
            IntegratedUtilsScript._instance = None

# --- 統合パッチ済みforward関数 ---
@wraps(UNetModel.forward)
def patched_unet_forward(self_unet: UNetModel, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, y=None, control=None, transformer_options:dict =None, **kwargs):
    script_instance = IntegratedUtilsScript._instance
    # オリジナルの関数を呼び出すための準備
    original_forward = script_instance.original_forward
    
    # スクリプトが無効な場合はオリジナルを呼び出す
    if not script_instance or not original_forward:
        # この状況はありえないはずだが、念のため
        return UNetModel.forward(self_unet, x, timesteps, context, y, control, transformer_options, **kwargs)

    fb_params = script_instance.fb_params_runtime
    freeu_params = script_instance.freeu_params_runtime
    
    is_fb_enabled = fb_params.get('enabled', False)
    is_freeu_enabled = freeu_params.get('enabled', False) and freeu_params.get('start_at', 1.0) < freeu_params.get('stop_at', 0.0)

    if not is_fb_enabled and not is_freeu_enabled:
        return original_forward(self_unet, x, timesteps, context, y, control, transformer_options, **kwargs)

    # --- 以下、パッチロジック ---
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

    num_initial_blocks_for_cache = fb_params.get('blocks', 3) if is_fbcache_active_for_step else 0
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
    
    return final_output.type(x.dtype)

# --- Forgeのライフサイクルコールバックへの登録 (変更なし) ---
script_callbacks.on_script_unloaded(lambda: IntegratedUtilsScript._instance.on_script_unloaded() if IntegratedUtilsScript._instance else None)
