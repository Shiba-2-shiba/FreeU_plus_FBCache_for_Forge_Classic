# /scripts/integrate_utils.py
#
# FBCacheとFreeUを統合し、単一のパッチで両機能を制御するスクリプト。
#
# 改修履歴:
# - v2.3 (パラメータ受け渡し修正版):
#   - Forgeの内部処理でtransformer_optionsが上書きされる問題に対応。
#   - パラメータをtransformer_options経由ではなく、U-Netインスタンス自体にアタッチして直接受け渡す方式に変更。
#   - setup_patches: target_unet.integrated_paramsにパラメータを設定。
#   - patched_unet_forward_integrated: self_unet.integrated_paramsからパラメータを取得。
#   - teardown_patches: クリーンアップ時にintegrated_params属性を削除。

import torch
import gradio as gr
import traceback
import weakref
import datetime
import sys
import os
import inspect

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
    # 実行時エラーを防ぐためのダミー関数とクラスを定義
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


# --- 統合パッチ済みforward関数 ---
def patched_unet_forward_integrated(self_unet: UNetModel, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, y=None, control=None, transformer_options:dict =None, **kwargs):
    """
    FBCacheとFreeUの両方のロジックを組み込んだ、U-Netの新しいforwardメソッド。
    """
    script_instance = IntegratedUtilsScript._instance
    original_forward = getattr(self_unet, 'integrated_original_forward', None)

    def _call_original(current_unet, original_func):
        if not original_func:
            raise RuntimeError("IntegratedUtils: Cannot call original forward method because it's not set.")
        if inspect.ismethod(original_func):
            return original_func(x, timesteps, context, y=y, control=control, transformer_options=transformer_options, **kwargs)
        else:
            return original_func(current_unet, x, timesteps, context, y=y, control=control, transformer_options=transformer_options, **kwargs)

    # ▼▼▼【変更点】U-Netインスタンスから直接パラメータを取得 ▼▼▼
    integrated_params = getattr(self_unet, 'integrated_params', {})
    fb_params = integrated_params.get('fb_cache_params', {})
    freeu_params = integrated_params.get('freeu_params', {})

    # 必須のオブジェクトが存在しない、または両機能が無効な場合は、元のforwardを呼び出して終了
    if not script_instance or not original_forward or (not fb_params.get('enabled', False) and not freeu_params.get('enabled', False)):
        return _call_original(self_unet, original_forward)

    # --- ここから下はFBCacheまたはFreeUが有効な場合のメインロジック ---
    fb_state = script_instance.active_fb_state_object

    is_fbcache_active_for_step = False
    if fb_params.get('enabled', False) and fb_state:
        current_pass_type = fb_params.get('current_pass_type', 'first')
        current_batch_size = x.shape[0]
        fb_state.record_call(current_batch_size, current_pass_type) # これでcallが記録されるはず
        current_step_index = shared.state.sampling_step if hasattr(shared.state, 'sampling_step') else 0
        is_fbcache_active_for_step = (fb_params.get('start_step', 0) <= current_step_index < fb_params.get('end_step', float('inf')))
        script_instance.log_debug(f"FBCache Check: BS {current_batch_size} ({current_pass_type}), Step {current_step_index}. Active: {is_fbcache_active_for_step}.")
    
    # --- 標準的なU-Netの前処理 ---
    hs = []
    t_emb = timestep_embedding(timesteps, self_unet.model_channels, repeat_only=False).to(x.dtype)
    emb = self_unet.time_embed(t_emb)
    if self_unet.num_classes is not None:
        emb = emb + self_unet.label_emb(y)
    h = x
    
    # transformer_optionsはNoneの場合があるのでデフォルト値を設定
    current_transformer_options = {} if transformer_options is None else transformer_options.copy()


    # --- FBCache用の初期ブロック計算 ---
    num_initial_blocks_for_cache = fb_params.get('num_initial_blocks', 3) if is_fbcache_active_for_step else 0
    for block_idx in range(num_initial_blocks_for_cache):
        module_block = self_unet.input_blocks[block_idx]
        current_transformer_options["block"] = ("input", block_idx)
        h = forward_timestep_embed(module_block, h, emb, context, current_transformer_options, **kwargs)
        if control is not None: h = apply_control(h, control, "input")
        hs.append(h)
    
    h_after_initial_blocks = h.clone() if is_fbcache_active_for_step else None
    
    # --- FBCacheのキャッシュヒット判定 ---
    use_cached_result = False
    if is_fbcache_active_for_step:
        current_batch_size = x.shape[0]
        current_pass_type = fb_params.get('current_pass_type', 'first')
        previous_key = fb_state.get_key(current_batch_size, current_pass_type)
        if previous_key is not None:
            max_hits = fb_params.get('max_hits', -1)
            consecutive_hits = fb_state.get_consecutive_hits(current_batch_size, current_pass_type)
            if max_hits < 0 or consecutive_hits < max_hits:
                is_similar = are_two_tensors_similar(previous_key, h_after_initial_blocks, fb_params.get('threshold', 0.1), current_batch_size, current_pass_type, script_instance.is_debug_logging_enabled)
                if is_similar and fb_state.get_residual(current_batch_size, current_pass_type) is not None:
                    use_cached_result = True
                    script_instance.log_debug(f"FBCache: HIT on BS {current_batch_size} ({current_pass_type}).")

    # --- キャッシュ適用 or フル計算 ---
    if use_cached_result:
        try:
            current_batch_size = x.shape[0]
            current_pass_type = fb_params.get('current_pass_type', 'first')
            cached_residual = fb_state.get_residual(current_batch_size, current_pass_type)
            h = h_after_initial_blocks + cached_residual.to(h_after_initial_blocks.device, dtype=h_after_initial_blocks.dtype)
            fb_state.increment_consecutive_hits(current_batch_size, current_pass_type)
            script_instance.log_debug(f"FBCache: Applied residual. Consecutive hits: {fb_state.get_consecutive_hits(current_batch_size, current_pass_type)}.")
        except Exception as e:
            script_instance.log_error(f"FBCache failed to apply residual: {e}. Falling back to full calculation.")
            use_cached_result = False
            current_batch_size = x.shape[0]
            current_pass_type = fb_params.get('current_pass_type', 'first')
            fb_state.reset_consecutive_hits(current_batch_size, current_pass_type)
    
    if not use_cached_result:
        if is_fbcache_active_for_step:
            current_batch_size = x.shape[0]
            current_pass_type = fb_params.get('current_pass_type', 'first')
            script_instance.log_debug(f"FBCache: MISS on BS {current_batch_size} ({current_pass_type}). Storing new key.")
            fb_state.reset_consecutive_hits(current_batch_size, current_pass_type)
            fb_state.store_key(h_after_initial_blocks, current_batch_size, current_pass_type)
        
        # --- U-Netの残りのブロック計算 ---
        # input blocks
        for block_idx in range(num_initial_blocks_for_cache, len(self_unet.input_blocks)):
            module_block = self_unet.input_blocks[block_idx]
            current_transformer_options["block"] = ("input", block_idx)
            h = forward_timestep_embed(module_block, h, emb, context, current_transformer_options, **kwargs)
            if control is not None: h = apply_control(h, control, "input")
            hs.append(h)

        # middle block
        current_transformer_options["block"] = ("middle", 0)
        h = forward_timestep_embed(self_unet.middle_block, h, emb, context, current_transformer_options, **kwargs)
        if control is not None: h = apply_control(h, control, "middle")
        
        # output blocks
        for block_idx, module_block in enumerate(self_unet.output_blocks):
            current_transformer_options["block"] = ("output", block_idx)
            hsp = hs.pop()
            if control is not None: hsp = apply_control(hsp, control, "output")

            # ▼▼▼ FreeUロジックの注入箇所 ▼▼▼
            if freeu_params.get('enabled', False):
                current_step = shared.state.sampling_step if hasattr(shared.state, 'sampling_step') else 0
                total_steps = shared.state.sampling_steps if hasattr(shared.state, 'sampling_steps') and shared.state.sampling_steps > 0 else 1
                progress = current_step / (total_steps - 1) if total_steps > 1 else 1.0
                
                if (progress >= freeu_params.get('start_at', 0.0)) and (progress <= freeu_params.get('stop_at', 1.0)):
                    script_instance.log_debug(f"FreeU Applying on output block {block_idx}")
                    h, hsp = apply_freeu_scaling(
                        h, hsp, freeu_params.get('scale_dict', {}), freeu_params.get('on_cpu_devices_ref', {})
                    )
            # ▲▲▲ FreeUロジックの注入完了 ▲▲▲

            h = torch.cat([h, hsp], dim=1)
            output_shape = hs[-1].shape if hs else None
            h = forward_timestep_embed(module_block, h, emb, context, current_transformer_options, output_shape=output_shape, **kwargs)
        
        if is_fbcache_active_for_step:
            current_batch_size = x.shape[0]
            current_pass_type = fb_params.get('current_pass_type', 'first')
            calculated_residual = h - h_after_initial_blocks
            fb_state.store_residual(calculated_residual, current_batch_size, current_pass_type)
            script_instance.log_debug(f"FBCache: Stored new residual for BS {current_batch_size} ({current_pass_type}).")

    # --- U-Netの最終出力 ---
    if self_unet.predict_codebook_ids:
        final_output = self_unet.id_predictor(h)
    else:
        final_output = self_unet.out(h)
    
    return final_output.type(x.dtype)


# --- メインのスクリプトクラス ---
class IntegratedUtilsScript(scripts.Script):
    _instance = None
    
    def __init__(self):
        super().__init__()
        if IntegratedUtilsScript._instance is None:
            IntegratedUtilsScript._instance = self
        
        self.active_fb_state_object = None
        self.last_patched_unet_ref = None
        self.is_debug_logging_enabled = False
        self.fb_params_ui = {}
        self.freeu_params_ui = {}
    
    def title(self):
        return "FBCache + FreeU"

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
        # (UI部分は変更なし)
        ui_components = []
        with gr.Accordion(self.title(), open=False):
            with gr.Row():
                enable_debug_logging = gr.Checkbox(label="Enable Debug Logging (Console)", value=False)
                ui_components.append(enable_debug_logging)

            with gr.Tabs():
                with gr.TabItem("First Block Cache"):
                    gr.Markdown("U-Netの初期ブロックの計算結果をキャッシュし、類似した入力に対して再利用します。SDXL Baseモデル推奨。")
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
                    gr.Markdown("U-Netのバックボーンとスキップ接続の特徴量を調整し、生成品質を向上させます。")
                    freeu_enabled = gr.Checkbox(label="Enable FreeU", value=False)
                    freeu_b1 = gr.Slider(label="Backbone 1 (b1)", minimum=0, maximum=2, step=0.01, value=1.3)
                    freeu_b2 = gr.Slider(label="Backbone 2 (b2)", minimum=0, maximum=2, step=0.01, value=1.4)
                    freeu_s1 = gr.Slider(label="Skip 1 (s1)", minimum=0, maximum=4, step=0.01, value=1.2)
                    freeu_s2 = gr.Slider(label="Skip 2 (s2)", minimum=0, maximum=4, step=0.01, value=0.7)
                    freeu_start_at = gr.Slider(label="Start At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.01)
                    freeu_stop_at = gr.Slider(label="Stop At % of Steps", minimum=0.0, maximum=1.0, step=0.01, value=0.2)
                    ui_components.extend([freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_start_at, freeu_stop_at])

        return ui_components

    def get_target_unet(self):
        # (get_target_unetは変更なし)
        if not (shared.sd_model and hasattr(shared.sd_model, 'forge_objects') and hasattr(shared.sd_model.forge_objects, 'unet')):
             self.log_info("Could not find forge_objects.unet.")
             return None
        unet_candidate = shared.sd_model.forge_objects.unet
        q = [(unet_candidate, 0)]; visited_ids = {id(unet_candidate)}
        while q:
            current_obj, depth = q.pop(0)
            if depth > 5: continue
            if isinstance(current_obj, UNetModel) or (hasattr(current_obj, 'input_blocks') and hasattr(current_obj, 'output_blocks')):
                 self.log_debug(f"Found UNet-like object: {type(current_obj).__name__} at depth {depth}.")
                 return current_obj
            for attr_name in ['model', 'diffusion_model', 'wrapped', '_model', 'unet']:
                 if hasattr(current_obj, attr_name):
                     inner_obj = getattr(current_obj, attr_name)
                     if inner_obj is not None and id(inner_obj) not in visited_ids:
                          visited_ids.add(id(inner_obj)); q.append((inner_obj, depth + 1))
        self.log_error(f"Could not find a valid UNetModel instance inside wrapper: {type(unet_candidate).__name__}")
        return None

    def setup_patches(self, p):
        """U-Netにパッチを適用し、各機能のパラメータを設定する。"""
        target_unet = self.get_target_unet()
        if not target_unet:
            self.log_info("No UNet found, skipping patch setup.")
            return

        pass_type = "hires" if getattr(p, 'is_hr_pass', False) else "first"
        is_hires_pass = (pass_type == "hires")
        
        # ▼▼▼【変更点】統合パラメータ辞書を準備 ▼▼▼
        integrated_params = {'fb_cache_params': {'enabled': False}, 'freeu_params': {'enabled': False}}

        # --- FBCache パラメータ設定 ---
        fb_params_for_pass = self.fb_params_ui.get('hires' if is_hires_pass else 'first', {})
        fb_enabled = fb_params_for_pass.get('enabled', False)
        
        if fb_enabled:
            if not self.active_fb_state_object or not self.active_fb_state_object.unet_instance_ref() or self.active_fb_state_object.unet_instance_ref() is not target_unet:
                self.log_info(f"Creating new FBCacheState for current UNet ({type(target_unet).__name__}).")
                cache_dtype = getattr(target_unet, 'dtype', torch.float16)
                self.active_fb_state_object = FBCacheState(weakref.ref(target_unet), cache_dtype, self.is_debug_logging_enabled)
            
            total_steps = p.hr_second_pass_steps if is_hires_pass and hasattr(p, 'hr_second_pass_steps') and p.hr_second_pass_steps > 0 else p.steps
            fb_params_for_unet = {
                'enabled': True, 'current_pass_type': pass_type,
                'threshold': fb_params_for_pass.get('threshold'), 'num_initial_blocks': fb_params_for_pass.get('blocks'),
                'start_step': int(fb_params_for_pass.get('start', 0.0) * total_steps),
                'end_step': int(fb_params_for_pass.get('end', 1.0) * total_steps),
                'max_hits': int(fb_params_for_pass.get('max_hits')),
            }
            self.active_fb_state_object.check_and_clear_if_critical_params_changed(pass_type, fb_params_for_unet)
            integrated_params['fb_cache_params'] = fb_params_for_unet
        
        # --- FreeU パラメータ設定 ---
        freeu_enabled = self.freeu_params_ui.get('enabled', False)
        if freeu_enabled:
            model_channels = getattr(target_unet, 'model_channels', 320)
            b1, b2, s1, s2 = self.freeu_params_ui.get('b1'), self.freeu_params_ui.get('b2'), self.freeu_params_ui.get('s1'), self.freeu_params_ui.get('s2')
            
            freeu_params_for_unet = {
                'enabled': True, 'start_at': self.freeu_params_ui.get('start_at'), 'stop_at': self.freeu_params_ui.get('stop_at'),
                'scale_dict': {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)},
                'on_cpu_devices_ref': {}
            }
            integrated_params['freeu_params'] = freeu_params_for_unet
                
        # --- パッチの適用とパラメータのアタッチ ---
        if fb_enabled or freeu_enabled:
            # ▼▼▼【変更点】U-Netインスタンスに直接パラメータをアタッチ ▼▼▼
            target_unet.integrated_params = integrated_params
            self.log_debug(f"Attached integrated_params to UNet: {target_unet.integrated_params}")
            
            if not hasattr(target_unet, 'integrated_original_forward'):
                self.log_info(f"Patching UNet {type(target_unet).__name__} with integrated forward method.")
                target_unet.integrated_original_forward = target_unet.forward
                target_unet.forward = patched_unet_forward_integrated.__get__(target_unet, UNetModel)
                self.last_patched_unet_ref = weakref.ref(target_unet)
        else:
            self.teardown_patches()

    def teardown_patches(self):
        """U-Netからパッチを安全に解除し、関連する状態をクリーンアップする。"""
        unet_instance = self.last_patched_unet_ref() if self.last_patched_unet_ref else None
        
        if unet_instance:
            if hasattr(unet_instance, 'integrated_original_forward'):
                self.log_info(f"Unpatching UNet {type(unet_instance).__name__}.")
                unet_instance.forward = unet_instance.integrated_original_forward
                delattr(unet_instance, 'integrated_original_forward')
            
            # ▼▼▼【変更点】アタッチしたパラメータを削除 ▼▼▼
            if hasattr(unet_instance, 'integrated_params'):
                delattr(unet_instance, 'integrated_params')

            if self.active_fb_state_object:
                if self.active_fb_state_object.unet_instance_ref and self.active_fb_state_object.unet_instance_ref() is unet_instance:
                    summary = self.active_fb_state_object.get_hit_rate_summary()
                    if summary and "not loaded" not in summary and "N/A (0 calls)" not in summary:
                        self.log_info(f"Final FBCache Stats: {summary}")
                    self.active_fb_state_object.clear_all_data()
                    self.active_fb_state_object = None
        
        self.last_patched_unet_ref = None

    def process(self, p, enable_debug_logging,
                fb_enabled_first, fb_threshold_first, fb_blocks_first, fb_start_first, fb_end_first, fb_max_hits_first,
                fb_enabled_hires, fb_use_first_settings, fb_threshold_hires, fb_blocks_hires, fb_start_hires, fb_end_hires, fb_max_hits_hires,
                freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_start_at, freeu_stop_at
                ):
        # (processメソッドは変更なし)
        self.is_debug_logging_enabled = enable_debug_logging
        self.log_info(f"UI settings received. Debug Logging: {self.is_debug_logging_enabled}")
        self.fb_params_ui['first'] = {
            'enabled': fb_enabled_first, 'threshold': fb_threshold_first, 'blocks': fb_blocks_first,
            'start': fb_start_first, 'end': fb_end_first, 'max_hits': fb_max_hits_first
        }
        if fb_use_first_settings:
            self.fb_params_ui['hires'] = self.fb_params_ui['first'].copy()
            self.fb_params_ui['hires']['enabled'] = fb_enabled_hires
        else:
            self.fb_params_ui['hires'] = {
                'enabled': fb_enabled_hires, 'threshold': fb_threshold_hires, 'blocks': fb_blocks_hires,
                'start': fb_start_hires, 'end': fb_end_hires, 'max_hits': fb_max_hits_hires
            }
        self.freeu_params_ui = {
            'enabled': freeu_enabled, 'b1': freeu_b1, 'b2': freeu_b2, 's1': freeu_s1, 's2': freeu_s2,
            'start_at': freeu_start_at, 'stop_at': freeu_stop_at
        }
        p.extra_generation_params["Integrated FBCache Enabled"] = fb_enabled_first or (fb_enabled_hires and getattr(p, 'enable_hr', False))
        p.extra_generation_params["Integrated FreeU Enabled"] = freeu_enabled

    def process_before_every_sampling(self, p, *args, **kwargs):
        """サンプリングが開始される直前に呼び出され、パッチをセットアップする。"""
        pass_name = 'Hires' if getattr(p, 'is_hr_pass', False) else 'First'
        self.log_info(f"Starting setup for pass: {pass_name}")
        
        # ▼▼▼【変更点】p.transformer_optionsの初期化は不要になったため削除▼▼▼
        # if not hasattr(p, 'transformer_options'):
        #     p.transformer_options = {}
            
        self.setup_patches(p)
        
    def postprocess(self, p, processed, *args):
        # (postprocessメソッドは変更なし)
        self.log_debug("Postprocess started. Tearing down patches.")
        self.teardown_patches()
        return processed
        
    def on_script_unloaded(self):
        # (on_script_unloadedメソッドは変更なし)
        self.log_info("Script unloading. Tearing down all patches.")
        self.teardown_patches()
        if IntegratedUtilsScript._instance == self:
            IntegratedUtilsScript._instance = None

# --- Forgeのライフサイクルコールバックへの登録 ---
def on_model_loaded(model):
    # (on_model_loadedメソッドは変更なし)
    if IntegratedUtilsScript._instance:
        model_name = "Unknown"
        if hasattr(model, 'sd_checkpoint_info') and model.sd_checkpoint_info:
             model_name = getattr(model.sd_checkpoint_info, 'name_for_extra', 'Unknown')
        IntegratedUtilsScript._instance.log_info(f"New model loaded: {model_name}. Ensuring old patches are removed.")
        IntegratedUtilsScript._instance.teardown_patches()

script_callbacks.on_model_loaded(on_model_loaded)
script_callbacks.on_script_unloaded(lambda: IntegratedUtilsScript._instance.on_script_unloaded() if IntegratedUtilsScript._instance else None)
