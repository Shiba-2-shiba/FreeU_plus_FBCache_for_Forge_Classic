# fb_cache_core.py
#
# このスクリプトは、First Block Cache (FBCache) のコア機能を提供します。
# 元の fb_cache_script.py からUI定義やスクリプトイベント処理を分離し、
# 状態管理クラスとユーティリティ関数のみを抽出したライブラリモジュールです。
# 統合スクリプト (integrate_utils.py) からインポートして使用されることを想定しています。

import torch
import weakref
import traceback
from ldm_patched.modules import model_management

# FBCacheの状態を管理するクラス
# 元のスクリプトからほぼそのまま流用し、ロギング部分を単純なprintに置き換えています。
class FBCacheState:
    def __init__(self, unet_instance_ref: weakref.ReferenceType, cache_dtype: torch.dtype, debug_logging: bool = False):
        self.unet_instance_ref = unet_instance_ref
        # cache_data_by_pass_and_batch_size: { "first": {batch_size: cache}, "hires": {batch_size: cache} }
        self.cache_data_by_pass_and_batch_size = {"first": {}, "hires": {}}
        self.cache_device = model_management.get_torch_device()
        self.cache_dtype = cache_dtype
        self.is_debug_logging_enabled = debug_logging
        
        self.total_overall_calls_by_pass = {"first": 0, "hires": 0}
        self.total_overall_hits_by_pass = {"first": 0, "hires": 0}
        self.last_applied_params_by_pass = {"first": {}, "hires": {}}

        unet_instance = self.unet_instance_ref()
        print(f"FBCache Core(Info): FBCacheState created for UNet {type(unet_instance).__name__ if unet_instance else 'Unknown'}. Cache Device: {self.cache_device}, Cache Dtype: {self.cache_dtype}")

    def _log_debug(self, message: str):
        if self.is_debug_logging_enabled:
            print(f"FBCache Core(Debug): {message}")
            
    def _get_bs_cache(self, batch_size: int, pass_type: str, create_if_missing: bool = False):
        if pass_type not in self.cache_data_by_pass_and_batch_size:
            self.cache_data_by_pass_and_batch_size[pass_type] = {}
            
        pass_cache_for_bs_size = self.cache_data_by_pass_and_batch_size[pass_type]
        if batch_size not in pass_cache_for_bs_size and create_if_missing:
            pass_cache_for_bs_size[batch_size] = {
                'key': None, 'residual': None, 'consecutive_hits': 0,
                'total_calls_for_bs': 0, 'total_hits_for_bs': 0
            }
        return pass_cache_for_bs_size.get(batch_size)

    def store_key(self, key_tensor: torch.Tensor, batch_size: int, pass_type: str):
        bs_cache = self._get_bs_cache(batch_size, pass_type, create_if_missing=True)
        if not bs_cache: return
        try:
            bs_cache['key'] = key_tensor.to(device=self.cache_device, dtype=self.cache_dtype, copy=True)
            self._log_debug(f"BS {batch_size} ({pass_type}): Stored key. Shape: {bs_cache['key'].shape}, Device: {bs_cache['key'].device}, Dtype: {bs_cache['key'].dtype}")
        except Exception as e:
            print(f"FBCache Core(Error): BS {batch_size} ({pass_type}) storing key: {e}\n{traceback.format_exc()}")
            bs_cache['key'] = None

    def store_residual(self, residual_tensor: torch.Tensor, batch_size: int, pass_type: str):
        bs_cache = self._get_bs_cache(batch_size, pass_type, create_if_missing=True)
        if not bs_cache: return
        try:
            bs_cache['residual'] = residual_tensor.to(device=self.cache_device, dtype=self.cache_dtype, copy=True)
            self._log_debug(f"BS {batch_size} ({pass_type}): Stored residual. Shape: {bs_cache['residual'].shape}, Device: {bs_cache['residual'].device}, Dtype: {bs_cache['residual'].dtype}")
        except Exception as e:
            print(f"FBCache Core(Error): BS {batch_size} ({pass_type}) storing residual: {e}\n{traceback.format_exc()}")
            bs_cache['residual'] = None
            
    def get_key(self, batch_size: int, pass_type: str):
        bs_cache = self._get_bs_cache(batch_size, pass_type)
        return bs_cache['key'] if bs_cache else None

    def get_residual(self, batch_size: int, pass_type: str):
        bs_cache = self._get_bs_cache(batch_size, pass_type)
        return bs_cache['residual'] if bs_cache else None

    def get_consecutive_hits(self, batch_size: int, pass_type: str) -> int:
        bs_cache = self._get_bs_cache(batch_size, pass_type)
        return bs_cache['consecutive_hits'] if bs_cache else 0

    def increment_consecutive_hits(self, batch_size: int, pass_type: str):
        bs_cache = self._get_bs_cache(batch_size, pass_type, create_if_missing=True)
        if bs_cache:
            bs_cache['consecutive_hits'] += 1
            bs_cache['total_hits_for_bs'] +=1
            self.total_overall_hits_by_pass[pass_type] = self.total_overall_hits_by_pass.get(pass_type, 0) + 1
    
    def reset_consecutive_hits(self, batch_size: int, pass_type: str):
        bs_cache = self._get_bs_cache(batch_size, pass_type)
        if bs_cache:
            bs_cache['consecutive_hits'] = 0

    def record_call(self, batch_size: int, pass_type: str):
        bs_cache = self._get_bs_cache(batch_size, pass_type, create_if_missing=True)
        if bs_cache:
            bs_cache['total_calls_for_bs'] += 1
        self.total_overall_calls_by_pass[pass_type] = self.total_overall_calls_by_pass.get(pass_type, 0) + 1

    def clear_pass_data(self, pass_type: str):
        if pass_type in self.cache_data_by_pass_and_batch_size:
            self.cache_data_by_pass_and_batch_size[pass_type].clear()
        self.total_overall_calls_by_pass[pass_type] = 0
        self.total_overall_hits_by_pass[pass_type] = 0
        self.last_applied_params_by_pass[pass_type] = {}
        print(f"FBCache Core(Info): FBCacheState data cleared for pass: {pass_type}.")

    def clear_all_data(self):
        self.cache_data_by_pass_and_batch_size = {"first": {}, "hires": {}}
        self.total_overall_calls_by_pass = {"first": 0, "hires": 0}
        self.total_overall_hits_by_pass = {"first": 0, "hires": 0}
        self.last_applied_params_by_pass = {"first": {}, "hires": {}}
        print(f"FBCache Core(Info): FBCacheState: All pass data cleared.")

    def get_hit_rate_summary(self) -> str:
        summaries = []
        for pass_type in ["first", "hires"]:
            calls = self.total_overall_calls_by_pass.get(pass_type, 0)
            if calls == 0:
                if self.last_applied_params_by_pass.get(pass_type):
                     summaries.append(f"Pass '{pass_type}' Hit Rate: N/A (0 calls)")
                continue
            
            hits = self.total_overall_hits_by_pass.get(pass_type, 0)
            hit_rate = (hits / calls) * 100
            summary_line = f"Pass '{pass_type}' Hit Rate: {hit_rate:.2f}% ({hits}/{calls} hits/calls)."
            
            if self.is_debug_logging_enabled:
                pass_cache_data = self.cache_data_by_pass_and_batch_size.get(pass_type, {})
                for bs, data in pass_cache_data.items():
                    if data['total_calls_for_bs'] > 0:
                        bs_hit_rate = (data['total_hits_for_bs'] / data['total_calls_for_bs']) * 100
                        summary_line += f"\n    BS {bs}: {bs_hit_rate:.2f}% ({data['total_hits_for_bs']}/{data['total_calls_for_bs']})"
                    else:
                        summary_line += f"\n    BS {bs}: N/A (0 calls)"
            summaries.append(summary_line)
        return " | ".join(summaries) if summaries else "No cache activity recorded."

    def check_and_clear_if_critical_params_changed(self, pass_type: str, new_params: dict):
        critical_keys = ['threshold', 'num_initial_blocks']
        last_params = self.last_applied_params_by_pass.get(pass_type, {})
        
        params_changed = False
        for key in critical_keys:
            if last_params.get(key) != new_params.get(key):
                params_changed = True
                print(f"FBCache Core(Info): ({pass_type}): Critical param '{key}' changed from '{last_params.get(key)}' to '{new_params.get(key)}'.")
                break
        
        if params_changed:
            self.clear_pass_data(pass_type)
            print(f"FBCache Core(Info): ({pass_type}): Cache cleared due to critical parameter change.")
        
        self.last_applied_params_by_pass[pass_type] = new_params.copy()

# テンソルの類似度を比較するユーティリティ関数
def are_two_tensors_similar(tensor1: torch.Tensor, tensor2: torch.Tensor, threshold: float, current_bs: int, pass_type: str, debug_logging: bool = False) -> bool:
    if tensor1 is None or tensor2 is None:
        if debug_logging: print(f"FBCache Core(Debug): BS {current_bs} ({pass_type}): Similarity check: One or both tensors are None.")
        return False
    if tensor1.shape != tensor2.shape:
        if debug_logging: print(f"FBCache Core(Debug): BS {current_bs} ({pass_type}): Similarity check: Tensor shapes differ: {tensor1.shape} vs {tensor2.shape}")
        return False
    
    try:
        dev = tensor2.device
        t1_comp = tensor1.to(device=dev, dtype=torch.float32, non_blocking=True)
        t2_comp = tensor2.to(device=dev, dtype=torch.float32, non_blocking=True)
        
        norm_t1 = torch.linalg.norm(t1_comp) + 1e-9 
        diff_norm = torch.linalg.norm(t1_comp - t2_comp)
        
        relative_diff = diff_norm / norm_t1
        similarity_result = relative_diff.item() < threshold

        if debug_logging:
            print(f"FBCache Core(Debug): BS {current_bs} ({pass_type}): Similarity relative_diff: {relative_diff.item():.6f}, Threshold: {threshold:.6f}, Similar: {similarity_result}")
        return similarity_result
    except Exception as e:
        print(f"FBCache Core(Error): BS {current_bs} ({pass_type}) aligning/comparing tensors: {e}\n{traceback.format_exc()}")
        return False