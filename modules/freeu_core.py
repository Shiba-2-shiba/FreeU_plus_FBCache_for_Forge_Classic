import torch

# freeu.pyからFourier_filter関数をそのままコピー
def fourier_filter(x, threshold, scale):
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)
    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered.to(x.dtype)

# FreeUのコアロジックを独立した関数として定義
def apply_freeu_scaling(h: torch.Tensor, hsp: torch.Tensor, scale_dict: dict, on_cpu_devices_ref: dict):
    """
    U-Netのバックボーン(h)とスキップ接続(hsp)にFreeUのスケーリングを適用する。
    
    Args:
        h (torch.Tensor): U-Netのoutput_blockに入力されるテンソル。
        hsp (torch.Tensor): U-Netのinput_blockから渡されたスキップ接続テンソル。
        scale_dict (dict): チャンネル数に応じた(b, s)スケール値の辞書。
        on_cpu_devices_ref (dict): FFTがGPUでサポートされないデバイスを記録するための辞書(参照渡し)。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: スケーリング適用後の (h, hsp)。
    """
    scale = scale_dict.get(h.shape[1], None)
    if scale is None:
        return h, hsp

    # Backbone scaling (b1, b2)
    b_scale = scale[0]
    hidden_mean = h.mean(1).unsqueeze(1)
    B = hidden_mean.shape[0]
    hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
    hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
    hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max.unsqueeze(2).unsqueeze(3) - hidden_min.unsqueeze(2).unsqueeze(3) + 1e-5)
    h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((b_scale - 1) * hidden_mean + 1)
    
    # Skip feature scaling (s1, s2)
    s_scale = scale[1]
    if hsp.device not in on_cpu_devices_ref:
        try:
            hsp = fourier_filter(hsp, threshold=1, scale=s_scale)
        except Exception:
            print(f"FreeU Core(Info): Device {hsp.device} does not support torch.fft, switching to CPU for this pass.")
            on_cpu_devices_ref[hsp.device] = True
            hsp = fourier_filter(hsp.cpu(), threshold=1, scale=s_scale).to(hsp.device)
    else:
        hsp = fourier_filter(hsp.cpu(), threshold=1, scale=s_scale).to(hsp.device)
        
    return h, hsp