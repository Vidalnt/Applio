import torch

if torch.cuda.is_available() and torch.cuda.get_device_name().endswith("[ZLUDA]"):
    _torch_stft = torch.stft

    def z_stft(
        audio: torch.Tensor,
        n_fft: int,
        hop_length: int = None,
        win_length: int = None,
        window: torch.Tensor = None,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = None,
        return_complex: bool = None,
    ):
        sd = audio.device
        return _torch_stft(
            audio.to("cpu"),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window.to("cpu"),
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            return_complex=return_complex,
        ).to(sd)

    def z_jit(f, *_, **__):
        f.graph = torch._C.Graph()
        return f

    # hijacks
    torch.stft = z_stft
    torch.jit.script = z_jit
    # disabling unsupported cudnn
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
