import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    Dynamic range compression using log10.

    Args:
        x (torch.Tensor): Input tensor.
        C (float, optional): Scaling factor. Defaults to 1.
        clip_val (float, optional): Minimum value for clamping. Defaults to 1e-5.
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    Dynamic range decompression using exp.

    Args:
        x (torch.Tensor): Input tensor.
        C (float, optional): Scaling factor. Defaults to 1.
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    """
    Spectral normalization using dynamic range compression.

    Args:
        magnitudes (torch.Tensor): Magnitude spectrogram.
    """
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    """
    Spectral de-normalization using dynamic range decompression.

    Args:
        magnitudes (torch.Tensor): Normalized spectrogram.
    """
    return dynamic_range_decompression_torch(magnitudes)


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    """
    Compute the spectrogram of a signal using STFT.

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT window size.
        hop_size (int): Hop size between frames.
        win_size (int): Window size.
        center (bool, optional): Whether to center the window. Defaults to False.
    """
    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Convert a spectrogram to a mel-spectrogram.

    Args:
        spec (torch.Tensor): Magnitude spectrogram.
        n_fft (int): FFT window size.
        num_mels (int): Number of mel frequency bins.
        sample_rate (int): Sampling rate of the audio signal.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    melspec = spectral_normalize_torch(melspec)
    return melspec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False
):
    """
    Compute the mel-spectrogram of a signal.

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT window size.
        num_mels (int): Number of mel frequency bins.
        sample_rate (int): Sampling rate of the audio signal.
        hop_size (int): Hop size between frames.
        win_size (int): Window size.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        center (bool, optional): Whether to center the window. Defaults to False.
    """
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)

    melspec = spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax)

    return melspec


def compute_window_length(n_mels: int, sample_rate: int):
    f_min = 0
    f_max = sample_rate / 2
    window_length_seconds = 8 * n_mels / (f_max - f_min)
    window_length = int(window_length_seconds * sample_rate)
    return 2 ** (window_length.bit_length() - 1)


class MultiScaleMelSpectrogramLoss(torch.nn.Module):

    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels=[5, 10, 20, 40, 80, 160, 320, 480],
        loss_fn=torch.nn.L1Loss(),
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.loss_fn = loss_fn
        self.log_base = torch.log(torch.tensor(10.0))
        self.stft_params = {}
        self.mel_banks = {}

        window_lengths = [compute_window_length(mel, sample_rate) for mel in n_mels]
        # print(window_lengths)

        for n_mels, window_length in zip(n_mels, window_lengths):
            self.stft_params[n_mels] = {
                "n_mels": n_mels,
                "window_length": window_length,
                "hop_length": self.sample_rate // 100,
            }
            self.mel_banks[n_mels] = torch.from_numpy(
                librosa_mel_fn(
                    sr=self.sample_rate,
                    n_mels=n_mels,
                    n_fft=window_length,
                    fmin=0,
                    fmax=None,
                )
            )

    def mel_spectrogram(
        self,
        wav,
        n_mels,
        window_length,
        hop_length,
    ):
        wav = wav.squeeze(1)  # -> torch(B, T)
        window = torch.hann_window(window_length).to(wav.device).to(wav.dtype)
        stft = torch.stft(
            wav.float(),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )  # -> torch (B, window_length // 2 + 1, (T - window_length)/hop_length + 1)
        magnitude = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-6)
        mel_basis = self.mel_banks[n_mels].to(
            wav.device
        )  # torch(n_mels, window_length // 2 + 1)
        mel_spectrogram = torch.matmul(
            mel_basis, magnitude
        )  # torch(B, n_mels, stft.frames)
        return mel_spectrogram

    def forward(self, real, fake):  # real: torch(B, 1, T) , fake: torch(B, 1, T)
        loss = 0.0
        for p in self.stft_params.values():
            real_mels = self.mel_spectrogram(real, **p)
            fake_mels = self.mel_spectrogram(fake, **p)
            real_logmels = torch.log(real_mels.clamp(min=1e-5).pow(1)) / self.log_base
            fake_logmels = torch.log(fake_mels.clamp(min=1e-5).pow(1)) / self.log_base
            loss += self.loss_fn(real_logmels, fake_logmels)
        return loss
