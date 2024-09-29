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
        n_fft,
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
