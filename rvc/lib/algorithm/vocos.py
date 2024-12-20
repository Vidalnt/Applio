import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, remove_parametrizations
from torchaudio.transforms import InverseSpectrogram


class ConvNeXtLayer(nn.Module):
    def __init__(self, channel, h_channel, scale):
        super().__init__()
        self.dw_conv = nn.Conv1d(channel, channel, kernel_size=7, padding=3, groups=channel)
        self.norm = nn.LayerNorm(channel)
        self.pw_conv1 = nn.Linear(channel, h_channel)
        self.pw_conv2 = nn.Linear(h_channel, channel)
        self.scale = nn.Parameter(torch.full(size=(channel,), fill_value=scale), requires_grad=True)

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = x.transpose(1, 2)
        x = res + x
        return x


class SineGen(torch.nn.Module):
    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = torch.ones_like(f0) * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] += rand_ini
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        return sines

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            uv = self._f02uv(f0)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise * (1 - uv)
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class VocosGAN(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        h_channel,
        out_channel,
        num_layers,
        sample_rate,
        gin_channels=0,
        leaky_relu_slope=0.2,
        istft_config=None,
    ):
        super(VocosGAN).__init__()
        self.pad = nn.ReflectionPad1d([1, 0])
        self.in_conv = weight_norm(
            nn.Conv1d(in_channel, channel, kernel_size=7, padding=3)
        )
        self.norm = nn.LayerNorm(channel)
        scale = 1 / num_layers
        self.layers = nn.ModuleList(
            [
                ConvNeXtLayer(channel, h_channel, scale)
                for _ in range(num_layers)
            ]
        )
        self.norm_last = nn.LayerNorm(channel)
        self.out_conv = weight_norm(nn.Conv1d(channel, out_channel, 1))
        self.source_module = SourceModuleHnNSF(
            sample_rate, harmonic_num=8, sine_amp=0.1
        )
        self.cond = (
            weight_norm(nn.Conv1d(gin_channels, channel, 1))
            if gin_channels != 0
            else None
        )
        self.istft = InverseSpectrogram(**istft_config)
        self.leaky_relu_slope = leaky_relu_slope
       

    def forward(self, x, g=None, f0=None):
        if f0 is not None:
            sine_wave, _, _ = self.source_module(f0)
            x += sine_wave.unsqueeze(1)

        x = F.leaky_relu(self.in_conv(self.pad(x)), self.leaky_relu_slope)
        if g is not None and self.cond is not None:
            x += self.cond(g)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_last(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_conv(x)
        mag, phase = x.chunk(2, dim=1)
        mag = mag.exp().clamp_max(max=1e2)
        s = mag * (phase.cos() + 1j * phase.sin())
        o = self.istft(s).unsqueeze(1)
        return o

    def remove_parametrizations(self):
        remove_parametrizations(self.in_conv)
        remove_parametrizations(self.out_conv)
        if self.cond:
            remove_parametrizations(self.cond)