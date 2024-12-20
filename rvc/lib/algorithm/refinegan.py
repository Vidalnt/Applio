from typing import Callable
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)

class ResBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super(ResBlock, self).__init__()

        self.leaky_relu_slope = leaky_relu_slope
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        in_channels=in_channels if idx == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for idx, d in enumerate(dilation)
            ]
        )
        self.convs1.apply(self.init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for idx, d in enumerate(dilation)
            ]
        )
        self.convs2.apply(self.init_weights)

    def forward(self, x):
        for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)

            if idx != 0 or self.in_channels == self.out_channels:
                x = xt + x
            else:
                x = xt

        return x

    def remove_parametrizations(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_parametrizations(c1)
            remove_parametrizations(c2)

    def init_weights(self, m):
        if type(m) == nn.Conv1d:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0.0)


class AdaIN(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels))
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gaussian = torch.randn_like(x) * self.weight[None, :, None]

        return self.activation(x + gaussian)


class ParallelResBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: int = (3, 7, 11),
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    AdaIN(channels=out_channels),
                    ResBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    AdaIN(channels=out_channels),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        results = [block(x) for block in self.blocks]

        return torch.mean(torch.stack(results), dim=0)

    def remove_parametrizations(self):
        for block in self.blocks:
            block[1].remove_parametrizations()

class SineGen(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)

    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)

    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

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
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
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

        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        return sine_merge, None, None

        
class RefineGAN(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        downsample_rates: tuple[int] = (2, 2, 8, 8),
        upsample_rates: tuple[int] = (8, 8, 2, 2),
        leaky_relu_slope: float = 0.2,
        num_mels: int = 128,
        start_channels: int = 16,
        gin_channels: int = 256,
    ) -> None:
        super().__init__()

        self.downsample_rates = downsample_rates
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope

        self.f0_upsample = nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num=8)

        # expands
        self.source_conv = weight_norm(nn.Conv1d(in_channels=1, out_channels=start_channels, kernel_size=7, stride=1, padding=3,))

        channels = start_channels
        self.downsample_blocks = nn.ModuleList([])
        for rate in downsample_rates:
            new_channels = channels * 2

            self.downsample_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=1 / rate, mode="linear"),
                    ResBlock(in_channels=channels, out_channels=new_channels, kernel_size=7, dilation=(1, 3, 5), leaky_relu_slope=leaky_relu_slope,),
                )
            )

            channels = new_channels

        self.mel_conv = weight_norm(nn.Conv1d(in_channels=num_mels,out_channels=channels,kernel_size=7,stride=1,padding=3,))
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(256, channels, 1)
        
        channels *= 2

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        for rate in upsample_rates:
            new_channels = channels // 2

            self.upsample_blocks.append(nn.Upsample(scale_factor=rate, mode="linear"))

            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4,
                    out_channels=new_channels,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                )
            )

            channels = new_channels

        self.conv_post = weight_norm(nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=7, stride=1, padding=3,))

    def forward(self, mel: torch.Tensor, f0: torch.Tensor, g: torch.Tensor=None) -> torch.Tensor:
        f0 = self.f0_upsample(f0[:, None, :]).transpose(-1, -2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(-1, -2)

        # expanding pitch source to 16 channels
        x = self.source_conv(har_source)
        # making a downscaled version to match upscaler stages
        downs = []
        for i, block in enumerate(self.downsample_blocks):
            x = F.leaky_relu(x, self.leaky_relu_slope, inplace=True)
            downs.append(x)
            x = block(x)

        # expanding spectrogram from 192 to 256 channels
        mel = self.mel_conv(mel)
        
        if g is not None:
            # adding expanded speaker embedding
              x = x + self.cond(g)
        x = torch.cat([x, mel], dim=1)
        i = 1
        for up, res, down in zip(self.upsample_blocks, self.upsample_conv_blocks, reversed(downs),):
            x = F.leaky_relu(x, self.leaky_relu_slope, inplace=True)
            x = up(x)
            x = torch.cat([x, down], dim=1)
            x = res(x)

        x = F.leaky_relu(x, self.leaky_relu_slope, inplace=True)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_parametrizations(self) -> None:
        remove_parametrizations(self.source_conv)
        remove_parametrizations(self.mel_conv)
        remove_parametrizations(self.conv_post)

        for block in self.downsample_blocks:
            block[1].remove_parametrizations()

        for block in self.upsample_conv_blocks:
            block.remove_parametrizations()