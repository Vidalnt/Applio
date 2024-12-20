import math
import torch
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from rvc.lib.algorithm.generators import SineGenerator
from rvc.lib.algorithm.commons import init_weights
from dataclasses import dataclass
from einops import rearrange
from typing import Optional
from vector_quantize_pytorch import GroupedResidualFSQ

LRELU_SLOPE = 0.1


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class LayerNorm(torch.nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ConvNeXtBlock(torch.nn.Module):
    r"""ConvNeXt Block."""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()

        self.dwconv = Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(dim, int(mlp_ratio * dim))
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            torch.nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()

    def forward(self, x: torch.Tensor, apply_residual: bool = True) -> torch.Tensor:
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)
        x = self.drop_path(x)

        if apply_residual:
            x = input + x

        return x


class ConvNeXtEncoder(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.downsample_layers = torch.nn.ModuleList()
        stem = torch.nn.Sequential(
            Conv1d(
                input_channels,
                dims[0],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode="zeros",
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(len(depths) - 1):
            mid_layer = torch.nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                Conv1d(dims[i], dims[i + 1], kernel_size=1),
            )
            self.downsample_layers.append(mid_layer)

        self.stages = torch.nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(len(depths)):
            stage = torch.nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        kernel_size=kernel_size,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (Conv1d, torch.nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x)


@dataclass
class FSQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor


class DownsampleFiniteScalarQuantize(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 1,
        n_groups: int = 1,
        levels: tuple[int] = (8, 5, 5, 5),
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: Optional[tuple[int]] = None,
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        self.residual_fsq = GroupedResidualFSQ(
            dim=all_dims[-1],
            levels=levels,
            num_quantizers=n_codebooks,
            groups=n_groups,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        self.downsample = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    Conv1d(
                        all_dims[idx],
                        all_dims[idx + 1],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(
                        all_dims[idx + 1],
                        all_dims[idx],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (Conv1d, torch.nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> FSQResult:
        original_shape = z.shape
        z = self.downsample(z)
        quantized, indices = self.residual_fsq(z.mT)
        result = FSQResult(
            z=quantized.mT,
            codes=indices.mT,
            latents=z,
        )
        result.z = self.upsample(result.z)

        diff = original_shape[-1] - result.z.shape[-1]
        left = diff // 2
        right = diff - left

        if diff > 0:
            result.z = F.pad(result.z, (left, right))
        elif diff < 0:
            result.z = result.z[..., left:-right]

        return result

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.downsample(z)
        _, indices = self.residual_fsq(z.mT)
        indices = rearrange(indices, "g b l r -> b (g r) l")
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        indices = rearrange(indices, "b (g r) l -> g b l r", g=self.residual_fsq.groups)
        z_q = self.residual_fsq.get_output_from_indices(indices)
        z_q = self.upsample(z_q.mT)
        return z_q



def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int] = (1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_parametrizations(self):
        for conv in self.convs1:
            remove_parametrizations(conv)
        for conv in self.convs2:
            remove_parametrizations(conv)


class ParralelBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        assert len(kernel_sizes) == len(dilation_sizes)

        self.blocks = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilation_sizes):
            self.blocks.append(ResBlock1(channels, k, d))

    def forward(self, x):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)

    def remove_parametrizations(self):
        for block in self.blocks:
            block.remove_parametrizations()


class SourceModuleHnNSF(torch.nn.Module):
    """
    Source Module for harmonic-plus-noise excitation.

    Args:
        sample_rate (int): Sampling rate in Hz.
        harmonic_num (int, optional): Number of harmonics above F0. Defaults to 0.
        sine_amp (float, optional): Amplitude of sine source signal. Defaults to 0.1.
        add_noise_std (float, optional): Standard deviation of additive Gaussian noise. Defaults to 0.003.
        voiced_threshod (float, optional): Threshold to set voiced/unvoiced given F0. Defaults to 0.
        is_half (bool, optional): Whether to use half precision. Defaults to True.
    """

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
        is_half: bool = True,
    ):
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half

        self.l_sin_gen = SineGenerator(
            sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upsample_factor: int = 1) -> tuple[torch.Tensor, None, None]:
        sine_wavs, uv, _ = self.l_sin_gen(x, upsample_factor)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class FireFlyVQGAN(torch.nn.Module):
    """
    Generator for synthesizing audio using the NSF (Neural Source Filter) approach.

    Args:
        initial_channel (int): Number of channels in the initial convolutional layer.
        resblock (str): Type of residual block to use (1 or 2).
        resblock_kernel_sizes (list): Kernel sizes of the residual blocks.
        resblock_dilation_sizes (list): Dilation rates of the residual blocks.
        upsample_rates (list): Upsampling rates.
        upsample_initial_channel (int): Number of channels in the initial upsampling layer.
        upsample_kernel_sizes (list): Kernel sizes of the upsampling layers.
        gin_channels (int): Number of channels for the global conditioning input.
        sr (int): Sampling rate.
        is_half (bool, optional): Whether to use half precision. Defaults to False.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        is_half: bool = False,
        use_convnext_encoder=True,
        convnext_depths=[3, 3, 9, 3],
        convnext_dims=[128, 256, 384, 512],
        convnext_drop_path_rate=0.2,
        convnext_kernel_size=7,
        use_vq=True,
        vq_n_codebooks=1,
        vq_n_groups=1,
        vq_levels=[8, 5, 5, 5],
        vq_downsample_factor=[2,2],
    ):
        super(FireFlyVQGAN, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sample_rate=sr, harmonic_num=0, is_half=is_half
        )
        self.use_convnext_encoder = use_convnext_encoder
        if use_convnext_encoder:
            self.backbone = ConvNeXtEncoder(
                input_channels=initial_channel,
                depths=convnext_depths,
                dims=convnext_dims,
                drop_path_rate=convnext_drop_path_rate,
                kernel_size=convnext_kernel_size,
            )
        else:
            self.conv_pre = weight_norm(
                Conv1d(
                    initial_channel, upsample_initial_channel, 7, 1, padding=3
                )
            )

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # handling odd upsampling rates
            if u % 2 == 0:
                # old method
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2
                
            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        channels[i],
                        k,
                        u,
                        padding=padding,
                        output_padding=u % 2,
                    )
                )
            )
            
            stride = stride_f0s[i]
            kernel = (1 if stride == 1 else stride * 2 - stride % 2)
            padding = (0 if stride == 1 else (kernel - stride) // 2)
            
            self.noise_convs.append(
                Conv1d(
                    1,
                    channels[i],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

        self.resblocks = torch.nn.ModuleList(
            [
                ParralelBlock(
                    channels[i], resblock_kernel_sizes, resblock_dilation_sizes
                )
                for i in range(len(self.ups))
            ]
        )

        self.conv_post = weight_norm(
            Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        if gin_channels != 0:
            self.cond = weight_norm(
                Conv1d(gin_channels, upsample_initial_channel, 1)
            )

        self.use_vq = use_vq
        if use_vq:
            self.vq = DownsampleFiniteScalarQuantize(
                input_dim=upsample_initial_channel,
                n_codebooks=vq_n_codebooks,
                n_groups=vq_n_groups,
                levels=vq_levels,
                downsample_factor=vq_downsample_factor,
            )
        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        if self.use_convnext_encoder:
            x = self.backbone(x)
        else:
            x = self.conv_pre(x)

            if g is not None:
                x += self.cond(g)

        if self.use_vq:
            vq_result = self.vq(x)
            x = vq_result.z
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = ups(x)
            x += noise_convs(har_source)

            x = self.resblocks[i](x)

        x = F.leaky_relu(x)
        x = torch.tanh(self.conv_post(x))

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        if hasattr(self, "conv_pre") and self.conv_pre is not None:
            remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond") and self.cond is not None:
            remove_weight_norm(self.cond)
        for l in self.resblocks:
            l.remove_parametrizations()
        if hasattr(self, "backbone"):
            self.backbone.remove_parametrizations()

    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        if hasattr(self, "conv_pre"):
            for hook in self.conv_pre._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(self.conv_pre)

        for hook in self.conv_post._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm(self.conv_post)

        if hasattr(self, "cond"):
            for hook in self.cond._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(self.cond)

        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    l.remove_parametrizations()
        if hasattr(self, "backbone"):
            self.backbone.remove_parametrizations()

        return self