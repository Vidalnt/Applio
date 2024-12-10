import torch
from typing import Optional
from rvc.lib.algorithm.nsf import GeneratorNSF
from rvc.lib.algorithm.generators import Generator
from rvc.lib.algorithm.commons import slice_segments, rand_slice_segments
from rvc.lib.algorithm.residuals import ResidualCouplingBlock
from rvc.lib.algorithm.encoders import TextEncoder, PosteriorEncoder


class Synthesizer(torch.nn.Module):
    """
    Base Synthesizer model.

    Args:
        spec_channels (int): Number of channels in the spectrogram.
        segment_size (int): Size of the audio segment.
        inter_channels (int): Number of channels in the intermediate layers.
        hidden_channels (int): Number of channels in the hidden layers.
        filter_channels (int): Number of channels in the filter layers.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of layers in the encoder.
        kernel_size (int): Size of the convolution kernel.
        p_dropout (float): Dropout probability.
        resblock (str): Type of residual block.
        resblock_kernel_sizes (list): Kernel sizes for the residual blocks.
        resblock_dilation_sizes (list): Dilation sizes for the residual blocks.
        upsample_rates (list): Upsampling rates for the decoder.
        upsample_initial_channel (int): Number of channels in the initial upsampling layer.
        upsample_kernel_sizes (list): Kernel sizes for the upsampling layers.
        spk_embed_dim (int): Dimension of the speaker embedding.
        gin_channels (int): Number of channels in the global conditioning vector.
        sr (int): Sampling rate of the audio.
        use_f0 (bool): Whether to use F0 information.
        text_enc_hidden_dim (int): Hidden dimension for the text encoder.
        kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        **kwargs,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.use_f0 = use_f0

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
            f0=use_f0,
        )

        if use_f0:
            self.dec = GeneratorNSF(
                inter_channels,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
                sr=sr,
                is_half=kwargs["is_half"],
            )
        else:
            self.dec = Generator(
                inter_channels,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
            )

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

    def _remove_weight_norm_from(self, module):
        """Utility to remove weight normalization from a module."""
        for hook in module._forward_pre_hooks.values():
            if getattr(hook, "__class__", None).__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(module)

    def remove_weight_norm(self):
        """Removes weight normalization from the model."""
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        y_lengths: Optional[torch.Tensor] = None,
        ds: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            pitchf (torch.Tensor, optional): Fine-grained pitch sequence.
            y (torch.Tensor, optional): Target spectrogram.
            y_lengths (torch.Tensor, optional): Lengths of the target spectrograms.
            ds (torch.Tensor, optional): Speaker embedding.
        """
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        if y is not None:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_p = self.flow(z, y_mask, g=g)
            z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)

            if self.use_f0 and pitchf is not None:
                pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                o = self.dec(z_slice, pitchf, g=g)
            else:
                o = self.dec(z_slice, g=g)

            return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

        return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
    ):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            rate (torch.Tensor, optional): Rate for time-stretching.
        """
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]

        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = (
            self.dec(z * x_mask, nsff0, g=g)
            if self.use_f0
            else self.dec(z * x_mask, g=g)
        )

        return o, x_mask, (z, z_p, m_p, logs_p)
