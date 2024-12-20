from typing import Optional, Tuple
import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from rvc.lib.algorithm.modules import WaveNet
from rvc.lib.algorithm.commons import get_padding, init_weights

LRELU_SLOPE = 0.1


def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(
        torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            1,
            dilation=dilation,
            padding=get_padding(kernel_size, dilation),
        )
    )


def apply_mask(tensor, mask):
    return tensor * mask if mask is not None else tensor


class ResBlockBase(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: Tuple[int]):
        super(ResBlockBase, self).__init__()
        self.convs1 = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        self.convs1.apply(init_weights)

        self.convs2 = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, 1) for _ in dilations]
        )
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = apply_mask(xt, x_mask)
            xt = torch.nn.functional.leaky_relu(c1(xt), LRELU_SLOPE)
            xt = apply_mask(xt, x_mask)
            xt = c2(xt)
            x = xt + x
        return apply_mask(x, x_mask)

    def remove_weight_norm(self):
        for conv in self.convs1 + self.convs2:
            remove_weight_norm(conv)


class ResBlock(ResBlockBase):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: Tuple[int] = (1, 3, 5)
    ):
        super(ResBlock, self).__init__(channels, kernel_size, dilation)


class Flip(torch.nn.Module):
    """Flip module for flow-based models.

    This module flips the input along the time dimension.
    """

    def forward(self, x, *args, reverse=False, **kwargs):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            reverse (bool, optional): Whether to reverse the operation. Defaults to False.
        """
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ResidualCouplingBlock(torch.nn.Module):
    """Residual Coupling Block for normalizing flow.

    Args:
        channels (int): Number of channels in the input.
        hidden_channels (int): Number of hidden channels in the coupling layer.
        kernel_size (int): Kernel size of the convolutional layers.
        dilation_rate (int): Dilation rate of the convolutional layers.
        n_layers (int): Number of layers in the coupling layer.
        n_flows (int, optional): Number of coupling layers in the block. Defaults to 4.
        gin_channels (int, optional): Number of channels for the global conditioning input. Defaults to 0.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = torch.nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow.forward(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        """Removes weight normalization from the coupling layers."""
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()

    def __prepare_scriptable__(self):
        """Prepares the module for scripting."""
        for i in range(self.n_flows):
            for hook in self.flows[i * 2]._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.flows[i * 2])

        return self


class ResidualCouplingLayer(torch.nn.Module):
    """Residual coupling layer for flow-based models.

    Args:
        channels (int): Number of channels.
        hidden_channels (int): Number of hidden channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation_rate (int): Dilation rate of the convolution.
        n_layers (int): Number of convolutional layers.
        p_dropout (float, optional): Dropout probability. Defaults to 0.
        gin_channels (int, optional): Number of conditioning channels. Defaults to 0.
        mean_only (bool, optional): Whether to use mean-only coupling. Defaults to False.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = torch.nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = torch.nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1
        )
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
            x_mask (torch.Tensor): Mask tensor of shape (batch_size, 1, time_steps).
            g (torch.Tensor, optional): Conditioning tensor of shape (batch_size, gin_channels, time_steps).
                Defaults to None.
            reverse (bool, optional): Whether to reverse the operation. Defaults to False.
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def remove_weight_norm(self):
        """Remove weight normalization from the module."""
        self.enc.remove_weight_norm()
