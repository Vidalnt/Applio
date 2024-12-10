import torch
from rvc.lib.algorithm.commons import fused_add_tanh_sigmoid_multiply


class WaveNet(torch.nn.Module):
    """WaveNet residual blocks as used in WaveGlow.

    Args:
        hidden_channels (int): Number of hidden channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation_rate (int): Dilation rate of the convolution.
        n_layers (int): Number of convolutional layers.
        gin_channels (int, optional): Number of conditioning channels. Defaults to 0.
        p_dropout (float, optional): Dropout probability. Defaults to 0.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: int = 0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for proper padding."

        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.n_channels_tensor = torch.IntTensor([hidden_channels])  # Static tensor

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)

        # Conditional layer for global conditioning
        if gin_channels:
            self.cond_layer = torch.nn.utils.parametrizations.weight_norm(
                torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1),
                name="weight",
            )

        # Precompute dilations and paddings
        dilations = [dilation_rate**i for i in range(n_layers)]
        paddings = [(kernel_size * d - d) // 2 for d in dilations]

        # Initialize layers
        for i in range(n_layers):
            self.in_layers.append(
                torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(
                        hidden_channels,
                        2 * hidden_channels,
                        kernel_size,
                        dilation=dilations[i],
                        padding=paddings[i],
                    ),
                    name="weight",
                )
            )

            res_skip_channels = (
                hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            )
            self.res_skip_layers.append(
                torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(hidden_channels, res_skip_channels, 1),
                    name="weight",
                )
            )

    def forward(self, x, x_mask, g=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor (batch_size, hidden_channels, time_steps).
            x_mask (torch.Tensor): Mask tensor (batch_size, 1, time_steps).
            g (torch.Tensor, optional): Conditioning tensor (batch_size, gin_channels, time_steps).
        """
        output = x.clone().zero_()

        # Apply conditional layer if global conditioning is provided
        g = self.cond_layer(g) if g is not None else None

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = (
                g[
                    :,
                    i * 2 * self.hidden_channels : (i + 1) * 2 * self.hidden_channels,
                    :,
                ]
                if g is not None
                else 0
            )

            # Activation with fused Tanh-Sigmoid
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.n_channels_tensor)
            acts = self.drop(acts)

            # Residual and skip connections
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self):
        """Remove weight normalization from the module."""
        if self.gin_channels:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)
