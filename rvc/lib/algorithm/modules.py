import torch
from rvc.lib.algorithm.commons import fused_add_tanh_sigmoid_multiply


class WaveNet(torch.nn.Module):
    """WaveNet residual blocks as used in WaveGlow

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
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WaveNet, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.parametrizations.weight_norm(
                cond_layer, name="weight"
            )

        dilations = [dilation_rate**i for i in range(n_layers)]
        paddings = [(kernel_size * d - d) // 2 for d in dilations]

        for i in range(n_layers):
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilations[i],
                padding=paddings[i],
            )
            in_layer = torch.nn.utils.parametrizations.weight_norm(
                in_layer, name="weight"
            )
            self.in_layers.append(in_layer)

            res_skip_channels = (
                hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            )

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.parametrizations.weight_norm(
                res_skip_layer, name="weight"
            )
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_channels, time_steps).
            x_mask (torch.Tensor): Mask tensor of shape (batch_size, 1, time_steps).
            g (torch.Tensor, optional): Conditioning tensor of shape (batch_size, gin_channels, time_steps).
                Defaults to None.
        """
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

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
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)
