import torch


class LayerNorm(torch.nn.Module):
    """Layer normalization module.

    Args:
        channels (int): Number of channels.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).

        """
        # Transpose to (batch_size, time_steps, channels) for layer_norm
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(
            x, (x.size(-1),), self.gamma, self.beta, self.eps
        )
        # Transpose back to (batch_size, channels, time_steps)
        return x.transpose(1, -1)
