import torch


def feature_loss(fmap_r, fmap_g):
    """
    Compute the feature loss between reference and generated feature maps.

    Args:
        fmap_r (list of torch.Tensor): List of reference feature maps.
        fmap_g (list of torch.Tensor): List of generated feature maps.
    """
    return 2 * sum(
        torch.mean(torch.abs(rl - gl))
        for dr, dg in zip(fmap_r, fmap_g)
        for rl, gl in zip(dr, dg)
    )


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Compute the discriminator loss for real and generated outputs.

    Args:
        disc_real_outputs (list of torch.Tensor): List of discriminator outputs for real samples.
        disc_generated_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
    """
    r_losses = [(1 - dr).pow(2).mean() for dr in disc_real_outputs]
    g_losses = [dg.pow(2).mean() for dg in disc_generated_outputs]
    loss = sum(r_losses) + sum(g_losses)
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """
    Compute the generator loss based on discriminator outputs.

    Args:
        disc_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
    """
    gen_losses = [(1 - dg).pow(2).mean() for dg in disc_outputs]
    loss = sum(gen_losses)
    return loss, gen_losses


def discriminator_loss_scaled(disc_real, disc_fake, scale=1.0):
    """
    Compute the scaled discriminator loss for real and generated outputs.

    Args:
        disc_real (list of torch.Tensor): List of discriminator outputs for real samples.
        disc_fake (list of torch.Tensor): List of discriminator outputs for generated samples.
        scale (float, optional): Scaling factor applied to losses beyond the midpoint. Default is 1.0.
    """
    midpoint = len(disc_real) // 2
    losses = []
    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        real_loss = (1 - d_real).pow(2).mean()
        fake_loss = d_fake.pow(2).mean()
        total_loss = real_loss + fake_loss
        if i >= midpoint:
            total_loss *= scale
        losses.append(total_loss)
    loss = sum(losses)
    return loss, None, None


def generator_loss_scaled(disc_outputs, scale=1.0):
    """
    Compute the scaled generator loss based on discriminator outputs.

    Args:
        disc_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
        scale (float, optional): Scaling factor applied to losses beyond the midpoint. Default is 1.0.
    """
    midpoint = len(disc_outputs) // 2
    losses = []
    for i, d_fake in enumerate(disc_outputs):
        loss_value = (1 - d_fake).pow(2).mean()
        if i >= midpoint:
            loss_value *= scale
        losses.append(loss_value)
    loss = sum(losses)
    return loss, None, None


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Compute the Kullback-Leibler divergence loss.

    Args:
        z_p (torch.Tensor): Latent variable z_p [b, h, t_t].
        logs_q (torch.Tensor): Log variance of q [b, h, t_t].
        m_p (torch.Tensor): Mean of p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()
    return loss
