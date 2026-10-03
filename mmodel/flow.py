import torch

# Rectified flow. tau = 0 is a clean latent and tau = 1 is pure noise; in between the
# noisy latent is the straight line (1 - tau) * z + tau * eps, so the velocity along the
# path is eps - z and is the same at every tau. The network predicts that velocity, and
# sampling walks back from tau = 1 to tau = 0 with Euler steps.


def add_noise(z, tau, eps=None):
    # z shape: (batch, C, h, w); tau shape: (batch,)
    if (eps is None):
        eps = torch.randn_like(z)

    tau = tau[:, None, None, None]

    return (1 - tau) * z + tau * eps, eps


def sample_tau(batch_size, device):
    # Logit-normal: most of the training signal goes to the middle of the path, where
    # the velocity is hardest to predict, and little to tau near 0 or 1 where it is
    # almost trivial.
    return torch.sigmoid(torch.randn(batch_size, device=device))


def flow_loss(velocity, z, eps):
    return ((velocity - (eps - z)) ** 2).mean()


@torch.no_grad()
def sample(denoiser, context, cond_inputs, num_steps, tau_ctx, eps=None):
    """One generated latent, (batch, C, h, w), from num_steps Euler steps.

    context arrives clean and is noised here to tau_ctx, so the model sees the same kind
    of slightly-off context it was trained on and does not trust its own previous
    outputs too literally.
    """
    batch_size = context.shape[0]
    device = context.device

    tau_ctx = torch.full((batch_size,), tau_ctx, device=device)
    context, _ = add_noise(context.flatten(1, 2), tau_ctx)
    context = context.view(batch_size, -1, denoiser.props.latent_channels, context.shape[-2], context.shape[-1])

    if (eps is None):
        eps = torch.randn(batch_size, denoiser.props.latent_channels, context.shape[-2], context.shape[-1], device=device)

    z = eps
    taus = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        tau = taus[i].expand(batch_size)
        cond = denoiser.conditioning(tau, tau_ctx, **cond_inputs)

        velocity = denoiser.denoise(z, context, cond)
        z = z - (taus[i] - taus[i + 1]) * velocity

    return z
