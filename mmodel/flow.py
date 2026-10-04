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
def sample(denoiser, context, cond_inputs, num_steps, tau_ctx, eps=None, guide=None, tau_guide=1.0, loose=None, tau_loose=1.0):
    """One generated latent, (batch, C, h, w), from num_steps Euler steps.

    context arrives clean and is noised here to tau_ctx, so the model sees the same kind
    of slightly-off context it was trained on and does not trust its own previous
    outputs too literally.

    guide is an optional latent to start from instead of pure noise: it is noised to
    tau_guide and the walk begins there. At tau_guide = 1 nothing of it is left and this
    is ordinary sampling; at 0 it comes back unchanged.

    loose is an optional mask, (batch, 1, h, w), of cells that follow the guide less
    closely: they start from the noisier tau_loose. That costs one extra step, from
    tau_loose down to tau_guide, during which the other cells are held on the guide's
    path; from there every cell gets the same num_steps as without the mask.
    """
    batch_size = context.shape[0]
    device = context.device

    tau_ctx = torch.full((batch_size,), tau_ctx, device=device)
    context, _ = add_noise(context.flatten(1, 2), tau_ctx)
    context = context.view(batch_size, -1, denoiser.props.latent_channels, context.shape[-2], context.shape[-1])

    if (eps is None):
        eps = torch.randn(batch_size, denoiser.props.latent_channels, context.shape[-2], context.shape[-1], device=device)

    z = eps
    tau_start = 1.0

    if (guide is not None):
        tau_start = tau_guide if (loose is None) else max(tau_guide, tau_loose)
        z, _ = add_noise(guide, torch.full((batch_size,), tau_start, device=device), eps)

    taus = torch.linspace(min(tau_start, tau_guide), 0.0, num_steps + 1, device=device)

    if (tau_start > tau_guide):
        taus = torch.cat([torch.tensor([tau_start], device=device), taus])

    for i in range(len(taus) - 1):
        tau = taus[i].expand(batch_size)
        cond = denoiser.conditioning(tau, tau_ctx, **cond_inputs)

        velocity = denoiser.denoise(z, context, cond)
        z = z - (taus[i] - taus[i + 1]) * velocity

        if (loose is not None and taus[i + 1] >= tau_guide):
            held, _ = add_noise(guide, taus[i + 1].expand(batch_size), eps)
            z = torch.where(loose, z, held)

    return z
