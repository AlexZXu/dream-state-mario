from vmodel.tokenizer import (
    Tokenizer, TokenizerProps, tokenizer_loss, pixel_accuracy, colorize,
    FRAME_H, FRAME_W, DOWNSAMPLE, NUM_COLORS
)
import math

import torch


def main():
    torch.manual_seed(0)

    props = TokenizerProps()
    tokenizer = Tokenizer(props)

    encoder_params = sum(param.numel() for param in tokenizer.encoder.parameters())
    decoder_params = sum(param.numel() for param in tokenizer.decoder.parameters())
    print(f"encoder params {encoder_params / 1e6:.2f}M  decoder params {decoder_params / 1e6:.2f}M")

    X = torch.rand(2, props.in_channels, FRAME_H, FRAME_W)

    X_recon, mu, logvar = tokenizer(X)
    print(f"X_recon {tuple(X_recon.shape)}  mu {tuple(mu.shape)}  logvar {tuple(logvar.shape)}")

    latent_shape = (2, props.latent_channels, FRAME_H // DOWNSAMPLE, FRAME_W // DOWNSAMPLE)
    assert mu.shape == latent_shape
    assert logvar.shape == latent_shape
    assert X_recon.shape == (2, NUM_COLORS, FRAME_H, FRAME_W)

    # As the variance goes to zero the sample has to collapse onto mu.
    z = tokenizer.reparameterization(mu, torch.full_like(logvar, -100.0))
    assert torch.allclose(z, mu, atol=1e-6)

    # The head is sized by out_channels alone, so an RGB head is a config change.
    rgb = Tokenizer(TokenizerProps({"out_channels": 3}))
    assert rgb.decode(mu).shape == (2, 3, FRAME_H, FRAME_W)

    # Training runs on patches and inference on whole frames, so any size that divides
    # by 8 has to go through.
    patch_recon, patch_mu, _ = tokenizer(torch.rand(2, props.in_channels, 128, 128))
    assert patch_mu.shape == (2, props.latent_channels, 16, 16)
    assert patch_recon.shape == (2, NUM_COLORS, 128, 128)

    palette = torch.rand(NUM_COLORS, 3)
    frame = torch.randint(0, NUM_COLORS, (2, 16, 24))

    colors = colorize(frame, palette)
    assert colors.shape == (2, 3, 16, 24)
    assert torch.equal(colors[1, :, 5, 7], palette[frame[1, 5, 7]])

    # Logits that put everything on the right colour cost nothing, and logits that know
    # nothing cost log(64) per pixel.
    target = torch.randint(0, NUM_COLORS, (2, 8, 8))
    zeros = torch.zeros(2, 4, 1, 1)

    certain = 100.0 * torch.nn.functional.one_hot(target, NUM_COLORS).permute(0, 3, 1, 2).float()
    loss, reconstruct_loss, kl_loss = tokenizer_loss(certain, target, zeros, zeros, kl_weight=1.0)
    assert loss.item() < 1e-3

    uniform = torch.zeros(2, NUM_COLORS, 8, 8)
    loss, reconstruct_loss, kl_loss = tokenizer_loss(uniform, target, zeros, zeros, kl_weight=1.0)
    assert math.isclose(reconstruct_loss.item(), 64 * math.log(NUM_COLORS), rel_tol=1e-4)
    assert kl_loss.item() == 0

    # KL of N(1, 1) from N(0, 1) is 0.5 per dimension; there are 4 dimensions per sample.
    loss, reconstruct_loss, kl_loss = tokenizer_loss(certain, target, torch.ones(2, 4, 1, 1), zeros, kl_weight=1.0)
    assert math.isclose(kl_loss.item(), 2.0, rel_tol=1e-5)

    # One frame, 100 pixels: 90 background and 10 foreground. Getting 5 of the foreground
    # pixels wrong is 95% overall but only 50% of what matters.
    target = torch.zeros(1, 10, 10, dtype=torch.long)
    target[0, 0, :] = 5

    predicted = target.clone()
    predicted[0, 0, :5] = 7

    logits = torch.nn.functional.one_hot(predicted, NUM_COLORS).permute(0, 3, 1, 2).float()
    accuracy, foreground = pixel_accuracy(logits, target)

    assert math.isclose(accuracy, 0.95, rel_tol=1e-5)
    assert math.isclose(foreground, 0.5, rel_tol=1e-5)

    loss = X_recon.pow(2).mean() + mu.pow(2).mean() + logvar.pow(2).mean()
    loss.backward()

    missing = [name for name, param in tokenizer.named_parameters() if (param.grad is None)]
    assert missing == [], f"no gradient reached {missing}"

    print("\nAll tests passed.")

if (__name__ == "__main__"):
    main()
