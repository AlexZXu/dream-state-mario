from vmodel.tokenizer import Tokenizer, TokenizerProps, FRAME_H, FRAME_W, DOWNSAMPLE
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
    assert X_recon.shape == (2, props.out_channels, FRAME_H, FRAME_W)

    # As the variance goes to zero the sample has to collapse onto mu.
    z = tokenizer.reparameterization(mu, torch.full_like(logvar, -100.0))
    assert torch.allclose(z, mu, atol=1e-6)

    # The head is sized by out_channels alone, so a palette head is a config change.
    palette = Tokenizer(TokenizerProps({"out_channels": 64}))
    assert palette.decode(mu).shape == (2, 64, FRAME_H, FRAME_W)

    loss = X_recon.pow(2).mean() + mu.pow(2).mean() + logvar.pow(2).mean()
    loss.backward()

    missing = [name for name, param in tokenizer.named_parameters() if (param.grad is None)]
    assert missing == [], f"no gradient reached {missing}"

    print("\nAll tests passed.")

if (__name__ == "__main__"):
    main()
