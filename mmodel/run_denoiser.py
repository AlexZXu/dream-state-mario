from mmodel.denoiser import Denoiser, DenoiserProps, SelfAttention, scalar_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_H = 28
LATENT_W = 32


def make_inputs(props, batch_size):
    C = props.latent_channels
    L = props.context_frames

    return {
        "z_noisy": torch.randn(batch_size, C, LATENT_H, LATENT_W),
        "context": torch.randn(batch_size, L, C, LATENT_H, LATENT_W),
        "tau": torch.rand(batch_size),
        "tau_ctx": torch.rand(batch_size),
        "actions": torch.randint(0, 2, (batch_size, L + 1, props.action_dim)).float(),
        "level": torch.randint(0, props.num_levels, (batch_size,)),
        "x_pos": torch.rand(batch_size),
        "powerup": torch.randint(0, props.num_powerups, (batch_size,)),
        "state_mask": torch.ones(batch_size)
    }


def randomize_output(denoiser):
    # conv_out starts at zero, which makes the output zero whatever the input is. Every
    # check that asks "does the output change" needs a real output layer first.
    nn.init.normal_(denoiser.conv_out.weight, std=0.02)


def main():
    torch.manual_seed(0)

    props = DenoiserProps()
    denoiser = Denoiser(props).eval()

    num_params = sum(param.numel() for param in denoiser.parameters())
    print(f"denoiser params {num_params / 1e6:.1f}M")

    inputs = make_inputs(props, batch_size=2)

    with torch.no_grad():
        velocity = denoiser(**inputs)

    print(f"velocity {tuple(velocity.shape)}")
    assert velocity.shape == inputs["z_noisy"].shape
    assert torch.all(velocity == 0)

    randomize_output(denoiser)

    with torch.no_grad():
        velocity = denoiser(**inputs)

        # Sample 0 run alone has to match sample 0 run in a batch. GroupNorm and attention
        # are both per-sample, so a mismatch means a reshape mixed the batch axis.
        single = denoiser(**{name: value[:1] for name, value in inputs.items()})
        assert torch.allclose(velocity[:1], single, atol=1e-4)

        pressed = dict(inputs)
        pressed["actions"] = 1 - inputs["actions"]
        assert not torch.allclose(velocity, denoiser(**pressed), atol=1e-4)

        noisier = dict(inputs)
        noisier["tau"] = 1 - inputs["tau"]
        assert not torch.allclose(velocity, denoiser(**noisier), atol=1e-4)

        elsewhere = dict(inputs)
        elsewhere["level"] = (inputs["level"] + 1) % props.num_levels
        elsewhere["x_pos"] = 1 - inputs["x_pos"]
        assert not torch.allclose(velocity, denoiser(**elsewhere), atol=1e-4)

        # With the state dropped, the level and position must not leak through.
        dropped = dict(inputs)
        dropped["state_mask"] = torch.zeros(2)
        dropped_elsewhere = dict(elsewhere)
        dropped_elsewhere["state_mask"] = torch.zeros(2)
        assert torch.equal(denoiser(**dropped), denoiser(**dropped_elsewhere))

    print("conditioning checks passed")

    attention = SelfAttention(props.channels[-1], props).eval()
    X = torch.randn(2, props.channels[-1], 7, 8)

    with torch.no_grad():
        tokens = attention.norm(X).flatten(2).transpose(1, 2)
        q, k, v = attention.qkv(tokens).chunk(3, dim=-1)
        q, k, v = [
            part.reshape(2, 7 * 8, props.num_heads, attention.d_head).transpose(1, 2)
            for part in (q, k, v)
        ]

        reference = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 7 * 8, -1)
        reference = X + attention.out(reference).transpose(1, 2).reshape(X.shape)

        assert torch.allclose(attention(X), reference, atol=1e-5)

    print("attention matches F.scaled_dot_product_attention")

    embedding = scalar_embedding(torch.tensor([0.0, 0.5, 1.0]), 128)
    assert embedding.shape == (3, 128)
    assert not torch.allclose(embedding[1], embedding[2])

    denoiser.train()
    inputs["state_mask"] = torch.tensor([1.0, 0.0])

    loss = denoiser(**inputs).pow(2).mean()
    loss.backward()

    missing = [name for name, param in denoiser.named_parameters() if (param.grad is None)]
    assert missing == [], f"no gradient reached {missing}"

    print("\nAll tests passed.")

if (__name__ == "__main__"):
    main()
