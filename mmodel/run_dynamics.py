from mmodel.denoiser import Denoiser, DenoiserProps
from mmodel.flow import add_noise, sample_tau, flow_loss, sample
from mmodel import train_dynamics
from envs.smb_sequences import SmbSequences, FRAME_SKIP, CONTEXT_FRAMES, level_id, X_POS_SCALE
from envs.smb_frames import split
from envs.smb import load
from envs.mario import decode

import numpy as np
import torch


class Oracle:
    """Stands in for a denoiser that knows the answer: given the noisy latent at tau it
    returns the exact velocity eps - z, so the sampler has to land on z precisely."""

    def __init__(self, z):
        self.z = z
        self.tau = None

    def denoise(self, x, context, cond):
        eps = (x - (1 - self.tau) * self.z) / self.tau

        return eps - self.z


class FakeLatents:
    """Looks like the latents memmap, but the latent of frame i just holds i, so a window
    can be checked for pulling exactly the frames it should."""

    def __init__(self, N):
        self.shape = (N, 8, 28, 32)

    def __getitem__(self, frames):
        latents = np.zeros((len(frames), 8, 28, 32), dtype=np.float16)
        latents[:, 0, 0, 0] = np.asarray(frames) % 2048  # exact in float16

        return latents


def check_flow():
    z = torch.randn(4, 8, 28, 32)
    eps = torch.randn_like(z)

    clean, _ = add_noise(z, torch.zeros(4), eps)
    noise, _ = add_noise(z, torch.ones(4), eps)
    half, _ = add_noise(z, torch.full((4,), 0.5), eps)

    assert torch.equal(clean, z)
    assert torch.equal(noise, eps)
    assert torch.allclose(half, 0.5 * z + 0.5 * eps)
    assert flow_loss(eps - z, z, eps).item() == 0

    tau = sample_tau(100_000, "cpu")
    assert tau.min() > 0 and tau.max() < 1
    assert abs(tau.mean().item() - 0.5) < 0.01

    # The path is a straight line, so the oracle's Euler walk is exact with 1 step or 4.
    oracle = Oracle(z)

    for num_steps in (1, 4):
        taus = torch.linspace(1.0, 0.0, num_steps + 1)
        x = eps.clone()

        for i in range(num_steps):
            oracle.tau = taus[i]
            x = x - (taus[i] - taus[i + 1]) * oracle.denoise(x, None, None)

        assert torch.allclose(x, z, atol=1e-4)

    print("flow maths passed")


def check_sampler(props):
    denoiser = Denoiser(props).eval()
    context = torch.randn(2, props.context_frames, props.latent_channels, 28, 32)
    cond_inputs = {
        "actions": torch.rand(2, props.context_frames + 1, props.action_dim),
        "level": torch.zeros(2, dtype=torch.long),
        "x_pos": torch.rand(2),
        "powerup": torch.zeros(2, dtype=torch.long),
        "state_mask": torch.ones(2)
    }

    z = sample(denoiser, context, cond_inputs, num_steps=2, tau_ctx=0.1)
    assert z.shape == (2, props.latent_channels, 28, 32)
    assert torch.isfinite(z).all()

    # Same noise in, same latent out, when the context is not re-noised.
    eps = torch.randn(2, props.latent_channels, 28, 32)
    once = sample(denoiser, context, cond_inputs, num_steps=2, tau_ctx=0.0, eps=eps)
    twice = sample(denoiser, context, cond_inputs, num_steps=2, tau_ctx=0.0, eps=eps)
    assert torch.equal(once, twice)

    print("sampler passed")


def check_dataset():
    data = load()
    episodes = data["episodes"]
    train_episodes, val_episodes = split(episodes)
    fake = FakeLatents(data["frames"].shape[0])

    dset = SmbSequences(train=True, latents=fake)
    span = FRAME_SKIP * CONTEXT_FRAMES + FRAME_SKIP - 1

    assert len(dset) == (episodes["length"][train_episodes] - span).sum()

    item = dset[12345]
    t = dset.index[12345]
    frames = np.arange(t - FRAME_SKIP * CONTEXT_FRAMES, t + 1, FRAME_SKIP)

    assert item["context"].shape == (CONTEXT_FRAMES, 8, 28, 32)
    assert item["target"].shape == (8, 28, 32)
    assert item["actions"].shape == (CONTEXT_FRAMES + 1, FRAME_SKIP * 6)

    # The window is the target and the 8 frames before it, 3 apart, oldest first.
    assert item["target"][0, 0, 0].item() == t % 2048
    assert [int(value) for value in item["context"][:, 0, 0, 0]] == [frame % 2048 for frame in frames[:-1]]

    # Each frame's action is the buttons of the 3 frames leading up to it.
    expected = np.concatenate([np.asarray(data["actions"][frame - FRAME_SKIP + 1:frame + 1]).reshape(-1) for frame in frames])
    assert np.array_equal(item["actions"].numpy().reshape(-1), expected)

    # The whole window, buttons included, sits inside one episode, and the labels come
    # from the target's RAM.
    e = np.searchsorted(episodes["start"], t, side="right") - 1
    state = decode(np.asarray(data["ram"][t]))

    assert frames[0] - FRAME_SKIP + 1 >= episodes["start"][e]
    assert item["level"].item() == level_id(episodes["world"][e], episodes["stage"][e])
    assert abs(item["x_pos"].item() - state["x_pos"] / X_POS_SCALE) < 1e-6
    assert item["powerup"].item() == state["powerup"]

    # No target earlier than the first full window of its episode, and the very first
    # target of the whole dataset (episode 0 starts at frame 0) must not reach before
    # frame 0 for its buttons. That is the case that caught the off-by-two the first time.
    some_episode = episodes[train_episodes[3]]
    first = dset.index[np.searchsorted(dset.index, some_episode["start"])]
    assert first == some_episode["start"] + span
    assert dset.index[0] == span
    assert dset[0]["actions"].shape == (CONTEXT_FRAMES + 1, FRAME_SKIP * 6)

    world8 = SmbSequences(train=True, worlds=[7], latents=fake)
    assert np.all(world8.level // 4 == 7)
    assert 0 < len(world8) < len(dset)

    print(f"sequence dataset passed: {len(dset)} training targets, {len(world8)} of them in world 8")

    return dset


def check_training_step(props, dset):
    denoiser = Denoiser(props)
    optimizer = torch.optim.AdamW(params=denoiser.parameters(), lr=1e-4)

    batch = torch.utils.data.default_collate([dset[i] for i in range(4)])

    for _ in range(3):
        optimizer.zero_grad()
        loss = train_dynamics.compute_loss(denoiser, batch, torch.device("cpu"))
        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss)

    # With a zero-initialised output the prediction is 0 and the loss is the mean square
    # of the target velocity eps - z, which is 1 + mean(z^2) since eps is unit noise
    # independent of z.
    fresh = Denoiser(props)
    expected = 1 + batch["target"].pow(2).mean().item()
    assert abs(train_dynamics.compute_loss(fresh, batch, torch.device("cpu")).item() - expected) < 0.1

    print("training step passed")


def main():
    torch.manual_seed(0)

    props = DenoiserProps({"channels": (32, 64, 64), "num_groups": 8, "action_dim": FRAME_SKIP * 6})

    check_flow()
    check_sampler(props)
    dset = check_dataset()
    check_training_step(props, dset)

    print("\nAll tests passed.")

if (__name__ == "__main__"):
    main()
