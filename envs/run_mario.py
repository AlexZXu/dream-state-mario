from envs.mario import (
    make_env, step, rollout, clean, to_pad, decode, save_state, load_state, NoisyPolicy,
    BUTTONS, FRAME_H, FRAME_W, FRAME_SKIP, PLAYING, RAM_PLAYER_STATE, RAM_X_PAGE, RAM_X_ON_PAGE
)
from vmodel import tokenizer
import os
import tempfile

import numpy as np

NES_PAD = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]


def hold(*names):
    action = np.zeros(len(BUTTONS), dtype=np.int8)

    for name in names:
        action[BUTTONS.index(name)] = 1

    return lambda: action


def check_without_rom():
    assert (FRAME_H, FRAME_W) == (tokenizer.FRAME_H, tokenizer.FRAME_W)

    pad = to_pad(NES_PAD, hold("RIGHT", "A")())
    assert [NES_PAD[i] for i in np.flatnonzero(pad)] == ["RIGHT", "A"]
    assert to_pad(NES_PAD, np.ones(len(BUTTONS), dtype=np.int8)).sum() == len(BUTTONS)

    assert clean(hold("LEFT", "RIGHT", "A")()).tolist() == hold("A")().tolist()
    assert clean(hold("UP", "DOWN", "LEFT")()).tolist() == hold("LEFT")().tolist()
    assert clean(hold("RIGHT", "B")()).tolist() == hold("RIGHT", "B")().tolist()

    ram = np.zeros((5, 2048), dtype=np.uint8)
    ram[:, RAM_X_PAGE] = 3
    ram[:, RAM_X_ON_PAGE] = 200
    ram[:, RAM_PLAYER_STATE] = PLAYING

    state = decode(ram)
    assert state["x_pos"].tolist() == [3 * 256 + 200] * 5
    assert decode(ram[0])["player_state"] == PLAYING

    policy = NoisyPolicy(np.random.default_rng(0))
    actions = np.stack([clean(policy()) for _ in range(20_000)])
    left, right, a = [actions[:, BUTTONS.index(name)] for name in ("LEFT", "RIGHT", "A")]

    assert actions.shape == (20_000, len(BUTTONS))
    assert set(np.unique(actions).tolist()) == {0, 1}
    assert not np.any(left & right)
    assert 0.55 < right.mean() < 0.8
    assert 0.05 < left.mean() < 0.25
    assert 0.2 < a.mean() < 0.5

    # Jump holds have to cover a tap through a full jump, since hold length is jump height.
    edges = np.flatnonzero(np.diff(np.concatenate([[0], a, [0]])))
    holds = edges[1::2] - edges[0::2]
    assert holds.min() == 1
    assert holds.max() >= 14

    # The same seed has to give the same buttons, or a rollout index would not pin down
    # its rollout.
    again = NoisyPolicy(np.random.default_rng(0))
    assert np.array_equal(actions, np.stack([clean(again()) for _ in range(20_000)]))

    print("button mapping, RAM decode and policy checks passed")


def check_with_rom(env):
    obs, info = env.reset()
    print(f"raw frame {obs.shape}  ram {env.get_ram().shape}  buttons {env.buttons}")
    assert env.buttons == NES_PAD

    start = decode(env.get_ram())
    print(f"start state {start}")
    assert start["world"] == 0 and start["stage"] == 0

    frames, actions, rams = rollout(env, 40, hold("RIGHT"))
    state = decode(rams)
    print(f"held right for 40 steps: x_pos {state['x_pos'][0]} -> {state['x_pos'][-1]}")

    assert frames.shape == (40, FRAME_H, FRAME_W, 3)
    assert frames.dtype == np.uint8
    assert actions.shape == (40, len(BUTTONS))
    assert np.all(np.diff(state["x_pos"]) >= 0)
    assert state["x_pos"][-1] - state["x_pos"][0] > 40
    assert np.all(state["player_state"][-10:] == PLAYING)

    # Restoring a savestate has to replay identically, or rollouts started from the bank
    # would not be reproducible.
    path = os.path.join(tempfile.mkdtemp(), "check.state")
    save_state(env, path)

    policy = NoisyPolicy(np.random.default_rng(1))
    frames_a, actions_a, rams_a = rollout(env, 60, policy)

    load_state(env, path)
    assert np.array_equal(env.get_ram(), rams[-1])

    replay = iter(actions_a)
    frames_b, actions_b, rams_b = rollout(env, 60, lambda: next(replay))

    assert np.array_equal(frames_a, frames_b)
    assert np.array_equal(rams_a, rams_b)
    print("savestate replay is deterministic")

    # Walking right without jumping runs into the first Goomba. That has to show up as
    # the player state leaving PLAYING and then a life being lost.
    env.reset()
    frames, actions, rams = rollout(env, 300, hold("RIGHT"))
    state = decode(rams)

    print(f"walked into the Goomba: player states seen {np.unique(state['player_state']).tolist()}  lives {state['lives'][0]} -> {state['lives'].min()}")
    assert np.any(state["player_state"] != PLAYING)
    assert state["lives"].min() < state["lives"][0]

    jump = rollout(env, 20, hold("A"))[2]
    print(f"held A for 20 steps: y_pos {decode(jump)['y_pos'].tolist()}")


def main():
    check_without_rom()

    try:
        env = make_env()
    except FileNotFoundError:
        print("\nThe Super Mario Bros ROM is not imported, so the emulator checks did not run.")
        print("Import it with: python -m stable_retro.import <folder containing the ROM>")
        return

    check_with_rom(env)
    env.close()

    print(f"\nAll tests passed. ({FRAME_SKIP} NES frames per step)")

if (__name__ == "__main__"):
    main()
