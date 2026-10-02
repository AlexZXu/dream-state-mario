import gzip

import numpy as np
import stable_retro as retro

GAME = "SuperMarioBros-Nes-v0"
START_STATE = "Level1-1"

FRAME_SKIP = 3  # NES frames per model step: the console runs at 60 fps, the model at 20
CROP_TOP = 8  # the NES draws 240 rows; the top and bottom 8 are overscan
FRAME_H = 224
FRAME_W = 256

BUTTONS = ["LEFT", "RIGHT", "UP", "DOWN", "A", "B"]  # order of the 6 entries in an action

# Addresses in the console's 2 KB of RAM. The rollouts store the whole RAM for every step
# and these are only applied afterwards, so a wrong address here means fixing decode(),
# not collecting the data again.
RAM_PLAYER_STATE = 0x000E
RAM_X_PAGE = 0x006D
RAM_X_ON_PAGE = 0x0086
RAM_Y_ON_SCREEN = 0x00CE
RAM_POWERUP = 0x0756
RAM_LIVES = 0x075A
RAM_STAGE = 0x075C
RAM_WORLD = 0x075F

PLAYING = 0x08  # player state while Mario is under the player's control
DYING = 0x0B  # the death animation, Mario flipping off the screen
DEAD = 0x06  # the single frame that ends a life


def make_env(state=START_STATE):
    # Actions.ALL exposes the pad as one on/off bit per button. The default mode filters
    # combinations through a table, which is one more thing between the keyboard and the game.
    return retro.make(
        game=GAME,
        state=state,
        render_mode="rgb_array",
        use_restricted_actions=retro.Actions.ALL
    )


def preprocess(frame):
    # frame shape: (240, 256, 3) uint8 -> (224, 256, 3) uint8. Some builds of the emulator
    # core crop the overscan themselves and hand back 224 rows already.
    if (frame.shape[0] != FRAME_H):
        frame = frame[CROP_TOP:CROP_TOP + FRAME_H]

    assert frame.shape == (FRAME_H, FRAME_W, 3)

    return frame


def clean(action):
    # A real d-pad cannot press two opposite directions, and the game glitches when an
    # emulator lets it happen. A keyboard can, so both collection and the website cancel
    # the pair out and the model never has to learn what it means.
    action = np.array(action, dtype=np.int8)
    left, right, up, down = [BUTTONS.index(name) for name in ("LEFT", "RIGHT", "UP", "DOWN")]

    if (action[left] and action[right]):
        action[left] = action[right] = 0

    if (action[up] and action[down]):
        action[up] = action[down] = 0

    return action


def to_pad(buttons, action):
    # action shape: (6,) in BUTTONS order -> (len(buttons),) in the emulator's pad order.
    # Start and select stay unpressed, so the player can never pause or leave the level.
    pad = np.zeros(len(buttons), dtype=np.int8)

    for i, name in enumerate(BUTTONS):
        pad[buttons.index(name)] = action[i]

    return pad


def step(env, action):
    pad = to_pad(env.buttons, action)

    # The buttons are held for every NES frame of the step and only the last frame is
    # kept. The website samples the keyboard once per generated frame, so this is exactly
    # the control the model will be given there.
    for _ in range(FRAME_SKIP):
        obs, reward, terminated, truncated, info = env.step(pad)

        if (terminated or truncated):
            break

    return preprocess(obs), env.get_ram().copy(), (terminated or truncated)


def rollout(env, num_steps, policy):
    frames = []
    actions = []
    rams = []

    for _ in range(num_steps):
        action = clean(policy())
        frame, ram, done = step(env, action)

        # actions[i] is what was held while frames[i] was being produced, not what was
        # pressed after seeing it.
        frames.append(frame)
        actions.append(action)
        rams.append(ram)

        if (done):
            break

    return np.stack(frames), np.stack(actions), np.stack(rams)


def save_state(env, path):
    # gzip is the format the emulator's own .state files use.
    with gzip.open(path, "wb") as f:
        f.write(env.em.get_state())


def load_state(env, path):
    with gzip.open(path, "rb") as f:
        env.em.set_state(f.read())

    env.data.update_ram()


def decode(ram):
    # ram shape: (..., ram_size) uint8, one step or a whole rollout
    ram = ram.astype(np.int32)

    return {
        "world": ram[..., RAM_WORLD],
        "stage": ram[..., RAM_STAGE],
        "x_pos": ram[..., RAM_X_PAGE] * 256 + ram[..., RAM_X_ON_PAGE],
        "y_pos": ram[..., RAM_Y_ON_SCREEN],
        "powerup": np.minimum(ram[..., RAM_POWERUP], 2),
        "lives": ram[..., RAM_LIVES],
        "player_state": ram[..., RAM_PLAYER_STATE]
    }


class NoisyPolicy:
    def __init__(self, rng):
        self.rng = rng

        self.direction = 1  # -1 left, 0 neither, 1 right
        self.direction_steps = 0

        self.run = False
        self.run_steps = 0

        self.jump_steps = 0
        self.duck_steps = 0
        self.random_steps = 0

    def __call__(self):
        rng = self.rng
        action = np.zeros(len(BUTTONS), dtype=np.int8)

        # Every so often the buttons are pure noise for a second or two. Nobody plays like
        # that, but a visitor mashing the keyboard should not be the first time the model
        # sees a strange combination.
        if (self.random_steps == 0 and rng.random() < 0.005):
            self.random_steps = int(rng.integers(10, 40))

        if (self.random_steps > 0):
            self.random_steps -= 1
            return (rng.random(len(BUTTONS)) < 0.3).astype(np.int8)

        # The screen only scrolls forward, so the policy leans right or it would never
        # leave the start of the level. Standing still and walking left still have to
        # show up, because a player will do both.
        if (self.direction_steps == 0):
            self.direction = int(rng.choice([-1, 0, 1], p=[0.15, 0.15, 0.7]))
            self.direction_steps = int(rng.integers(10, 80))

        self.direction_steps -= 1

        if (self.run_steps == 0):
            self.run = bool(rng.random() < 0.6)
            self.run_steps = int(rng.integers(20, 100))

        self.run_steps -= 1

        # How long A is held sets how high Mario jumps, so the hold length is drawn from
        # a tap up to a full jump instead of a fixed press.
        if (self.jump_steps == 0 and rng.random() < 0.08):
            self.jump_steps = int(rng.integers(1, 15))

        if (self.duck_steps == 0 and rng.random() < 0.01):
            self.duck_steps = int(rng.integers(5, 20))

        action[BUTTONS.index("LEFT")] = (self.direction == -1)
        action[BUTTONS.index("RIGHT")] = (self.direction == 1)
        action[BUTTONS.index("B")] = self.run

        if (self.jump_steps > 0):
            self.jump_steps -= 1
            action[BUTTONS.index("A")] = 1

        if (self.duck_steps > 0):
            self.duck_steps -= 1
            action[BUTTONS.index("DOWN")] = 1

        return action
