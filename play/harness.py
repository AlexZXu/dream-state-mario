from envs.smb_sequences import FRAME_SKIP, CONTEXT_FRAMES, X_POS_SCALE, level_id

START_LIVES = 3
NUM_LEVELS = 32
DIED_STEPS = 10  # consecutive steps the reader must say "died" before the respawn: half a second
CLEAR_STEPS = 40  # two seconds of "clear", so the flagpole slide gets to play out
THRESHOLD = 0.5


def level_starts(episodes):
    # level id -> the first frame of that level that has a full context window before
    # it. These are the seed clips: the model cannot draw a level from nothing, so every
    # (re)start is the real opening second of the level.
    span = FRAME_SKIP * CONTEXT_FRAMES + FRAME_SKIP - 1
    starts = {}

    for episode in episodes:
        level = level_id(int(episode["world"]), int(episode["stage"]))

        if (level not in starts):
            starts[level] = int(episode["start"]) + span

    return starts


class Harness:
    """The part of the game that is ordinary code: lives, which level, how far along it,
    and what to do on a death or a flagpole. update() is called once per generated frame
    with what the reader saw, and answers with the event the player loop must act on."""

    def __init__(self, level=0):
        self.level = level
        self.lives = START_LIVES

        self.x_pos = 0.0  # pixels into the level, integrated from the reader's dx
        self.powerup = 0

        self.died_steps = 0
        self.clear_steps = 0

    def restart_level(self):
        self.x_pos = 0.0
        self.powerup = 0
        self.died_steps = 0
        self.clear_steps = 0

    def update(self, dx, died, clear, powerup):
        # dx is in pixels; died and clear are probabilities; powerup is the class index.
        # Returns None, "respawn", "next_level" or "game_over".
        self.x_pos = max(0.0, self.x_pos + dx)
        self.powerup = powerup

        # One confident frame is not enough: a generated frame can flicker, and a respawn
        # the player did not earn is worse than one that arrives half a second late.
        self.died_steps = self.died_steps + 1 if (died > THRESHOLD) else 0
        self.clear_steps = self.clear_steps + 1 if (clear > THRESHOLD) else 0

        if (self.died_steps >= DIED_STEPS):
            self.lives -= 1
            self.restart_level()

            if (self.lives == 0):
                self.lives = START_LIVES
                self.level = 0

                return "game_over"

            return "respawn"

        if (self.clear_steps >= CLEAR_STEPS):
            # After 8-4 the game wraps round to 1-1.
            self.level = (self.level + 1) % NUM_LEVELS
            self.restart_level()

            return "next_level"

        return None

    def conditioning(self):
        # What the denoiser is told about where the player is.
        return {
            "level": self.level,
            "x_pos": min(self.x_pos / X_POS_SCALE, 1.0),
            "powerup": self.powerup
        }
