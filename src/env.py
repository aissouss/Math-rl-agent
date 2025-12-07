import numpy as np
import random

class MathEnv:
    def __init__(self, max_value=10, episode_len=6):
        self.max_value = max_value
        self.episode_len = episode_len

    def _sample(self):
        x = random.randint(0, self.max_value)
        y = random.randint(0, self.max_value)
        self.current_answer = x + y
        self.state = np.array([x, y], dtype=np.float32)

    def reset(self):
        self.steps = 0
        self._sample()
        return self.state

    def step(self, action):
        correct = (action == self.current_answer)
        reward = 2.0 if correct else -1.0

        self.steps += 1
        done = self.steps >= self.episode_len

        if not done:
            self._sample()
            return self.state, reward, False, {}

        return None, reward, True, {}

