import random
import torch.nn as nn
import numpy as np
import random
from collections import deque

STATE_DIM = 2
N_ACTIONS = 21

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, size=5000):
        self.buffer = deque(maxlen=size)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch=64):
        batch = random.sample(self.buffer, batch)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(ns),
            np.array(d)
        )


def choose_action(model, state, eps):
    if random.random() < eps:
        return random.randint(0, 20)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return model(s).argmax().item()

