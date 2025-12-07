import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from src.env import MathEnv
from src.model import DQN, ReplayBuffer, choose_action

def train(episodes=300):
    env = MathEnv()
    q = DQN()
    target = DQN()
    target.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    gamma = 0.95
    eps = 1.0
    decay = 0.993

    rewards = []
    transitions_log = []

    for ep in range(episodes):
        s = env.reset()
        total = 0
        done = False

        while not done:
            a = choose_action(q, s, eps)
            ns, r, done, _ = env.step(a)

            if ns is None:
                ns = np.zeros_like(s)

            buffer.push(s, a, r, ns, done)

            transitions_log.append({
                "x": s[0],
                "y": s[1],
                "action": a,
                "reward": r,
                "correct_answer": int(s[0] + s[1]),
                "done": done
            })

            s = ns
            total += r

            if len(buffer.buffer) > 64:
                bs, ba, br, bns, bd = buffer.sample()

                bs  = torch.tensor(bs, dtype=torch.float32)
                ba  = torch.tensor(ba, dtype=torch.long).unsqueeze(1)
                br  = torch.tensor(br, dtype=torch.float32).unsqueeze(1)
                bns = torch.tensor(bns, dtype=torch.float32)
                bd  = torch.tensor(bd, dtype=torch.float32).unsqueeze(1)

                qvals = q(bs).gather(1, ba)

                with torch.no_grad():
                    target_q = br + gamma * (1 - bd) * target(bns).max(1, keepdim=True)[0]

                loss = nn.MSELoss()(qvals, target_q)

                opt.zero_grad()
                loss.backward()
                opt.step()

        eps = max(0.05, eps * decay)
        rewards.append(total)

        if ep % 20 == 0:
            target.load_state_dict(q.state_dict())
            print(f"Épisode {ep} | Reward = {total}")

    df = pd.DataFrame(transitions_log)
    return q, rewards, df
