import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from kaggle_environments import make


# DQN model
class DQN(nn.Module):
    def __init__(self, input_size=42, output_size=7):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


def preprocess(board, mark):
    state = np.array(board, dtype=np.float32)
    state = np.where(state == mark, 1, state)
    state = np.where((state != 1) & (state != 0), -1, state)
    return state

def train_dqn(
    episodes=300,
    batch_size=64,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995
):
    env = make("connectx", debug=True)
    model = DQN()
    target_model = DQN()
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    memory = ReplayBuffer()

    epsilon = epsilon_start
    update_target_every = 10

    for episode in range(episodes):
        env.reset()
        state = preprocess(env.state[0].observation.board, env.state[0].observation.mark)
        done = False
        total_reward = 0

        while not done:
            valid_moves = [c for c in range(7) if env.state[0].observation.board[c] == 0]

            if random.random() < epsilon:
                action = random.choice(valid_moves)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state).float().unsqueeze(0))
                    q_values = q_values.numpy().flatten()
                    for c in range(7):
                        if c not in valid_moves:
                            q_values[c] = -np.inf
                    action = int(np.argmax(q_values))

            env.step([action, "random"])
            next_state = preprocess(env.state[0].observation.board, env.state[0].observation.mark)
            reward = env.state[0].reward or 0
            done = env.done

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Training
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                states = torch.tensor(states, dtype=torch.float)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float)
                next_states = torch.tensor(next_states, dtype=torch.float)
                dones = torch.tensor(dones, dtype=torch.bool)

                q_vals = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    max_q_next = target_model(next_states).max(1)[0]
                    target = rewards + gamma * max_q_next * (~dones)

                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/dqn_model.pth")
    print("Model saved")


if __name__ == "__main__":
    train_dqn(episodes=300)
