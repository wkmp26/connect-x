# import torch
# import random
# import math
# import numpy as np
# from itertools import count
# from kaggle_environments import make
# import torch.nn.functional as F
# import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyperparameters
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 2000
# BATCH_SIZE = 256
# GAMMA = 0.999
# TARGET_UPDATE = 10
# NUM_EPISODES = 20000

# class ReplayMemory:
#     def __init__(self, capacity=10000):
#         self.capacity = capacity
#         self.memory = []

#     def dump(self, transition):
#         self.memory.append(transition)
#         if len(self.memory) > self.capacity:
#             self.memory.pop(0)

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)

# # DQN Model
# class DQN(nn.Module):
#     def __init__(self, outputs):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(6 * 7 * 64, 128)
#         self.fc2 = nn.Linear(128, outputs)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

# # Initialize environment
# env = make("connectx", debug=True)
# env.reset()
# n_actions = env.configuration.columns

# # instantiate models and training 
# policy_net = DQN(n_actions).to(device)
# target_net = DQN(n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

# optimizer = torch.optim.Adam(policy_net.parameters())
# memory = ReplayMemory()
# steps_done = 0

# def get_available_actions(board, columns):
#     return [c for c in range(columns) if board[c] == 0]

# def select_action(state, available_actions, training=True):
#     global steps_done
#     steps_done += 1
#     state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

#     if training:
#         eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
#         if random.random() < eps_threshold:
#             return random.choice(available_actions)

#     with torch.no_grad():
#         q_values = policy_net(state_tensor)[0]
#         q_values[~torch.tensor([i in available_actions for i in range(n_actions)], dtype=torch.bool)] = -float('inf')
#         return torch.argmax(q_values).item()

# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)

#     state_batch = torch.tensor(np.array([np.expand_dims(t[0], 0) for t in transitions]), dtype=torch.float32, device=device)
#     action_batch = torch.tensor([t[1] for t in transitions], dtype=torch.int64, device=device).unsqueeze(1)
#     reward_batch = torch.tensor([t[2] for t in transitions], dtype=torch.float32, device=device)

#     non_final_mask = torch.tensor([t[3] is not None for t in transitions], dtype=torch.bool, device=device)
#     non_final_next_states = torch.tensor(np.array([np.expand_dims(t[3], 0) for t in transitions if t[3] is not None]),
#                                          dtype=torch.float32, device=device)

#     state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze()
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

#     expected_state_action_values = reward_batch + (GAMMA * next_state_values)

#     loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# # Training loop
# for episode in range(NUM_EPISODES):
#     env.reset()
#     state = np.array(env.state[0].observation['board']).reshape(6, 7)

#     for t in count():
#         available_actions = get_available_actions(env.state[0].observation['board'], env.configuration.columns)
#         action = select_action(state, available_actions, training=True)

#         # Step for agent
#         env.step([action, None])
#         new_state = env.state[0]
#         done = new_state.status != "ACTIVE"
#         reward = new_state.reward if new_state.reward is not None else 0

#         if done:
#             memory.dump((state, action, reward, None))
#             break

#         # Opponent (random)
#         opp_available = get_available_actions(env.state[1].observation['board'], env.configuration.columns)
#         opp_action = random.choice(opp_available)
#         env.step([None, opp_action])

#         new_state = env.state[0]
#         reward = new_state.reward if new_state.reward is not None else 0
#         done = new_state.status != "ACTIVE"
#         next_state = np.array(new_state.observation['board']).reshape(6, 7)

#         if done:
#             reward = -1 if new_state.status == "LOST" else 0.5
#             memory.dump((state, action, reward, None))
#             break

#         memory.dump((state, action, -0.05, next_state))
#         state = next_state
#         optimize_model()

#     if episode % TARGET_UPDATE == 0:
#         target_net.load_state_dict(policy_net.state_dict())

#     if episode % 500 == 0:
#         print(f"Episode {episode} complete")

# # Save model
# torch.save(policy_net.state_dict(), "connectx_dqn.pth")
# print("Training complete and model saved.")

import torch
import random
import math
import numpy as np
from itertools import count
from kaggle_environments import make
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000
BATCH_SIZE = 256
GAMMA = 0.999
TARGET_UPDATE = 20
NUM_EPISODES = 20000
NEGAMAX_START = 5000
BLOCKER_START = 2000

# Replay memory
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []

    def dump(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN with dropout
class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(6 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Encode state into 2-channel format
def encode_board(board, mark):
    board = np.array(board).reshape(6, 7)
    p1 = (board == mark).astype(np.float32)
    p2 = (board == (3 - mark)).astype(np.float32)
    return np.stack([p1, p2])

# Environment setup
env = make("connectx", debug=True)
env.reset()
n_actions = env.configuration.columns

# Models
policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = torch.optim.Adam(policy_net.parameters())
memory = ReplayMemory()
steps_done = 0

def get_available_actions(board, columns):
    return [c for c in range(columns) if board[c] == 0]

def select_action(state, available_actions, training=True):
    global steps_done
    steps_done += 1
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    if training:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        if random.random() < eps_threshold:
            return random.choice(available_actions)
    with torch.no_grad():
        q_values = policy_net(state_tensor)[0]
        mask = torch.tensor([i in available_actions for i in range(n_actions)], dtype=torch.bool, device=device)
        q_values[~mask] = -float('inf')
        return torch.argmax(q_values).item()

def check_threat(board, opponent):
    board = np.array(board).reshape(6, 7)
    for col in range(7):
        temp = board.copy()
        for row in reversed(range(6)):
            if temp[row, col] == 0:
                temp[row, col] = opponent
                if check_win(temp, opponent):
                    return col
                break
    return None

def check_win(board, mark):
    for r in range(6):
        for c in range(4):
            if all(board[r, c+i] == mark for i in range(4)): return True
    for r in range(3):
        for c in range(7):
            if all(board[r+i, c] == mark for i in range(4)): return True
    for r in range(3):
        for c in range(4):
            if all(board[r+i, c+i] == mark for i in range(4)): return True
            if all(board[r+3-i, c+i] == mark for i in range(4)): return True
    return False

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    state_batch = torch.tensor(np.array([t[0] for t in transitions]), dtype=torch.float32, device=device)
    action_batch = torch.tensor([t[1] for t in transitions], dtype=torch.int64, device=device).unsqueeze(1)
    reward_batch = torch.tensor([t[2] for t in transitions], dtype=torch.float32, device=device)
    non_final_mask = torch.tensor([t[3] is not None for t in transitions], dtype=torch.bool, device=device)
    non_final_next_states = torch.tensor(np.array([t[3] for t in transitions if t[3] is not None]), dtype=torch.float32, device=device)
    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze()
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_values = reward_batch + GAMMA * next_state_values
    loss = F.smooth_l1_loss(state_action_values, expected_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
for episode in range(NUM_EPISODES):
    env.reset()
    mark = 1
    raw_board = env.state[0].observation['board']
    state = encode_board(raw_board, mark)
    for t in count():
        available_actions = get_available_actions(env.state[0].observation['board'], env.configuration.columns)
        action = select_action(state, available_actions, training=True)
        env.step([action, None])
        new_state = env.state[0]
        done = new_state.status != "ACTIVE"
        reward = new_state.reward if new_state.reward is not None else 0
        if done:
            memory.dump((state, action, reward, None))
            break
        # Opponent strategy
        if episode < BLOCKER_START:
            opp_action = random.choice(get_available_actions(env.state[1].observation['board'], env.configuration.columns))
        elif episode < NEGAMAX_START:
            threat = check_threat(env.state[1].observation['board'], 1)
            opp_action = threat if threat is not None else random.choice(get_available_actions(env.state[1].observation['board'], env.configuration.columns))
        else:
            from kaggle_environments.envs.connectx.connectx import negamax_agent
            obs = env.state[1].observation
            conf = env.configuration
            opp_action = negamax_agent(obs, conf)
        env.step([None, opp_action])
        new_state = env.state[0]
        done = new_state.status != "ACTIVE"
        next_state = encode_board(new_state.observation['board'], mark)
        if done:
            reward = -1 if new_state.status == "LOST" else 0.5
            memory.dump((state, action, reward, None))
            break
        memory.dump((state, action, -0.05, next_state))
        state = next_state
        optimize_model()
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if episode % 500 == 0:
        print(f"Episode {episode} complete")

# Save model
torch.save(policy_net.state_dict(), "connectx_dqn.pth")
print("Training complete and model saved.")
