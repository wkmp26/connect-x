import torch
import torch.nn as nn
import numpy as np

# DQN model
class DQN(nn.Module):
    def __init__(self, input_size=42, output_size=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

# Load the model
model = DQN()
try:
    model.load_state_dict(torch.load("models/dqn_model.pth"))
    model.eval()
except FileNotFoundError:
    print("Trained model not found. Random actions will be returned.")
    model = None

def preprocess(board, mark):
    state = np.array(board, dtype=np.float32)
    state = np.where(state == mark, 1, state)
    state = np.where((state != 1) & (state != 0), -1, state)
    return state

def my_agent(observation, configuration):
    board = observation.board
    mark = observation.mark
    state = preprocess(board, mark)
    state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)

    valid_moves = [c for c in range(configuration.columns) if board[c] == 0]
    if not valid_moves:
        return 0  # fallback

    if model:
        with torch.no_grad():
            q_values = model(state_tensor).squeeze().numpy()
            for c in range(configuration.columns):
                if c not in valid_moves:
                    q_values[c] = -float("inf")
            action = int(np.argmax(q_values))
    else:
        action = np.random.choice(valid_moves)

    return action
