# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.nn as nn

# device = torch.device("cpu")

# # class DQN(nn.Module):
# #     def __init__(self, outputs):
# #         super(DQN, self).__init__()
# #         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
# #         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
# #         self.fc1 = nn.Linear(6 * 7 * 64, 128)
# #         self.fc2 = nn.Linear(128, outputs)

# #     def forward(self, x):
# #         x = F.relu(self.conv1(x))
# #         x = F.relu(self.conv2(x))
# #         x = x.view(x.size(0), -1)
# #         x = F.relu(self.fc1(x))
# #         return self.fc2(x)

# class DQN(nn.Module):
#     def __init__(self, outputs):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.dropout = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(6 * 7 * 64, 128)
#         self.fc2 = nn.Linear(128, outputs)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

# model = DQN(7)
# model.load_state_dict(torch.load("connectx_dqn.pth", map_location=device))
# model.eval()

# def get_available_actions(board, width):
#     return [col for col in range(width) if board[col] == 0]

# def my_agent(observation, configuration):
#     board = np.array(observation.board).reshape(configuration.rows, configuration.columns)

#     board_input = torch.tensor(board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#     available_actions = get_available_actions(observation.board, configuration.columns)
#     with torch.no_grad():
#         action_values = model(board_input)[0]
#         mask = torch.full_like(action_values, float('-inf'))
#         for a in available_actions:
#             mask[a] = action_values[a]
#         action = torch.argmax(mask).item()
#     return action


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cpu")

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

model = DQN(7)
model.load_state_dict(torch.load("connectx_dqn.pth", map_location=device))
model.eval()

def encode_board(board, mark):
    board = np.array(board).reshape(6, 7)
    p1 = (board == mark).astype(np.float32)
    p2 = (board == (3 - mark)).astype(np.float32)
    return np.stack([p1, p2])

def get_available_actions(board, width):
    return [col for col in range(width) if board[col] == 0]

def my_agent(observation, configuration):
    board = encode_board(observation.board, observation.mark)
    board_input = torch.tensor(board, dtype=torch.float32).unsqueeze(0)

    available_actions = get_available_actions(observation.board, configuration.columns)
    with torch.no_grad():
        action_values = model(board_input)[0]
        mask = torch.full_like(action_values, float('-inf'))
        for a in available_actions:
            mask[a] = action_values[a]
        action = torch.argmax(mask).item()
    return action
