from kaggle_environments import evaluate, make
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cpu")

# DQN
class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        # input: 2 channels (agent and opponent boards), 6x7 ConnectX grid
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)  # First convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolution layer
        self.dropout = nn.Dropout(0.3)  # Dropout
        self.fc1 = nn.Linear(6 * 7 * 64, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, outputs)  # Output layer for Q-values 

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x)) 
        x = self.dropout(x)      
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))  
        return self.fc2(x)  

# load trained model weights
model = DQN(7)  # 7 columns = 7 possible actions in ConnectX
model.load_state_dict(torch.load("connectx_dqn.pth", map_location=device))  # load model from file
model.eval()


# convert flat board into a 2-channel format:
# channel 1: agent's pieces
# channel 2: opponent's pieces
def encode_board(board, mark):
    board = np.array(board).reshape(6, 7)
    p1 = (board == mark).astype(np.float32)
    p2 = (board == (3 - mark)).astype(np.float32)
    return np.stack([p1, p2])

# return indices of columns where a move is possible
def get_available_actions(board, width):

    return [col for col in range(width) if board[col] == 0]

# this is called by the Kaggle environment at each move
# selects best action using the trained DQN model
def my_agent(observation, configuration):
    # encode current board for the DQN
    board = encode_board(observation.board, observation.mark)
    board_input = torch.tensor(board, dtype=torch.float32).unsqueeze(0)

    # get list of valid columns 
    available_actions = get_available_actions(observation.board, configuration.columns)

    # use model to select action with the highest Q-value among valid actions
    with torch.no_grad():
        action_values = model(board_input)[0] 
        mask = torch.full_like(action_values, float('-inf'))
        for a in available_actions:
            mask[a] = action_values[a]
        action = torch.argmax(mask).item()  # choose action with highest valid Q-value

    return action  # return action to the environment
if __name__ == "__main__":
    env = make("connectx", debug=True)
    env.run([my_agent, "negamax"])
    env.render(mode="ipython", width=500, height=450)