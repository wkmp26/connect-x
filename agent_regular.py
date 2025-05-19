
# Helper functions to convert between the board and a column ,row tuple


# Helper function to better print the board
def printBoard(board, rows, columns):
    boardString = ""
    for i in range(rows):
        row = []
        for j in range(columns):
            row.append(str(board[i * columns + j]))
        boardString += " ".join(row) + "\n"
    return boardString

from kaggle_environments import evaluate, make

# for now: agent is always 1, always max
agent_num=1
opponent_num=2
agent_minimax="max"

def find_best_move(board,configuration):
    next_states=find_available_boards(board,[c for c in range(configuration.columns)],agent_num)
    values = [minimax(board,configuration,agent_minimax) for board in next_states]
    best_state,best_val=(None,float("-inf"))
    for i in range(len(values)):
        if values[i]>best_val:
            best_val=values[i]
            best_state=next_states[i]
    # find the best move
    diff=[a - b for a, b in zip(board, best_state)]
    for j in range(len(diff)):
        if diff[j]!=0: 
            return j%7
        

def my_agent(observation, configuration):
    return find_best_move(observation.board,configuration)

def print_board(board):
    print(board[:6])
    print(board[7:13])
    print(board[14:20])
    print(board[21:27])
    print(board[28:34])
    print(board[35:41])

def minimax(board,configuration,player,alpha=float('-inf'), beta=float('inf'), depth=0):
    if depth==7:
        return get_value_better(board)
    if player == "max":
      # get max of its kids(when min runs)
      next_states=find_available_boards(board,[c for c in range(configuration.columns)],agent_num)
      best=float('-inf')
      for pos in next_states:
        best=max(minimax(pos,configuration,"min",max(best,alpha), beta,depth+1),best)
        # if this child is greater than beta, give up & send back
        if best>beta:
          return best
      return best
    if player == "min":
      # get the min of its kids
      next_states=find_available_boards(board,[c for c in range(configuration.columns)],opponent_num)
      best=float('inf')
      for pos in next_states:
        best=min(minimax(pos,configuration,"max",alpha,min(best,beta),depth+1),best)
        # if this child is less than alpha, give up & send this back
        if best<alpha:
          return best
      return best

def find_available_boards(board_state,configuration_columns, player):

    states = []
    available_columns=[]
    for c in configuration_columns:
        if board_state[c] == 0:
            available_columns.append(c)

    for c in available_columns:
        
        while board_state[c] == 0:
            c = c + 7
            if c >= 41:
                break
        
        c = c - 7
        if board_state[c]==0:
            board_copy = board_state.copy()
            board_copy[c] = player
            states.append(board_copy)
    return states

#interates through each row and column one and calcualtes score, a connect 4 triggers an instant (+/-) 1000 returned
def get_value_better(board):
    rows = [board[i*7:(i+1)*7] for i in range(6)]
    columns = [[board[p + 7*r] for r in range(6)] for p in range(7)]

    count_1 = 0
    count_2 = 0

    score = 0

    scoring = {
        0: 0,
        1: 0,
        2: 10,
        3: 40,
        4: 1000
    }
   
    candidates = rows + columns

    for candidate in candidates:
        if sum(candidate) == 0:
            next
        for r in candidate:
            if r == 1:
                count_1 += 1

                score -= scoring[count_2]
                count_2 = 0
            elif r == 2:
                count_2 += 1

                score += scoring[count_1]
                count_1 = 0
            else:
                score -= scoring[count_2]
                score += scoring[count_1]
                count_1 = 0
                count_2 = 0

            if count_1 == 4:
                return 1000
            if count_2 == 4:
                return -1000
    
    return score

def human_agent(observation, configuration):
    print_board(observation.board)
    return int(input("Enter move (0-6): "))


if __name__ == "__main__":
    env = make("connectx", debug=True)
    #env.render()

    env.reset()
    # Play as the first agent against default "random"/"negamax" agent.
    #env.run([human_agent, my_agent])

    #env.render(mode="ipython", width=500, height=450)

    agents = [human_agent,my_agent]

    while not env.done:
        current_player = env.state[env.state.index(next(p for p in env.state if p.status == "ACTIVE"))].index
        observation = env.state[current_player].observation
        action = agents[current_player](observation, env.configuration)
        env.step(action)


#### Heuristic Graveyard

def get_value_simple(board):
    vertical_points = list(range(21))
    horizontal_points = [n + 7*i for i in range(1,6) for n in list(range(4))]
    score = 0

    #vertical
    for p in vertical_points:
        points = [p +7*i for i in range(4)]
        candidate = [board[i] for i in points]
        if candidate == [1,1,1,1]:
            return 1000
        if candidate == [2,2,2,2]:
            return -1000
        score += count_consecutive_score(p)
        
    #horizontal
    for p in horizontal_points:
        candidate = board[p:p+4]
        if candidate == [1,1,1,1]:
            return 1000
        if candidate == [2,2,2,2]:
            return -1000
        score += count_consecutive_score(p)
        
    return score

def count_consecutive_score(row):
    max_count_1 = count_1 = 0
    max_count_2 = count_2 = 0

    scoring = {
        0: 0,
        1: 0,
        2: 10,
        3: 40,
    }
   
    for r in row:
        if r == 1:
            count_1 += 1
            count_2 = 0
            max_count_1 = max(max_count_1,count_1)
        elif r == 2:
            count_2 += 1
            count_1 = 0
            max_count_2 = max(max_count_2,count_2)
        else:
            count_1 = 0
            count_2 = 0

    return scoring[max_count_1], -scoring[max_count_2]
