# %%
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


# %%
from kaggle_environments import evaluate, make


def my_agent(observation, configuration):

    print(observation.board)

    print_board(observation.board) 

    states = find_available_boards(observation.board, list(range(configuration.columns)), 1)
    for s in states:
        print("---------------------")
        print_board(s)

    return 1
    from random import choice

    SELECTED_COLUMN = 1

    ##return 0
    for n in range(configuration.columns):
        print(
            f"""
            *********************************************************
              Number of columns: {n}
              Number of rows: {configuration.rows}
              Total number of cells: {len(observation.board)}
              Number of Filled cells: {len([c for c in observation.board if c != 0])}
              -------------------------
              Board: \n{printBoard(observation.board, configuration.rows, configuration.columns)}\n
            *********************************************************
              """
        )

        return SELECTED_COLUMN

    # Not exactly what this for if 1 is always returned?
    return choice(
        [c for c in range(configuration.columns) if observation.board[c] == 0]
    )

def print_board(board):
    print(board[:6])
    print(board[7:13])
    print(board[14:20])
    print(board[21:27])
    print(board[28:34])
    print(board[35:41])

def find_available_boards(board_state,configuration_columns, player):

    states = []
    #remove full columns
    for c in configuration_columns:
        if board_state[c] != 0:
            configuration_columns.remove(c)

    for c in configuration_columns:
        board_copy = board_state.copy()
        while board_state[c] == 0:
            c = c + 7
            if c >= 41:
                break
        c = c - 7
        board_copy[c] = player
        # print("Adding the following state:")
        # print_board(board_copy)
        states.append(board_copy)

    return states

def get_value():
    pass

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
            print("Considering Row: ", r)
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
                print(1000)
                return 1000
            if count_2 == 4:
                print(-1000)
                return -1000
    
    print(score)
    return score


if __name__ == "__main__":
    env = make("connectx", debug=True)
    env.render()

    env.reset()
    # Play as the first agent against default "random" agent.
    env.run([my_agent, "random"])
    env.render(mode="ipython", width=500, height=450)




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

    if max_count_1 > max_count_2:
        return max_count_1
    else:
        return -max_count_2

# %%
