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

def get_value_simple(board):
    vertical_points = list(range(21))
    horizontal_points = [n + 7*i for i in range(1,6) for n in list(range(4))]

    #vertical
    for p in vertical_points:
        points = [p +7*i for i in range(4)]
        candidate = [board[i] for i in points]
        if candidate[0] == 0:
            continue
        sum_candidate = sum(candidate)
        if sum_candidate == 4:
            return 1000
        if sum_candidate == 8:
            return -1000
        
    #horizontal
    for p in horizontal_points:
        candidate = board[p:p+4]
        if candidate[0] == 0:
            continue
        sum_candidate = sum(candidate)
        if sum_candidate == 4:
            return 1000
        if sum_candidate == 8:
            return -1000
        
    return 0

            

if __name__ == "__main__":
    env = make("connectx", debug=True)
    env.render()

    env.reset()
    # Play as the first agent against default "random" agent.
    env.run([my_agent, "random"])
    env.render(mode="ipython", width=500, height=450)


# %%
