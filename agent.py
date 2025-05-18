# %%
# Helper functions to convert between the board and a column ,row tuple


# Helper function to better print the board
# NOTE : Please read the following
"""
The board is setup in a 1D array where it goes from top left to bottom right
The board is setup in a 1D array where it goes from top left to bottom right
Row 0 is the top row and Row n-1 is the bottom row where n is the number of rows
Col 1 is the left column and Col m is the right column where m is the number of columns
To get an item at row n and column m, you can use the formula: (Column * n) -1 + m
Ex : To get the lower left corner, you where n = 5 and m = 1
(5 * 7) - 1 + 1 = 35
To get the upper left corner, you where n = 0 and m = 1
Ex : To get the upper left corner, you where n = 0 and m = 1
(0 * 7) - 1 + 1 = 0
To get the upper right corner, you where n = 0 and m = 6
Ex : To get the upper right corner, you where n = 0 and m = 6
(0 * 7) - 1 + 6 = 6
To get the lower right corner, you where n = 5 and m = 6
Ex : To get the lower right corner, you where n = 5 and m = 6
(5 * 7) - 1 + 6 = 41
"""


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
from multiprocessing import Queue, Manager, Process

# Key : Board State
# Value : (CalculatedValue, Alpha, Beta, Depth)
# NOTE : Depth is global depth not the depth of the current search


# for now: agent is always 1, always max
agent_num = 1
opponent_num = 2
agent_minimax = "max"
global cache
cache = {}


def find_best_move(board, configuration):

    global cache

    # Finds all possible next states (moves) in the game
    next_states = find_available_boards(board, configuration.columns, agent_num)

    # Calculate the minimax value for each next state
    values = [
        minimax(board, configuration, "min", cache=cache) for board in next_states
    ]

    # Select the best state based on the minimax value

    best_state, best_val = (None, float("-inf"))

    print("Best State: ", next_states)
    print("Best Value: ", values)

    # For each state, if the value is greater than the best value, update the best state and value
    for i in range(len(values)):
        if values[i] > best_val:
            best_val = values[i]
            best_state = next_states[i]

    # Return the column of the best state

    diff = [a - b for a, b in zip(board, best_state)]
    for j in range(len(diff)):
        if diff[j] != 0:
            return j % 7


def my_agent(observation, configuration):
    print(
        "Board State: ",
        printBoard(observation.board, configuration.rows, configuration.columns),
    )
    global total_depth
    total_depth += 2
    return find_best_move(observation.board, configuration)


def minimax(
    board,
    configuration,
    player,
    alpha=float("-inf"),
    beta=float("inf"),
    depth=0,
    cache=None,
):
    global total_depth
    if depth == 8:

        if tuple(board) in cache and cache[tuple(board)][3] >= (total_depth + depth):
            return cache[tuple(board)][0]
        else:
            score = get_value_window(board)
            cache[tuple(board)] = (score, alpha, beta, total_depth + depth, board)
            return score
    if player == "max":
        if tuple(board) in cache and cache[tuple(board)][3] >= (total_depth + depth):
            return cache[tuple(board)][0]
        # get max of its kids(when min runs)
        next_states = find_available_boards(board, configuration.columns, agent_num)
        if len(next_states) == 0:
            score = get_value_window(board)
            cache[tuple(board)] = (score, alpha, beta, total_depth + depth, board)
            return score
        best = float("-inf")
        for pos in next_states:
            best = max(
                minimax(
                    pos, configuration, "min", max(best, alpha), beta, depth + 1, cache
                ),
                best,
            )
            # if this child is greater than beta, give up & send back

            if best > beta:
                return best
        cache[tuple(board)] = (best, alpha, beta, total_depth + depth, board)
        return best
    if player == "min":

        if tuple(board) in cache and cache[tuple(board)][3] >= (total_depth + depth):
            return cache[tuple(board)][0]
        # get the min of its kids
        next_states = find_available_boards(board, configuration.columns, opponent_num)
        if len(next_states) == 0:
            score = get_value_window(board)
            cache[tuple(board)] = (score, alpha, beta, total_depth + depth, board)
            return score
        best = float("inf")
        for pos in next_states:
            best = min(
                minimax(
                    pos, configuration, "max", alpha, min(best, beta), depth + 1, cache
                ),
                best,
            )
            # if this child is less than alpha, give up & send this back
            if best < alpha:
                return best
        cache[tuple(board)] = (best, alpha, beta, total_depth + depth, board)
        return best


def find_available_boards(board_state, configuration_columns, player):
    states = []
    available_columns = []

    for c in range(configuration_columns):

        # Checks if the column is available IE top row is empty in that column
        if board_state[c] == 0:
            available_columns.append(c)

    for c in available_columns:

        # OPTIMIZE: Avoid while loops Slow in Python ...
        for i in range(41 - c, -1, -7):
            if board_state[i] == 0:
                board_copy = board_state.copy()
                board_copy[i] = player
                states.append(board_copy)
                break
    return states


# iterates through each row and column one and calculates score, a connect 4 triggers an instant (+/-) 1000 returned
def get_value_window(board):

    max_count_1 = 0
    max_count_2 = 0
    # rows
    scoring = {
        0: 0,
        1: 0,
        2: 10,
        3: 70,
    }
    r = 0
    while r < len(board):
        if r % 7 == 4:
            r += 3
        if r >= len(board):
            break
        window = 0
        count1 = 0
        count2 = 0
        one_stuck = False
        two_stuck = False
        while window < 4:
            if board[r + window] == 1:
                if not one_stuck:
                    count1 += 1
                count2 = 0
                two_stuck = True
            if board[r + window] == 2:
                count1 = 0
                one_stuck = True
                if not two_stuck:
                    count2 += 1
            window += 1
        r += 1
        if count1 > 0 and count2 > 0:
            continue
        max_count_1 = max(max_count_1, count1)
        max_count_2 = max(max_count_2, count2)
    # columns
    row = 0
    while row < 4:
        for c in range(len(board)):
            if c % 7 == 0:
                row += 1
            if row == 4:
                break
            window = 0
            count1 = 0
            count2 = 0
            one_stuck = False
            two_stuck = False
            while window < 4:
                if board[c + window * 7] == 1:
                    if not one_stuck:
                        count1 += 1
                    count2 = 0
                    two_stuck = True
                if board[c + window * 7] == 2:
                    if not two_stuck:
                        count2 += 1
                    count1 = 0
                    one_stuck = True

                window += 1
            max_count_1 = max(max_count_1, count1)
            max_count_2 = max(max_count_2, count2)

    if max_count_1 == 4:
        return 1000
    if max_count_2 == 4:
        return -1000

    return scoring[max_count_1] - scoring[max_count_2]


if __name__ == "__main__":
    env = make("connectx", debug=True)
    env.render()

    env.reset()

    # Total number of moves occurred in the game
    global total_depth
    total_depth = 0

    # Play as the first agent against default "random" agent.
    env.run([my_agent, "negamax"])
    env.render(mode="ipython", width=500, height=450)


def get_value_better(board):
    rows = [board[i * 7 : (i + 1) * 7] for i in range(6)]
    columns = [[board[p + 7 * r] for r in range(6)] for p in range(7)]

    count_1 = 0
    count_2 = 0

    score = 0

    scoring = {0: 0, 1: 0, 2: 10, 3: 40, 4: 1000}

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


#### Heuristic Graveyard
def get_value_simple(board):
    vertical_points = list(range(21))
    horizontal_points = [n + 7 * i for i in range(1, 6) for n in list(range(4))]
    score = 0

    # vertical
    for p in vertical_points:
        points = [p + 7 * i for i in range(4)]
        candidate = [board[i] for i in points]
        if candidate == [1, 1, 1, 1]:
            return 1000
        if candidate == [2, 2, 2, 2]:
            return -1000
        score += count_consecutive_score(p)

    # horizontal
    for p in horizontal_points:
        candidate = board[p : p + 4]
        if candidate == [1, 1, 1, 1]:
            return 1000
        if candidate == [2, 2, 2, 2]:
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
            max_count_1 = max(max_count_1, count_1)
        elif r == 2:
            count_2 += 1
            count_1 = 0
            max_count_2 = max(max_count_2, count_2)
        else:
            count_1 = 0
            count_2 = 0

    return scoring[max_count_1], -scoring[max_count_2]


# %%
