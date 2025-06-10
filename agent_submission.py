import time as python_time

print("Top-level script executed", flush=True)

# Key : Board State
# Value : (CalculatedValue, Alpha, Beta, Depth)
# NOTE : Depth is global depth not the depth of the current search


# for now: agent is always 1, always max


def init_terminal():

    X = 4
    COLUMNS = 7
    ROWS = 6

    global terminal_cache
    terminal_cache = []

    # Diagonal from top left to bottom right
    for i in range(ROWS - X + 1):
        for j in range(COLUMNS - X + 1):
            startIndex = i * COLUMNS + j
            pieces = [
                k
                for k in range(
                    startIndex, startIndex + (X - 1) * (COLUMNS + 1), COLUMNS + 1
                )
            ]
            terminal_cache.append(pieces)
    # Diagonal from bottom left to top right
    for i in range(X - 1, ROWS):
        for j in range(COLUMNS - X + 1):
            startIndex = i * COLUMNS + j
            pieces = [
                k
                for k in range(
                    startIndex, startIndex - (X - 1) * (COLUMNS - 1), -(COLUMNS - 1)
                )
            ]
            terminal_cache.append(pieces)

    # Horizontals
    for i in range(ROWS):
        for j in range(COLUMNS - X + 1):
            startIndex = i * COLUMNS + j
            pieces = [k for k in range(startIndex, startIndex + X)]
            terminal_cache.append(pieces)

    # Verticals
    for i in range(COLUMNS):
        for j in range(ROWS - X + 1):
            startIndex = j * COLUMNS + i
            pieces = [k for k in range(startIndex, startIndex + (X * COLUMNS), COLUMNS)]
            terminal_cache.append(pieces)


def find_best_move(board, configuration, total_depth):

    global cache

    # Finds all possible next states (moves) in the game
    next_states = find_available_boards(board, configuration.columns, agent_num)

    # Calculate the minimax value for each next state
    values = [
        minimax(board, configuration, "min", total_depth=total_depth, cache=cache)
        for board in next_states
    ]

    # Select the best state based on the minimax value
    best_state, best_val = (None, float("-inf") if agent_minimax == 'max' else float("inf"))

    # For each state, if the value is greater than the best value, update the best state and value
    for i in range(len(values)):
        if agent_minimax == 'max':
            if values[i] > best_val:
                best_val = values[i]
                best_state = next_states[i]
        else:
            if values[i] < best_val:
                best_val = values[i]
                best_state = next_states[i] 

    # Return the column of the best state

    diff = [a - b for a, b in zip(board, best_state)]
    for j in range(len(diff)):
        if diff[j] != 0:
            return j % 7


def determine_max_depth(children, numberOfStepsIn):
    # Determine the max depth based on the number of children and the number of steps in
    if numberOfStepsIn < 10 or children >= 6:
        return 5
    elif numberOfStepsIn < 20 and children <= 4:
        return 9
    elif numberOfStepsIn < 30 and children <= 2:
        return 10
    elif numberOfStepsIn < 30:
        return 7
    else:
        return 64


def minimax(
    board,
    configuration,
    player,
    alpha=float("-inf"),
    beta=float("inf"),
    total_depth=0,
    depth=0,
    cache=None,
    maxDepth=5,
):

    global start_time
    # global logger

    # logger.info(
    #     f"Minimax called with player: {player}, depth: {depth}, total_depth: {total_depth}, alpha: {alpha}, beta: {beta}, time used : {(python_time.time_ns() - start_time)/1_000_000} ms"
    # )

    print(
        f"Minimax called with player: {player}, depth: {depth}, total_depth: {total_depth}, alpha: {alpha}, beta: {beta}, time used : {(python_time.time_ns() - start_time)/1_000_000} ms",
        flush=True,
    )

    board_hash = (tuple(board), player)

    if (
        board_hash in cache
        and cache[board_hash][1] >= (total_depth + depth)
        and (
            cache[board_hash][2] == "EXACT"
            or (cache[board_hash][2] == "LOWER" and beta <= cache[board_hash][0])
            or (cache[board_hash][2] == "UPPER" and alpha >= cache[board_hash][0])
        )
    ):
        return cache[board_hash][0]

    # Check if the game is over
    score = isTerminal(board)
    if score != 0:
        if player == "max":
            if score == 1000:
                score = score - (total_depth + depth)
            else:
                score = score + (total_depth + depth)
        else:
            if score == -1000:
                score = score + (total_depth + depth)
            else:
                score = score - (total_depth + depth)
        cache[board_hash] = (score, total_depth + depth, "EXACT")
        return score

    if depth >= maxDepth:
        score = get_value_window(board)
        cache[board_hash] = (score, total_depth + depth, "EXACT")
        return score
    else:

        # get max of its kids(when min runs)
        if player == "max":

            next_states = find_available_boards(board, configuration.columns, agent_num)
            if len(next_states) == 0:
                score = get_value_window(board)
                cache[board_hash] = (score, total_depth + depth, "EXACT")
                return score

            # Check if min already has eliminated this branch

            best = float("-inf")
            for pos in next_states:
                best = max(
                    minimax(
                        pos,
                        configuration,
                        "min",
                        max(best, alpha),
                        beta,
                        total_depth,
                        depth + 1,
                        cache,
                        maxDepth=determine_max_depth(
                            len(next_states), total_depth + depth
                        ),
                    ),
                    best,
                )
                # if this child is greater than beta, give up & send back
                if best > beta:
                    cache[board_hash] = (best, total_depth + depth, "LOWER")
                    return best
            cache[board_hash] = (best, total_depth + depth, "EXACT")
            return best
        if player == "min":

            next_states = find_available_boards(
                board, configuration.columns, opponent_num
            )
            if len(next_states) == 0:
                score = get_value_window(board)
                cache[board_hash] = (score, total_depth + depth, "EXACT")
                return score

            best = float("inf")
            for pos in next_states:
                best = min(
                    minimax(
                        pos,
                        configuration,
                        "max",
                        alpha,
                        min(best, beta),
                        total_depth,
                        depth + 1,
                        cache,
                        maxDepth=determine_max_depth(
                            len(next_states), total_depth + depth
                        ),
                    ),
                    best,
                )
                # if this child is less than alpha, give up & send this back
                if best < alpha:
                    cache[board_hash] = (best, total_depth + depth, "UPPER")
                    return best
            cache[board_hash] = (best, total_depth + depth, "EXACT")
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
        for i in range(41 - (6 - c), -1, -7):
            if board_state[i] == 0:
                board_copy = board_state.copy()
                board_copy[i] = player
                states.append(board_copy)
                break
    return states


def isTerminal(board):
    X = 4

    global terminal_cache

    try:
        terminal_cache
    except NameError:
        init_terminal()

    for state in terminal_cache:
        pieces = [board[i] for i in state]
        count1 = pieces.count(1)
        count2 = pieces.count(2)
        if count1 == X:
            return 1000
        if count2 == X:
            return -1000
    return 0


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
    
    
    diagonals_1, diagonals_2 = diagonals_windows(board)
    max_count_1 = max(max_count_1, diagonals_1)
    max_count_2 = max(max_count_2, diagonals_2)
    
    if max_count_1 == 4:
        return 1000
    if max_count_2 == 4:
        return -1000

    return scoring[max_count_1] - scoring[max_count_2]

def find_available_boards(board_state,configuration_columns, player):

    states = []
    available_columns=[]
    for c in range(configuration_columns):
        if board_state[c] == 0:
            available_columns.append(c)

    for c in available_columns:
        
        while board_state[c] == 0:
            c = c + 7
            if c > 41:
                break
        
        c = c - 7

        """
        if c == 41:
            print("IM AT 41")
        """

        if board_state[c]==0:
            board_copy = board_state.copy()
            board_copy[c] = player
            states.append(board_copy)
    return states

def diagonals_windows(board):
    #sector 1 (down left)
    sec_1 = [0,1,2,3,7,14,8]
    iter_1 = 8
    #sector 2 (down right)
    sec_2 = [3,4,5,6,13,20,12]
    iter_2 = 6
    #sector 3 (up right)
    sec_3 = [35,36,37,38,28,21,29]
    iter_3 = -6
    #sector 4 (up left)
    sec_4 = [38,39,40,41,34,27,33]
    iter_4 = -8

    max_count_1 = 0
    max_count_2 = 0

    curr_1, curr_2 = sector_check(board, sec_1, iter_1)
    max_count_1 = max(max_count_1, curr_1)
    max_count_2 = max(max_count_2, curr_2)

    if max_count_1 == 4 or max_count_2 == 4:
        return max_count_1, max_count_2

    curr_1, curr_2 = sector_check(board, sec_2, iter_2)
    max_count_1 = max(max_count_1, curr_1)
    max_count_2 = max(max_count_2, curr_2)

    if max_count_1 == 4 or max_count_2 == 4:
        return max_count_1, max_count_2

    curr_1, curr_2 = sector_check(board, sec_3, iter_3)
    max_count_1 = max(max_count_1, curr_1)
    max_count_2 = max(max_count_2, curr_2)

    if max_count_1 == 4 or max_count_2 == 4:
        return max_count_1, max_count_2

    curr_1, curr_2 = sector_check(board, sec_4, iter_4)
    max_count_1 = max(max_count_1, curr_1)
    max_count_2 = max(max_count_2, curr_2)


    return max_count_1, max_count_2

def sector_check(board, sector, iter):
    max_count_1 = 0
    max_count_2 = 0

    for s in sector:
        window = 0
        count1 = 0
        count2 = 0
        one_stuck = False
        two_stuck = False
        while window < 4:
            if board[s + window * iter] == 1:
                if not one_stuck:
                    count1 += 1
                count2 = 0
                two_stuck = True
            if board[s + window * iter] == 2:
                if not two_stuck:
                    count2 += 1
                count1 = 0
                one_stuck = True

            window += 1

        max_count_1 = max(max_count_1, count1)
        max_count_2 = max(max_count_2, count2) 

        if max_count_1 == 4 or max_count_2 == 4:
            return max_count_1, max_count_2

    return max_count_1, max_count_2


def main():
    pass
    """
    env = make("connectx", debug=True)
    env.render()

    env.reset()

    # Play as the first agent against default "random" agent.
    env.run([my_agent, "negamax"])

    # Print who wins
    # env.render(mode="ipython", width=500, height=450)
    """


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

def my_agent(observation, configuration):

    # Checks if the necessary variables are initialized
    global cache
    global time_spent
    global start_time
    global logger
    global agent_num
    global opponent_num
    global agent_minimax
    if observation.step == 0 or observation.step == 1:
        if observation.step == 0:
            agent_num = 1
            opponent_num = 2
            agent_minimax = "max"
        else:
            agent_num = 2
            opponent_num = 1
            agent_minimax = "min"
        time_spent = 0
        cache = {}
    else:
        pass
    start_time = python_time.time_ns()
    move = find_best_move(observation.board, configuration, observation.step)
    time_spent += python_time.time_ns() - start_time
    print(f"Move: {move}, Time Spent: {time_spent/(10**9)} seconds")
    return move
