from rl_agent import my_agent

class Configuration:
    def __init__(self, columns: int = 7, rows: int = 6):
        self.columns = columns
        self.rows = rows

class Observation:
    def __init__(self, board: list[int], mark: int):
        self.board = board
        self.mark = mark  # Who is playing: 1 = agent, 2 = opponent

def find_available_index(board, column):
    for r in reversed(range(6)):  # from bottom to top
        idx = r * 7 + column
        if board[idx] == 0:
            return idx
    raise ValueError(f"Column {column} is full.")

def check_win(board):
    # Vertical check
    for c in range(7):
        for r in range(3):
            idxs = [r * 7 + c, (r + 1) * 7 + c, (r + 2) * 7 + c, (r + 3) * 7 + c]
            line = [board[i] for i in idxs]
            if line == [1] * 4:
                return 1
            if line == [2] * 4:
                return 2

    # Horizontal check
    for r in range(6):
        for c in range(4):
            idxs = [r * 7 + c + i for i in range(4)]
            line = [board[i] for i in idxs]
            if line == [1] * 4:
                return 1
            if line == [2] * 4:
                return 2

    # Diagonal \ check
    for r in range(3):
        for c in range(4):
            idxs = [r * 7 + c + i * 8 for i in range(4)]
            line = [board[i] for i in idxs]
            if line == [1] * 4:
                return 1
            if line == [2] * 4:
                return 2

    # Diagonal / check
    for r in range(3, 6):
        for c in range(4):
            idxs = [r * 7 + c + i * -6 for i in range(4)]
            line = [board[i] for i in idxs]
            if line == [1] * 4:
                return 1
            if line == [2] * 4:
                return 2

    return 0  # no winner

def print_board(board):
    print(" 0 1 2 3 4 5 6")
    print("------------------------------------")
    for r in range(6):
        print(" ".join(str(board[r * 7 + c]) for c in range(7)))
    print("------------------------------------")

def game_simulation_human(agent):
    config = Configuration()
    board = [0] * 42
    observation = Observation(board, mark=1)  # agent starts as player 1

    while True:
        # Agent's move
        print("Agent's Turn")
        move_1 = agent(observation=observation, configuration=config)
        index_1 = find_available_index(observation.board, move_1)
        observation.board[index_1] = 1
        print_board(observation.board)

        if check_win(observation.board) == 1:
            print("Player One (Agent) Wins!")
            break

        # Switch to player 2
        observation.mark = 2

        # Human's move
        while True:
            try:
                move_2 = int(input("Enter move (0–6): "))
                index_2 = find_available_index(observation.board, move_2)
                break
            except (ValueError, IndexError):
                print("Invalid move. Try another column.")

        observation.board[index_2] = 2
        print_board(observation.board)

        if check_win(observation.board) == 2:
            print("Player Two (You) Win!")
            break

        # Switch back to agent
        observation.mark = 1

if __name__ == "__main__":
    game_simulation_human(my_agent)
