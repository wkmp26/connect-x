from agent import my_agent

class Configuration():
    def __init__(self,columns:int):
        self.columns = columns

class Observation():
    def __init__(self, board: list[int]):
        self.board = board

def find_available_index(board, column):

    c = column 
    
    if board[c] != 0:
        raise Exception("Invalid Move Played")

    while board[c] == 0:
        c = c + 7
        if c >= 41:
            break   
    c = c - 7

    return c

def check_win(board):
    vertical_points = list(range(21))
    horizontal_points = [n + 7*i for i in range(1,6) for n in list(range(4))]

    #vertical
    for p in vertical_points:
        points = [p +7*i for i in range(4)]
        candidate = [board[i] for i in points]
        if candidate == [1,1,1,1]:
            return 1
        if candidate == [2,2,2,2]:
            return 2
        
    #horizontal
    for p in horizontal_points:
        candidate = board[p:p+4]
        if candidate == [1,1,1,1]:
            return 1
        if candidate == [2,2,2,2]:
            return 2
        
def print_board(board):
    print("------------------------------------")
    print(board[:6])
    print(board[7:13])
    print(board[14:20])
    print(board[21:27])
    print(board[28:34])
    print(board[35:41])
    print("------------------------------------")


def game_simulation_human(agent):
    config = Configuration(7)

    game_state = Observation([0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0])
    
    while True:
        print("Agent's Turn")
        move_1 = agent(observation=game_state,configuration=config)
        index_1 = find_available_index(board=game_state.board,column=move_1)
        game_state.board[index_1] = 1
        print_board(game_state.board)

        if check_win(game_state.board) == 1:
            print("Player One Wins!")
            break

        move_2 = int(input("Enter move (0-6): "))
        index_2 = find_available_index(board=game_state.board, column=move_2)
        game_state.board[index_2] = 2
        print_board(game_state.board)

        if check_win(game_state.board) == 2:
            print("Player Two Wins!")
            break
    

    
if __name__ == "__main__":
    game_simulation_human(my_agent)