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

    curr_1, curr_2 = sector_check(board, sec_1, iter_1)

    if curr_1 == 4:
        return 1
    elif curr_2 == 4:
        return 2

    curr_1, curr_2 = sector_check(board, sec_2, iter_2)

    if curr_1 == 4:
        return 1
    elif curr_2 == 4:
        return 2

    curr_1, curr_2 = sector_check(board, sec_3, iter_3)

    if curr_1 == 4:
        return 1
    elif curr_2 == 4:
        return 2

    curr_1, curr_2 = sector_check(board, sec_4, iter_4)

    if curr_1 == 4:
        return 1
    elif curr_2 == 4:
        return 2
        

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

    curr_1, curr_2 = sector_check(board, sec_2, iter_2)
    max_count_1 = max(max_count_1, curr_1)
    max_count_2 = max(max_count_2, curr_2)

    curr_1, curr_2 = sector_check(board, sec_3, iter_3)
    max_count_1 = max(max_count_1, curr_1)
    max_count_2 = max(max_count_2, curr_2)

    curr_1, curr_2 = sector_check(board, sec_3, iter_3)
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

    return max_count_1, max_count_2

        
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

        move_2 = int(input("Enter move (1-6): "))
        index_2 = find_available_index(board=game_state.board, column=(move_2-1))
        game_state.board[index_2] = 2
        print_board(game_state.board)

        if check_win(game_state.board) == 2:
            print("Player Two Wins!")
            break
    

    
if __name__ == "__main__":
    game_simulation_human(my_agent)