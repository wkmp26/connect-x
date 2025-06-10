
# from agent import get_value_simple

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
                # max_count_1 = max(max_count_1,count_1)
            elif r == 2:
                count_2 += 1

                score += scoring[count_1]
                count_1 = 0
                # max_count_2 = max(max_count_2,count_2)
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

def get_value_simple(board):
    vertical_points = list(range(21))
    horizontal_points = [n + 7*i for i in range(1,6) for n in list(range(4))]
    score = 0

    #vertical
    for p in vertical_points:
        points = [p +7*i for i in range(4)]
        candidate = [board[i] for i in points]
        print("Candidate: ", candidate)
        # if candidate[0] == 0:
            # continue
        if candidate == [1,1,1,1]:
            return 1000
        if candidate == [2,2,2,2]:
            return -1000
        
        print(count_consecutive_score(candidate))
        score += count_consecutive_score(candidate)
        
        
    #horizontal
    for p in horizontal_points:
        candidate = board[p:p+4]
        print("Candidate: ", candidate)
        # if candidate[0] == 0:
            # continue
        if candidate == [1,1,1,1]:
            return 1000
        if candidate == [2,2,2,2]:
            return -1000
        
        print(count_consecutive_score(candidate))
        score += count_consecutive_score(candidate)
        
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
    return scoring[max_count_1]- scoring[max_count_2]
    return scoring[max_count_1], -scoring[max_count_2]

    if max_count_1 > max_count_2:
        return max_count_1
    else:
        return -max_count_2

def find_available_boards(board_state,configuration_columns, player):

    states = []
    available_columns=[]
    for c in configuration_columns:
        if board_state[c] == 0:
            available_columns.append(c)

    for c in available_columns:
        
        while board_state[c] == 0:
            c = c + 7
            if c > 41:
                break
        
        c = c - 7
        if c == 41:
            print("IM AT 41")

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

    curr_1, curr_2 = sector_check(board, sec_2, iter_2)
    max_count_1 = max(max_count_1, curr_1)
    max_count_2 = max(max_count_2, curr_2)

    curr_1, curr_2 = sector_check(board, sec_3, iter_3)
    max_count_1 = max(max_count_1, curr_1)
    max_count_2 = max(max_count_2, curr_2)

    curr_1, curr_2 = sector_check(board, sec_4, iter_4)
    max_count_1 = max(max_count_1, curr_1)
    max_count_2 = max(max_count_2, curr_2)

    return max_count_1, max_count_2

def sector_check(board, sector, iter):
    max_count_1 = 0
    max_count_2 = 0

    for s in sector:
        print(f"Current Sector Point {s}")
        window = 0
        count1 = 0
        count2 = 0
        one_stuck = False
        two_stuck = False
        while window < 4:
            print(f"Checking cell {board[s + window * iter]}")
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


import time

def test_board_moves_1():
    board1 = [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0
                ]
        
    boards = [
            [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                1,0,0,0,0,0,0
                ],
                [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,1,0,0,0,0,0
                ],
                [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,1,0,0,0,0
                ],
                [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,1,0,0,0
                ],
                [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,1,0,0
                ],
                [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,1,0
                ],
                [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,1
                ]
    ]

    assert find_available_boards(board1,[0,1,2,3,4,5,6],1) == boards

def test_vertical_1():
    board1 = [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                1,1,1,1,0,0,0
                ]
        
    start = time.time()
    assert get_value_better(board1) == 1000
    print("Took: ", time.time() - start)

def test_vertical_2():
    board2 = [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,1,0,0,0,0,
                0,0,1,0,0,0,0,
                0,0,1,0,0,0,0,
                0,0,1,0,0,0,0
                ]
        
    assert get_value_better(board2) == 1000

def test_horizontal_1():
    board3 = [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,2,2,2,2,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0
                ]
            
    assert get_value_better(board3) == -1000

def test_score_1():
    board3 = [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                1,1,1,0,0,0,0
                ]
            
    assert get_value_better(board3) == 40

def test_score_2():
    print("Starting scoring 2")
    board3 = [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,2,0,
                1,2,2,0,0,2,0,
                1,1,1,0,0,2,0
                ]
            
    assert get_value_better(board3) == 0

def test_score_4():
    print("Starting scoring 2")
    board3 = [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,2,0,
                0,0,0,0,0,2,0,
                1,2,2,0,0,2,0,
                1,1,1,0,0,2,0
                ]
            
    assert get_value_better(board3) == -1000

def test_score_5():
    print("Starting scoring 2")
    board3 = [
                1,1,0,0,0,0,0,
                1,2,0,0,0,0,0,
                2,2,1,0,0,0,2,
                1,1,1,0,0,0,2,
                2,1,1,0,0,0,2,
                1,1,2,0,2,0,2
                ]
            
    assert get_value_better(board3) == -1000

def test_diagonal_1():
    board1 = [
                0,0,0,2,0,0,0,
                0,0,0,0,2,0,0,
                0,0,0,1,0,2,0,
                0,0,1,0,0,0,2,
                0,1,0,0,0,0,0,
                1,0,0,0,0,0,0,
                ]
            
    assert diagonals_windows(board1)[0] == 4
    assert diagonals_windows(board1)[1] == 4

def test_diagonal_2():
    board2 = [
                0,2,0,0,0,0,0,
                0,0,2,0,0,0,0,
                0,0,0,2,0,2,0,
                0,0,1,0,0,1,2,
                0,1,0,1,1,0,0,
                0,0,0,1,0,0,0,
                ]
            
    assert diagonals_windows(board2)[0] == 3
    assert diagonals_windows(board2)[1] == 3