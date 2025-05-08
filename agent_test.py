
# from agent import get_value_simple


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

import time


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
    assert get_value_simple(board1) == 1000
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
        
    assert get_value_simple(board2) == 1000

def test_horizontal_1():
    board3 = [
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,2,2,2,2,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0
                ]
            
    assert get_value_simple(board3) == -1000
