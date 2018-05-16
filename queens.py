import random 
import numpy as np

def initialize_board(n):
    """
    Places randomly queens on each row
    """
    up_boundry = n-1
    queens =[(row, random.randint(0, up_boundry)) for row  in range(n)]
    return queens


def get_conflicts(queens_positions):
    """
    Returns a random set of all queens in conflict
    """
    conflicted = set()
    for first_queen in range(len(queens_positions)):
        for second_queen in range(first_queen+1, len(queens_positions)):
            delta_col = queens_positions[first_queen][1] - queens_positions[second_queen][1]
            if delta_col == 0:
                conflicted.add(first_queen)
                conflicted.add(second_queen)
            elif abs((queens_positions[first_queen][0] - queens_positions[second_queen][0])/delta_col) == 1:
                conflicted.add(first_queen)
                conflicted.add(second_queen)
    return conflicted
   

def rearange_board(queens_positions, queen):
    """
    chose a place to put the choosen queen and place it on the board
    """
    queen_row = queen[0]
    minimum_conflicts = []
    minimum_conflict = float('inf')
    for test_col in range(len(queens_positions)+1):
        num_conflicts = 0
        for second_queen in range(len(queens_positions)):
            delta_col = test_col - queens_positions[second_queen][1]
            if delta_col == 0:
                num_conflicts += 1
            elif abs((queen_row-queens_positions[second_queen][0])/delta_col) == 1:
                num_conflicts += 1
        if num_conflicts < minimum_conflict:
            minimum_conflicts = [test_col]
            minimum_conflict = num_conflicts
        elif num_conflicts == minimum_conflict:
            minimum_conflicts.append(test_col)
    new_col = random.sample(minimum_conflicts, 1)[0]
    queens_positions.append((queen_row, new_col))
    return queens_positions


def board_print(queens):
    """
    Create the board and print it
    """
    charar = np.empty((len(queens), len(queens)), dtype = str)
    charar[:] = '_'
    for queen in queens:
        charar[queen[0],queen[1]] = '*'
    print(charar)
    
    
def main():
    #n is number of queens
    #iterations is maximum number of iterations
    n = int(input("Enter  number of queens: "))
    iterations = 1000
    queens_positions = initialize_board(n)
    for dummy in range(iterations):
        conflicts = get_conflicts(queens_positions)

        if conflicts == set():
            board_print(queens_positions)
            print('Solution found')
            return
        
        queen_idx = random.sample(conflicts, 1)[0]
        queen = queens_positions.pop(queen_idx)
        queens_positions = rearange_board(queens_positions, queen)

    print("can't find solition")

if __name__ == '__main__':
    main()