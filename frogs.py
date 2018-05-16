"""
Frog's game recursive implementation
Input: number of frogs faceing in one direction.
"""
import re
def inital_moves(number_frogs):
    """
    Inital posible moves
    """
    if number_frogs == 1:
        return [0, 2]
    return [number_frogs-2, number_frogs-1, number_frogs+1, number_frogs+2]


def get_moves(empty_pad, config, num_pads):
    """
    Find all possible moves for the current board
    """
    minimal = max(0, empty_pad-2)
    maximun = min(num_pads-1, empty_pad+2)
    moves = []
    for idx in range(minimal,empty_pad):
        if config[idx] == '>':
            moves.append(idx)
    for idx in range(empty_pad+1, maximun+1):
        if config[idx] == '<':
            moves.append(idx)    
    return moves


def get_curr_config(curr_move, last_cofig, empty_pos):
    """
    Swap frog and empty place after a move
    """
    last_cofig[curr_move], last_cofig[empty_pos]=last_cofig[empty_pos], last_cofig[curr_move]
    return last_cofig


def recursive_frogs(moves, empty_idx, game_state, num_pads, new_moves, orderd_config):
    """
    All possible moves are played recursively  untill the game is won
    There is no need to keep track of the the whole board, only the moves
    of the empty square is needed.
    """
    for new_move in new_moves:
        new_config = get_curr_config(new_move, list(game_state), empty_idx)
        if new_config == orderd_config:
            return moves+[new_move]
        else:
            next_moves = get_moves(new_move, new_config, num_pads)
            if next_moves != []:
                result = recursive_frogs(moves+[new_move], new_move, new_config, num_pads, next_moves, orderd_config)
                if result != None:
                    return result
def backtrack(moves, lilipads):
    """
    Given a set of moves and the inital condition of the board, reconstruct
    the whole board on each move.
    """
    boards = [lilipads]
    for idx in range(1, len(moves)):
        board = get_curr_config(moves[-idx-1], list(boards[-1]), moves[-idx])
        boards.append(board)
    return boards
                
def main():
    """
    Game frogs
    """
    number_frogs  =  input("Enter the number of frogs faceing in one direction: ")
    number_frogs = int(re.findall('[0-9]+',number_frogs)[0])
    first_moves = inital_moves(number_frogs)
    empty_idx = number_frogs
    lilipads = ['>']*number_frogs + ['_'] + ['<']*number_frogs
    num_pads = 2 * number_frogs + 1
    orderd_config = ['<']*number_frogs + ['_'] + ['>']*number_frogs
    moves = recursive_frogs([empty_idx], empty_idx, lilipads, num_pads, first_moves, orderd_config)
    boards = backtrack(moves, lilipads)
    for board in boards:
        print(board)
    return boards
    


if __name__ == '__main__':
    main()
