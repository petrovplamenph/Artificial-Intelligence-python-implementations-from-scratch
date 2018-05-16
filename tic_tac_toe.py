"""
Tic-Tac-Toe
"""

EMPTY = 1
PLAYERX = 2
PLAYERO = 3 
DRAW = 4

MAPING = {EMPTY: " ",
          PLAYERX: "X",
          PLAYERO: "O"}
SCORES = {PLAYERX: 1,
          DRAW: 0,
          PLAYERO: -1}

class Board:
    """
    Class Tic-Tac-Toe board.
    """

    def __init__(self, dim, board = None):
        """
        Initialize the Tic-Tac-Toe board
        """
 
        self._dim = dim
        if board == None:
            # Create empty board
            self._board = [[EMPTY for _ in range(dim)] 
                           for _ in range(dim)]
        else:
            # Copy board grid
            self._board = [[board[row][col] for col in range(dim)] 
                           for row in range(dim)]
            
    def __str__(self):
        """
        Human readable representation of the board.
        """
        str_board = ""
        for row in range(self._dim):
            for col in range(self._dim):
                str_board += MAPING[self._board[row][col]]
                if col == self._dim - 1:
                    str_board += "\n"
                else:
                    str_board += " | "
            if row != self._dim - 1:
                str_board += "-" * (4 * self._dim - 3)
                str_board += "\n"
        return str_board

    
    def square(self, row, col):
        """
        Get the condition of the square at position row,col
         """
        return self._board[row][col]

    def get_empty_squares(self):
        """
        Return a list of (row, col) tuples for all empty squares
        """
        empty = []
        for row in range(self._dim):
            for col in range(self._dim):
                if self._board[row][col] == EMPTY:
                    empty.append((row, col))
        return empty

    def move(self, row, col, player):
        """
        Make a move if the move is valid
        """
        if self._board[row][col] == EMPTY:
            self._board[row][col] = player

    def check_win(self):
        """
        Returns a constant associated with the state of the game
            If PLAYERX wins, returns PLAYERX.
            If PLAYERO wins, returns PLAYERO.
            If game is drawn, returns DRAW.
            If game is in progress, returns None.
        """
        board = self._board
        dim = self._dim
        dim_rng = range(dim)
        lines = []

        lines.extend(board)

        cols = [[board[row_idx][col_idx] for row_idx in dim_rng]
                for col_idx in dim_rng]
        lines.extend(cols)

        diag1 = [board[idx][idx] for idx in dim_rng]
        diag2 = [board[idx][dim - idx -1] 
                 for idx in dim_rng]
        lines.append(diag1)
        lines.append(diag2)

        for line in lines:
            if len(set(line)) == 1 and line[0] != EMPTY:
                return line[0]

        if len(self.get_empty_squares()) == 0:
            return DRAW

        return None
            
    def copy(self):
        """
        Return board copy.
        """
        return Board(self._dim, self._board)

def switch_player(player):
    """
    Convenience function to switch players.
    
    Returns other player.
    """
    if player == PLAYERX:
        return PLAYERO
    else:
        return PLAYERX




def mp_move(board, player):
    """
    Make a move on the board.
    
    Returns a tuple with two elements.  The first element is the score
    of the given board and the second element is the desired move as a
    tuple, (row, col).
    """
    scr = []
    moves = []
    for element in board.get_empty_squares():
        clone = board.copy()
        row, col = element
        clone.move(row, col, player)
        winner = clone.check_win()
        if player == winner:
            return SCORES[winner] ,(row, col)
        if winner == None:
            curplayer = switch_player(player)
            score = mp_move(clone, curplayer)
            score = score[0]
        else:
            score = SCORES[winner]
        scr.append(score)
        moves.append((row, col))
    if player == PLAYERX:
        idx = scr.index(max(scr))
    else:
        idx = scr.index(min(scr)) 
    return scr[idx], moves[idx]


def human_move(Flag = False):
    if Flag:
        print('invalid move')
    row = input("Enter row:")
    row = int(row)
    col = input("Enter column:")
    col = int(col)
    return (row, col)

def main():
    """
    Function to play a game with two MC players.
    """
    board = Board(3)
    curplayer = PLAYERX
    winner = None
    humanplayer = PLAYERX
    print(board)

    while winner == None:
        if humanplayer == curplayer:
            empty_sqr = board.get_empty_squares()
            row,col = human_move()
            while (row,col) not in empty_sqr:
                row, col = human_move(True)
            
        else:
            row, col = mp_move(board, curplayer)[1]

        board.move(row, col, curplayer)
        print(board)

        winner = board.check_win()
        curplayer = switch_player(curplayer)



    if winner == PLAYERX:
        print ("X wins")
    elif winner == PLAYERO:
        print ("O wins")
    elif winner == DRAW:
        print ("Draw")
    else:
        print ("Error")


if __name__ == '__main__':
    main()
