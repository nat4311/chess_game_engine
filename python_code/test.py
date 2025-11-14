import game_engine
import numpy as np

board = game_engine.BoardState()
moves = game_engine.MoveGenerator()
board.print()
board.reset()
board.print()

# moves.generate_pl_moves(board)
# moves.print_pl_moves()

def print_bool_bitboard(bitboard):
    print("    A  B  C  D  E  F  G  H\n")
    for y in range(8):
        print(8-y, end="   ")
        for x in range(8):
            sq = x + 8*y
            if bitboard[sq]:
                print("1  ", end="")
            else:
                print(".  ", end="")
        print(f"  {8-y}")
    print("\n    A  B  C  D  E  F  G  H")


# bb = board.get_bitboards()

# board.print()
