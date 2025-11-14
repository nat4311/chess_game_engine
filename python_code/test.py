import game_engine
import numpy as np
import time
from alphazero import GameStateNode

###############################################################

def basic_board_test():
    board = game_engine.BoardState()
    moves = game_engine.MoveGenerator()
    board.print()
    board.reset()
    board.print()


def get_model_input_test():
    meanings = [ "white_pawns", "white_knights", "white_bishops", "white_rooks", "white_queens", "white_kings", "black_pawns", "black_knights", "black_bishops", "black_rooks", "black_queens", "black_kings", "repetitions_1", "repetitions_2", "turn*64", "total_moves*64", "WHITE_KINGSIDE_CASTLE*64", "WHITE_QUEENSIDE_CASTLE*64", "BLACK_KINGSIDE_CASTLE*64", "BLACK_QUEENSIDE_CASTLE*64", "halfmove*64", ]
    node = GameStateNode()
    m = node.model_input()
    print(m.shape)
    print(m.dtype)
    for i in range(21):
        print(f"i={str(i).ljust(2)}   {str(sum(m[i,:,:].flatten())).ljust(5)}  {meanings[i].ljust(27)}")
# get_model_input_test()


def make_test():
    fen = "rnbqkbnr/pppppppp/8/8/7q/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    board = game_engine.BoardState()
    board.load(fen)
    moves = game_engine.MoveGenerator()
    move_list = moves.get_pl_move_list(board)
    for move in move_list:
        b = board.copy()
        if b.make(move):
            b.print()
            time.sleep(.5)
    board.print()



# moves.generate_pl_moves(board)
# moves.print_pl_moves()

# def print_bool_bitboard(bitboard):
#     print("    A  B  C  D  E  F  G  H\n")
#     for y in range(8):
#         print(8-y, end="   ")
#         for x in range(8):
#             sq = x + 8*y
#             if bitboard[sq]:
#                 print("1  ", end="")
#             else:
#                 print(".  ", end="")
#         print(f"  {8-y}")
#     print("\n    A  B  C  D  E  F  G  H")
#

# bb = board.get_bitboards()

# board.print()
