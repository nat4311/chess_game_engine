from stockfish import Stockfish
from constants import WHITE, BLACK, WHITE_WIN, BLACK_WIN, DRAW, NOTOVER, piece_chars, sq_strs, sq_ints
import game_engine
import time
import torch
from math import tanh, exp
import numpy as np

def get_promotion_piece_int(promotion_piece_str, turn):
    if promotion_piece_str == "n":
        if turn == WHITE:
            return 1
        else:
            return 7
    if promotion_piece_str == "b":
        if turn == WHITE:
            return 2
        else:
            return 8
    if promotion_piece_str == "r":
        if turn == WHITE:
            return 3
        else:
            return 9
    if promotion_piece_str == "q":
        if turn == WHITE:
            return 4
        else:
            return 10

def lsb_index(num):
    if num == 0:
        return None
    return (num & -num).bit_length() - 1

def generate_fen(board):
    """
    board is a BoardState object from engine.cpp
    """
    board_list = [[None for x in range(8)] for y in range(8)]
    bbs = board.get_bitboards_U64()
    for i, piece_char in enumerate(piece_chars):
        bb = int(bbs[i])
        while bb:
            sq = lsb_index(bb)
            if sq is None:
                break
            else:
                bb = bb ^ 1<<sq
                x = sq%8
                y = int(sq/8)
                board_list[y][x] = piece_char

    fen = ""
    for y in range(8):
        empty_sq_count = 0
        for x in range(8):
            piece = board_list[y][x]
            if piece is None:
                empty_sq_count += 1
            else:
                if empty_sq_count > 0:
                    fen += str(empty_sq_count)
                    empty_sq_count = 0
                fen += piece
        if empty_sq_count > 0:
            fen += str(empty_sq_count)
        if y < 7:
            fen += "/"

    fen += " "
    if board.turn == WHITE:
        fen += "w "
    else:
        fen += "b "

    no_castling = True
    if board.get_castle_K():
        fen += "K"
        no_castling = False
    if board.get_castle_Q():
        fen += "Q"
        no_castling = False
    if board.get_castle_k():
        fen += "k"
        no_castling = False
    if board.get_castle_q():
        fen += "q"
        no_castling = False
    if no_castling:
        fen += "-"
    fen += " "

    enpassant_sq = board.enpassant_sq
    if enpassant_sq == 64:
        fen += "-"
    else:
        fen += sq_strs[enpassant_sq]
    fen += " "

    fen += str(board.halfmove)
    fen += " "

    fen += str(board.turn_no+1)
    
    return fen

def get_human_move(game_state_node1):
    """
    for use with hand_tuned_model.cpp GameStateNode1 class
    returns a U32_move
    """
    board = game_state_node1.board

    human_move = input("input a move: ")
    source_sq_str = human_move[:2]
    source_sq = sq_ints[source_sq_str]
    target_sq_str = human_move[2:4]
    target_sq = sq_ints[target_sq_str]
    if len(human_move) > 4:
        promotion_piece_str = human_move[4]
    else:
        promotion_piece_str = None

    for move in board.get_l_move_list():
        a = game_engine.get_move_source_sq(move)
        b = game_engine.get_move_target_sq(move)
        if a == source_sq and b == target_sq:
            if promotion_piece_str is None:
                # print(stockfish_move, a, b)
                return move
            else:
                c = game_engine.get_move_promotion_piece_type(move)
                promotion_piece = get_promotion_piece_int(promotion_piece_str, board.turn)
                if c == promotion_piece:
                    # print(stockfish_move, a, b, c)
                    return move

    board.print()
    print(f"invalid move: {human_move}")
    return get_human_move(game_state_node1)

def get_stockfish_move1(stockfish, game_state_node1):
    """
    stockfish elo should already be set
    for use with hand_tuned_model.cpp GameStateNode1 class
    returns a U32_move
    """
    board = game_state_node1.board
    fen = generate_fen(board)
    if not stockfish.is_fen_valid(fen):
        board.print()
        print(fen)
        raise Exception("fen was invalid")
    else:
        stockfish.set_fen_position(fen)

    stockfish_move = stockfish.get_best_move_time(3000)
    source_sq_str = stockfish_move[:2]
    source_sq = sq_ints[source_sq_str]
    target_sq_str = stockfish_move[2:4]
    target_sq = sq_ints[target_sq_str]
    if len(stockfish_move) > 4:
        promotion_piece_str = stockfish_move[4]
    else:
        promotion_piece_str = None

    for move in board.get_l_move_list():
        a = game_engine.get_move_source_sq(move)
        b = game_engine.get_move_target_sq(move)
        if a == source_sq and b == target_sq:
            if promotion_piece_str is None:
                # print(stockfish_move, a, b)
                return move
            else:
                c = game_engine.get_move_promotion_piece_type(move)
                promotion_piece = get_promotion_piece_int(promotion_piece_str, board.turn)
                if c == promotion_piece:
                    # print(stockfish_move, a, b, c)
                    return move

    board.print()
    print(f"{stockfish_move = }")
    raise Exception("unable to find move")

def get_stockfish_move(stockfish, game_state_node):
    """
    stockfish elo should already be set
    game_state_node must contain a BoardState and MoveGenerator called board and moves
    returns a U32_move
    """
    board = game_state_node.board
    moves = game_state_node.moves
    fen = generate_fen(board)
    if not stockfish.is_fen_valid(fen):
        board.print()
        print(fen)
        raise Exception("fen was invalid")
    else:
        stockfish.set_fen_position(fen)

    stockfish_move = stockfish.get_best_move()
    source_sq_str = stockfish_move[:2]
    source_sq = sq_ints[source_sq_str]
    target_sq_str = stockfish_move[2:4]
    target_sq = sq_ints[target_sq_str]
    if len(stockfish_move) > 4:
        promotion_piece_str = stockfish_move[4]
    else:
        promotion_piece_str = None

    for move in moves.get_pl_move_list(board):
        a = game_engine.get_move_source_sq(move)
        b = game_engine.get_move_target_sq(move)
        if a == source_sq and b == target_sq:
            if promotion_piece_str is None:
                # print(stockfish_move, a, b)
                return move
            else:
                c = game_engine.get_move_promotion_piece_type(move)
                promotion_piece = get_promotion_piece_int(promotion_piece_str, board.turn)
                if c == promotion_piece:
                    # print(stockfish_move, a, b, c)
                    return move

    board.print()
    print(f"{stockfish_move = }")
    raise Exception("unable to find move")

def stockfish_move_to_U32_move(stockfish_move, pl_move_list, board):
    source_sq_str = stockfish_move[:2]
    source_sq = sq_ints[source_sq_str]
    target_sq_str = stockfish_move[2:4]
    target_sq = sq_ints[target_sq_str]
    if len(stockfish_move) > 4:
        promotion_piece_str = stockfish_move[4]
    else:
        promotion_piece_str = None

    for move in pl_move_list:
        a = game_engine.get_move_source_sq(move)
        b = game_engine.get_move_target_sq(move)
        if a == source_sq and b == target_sq:
            if promotion_piece_str is None:
                return move
            else:
                c = game_engine.get_move_promotion_piece_type(move)
                promotion_piece = get_promotion_piece_int(promotion_piece_str, board.turn)
                if c == promotion_piece:
                    return move

    print("ERROR MOVE NOT FOUND")
    board.print()
    raise Exception("ERROR MOVE NOT FOUND")
            
def np_softmax(arr, temperature=100):
    shifted_arr = arr - np.max(arr)
    exp_arr = np.exp(shifted_arr / temperature)
    probs = exp_arr / np.sum(exp_arr)
    return probs

def torch_softmax(arr, temperature=100):
    shifted_arr = arr - torch.max(arr)
    exp_arr = torch.exp(shifted_arr / temperature)
    probs = exp_arr / torch.sum(exp_arr)
    return probs

if __name__ == "__main__":
    # from alphazero import GameStateNode
    #
    # # game = GameStateNode()
    stockfish = Stockfish("/usr/games/stockfish")
    # # stockfish.set_elo_rating(2000)
    # stockfish.set_depth(10)
    # # stockfish.set_fen_position("rn1qkbnr/p1pppp1p/1p4p1/8/3PP3/2Pb1N2/PP3PPP/RNBQKB1R w KQkq - 1 5")
    # stockfish.set_fen_position("rn1qkbnr/p1pppp1p/1p4p1/8/3PP3/2Pb1N2/PP3PPP/RNBQKB1R w KQkq - 1 5")
    stockfish.set_fen_position("K7/8/r7/7r/7k/8/8/8 w - - 2 5")
    print(stockfish.get_board_visual())
    ms = stockfish.get_top_moves(200)
    for m in ms:
        if m["Centipawn"] is None:
            print(m)

    # x = np.array([-10000,100,200,300,1000])
    # y = np_softmax(x)
    #
    # for a, b in zip(x,y):
    #     print(str(a).ljust(5), b)

    # eval = stockfish.get_evaluation()["value"]
    # print(tanh(2000/700))


    # vs = []
    # vs2 = []
    # for m in ms:
    #     v2 = m["Centipawn"]
    #     vs2.append(v2)
    #     vs.append(v2)
    # vs.append(1000)
    # vs2.append(1000)
    # # vs = softmax(np.array(vs))
    # vs = np_softmax(np.array(vs))
    # print(len(vs))
    # print()
    # for i in range(30):
    #     print(np.random.choice(len(vs), p=vs))
    # for v,v2 in zip(vs,vs2):
    #     print(v2, v)
    # print(tanh(-100/777))
