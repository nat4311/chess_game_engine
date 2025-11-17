import os
import pickle
import torch
import time
import random
from stockfish import Stockfish
from alphazero import model as alphazero_model
from alphazero import choose_move as alphazero_choose_move
from alphazero import GameStateNode as alphazero_GameStateNode
from alphazero import load_objects as load_alphazero_objects
from alphazero import set_saved_objects_directory
from alphazero import WHITE, BLACK, WHITE_WIN, BLACK_WIN, DRAW, NOTOVER
import game_engine
import numpy as np

"""#############################################################
                   Section: elo_records
#############################################################"""

elo_records_savefile = "elo_records.pickle"
elo_records_dict = dict()

def load_elo_records():
    global elo_records_dict
    set_saved_objects_directory()
    print("Loading model_evaluation objects...")

    if os.path.exists(elo_records_savefile):
        with open(elo_records_savefile, 'rb') as f:
            elo_records_dict = pickle.load(f)
            print(f"    {elo_records_savefile} loaded")
    else:
        print(f" X  {elo_records_savefile} not found")

def save_elo_records():
    set_saved_objects_directory()
    print("Saving model_evaluation objects...")

    with open(elo_records_savefile, 'wb') as f:
        pickle.dump(elo_records_dict, f)
        print(f"    {elo_records_savefile} saved")

    print()

load_elo_records()
model_list = [
    "test",
    "alphazero",
]
for model_str in model_list:
    if f"{model_str}_elo" not in elo_records_dict.keys():
        print(f"adding {model_str} to elo_records")
        elo_records_dict[f"{model_str}_elo"] = 400
        elo_records_dict[f"{model_str}_elo_history"] = [] # list of tuples -> (epoch_time, elo)

def calculate_new_elo(old_elo, opponent_elo, score):
    E = 1/(1+10**((opponent_elo-old_elo)/400))
    return old_elo + 20*(score-E)

def record_new_elo(model_str, opponent_elo, score):
    old_elo = elo_records_dict[f"{model_str}_elo"]
    new_elo = calculate_new_elo(old_elo, opponent_elo, score)
    epoch_time = time.time()
    elo_records_dict[f"{model_str}_elo"] = new_elo
    elo_records_dict[f"{model_str}_elo_history"].append((epoch_time, new_elo))
    save_elo_records()

"""#############################################################
               Section: evaluate against stockfish
#############################################################"""
piece_chars = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
sq_strs = [
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
]
sq_ints = {
    "a8": 0, "b8": 1, "c8": 2, "d8": 3, "e8": 4, "f8": 5, "g8": 6, "h8": 7,
    "a7": 8, "b7": 9, "c7": 10, "d7": 11, "e7": 12, "f7": 13, "g7": 14, "h7": 15,
    "a6": 16, "b6": 17, "c6": 18, "d6": 19, "e6": 20, "f6": 21, "g6": 22, "h6": 23,
    "a5": 24, "b5": 25, "c5": 26, "d5": 27, "e5": 28, "f5": 29, "g5": 30, "h5": 31,
    "a4": 32, "b4": 33, "c4": 34, "d4": 35, "e4": 36, "f4": 37, "g4": 38, "h4": 39,
    "a3": 40, "b3": 41, "c3": 42, "d3": 43, "e3": 44, "f3": 45, "g3": 46, "h3": 47,
    "a2": 48, "b2": 49, "c2": 50, "d2": 51, "e2": 52, "f2": 53, "g2": 54, "h2": 55,
    "a1": 56, "b1": 57, "c1": 58, "d1": 59, "e1": 60, "f1": 61, "g1": 62, "h1": 63,
}
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

def alphazero_play_stockfish(model, info_str = None, min_stockfish_elo = 100):
    stockfish = Stockfish("/usr/games/stockfish")
    load_alphazero_objects()
    
    model_elo = elo_records_dict["alphazero_elo"]
    stockfish_elo = round(model_elo) + random.randint(-80,80)
    stockfish_elo = max(stockfish_elo, min_stockfish_elo)
    stockfish.set_elo_rating(stockfish_elo)

    curr_node = alphazero_GameStateNode()
    curr_node.generate_children()
    model_turn = random.random() > .5
    if model_turn:
        side_str = "alphazero white, stockfish black"
    else:
        side_str = "alphazero black, stockfish white"
    while True:
        print("------------------------------------")
        curr_node.print()
        if model_turn:
            _, new_node, _ = alphazero_choose_move(curr_node, greedy=True)
            curr_node = new_node
        else:
            U32_move = get_stockfish_move(stockfish, curr_node)
            for child in curr_node.children.values():
                if child.prev_move == U32_move:
                    curr_node = child
                    break
        curr_node.generate_children()
        print("\n\n")
        print(side_str)
        if info_str is not None:
            print(info_str)
            print(f"{stockfish_elo = }")

        if curr_node.state in (WHITE_WIN, BLACK_WIN):
            print("------------------------------------")
            curr_node.print()
            if model_turn:
                score = 1
                break
            else:
                score = 0
                break
        elif curr_node.state == DRAW:
            score = .5
            break
        else:
            model_turn = not model_turn

    old_elo = elo_records_dict["alphazero_elo"]
    record_new_elo("alphazero", stockfish_elo, score)
    new_elo = elo_records_dict["alphazero_elo"]
    print(f"alphazero_elo: {round(old_elo)} -> {round(new_elo)}")

    return score

if __name__ == "__main__":
    alphazero_n_games = 0
    alphazero_total_score = 0
    while True:
        if alphazero_n_games % 10 == 0:
            os.system("clear")
        info_str = f"total_score: {alphazero_total_score}/{alphazero_n_games}\nalphazero_elo = {round(elo_records_dict["alphazero_elo"])}"
        score = alphazero_play_stockfish(alphazero_model, info_str)
        alphazero_n_games += 1
        alphazero_total_score += score














