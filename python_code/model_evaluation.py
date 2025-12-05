import os
import pickle
import torch
import time
import random
import numpy as np
from stockfish import Stockfish
from memory_profiler import profile

from alphazero import model as alphazero_model
from alphazero import choose_move as alphazero_choose_move
from alphazero import GameStateNode as alphazero_GameStateNode
from alphazero import load_objects as load_alphazero_objects
from alphazero import set_saved_objects_directory

from alphazero2 import model as alphazero2_model
from alphazero2 import choose_move as alphazero2_choose_move
from alphazero2 import GameStateNode as alphazero2_GameStateNode
from alphazero2 import load_objects as load_alphazero2_objects

import game_engine
from constants import WHITE, BLACK, WHITE_WIN, BLACK_WIN, DRAW, NOTOVER
from stockfish_api import get_stockfish_move, get_stockfish_move1, get_human_move

"""#############################################################
                   Section: elo_calculation
#############################################################"""

def calculate_new_elo(old_elo, opponent_elo, score, min_elo=100):
    E = 1/(1+10**((opponent_elo-old_elo)/400))
    return max(old_elo + 20*(score-E), min_elo)

"""#############################################################
               Section: evaluate against stockfish
#############################################################"""

# @profile
def alphazero_play_stockfish(model_elo = 1200, info_str = None, min_stockfish_elo = 400, printing=False):
    """
    for evaluation purposes
    """
    stockfish = Stockfish("/usr/games/stockfish")
    load_alphazero_objects()
    
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
        if printing:
            print("------------------------------------")
            curr_node.print()
            print("\n\n")
            p, v = alphazero_model(curr_node.get_full_model_input())
            print(f"model eval: {round(v.item(), 4)}")
            print(side_str)
            if info_str is not None:
                print(info_str)
                print(f"{stockfish_elo = }")

        if curr_node.state in (WHITE_WIN, BLACK_WIN):
            if printing:
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

    new_model_elo = calculate_new_elo(model_elo, stockfish_elo, score)

    return score, new_model_elo

def alphazero2_play_stockfish(model_elo=1200, info_str = None, min_stockfish_elo = 400, printing=False):
    """
    for evaluation purposes
    """
    stockfish = Stockfish("/usr/games/stockfish")
    load_alphazero2_objects()
    
    stockfish_elo = round(model_elo) + random.randint(-80,80)
    stockfish_elo = max(stockfish_elo, min_stockfish_elo)
    stockfish.set_elo_rating(stockfish_elo)

    curr_node = alphazero2_GameStateNode()
    curr_node.generate_children()
    model_turn = random.random() > .5
    if model_turn:
        side_str = "alphazero2 white, stockfish black"
    else:
        side_str = "alphazero2 black, stockfish white"
    while True:
        if model_turn:
            _, new_node, _ = alphazero2_choose_move(curr_node, greedy=True)
            curr_node = new_node
        else:
            U32_move = get_stockfish_move(stockfish, curr_node)
            for child in curr_node.children.values():
                if child.prev_move == U32_move:
                    curr_node = child
                    break
        curr_node.generate_children()
        if printing:
            print("------------------------------------")
            curr_node.print()
            print("\n\n")
            p, v = alphazero2_model(curr_node.get_full_model_input())
            print(f"model eval: {round(v.item(), 4)}")
            print(side_str)
            if info_str is not None:
                print(info_str)
                print(f"{stockfish_elo = }")

        if curr_node.state in (WHITE_WIN, BLACK_WIN):
            if printing:
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

    new_model_elo = calculate_new_elo(model_elo, stockfish_elo, score)

    return score, new_model_elo

def hand_tuned_model_play_stockfish(model_elo = 1200, info_str = None, minmax_search_depth = 7, min_stockfish_elo = 400, printing=False):
    """
    for evaluation purposes
    """
    stockfish = Stockfish("/usr/games/stockfish")

    stockfish_elo = round(model_elo) + random.randint(-80,80)
    stockfish_elo = max(stockfish_elo, min_stockfish_elo)
    stockfish.set_elo_rating(stockfish_elo)

    curr_node = game_engine.GameStateNode1()
    model_turn = random.random() > .5
    if model_turn:
        side_str = "hand_tuned white, stockfish black"
    else:
        side_str = "hand_tuned black, stockfish white"
    while True:
        if model_turn:
            curr_node = curr_node.choose_node(minmax_search_depth)
        else:
            U32_move = get_stockfish_move1(stockfish, curr_node)
            for move in curr_node.board.get_l_move_list():
                if move == U32_move:
                    curr_node.make_move(move)
                    break
        if printing:
            print("------------------------------------")
            curr_node.board.print()
            print("\n\n")
            print(side_str)
            if info_str is not None:
                print(info_str)
                print(f"{stockfish_elo = }")

        if curr_node.board.get_state() in (WHITE_WIN, BLACK_WIN):
            if printing:
                print("------------------------------------")
                if curr_node.board.get_state() == WHITE_WIN:
                    print("game over: white win")
                else:
                    print("game over: black win")
                curr_node.board.print()
            if model_turn:
                score = 1
                break
            else:
                score = 0
                break
        elif curr_node.board.get_state() == DRAW:
            print("game over: draw")
            score = .5
            break
        else:
            model_turn = not model_turn

    new_model_elo = calculate_new_elo(model_elo, stockfish_elo, score)

    return score, new_model_elo

def human_play_hand_model(printing=True):
    """
    for evaluation purposes
    """

    curr_node = game_engine.GameStateNode1()
    model_turn = random.random() > .5
    if model_turn:
        side_str = "hand_tuned white, human black"
    else:
        side_str = "hand_tuned black, human white"
    while True:
        if model_turn:
            curr_node = curr_node.choose_node(7)
        else:
            U32_move = get_human_move(curr_node)
            for move in curr_node.board.get_l_move_list():
                if move == U32_move:
                    curr_node.make_move(move)
                    break
        if printing:
            print("------------------------------------")
            curr_node.board.print()
            print("\n\n")
            print(side_str)

        if curr_node.board.get_state() in (WHITE_WIN, BLACK_WIN):
            if printing:
                print("------------------------------------")
                if curr_node.board.get_state() == WHITE_WIN:
                    print("game over: white win")
                else:
                    print("game over: black win")
                curr_node.board.print()
            if model_turn:
                score = 1
                break
            else:
                score = 0
                break
        elif curr_node.board.get_state() == DRAW:
            print("game over: draw")
            score = .5
            break
        else:
            model_turn = not model_turn

"""#############################################################
                       Section: main
#############################################################"""

def main():
    n = 0
    model_elo = 1200
    while True:
        if n%10 == 9:
            os.system("clear")

        info_str = f"model_elo = {round(model_elo)}"
        score = alphazero_play_stockfish(info_str=info_str, printing=True)

        # info_str = f"model_elo = {round(model_elo)}"
        # score = alphazero2_play_stockfish(info_str=info_str, printing=True)

        # info_str = f"model_elo = {round(model_elo)}"
        # score, model_elo = hand_tuned_model_play_stockfish(model_elo, info_str=info_str, printing=True)

if __name__ == "__main__":
    main()
    # human_play_hand_model()
