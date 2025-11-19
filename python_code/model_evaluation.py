import os
import pickle
import torch
import time
import random
import numpy as np
from stockfish import Stockfish
from alphazero import model as alphazero_model
from alphazero import choose_move as alphazero_choose_move
from alphazero import GameStateNode as alphazero_GameStateNode
from alphazero import load_objects as load_alphazero_objects
from alphazero import save_objects as _save_alphazero_objects
from alphazero import set_saved_objects_directory
from memory_profiler import profile

import game_engine
from constants import WHITE, BLACK, WHITE_WIN, BLACK_WIN, DRAW, NOTOVER
from stockfish_api import get_stockfish_move

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

def calculate_new_elo(old_elo, opponent_elo, score, min_elo=100):
    E = 1/(1+10**((opponent_elo-old_elo)/400))
    return max(old_elo + 20*(score-E), min_elo)

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

# @profile
def alphazero_play_stockfish(info_str = None, min_stockfish_elo = 400, printing=False):
    """
    for evaluation purposes
    """
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

    record_new_elo("alphazero", stockfish_elo, score)

    return score

def main():
    while True:
        os.system("clear")
        info_str = f"alphazero_elo = {round(elo_records_dict["alphazero_elo"])}"
        score = alphazero_play_stockfish(info_str=info_str, printing=True)

if __name__ == "__main__":
    main()
