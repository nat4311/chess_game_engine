import game_engine
import numpy as np
import time
import torch
import random
import os
from alphazero import GameStateNode as alphazero_GameStateNode
from alphazero import load_objects as load_alphazero_objects
from alphazero import choose_move as alphazero_choose_move
from alphazero import feature_channels, ResNet, rollout, batch_size
from alphazero import model, load_objects, get_policy_move, mcts_n_sims
from alphazero import U32_move_to_policy_move_dict, self_play_one_game
from stockfish import Stockfish
from model_evaluation import get_stockfish_move
from constants import WHITE_WIN, BLACK_WIN, DRAW, piece_chars

###############################################################

def get_policy_move_test():
    # print(U32_move_to_policy_move_dict)
    node = alphazero_GameStateNode()
    pl_move_list = node.moves.get_pl_move_list(node.board)
    for U32_move in pl_move_list:
        policy_move = get_policy_move(U32_move)
        print(policy_move)
    node.print()
    # print(U32_move_to_policy_move_dict)

    # print("\nbefore")
    # print(U32_move_to_policy_move_dict)
    # print()
    # node = GameStateNode()
    # pl_move_list = node.moves.get_pl_move_list(node.board)
    # for U32_move in pl_move_list:
    #     policy_move = get_policy_move(U32_move)
    #     # print(policy_move)
    # # node.print()
    # print()
    # print("after")
    # print(U32_move_to_policy_move_dict)
    # print()
    # save_objects()


def test_net_shapes():
    model.eval()  # set to eval mode as default for testing

    input_tensor = torch.randn(batch_size, feature_channels, 8, 8)
    print("Input shape:", input_tensor.shape)

    # Forward pass through model
    t0 = time.time()
    for i in range(1000):
        policy_output, value_output = model(input_tensor)
    t1 = time.time()
    print(t1-t0)

    print("Policy output shape:", policy_output.shape)  # expected (1, 73, 8, 8)
    print("Value output shape:", value_output.shape)    # expected (1, 1)

    # # To get final policy as (64 x 73), reshape and permute policy output:
    # # policy_output shape is (batch, 73, 8, 8)
    # # Reshape to (batch, 73, 64) then permute to (batch, 64, 73)
    # policy_reshaped = policy_output.reshape(batch_size, 73, 64)
    # print()
    # print(f"Policy reshaped shape (should be {batch_size} x 73 x 64):", policy_reshaped.shape)

    # Value should be (batch, 1), verify the scalar value
    print()
    print("Value output:", value_output)


def test_rollout():
    curr_node = alphazero_GameStateNode()
    input_data = torch.zeros((1,feature_channels,8,8))
    d = curr_node.get_partial_model_input()
    input_data[0, -21:, :, :] = d
    curr_node.full_model_input = input_data

    p,v = model(curr_node.get_full_model_input())

    l = rollout(curr_node, model)
    print(l)
    print(curr_node.value_sum)

def model_input_test():
    input_data = torch.zeros((3000,8,8))
    curr_node = alphazero_GameStateNode()
    input_datum = curr_node.get_partial_model_input()
    # for i in range(12):
    #     print(input_datum[i,:,:])
    #     time.sleep(.8)
    # print(input_datum.shape)
    # print(type(input_datum))
    input_data[1:22, :, :] = torch.Tensor(input_datum)
    print(input_data.dtype)

def basic_board_test():
    board = game_engine.BoardState()
    moves = game_engine.MoveGenerator()
    fen = "rnbqkbnr/ppppp1pp/8/7B/7q/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1\n";
    board.load(fen)
    board.print()
    print(board.king_is_attacked())
    # print(board.halfmove)
    # board.reset()
    # board.print()

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


def get_partial_model_input_test():
    meanings = [ "white_pawns", "white_knights", "white_bishops", "white_rooks", "white_queens", "white_kings", "black_pawns", "black_knights", "black_bishops", "black_rooks", "black_queens", "black_kings", "repetitions_1", "repetitions_2", "turn*64", "turn_no*64", "WHITE_KINGSIDE_CASTLE*64", "WHITE_QUEENSIDE_CASTLE*64", "BLACK_KINGSIDE_CASTLE*64", "BLACK_QUEENSIDE_CASTLE*64", "halfmove*64", ]
    node = alphazero_GameStateNode()
    fen = "rnbqkbnr/pppppppp/8/8/7q/8/PPPPPPPP/RNBQKBNR b KQkq - 1 2\n";
    node.board.load(fen)
    node.board.print()
    m = node.get_partial_model_input()
    print(m.shape)
    print(m.dtype)
    for i in range(21):
        print(f"i={str(i).ljust(2)}   {str(sum(m[i,:,:].flatten())).ljust(5)}  {meanings[i].ljust(27)}")


def hand_eval_test():
    fen = "rnbqkbnr/p2ppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = game_engine.BoardState()
    board.load(fen)
    board.print()

    mat = board.material_score()
    print(f"{mat = }")
    dp = board.doubled_pawn_score()
    print(f"{dp = }")
    emob = board.enemy_mobility_score()
    print(f"{emob = }")

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

def self_play_one_game_test():
    try:
        input_data, policy_data, result = self_play_one_game()
        print(f"{input_data.shape = }")
        print(f"{policy_data.shape = }")
        print(f"{result = }")
    finally:
        print(f"{len(U32_move_to_policy_move_dict) = }")

def test_policy_and_input_datum():
    assert mcts_n_sims <= 30 # otherwise will take too long to run
    def print_input_datum(input_datum):
        for t in range(8):
            i = t*14 + 12
            assert input_datum[0, i:i+2, :, :].sum() == 0
            board = [['.' for _ in range(8)] for _ in range(8)]
            for bb in range(12):
                i = t*14 + bb
                for y in range(8):
                    for x in range(8):
                        if input_datum[0, i, y, x]:
                            board[y][x] = piece_chars[bb]
            if t==7:
                print(f"t = 0")
            else:
                print(f"t = -{7-t}")
            for line in board:
                print(line)
            print()
        print("turn")
        print(input_datum[0,-7,:,:])
        print("total_moves")
        print(input_datum[0,-6,:,:])
        print("castling")
        print(input_datum[0,-5,:,:])
        print(input_datum[0,-4,:,:])
        print(input_datum[0,-3,:,:])
        print(input_datum[0,-2,:,:])
        print("halfmove")
        print(input_datum[0,-1,:,:])

    def print_policy_datum(policy_datum):
        for i in range(73):
            for j in range(64):
                ap = policy_datum[0,i,j]
                if ap > 0:
                    print(f"{ap=}   ({i=},{j=})")


    input_data, policy_data, result = self_play_one_game()

    t = 0
    input_datum = input_data[t:t+1, :, :, :]
    policy_datum = policy_data[t:t+1, :, :]
    # print_input_datum(input_datum)
    print(policy_datum.shape)
    print_policy_datum(policy_datum)
    print(input_data.dtype)

def test_stockfish_api():
    node = alphazero_GameStateNode()
    stockfish = Stockfish("/usr/games/stockfish")
    stockfish.set_elo_rating(1000)
    for i in range(30):
        node.print()
        node.generate_children()
        U32_move = get_stockfish_move(stockfish, node)
        for child in node.children.values():
            if U32_move == child.prev_move:
                node = child
                break
        time.sleep(.4)

def test_alphazero_play_stockfish(info_str = None, stockfish_elo=3000):
    stockfish = Stockfish("/usr/games/stockfish")
    load_alphazero_objects()
    
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

    return score

def test_alphazero_play_stockfish_loop():
    print("======================================")
    alphazero_n_games = 0
    alphazero_total_score = 0
    while True:
        os.system("clear")
        info_str = f"total_score: {alphazero_total_score}/{alphazero_n_games}"
        score = test_alphazero_play_stockfish(info_str=info_str)
        alphazero_n_games += 1
        alphazero_total_score += score


if __name__ == "__main__":

    hand_eval_test()
    # test_alphazero_play_stockfish_loop()
    # model_input_test()
    # make_test()
    # basic_board_test()
    # get_partial_model_input_test()
    # test_rollout()
    # test_net_shapes()
    # get_policy_move_test()
    # test_stockfish_api()
    pass
