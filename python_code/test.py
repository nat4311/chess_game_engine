import game_engine
import numpy as np
import time
import torch
from alphazero import GameStateNode, feature_channels, ResNet, rollout, batch_size
from alphazero import model, load_objects, save_objects, get_policy_move
from alphazero import U32_move_to_policy_move_dict

###############################################################

def get_policy_move_test():
    # print(U32_move_to_policy_move_dict)
    node = GameStateNode()
    pl_move_list = node.moves.get_pl_move_list(node.board)
    for U32_move in pl_move_list:
        policy_move = get_policy_move(U32_move)
        print(policy_move)
    node.print()
    # print(U32_move_to_policy_move_dict)
    save_objects()

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

    # Forward pass through model
    t0 = time.time()
    for i in range(1000):
        policy_output, value_output = model(input_tensor)
    t1 = time.time()
    print(t1-t0)

    # Check shapes
    print()
    print("Input shape:", input_tensor.shape)
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
    curr_node = GameStateNode()
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
    curr_node = GameStateNode()
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
    node = GameStateNode()
    fen = "rnbqkbnr/pppppppp/8/8/7q/8/PPPPPPPP/RNBQKBNR b KQkq - 1 2\n";
    node.board.load(fen)
    node.board.print()
    m = node.get_partial_model_input()
    print(m.shape)
    print(m.dtype)
    for i in range(21):
        print(f"i={str(i).ljust(2)}   {str(sum(m[i,:,:].flatten())).ljust(5)}  {meanings[i].ljust(27)}")


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



if __name__ == "__main__":
    print("======================================")
    # model_input_test()
    # make_test()
    # basic_board_test()
    # get_partial_model_input_test()
    # test_rollout()
    # test_net_shapes()
    get_policy_move_test()

    pass
