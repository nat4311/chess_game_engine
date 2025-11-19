import time
import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
from alphazero import self_play_one_game, stockfish_play_one_game, set_saved_objects_directory
from alphazero import feature_channels, discount_factor, batch_size
from constants import piece_chars, WHITE, BLACK

test_input_data_tensor_savefile = "test_input_data_tensor.pth"
test_policy_data_tensor_savefile = "test_input_policy_tensor.pth"
test_value_data_tensor_savefile = "test_input_value_tensor.pth"

def generate_test_data(self_play=True):
    """
        *Description*
        set networks to None to use random rollout() and equal priors for expand()
        returns model scores - list of floats
        --------------------------------------------------------------------------
        *Input*
        self_play    bool    plays stockfish instead if False
    """
    model_data_list = []
    policy_data_list = []
    value_data_list = []
    t0 = time.time()

    #### SELF PLAY GAMES
    for i_game in range(1):
        input_data = None
        policy_data = None
        result = None

        # value_data_list came from monte carlo - going to use result instead for value training
        if self_play:
            input_data, policy_data, result = self_play_one_game()
        else:
            input_data, policy_data, result = stockfish_play_one_game()

        model_data_list.append(input_data.reshape(-1,feature_channels,8,8)) # shape: batch, channels, rows, cols
        policy_data_list.append(policy_data) # shape: batch, 73, 64
        value_data = torch.zeros(input_data.shape[0])
        if result != 0:
            r = result
            for i in range(len(value_data)):
                r *= discount_factor
                value_data[-i-1] = r
        value_data_list.append(value_data.reshape(-1,1)) # shape: batch, outputs

    input_data_tensor = torch.cat(model_data_list)
    policy_data_tensor = torch.cat(policy_data_list)
    value_data_tensor = torch.cat(value_data_list)
    print(f"size of input_data_tensor: {sys.getsizeof(input_data_tensor)/1000000} MB    {input_data_tensor.shape = }")
    print(f"size of policy_data_tensor: {sys.getsizeof(policy_data_tensor)/1000000} MB    {policy_data_tensor.shape = }")
    print(f"size of value_data_tensor: {sys.getsizeof(value_data_tensor)/1000000} MB    {value_data_tensor.shape = }")

    set_saved_objects_directory()
    torch.save(input_data_tensor, test_input_data_tensor_savefile)
    torch.save(policy_data_tensor, test_policy_data_tensor_savefile)
    torch.save(value_data_tensor, test_value_data_tensor_savefile)


def print_input_datum(input_datum):
    print("input")
    for t in range(8):
        i = t*14 + 12
        assert input_datum[i:i+2, :, :].sum() == 0
        board = [['.' for _ in range(8)] for _ in range(8)]
        for bb in range(12):
            i = t*14 + bb
            for y in range(8):
                for x in range(8):
                    if input_datum[i, y, x]:
                        board[y][x] = piece_chars[bb]
        if t==7:
            print(f"t = 0")
        else:
            print(f"t = -{7-t}")
        for line in board:
            print(line)
        print()

    for i in range(-7, 0):
        assert torch.all(input_datum[i,:,:] == input_datum[i,0,0])
    print("       turn:", "white" if (input_datum[-7,0,0].item()==WHITE) else "black")
    print("total_moves:", input_datum[-6,0,0].item())
    castle_str = ""
    if input_datum[-5,0,0]:
        castle_str += "K"
    if input_datum[-4,0,0]:
        castle_str += "Q"
    if input_datum[-3,0,0]:
        castle_str += "k"
    if input_datum[-2,0,0]:
        castle_str += "q"
    print("   castling:", castle_str)
    print("   halfmove:", input_datum[-1,0,0].item())

def print_policy_datum(policy_datum):
    print("policy")
    print(f"{policy_datum.sum() = }")
    print()
    print("probability     (73, 64)")
    for i in range(73):
        for j in range(64):
            ap = policy_datum[i,j]
            if ap > 0:
                print(f"ap = {str(round(ap.item(), 4)).ljust(7)}   ({i=},{j=})")

def visualize_data():
    set_saved_objects_directory()
    input_data = torch.load(test_input_data_tensor_savefile)
    policy_data = torch.load(test_policy_data_tensor_savefile)
    value_data = torch.load(test_value_data_tensor_savefile)

    i = 113
    print_input_datum(input_data[i,:,:,:])
    print("-----------------------------------------")
    print_policy_datum(policy_data[i,:,:]) 
    print("-----------------------------------------")
    print("value:", value_data[i,0].item())
    print("-----------------------------------------")
    print(f"{input_data.shape = }")
    print(f"{policy_data.shape = }")
    print(f"{value_data.shape = }")

if __name__ == "__main__":
    # generate_test_data()
    visualize_data()

