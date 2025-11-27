"""
run this file to train the alphazero resnet
"""

import os
import sys
import portalocker
import time
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stockfish import Stockfish
from math import inf, sqrt, log, tanh
from numpy.random import dirichlet
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from memory_profiler import profile
import subprocess

import game_engine
from utils import pretty_time_elapsed, pretty_datetime
from constants import WHITE, BLACK, WHITE_WIN, BLACK_WIN, DRAW, NOTOVER
from stockfish_api import stockfish_move_to_U32_move 

"""################################################################################
                    Section: Parameters
################################################################################"""

#### Training parameters
self_play = True # if false play stockfish, set elo on next line
stockfish_elo = 2000

n_games = 1024
n_epochs = 5
learning_rate = .0001
batch_size = 32
policy_loss_coeff = 1
value_loss_coeff = 1

#### MCTS parameters
mcts_n_sims = 200
ucb_exploration_constant = 1.414    # for ucb exploration score
alpha_dirichlet = 1.732
x_dirichlet = .75                   # p' = p*x_dirichlet + (1-x_dirichlet)*d; p ~ prior and d ~ dirichlet noise
exploration_temperature = 1.75

#### ResNet parameters
# 14*8 (12 bitboards + 2 repetition, 1 current + 7 past) + 7 (1 turn + 1 total_moves + 4 castling + 1 halfmove)
time_history = 7
feature_channels = 14*(time_history+1) + 7
# the following came from https://www.chessprogramming.org/AlphaZero#Network_Architecture
default_filters = 64 # 256
default_kernel_size = 3
res_block_layers = 6 # 19

"""################################################################################
                    Section: Residual Neural Network (ResNet)
################################################################################"""

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels=default_filters, out_channels=default_filters, kernel_size=default_kernel_size, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(default_filters)
        self.conv1 = nn.Conv2d(in_channels=default_filters, out_channels=default_filters, kernel_size=default_kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(default_filters)

    def forward(self, x):
        res = x
        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out += res
        out = F.relu(out)

        return out


class OutputHeads(nn.Module):
    """
    NOTE: policy head illegal move masking should be done after the output of the network

    outputs policy and value heads
    forward returns p,v
    """

    def __init__(self):
        super().__init__()

        self.p_conv0 = nn.Conv2d(in_channels=default_filters, out_channels=default_filters, kernel_size=default_kernel_size, stride=1, padding=1, bias=False)
        self.p_bn0 = nn.BatchNorm2d(default_filters)
        self.p_conv1 = nn.Conv2d(in_channels=default_filters, out_channels=73, kernel_size=default_kernel_size, stride=1, padding=1, bias=False)
        self.p_lsm = nn.LogSoftmax(dim=1)

        self.v_conv = nn.Conv2d(in_channels=default_filters, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_lin0 = nn.Linear(in_features=64, out_features=256)
        self.v_lin1 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        """
        returns p, v
        """

        p = self.p_conv0(x)
        p = self.p_bn0(p)
        p = F.relu(p)
        p = self.p_conv1(p)
        p = self.p_lsm(p).exp() # output is (batch_size, 73, 8, 8)
        p = p.view(-1, 73, 64)

        v = self.v_conv(x)
        v = self.v_bn(v)
        v = F.relu(v)
        v = v.view(-1, 64)
        v = self.v_lin0(v)
        v = F.relu(v)
        v = self.v_lin1(v)
        v = F.tanh(v) # output is (batch_size, 1)

        return p, v


class ResNet(nn.Module):
    """
    NOTE: policy head illegal move masking should be done after the output of the network

    outputs policy and value heads
    forward returns p,v
    p.shape = (batch_size or 1 for single value, 73, 64)
    v.shape = (batch_size or 1 for single value, 1)
    """
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=feature_channels, out_channels=default_filters, kernel_size=default_kernel_size, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(default_filters)
        for layer in range(res_block_layers):
            setattr(self, f"res_block_{layer}", ResBlock())
        self.output = OutputHeads()

    def forward(self, x):
        """
        returns p, v
        """

        # input conv layer
        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out)

        # ResBlocks
        for layer in range(res_block_layers):
            out = getattr(self, f"res_block_{layer}")(out)

        # policy and value head
        return self.output(out)

model = ResNet()

"""#############################################################
               Section: move_to_policy_move Cache
#############################################################"""

# for converting U32 move (game engine moves list) to 73x8x8 move (policy head output)
U32_move_to_policy_move_dict = dict()

def get_policy_move(U32_move):
    """
    returns (i, j)
    where 0 <= i <= 72 (queen move dir/dist OR knight move dir OR underpromotion dir)
    and   0 <= j <= 63 (source square)
    """
    if U32_move not in U32_move_to_policy_move_dict.keys():
        # print("U32_move not found, generating policy_move indices")
        i = game_engine.policy_move_index_0(U32_move)
        j = game_engine.policy_move_index_1(U32_move)
        U32_move_to_policy_move_dict[U32_move] = (i, j)
    return U32_move_to_policy_move_dict[U32_move]

"""#############################################################
               Section: Loading and saving objects
#############################################################"""
U32_move_to_policy_move_savefile = "U32_move_to_policy_move.pickle"
alphazero2_model_savefile = "alphazero2_model_weights.pth"

def set_saved_objects_directory():
    root_dir = __file__
    while root_dir[-17:] != "chess_game_engine":
        root_dir = root_dir[:-1]
        if len(root_dir) == 0:
            raise Exception("could not find project root directory")
    os.chdir(root_dir + r"/python_code/saved_objects")

def load_objects():
    global model, U32_move_to_policy_move_dict
    set_saved_objects_directory()
    print("Loading alphazero objects...")

    if os.path.exists(alphazero2_model_savefile):
        with portalocker.Lock(alphazero2_model_savefile, "rb", timeout=30) as f:
            model.load_state_dict(torch.load(f, weights_only=True))
            print("    model weights loaded")
    else:
        print(" X  no model weights found")

    if os.path.exists(U32_move_to_policy_move_savefile):
        with portalocker.Lock(U32_move_to_policy_move_savefile, "rb", timeout=30) as f:
            U32_move_to_policy_move_dict = pickle.load(f)
            print(f"    {U32_move_to_policy_move_savefile} loaded")
    else:
        print(f" X  {U32_move_to_policy_move_savefile} not found")

def save_objects():
    set_saved_objects_directory()
    print("Saving alphazero objects...")

    with portalocker.Lock(alphazero2_model_savefile, "wb", timeout=30) as f:
        torch.save(model.state_dict(), alphazero2_model_savefile)
        print("    model weights saved")

    print()

load_objects()

"""################################################################################
                    Section: GameStateNode
################################################################################"""

class GameStateNode:
    def __init__(self, parent=None, board=None, prev_move=0, generate_children=False):
        self.parent = parent
        self.children = dict() # indexed by (73, 64) move
        self.prev_move = prev_move
        self.moves = game_engine.MoveGenerator()
        self.prior = 0
        self.value_sum = 0
        self.n_visits = 0
        self.is_mcts_root = False

        if parent is None:
            self.board = game_engine.BoardState()
            input_datum = torch.zeros((1,feature_channels,8,8))
            input_datum[0, -21:, :, :] = self.get_partial_model_input()
            self._full_model_input = input_datum
        else:
            self.board = board
            self._full_model_input = None
        if generate_children:
            self.generate_children()
        else:
            self.state = None

    def generate_children(self):
        if self.board.halfmove == 100:
            self.state = DRAW
            return

        pl_move_list = self.moves.get_pl_move_list(self.board)
        for U32_move in pl_move_list:
            new_board = self.board.copy()
            if new_board.make(U32_move):
                policy_move = get_policy_move(U32_move)
                new_node = GameStateNode(self, new_board, U32_move)
                self.children[policy_move] = new_node

        if len(self.children) == 0:
            if self.board.king_is_attacked():
                if self.board.turn == WHITE:
                    self.state = BLACK_WIN
                else:
                    self.state = WHITE_WIN
            else:
                self.state = DRAW
        else:
            self.state = NOTOVER

        assert self.state is not None
        return pl_move_list

    def print_pl_moves(self):
        self.moves.generate_pl_moves(self.board)
        self.moves.print_pl_moves()

    def new_node(self, move):
        new_board = self.board.copy()
        if new_board.make(move):
            return GameStateNode(self, new_board, move)
        else:
            return None

    def get_partial_model_input(self):
        return torch.Tensor(self.board.get_partial_model_input()).view(1,-1,8,8)
    
    def get_full_model_input(self):
        """
        returns input_datum as torch.Tensor(1x119x8x8)
        """
        if self._full_model_input is None:
            self._full_model_input = torch.cat((self.parent._full_model_input[:, 14:-7, :, :], self.get_partial_model_input()), dim=1).detach()
        return self._full_model_input

    def print(self):
        print("GameStateNode: \n")
        self.board.print()
        print(f"prev_move: {self.prev_move}")


"""################################################################################
                    Section: Monte Carlo Tree Search (MCTS)
################################################################################"""

# @profile
def MCTS(root_node: GameStateNode):
    """
        *DESCRIPTION*
        Performs monte carlo tree search from a parent GameStateNode.
        The child of that parent with the most visits is selected and returned.
        ----------------------------------------------------------------------------------------------------------
        *INPUTS*
        root_node    GameStateNode    represents the gamestate to begin MCTS at    changes state in this function
        ----------------------------------------------------------------------------------------------------------
        *SIDE EFFECTS*
        - updates the attributes of start_node and its children
        - use n_visits of each child to choose a move
    """

    # reset the root_node MCTS values
    # root_node.n_visits = 0
    # root_node.value_sum = 0
    root_node.is_mcts_root = True

    for n in range(mcts_n_sims):
        curr_node = root_node
        while True:
            if curr_node.n_visits == 0 or curr_node.state in (WHITE_WIN, BLACK_WIN, DRAW): # only rollout for leaf nodes
                rollout(curr_node)
                break
            else:
                if curr_node.n_visits == 1: # only expand if need to
                    expand(curr_node)
                curr_node = select_child(curr_node)
                assert curr_node is not None

    return

# @profile
def rollout(curr_node: GameStateNode) -> int:
    '''
        This function is used when a new leaf node is reached.
        It gives us an estimate of that node's value which is then backpropped to all parent nodes
        --------------------------------------

        uses value network prediction as leaf node value estimate

        - backpropogates the terminal state to all nodes up to root with value_sum and n_visits
        - returns the leaf node value estimate
            1   WHITE_WIN
            0   DRAW
            -1  BLACK_WIN

        note: this is used only for monte carlo tree search, NOT for value training
    '''

    leaf_node = curr_node
    if curr_node.state is None:
        curr_node.generate_children()

    # get estimated value of leaf node (state)
    if curr_node.state in (WHITE_WIN, BLACK_WIN, DRAW):
        leaf_node_value_estimate = curr_node.state
    else:
        with torch.no_grad():
            _, leaf_node_value_estimate = model(leaf_node.get_full_model_input())
        leaf_node_value_estimate = leaf_node_value_estimate.item()

    # backprop leaf_node_value_estimate to all parent nodes
    while True:
        curr_node.value_sum += leaf_node_value_estimate
        curr_node.n_visits += 1
        if curr_node.is_mcts_root or curr_node.parent is None:
            break
        else:
            curr_node = curr_node.parent

    return leaf_node_value_estimate

def expand(curr_node):
    """
    unlike the connect4 version, expand here just needs to set the child priors
    children are already generated when the parent node was created because I needed to check for game over
    """
    with torch.no_grad():
        p,_ = model(curr_node.get_full_model_input())
    action_probs = torch.zeros(p.shape)

    for policy_move, child in curr_node.children.items():
        i, j = policy_move
        action_probs[0, i, j] = p[0, i, j]

    action_probs /= action_probs.sum()

    for policy_move, child in curr_node.children.items():
        i, j = policy_move
        child.prior = action_probs[0, i, j]

def select_child(parent_node: GameStateNode) -> GameStateNode:
    """
        This function is used when a root (expanded) node is reached while traversing tree - this helps us select a game path
        - the higher a child's avg value, the more likely it is to be selected
        - the fewer times a child has been selected relative to its siblings, the more likely it is to be selected
        Uses UCB1 formula to balance exploration and exploitation when choosing nodes in mctree
        - https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
    """
    best_UCB1 = -inf
    best_child = None

    noise = dirichlet([alpha_dirichlet for _ in range(len(parent_node.children))], 1)[0]
    for i, child in enumerate(parent_node.children.values()):
        if child.n_visits == 0:
            explore = inf
            return child

        d = noise[i]
        p = child.prior
        p_prime = p*x_dirichlet + (1-x_dirichlet)*d
        explore = p_prime * ucb_exploration_constant * sqrt(parent_node.n_visits - 1)/(child.n_visits+1)

        if parent_node.board.turn == WHITE:
            exploit = child.value_sum/child.n_visits
        else:
            exploit = -child.value_sum/child.n_visits

        UCB1 = exploit + explore
        if UCB1 > best_UCB1:
            best_UCB1 = UCB1
            best_child = child

    return best_child

# @profile
def choose_move(start_node: GameStateNode, greedy: bool) -> int:
    """
        Uses monte carlo tree search and value/policy networks to choose a move.
        -------------------------------------------------------------------------------------------------------
        INPUTS
        start_node*     GameStateNode       state of the game    *changes state inside this function
        greedy          bool                if greedy select move with most visits, else use visit distribution
        -------------------------------------------------------------------------------------------------------
        OUTPUTS
        move            U32_move            the chosen move
        new_node        GameStateNode       the node that results from making that move from start_node
        policy_datum    torch.Tensor        shape (1, 73, 64) 
        -------------------------------------------------------------------------------------------------------
        SIDE EFFECTS
        - updates n_visits and other attributes of start_node and its children
    """

    MCTS(start_node)
    policy_datum = torch.zeros(1,73,64)

    if greedy:
        most_visits = 0
        most_visited_move = None
        most_visited_child = None
        for policy_move, child in start_node.children.items():
            ap = child.n_visits/(start_node.n_visits-1)
            assert ap<=1 #todo: remove this
            i, j = policy_move
            policy_datum[0, i, j] = ap
            if child.n_visits > most_visits:
                most_visited_move = child.prev_move
                most_visited_child = child
                most_visits = child.n_visits
        # if most_visited_move is None:
        #     print("ERROR")
        #     start_node.print()
        #     print("children: ")
        #     print(start_node.children)
        # if most_visited_child is None:
        #     print("what the heck")
        #     print(most_visited_move)
        return most_visited_move, most_visited_child, None
    else:
        distribution = torch.zeros(1, 73, 64)
        for policy_move, child in start_node.children.items():
            ap = child.n_visits/(start_node.n_visits-1)
            i, j = policy_move
            policy_datum[0, i, j] = ap
            distribution[0, i, j] = child.n_visits**(1/exploration_temperature)
        distribution /= distribution.sum()
        distribution = distribution.view(-1)
        chosen_index = torch.multinomial(distribution, 1)
        chosen_policy_move = tuple([i.item() for i in torch.unravel_index(chosen_index, (73, 64))])
        chosen_child = start_node.children[chosen_policy_move]
        chosen_move = chosen_child.prev_move
        return chosen_move, chosen_child, policy_datum


"""################################################################################
                            Section: Training
################################################################################"""

value_loss_fn = nn.MSELoss()
policy_loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

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

def stockfish_only_one_game(stockfish, printouts = False):
    input_data_list = []
    policy_data_list = []
    value_data_list = []
    curr_node = GameStateNode()
    pl_move_list = curr_node.generate_children()
    stockfish.set_position([])

    nn = 0
    while True:
        if printouts:
            curr_node.print()
        # print(stockfish.get_board_visual())

        # calc new moves
        stockfish_moves = stockfish.get_top_moves(200)

        # generate input_datum
        input_datum = curr_node.get_full_model_input()
        input_data_list.append(input_datum)

        # generate value_datum
        eval_cp = stockfish.get_evaluation()["value"]
        value_datum = tanh(eval_cp/700)
        value_data_list.append(value_datum)

        # generate policy_datum
        U32_moves = np.zeros(len(stockfish_moves), dtype=np.uint32)
        policy_moves = [(0,0) for _ in range(len(stockfish_moves))]
        probs = np.zeros(len(stockfish_moves))
        for i, stockfish_move in enumerate(stockfish_moves):
            U32_move = stockfish_move_to_U32_move(stockfish_move["Move"], pl_move_list, curr_node.board)
            U32_moves[i] = U32_move
            policy_move = get_policy_move(U32_move)
            policy_moves[i] = policy_move
            prob = stockfish_move["Centipawn"]
            if prob == None:
                prob = 10000 * stockfish_move["Mate"]
                if curr_node.board.turn == BLACK:
                    prob *= -1
            probs[i] = prob
        probs = np_softmax(probs)
        policy_datum = torch.zeros((1, 73, 64))
        for i, ap in enumerate(probs):
            a, b = policy_moves[i]
            policy_datum[0,a,b] = ap
        policy_data_list.append(policy_datum)

        # make move
        try:
            chosen_index = np.random.choice(len(probs), p=probs)
        except:
            print("ERROR with random choice of probs")
            print(f"{stockfish_moves = }")
            print(f"{probs = }")
            raise Exception()
        stockfish_move = stockfish_moves[chosen_index]
        # print(stockfish_move["Move"])
        stockfish.make_moves_from_current_position([stockfish_move["Move"]])
        # print(stockfish.get_board_visual())
        policy_move = policy_moves[chosen_index]
        curr_node = curr_node.children[policy_move]

        # break if game over
        pl_move_list = curr_node.generate_children()
        if curr_node.state in (DRAW, WHITE_WIN, BLACK_WIN):
            if printouts:
                curr_node.print()
            break

    input_data = torch.cat(input_data_list)
    policy_data = torch.cat(policy_data_list)
    value_data = torch.Tensor(value_data_list).view(-1,1)

    return input_data, policy_data, value_data

# @profile
def trainloop(stockfish):
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
    for i_game in range(n_games):
        if n_games < 10 or (i_game % (n_games//10) == 0):
            log = f"game: {str(i_game+1).rjust(len(str(n_games)))}/{n_games} | time: {pretty_datetime()}"
            print(log)

        input_data, policy_data, value_data = stockfish_only_one_game(stockfish)
        model_data_list.append(input_data) # shape: batch, channels, rows, cols
        policy_data_list.append(policy_data) # shape: batch, 73, 64
        value_data_list.append(value_data) # shape: batch, outputs

    input_data_tensor = torch.cat(model_data_list)
    policy_data_tensor = torch.cat(policy_data_list)
    value_data_tensor = torch.cat(value_data_list)
    print(f"size of input_data_tensor: {sys.getsizeof(input_data_tensor)/1000000} MB    {input_data_tensor.shape = }")
    print(f"size of policy_data_tensor: {sys.getsizeof(policy_data_tensor)/1000000} MB    {policy_data_tensor.shape = }")
    print(f"size of value_data_tensor: {sys.getsizeof(value_data_tensor)/1000000} MB    {value_data_tensor.shape = }")

    #### TRAIN THE NETWORKS
    dataset = TensorDataset(input_data_tensor, policy_data_tensor, value_data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    t0 = time.time()

    for t in range(n_epochs):
        model.train()

        for batch, (x,yp,yv) in enumerate(dataloader):
            # fwd pass
            p_pred, v_pred = model(x)
            loss = policy_loss_coeff*policy_loss_fn(p_pred, yp) + value_loss_coeff*value_loss_fn(v_pred, yv)

            # bwd pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # evaluate training and print
        model.eval()
        with torch.no_grad():
            p_pred, v_pred = model(x)
            loss = policy_loss_coeff*policy_loss_fn(p_pred, yp) + value_loss_coeff*value_loss_fn(v_pred, yv)
            loss = loss.item()

        log = f"epoch: {t}/{n_epochs} | loss: {round(loss, 5)} | time: {pretty_datetime()}"
        print(log)
        # logfile.write(log + '\n')

    print("Training Loop Complete.")
    print("----------------------------------------")
    save_objects()

    return


# @profile
def main(stockfish):
    start_datetime = pretty_datetime()
    n_loops = 0
    try:
        while True:
            if n_loops % 100 == 99:
                os.system("clear")
            print("----------------------------------------")
            print(f"started at {start_datetime}")
            print(f"loops complete: {n_loops}")
            print("----------------------------------------")
            print("TRAINING PARAMETERS")
            print(f"        {stockfish_elo = }")
            print(f"              {n_games = }")
            print(f"             {n_epochs = }")
            print(f"        {learning_rate = }")
            print(f"           {batch_size = }")
            print(f"    {policy_loss_coeff = }")
            print(f"     {value_loss_coeff = }")
            print("----------------------------------------")
            trainloop(stockfish)
            n_loops += 1
    finally:
        print("stopping")

if __name__ == "__main__":
    stockfish = Stockfish("/usr/games/stockfish")
    stockfish.set_elo_rating(stockfish_elo)
    stockfish.set_depth(10)
    main(stockfish)


    # for n in range(100):
    #     print(n)
    #     i, p, v = stockfish_only_one_game(stockfish)
    # print(i.shape)
    # print(p.shape)
    # print(v.shape)


    # stockfish.make_moves_from_current_position(["e2e4"])
    # print(stockfish.get_board_visual())


