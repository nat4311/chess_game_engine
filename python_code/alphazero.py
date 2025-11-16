import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from math import inf, sqrt, log
from numpy.random import dirichlet
import time
from copy import deepcopy
import game_engine

# enums
WHITE = 0
BLACK = 1
WHITE_WIN = 1
BLACK_WIN = -1
DRAW = 0
NOTOVER = 2

"""################################################################################
                    Section: Residual Neural Network (ResNet)
################################################################################"""

batch_size = 1

# 14*8 (12 bitboards + 2 repetition, 1 current + 7 past) + 7 (1 turn + 1 total_moves + 4 castling + 1 halfmove)
time_history = 7
feature_channels = 14*(time_history+1) + 7

# the following came from https://www.chessprogramming.org/AlphaZero#Network_Architecture
default_filters = 64 # 256
default_kernel_size = 3
res_block_layers = 6 # 19


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

        v = self.v_conv(x)
        v = self.v_bn(v)
        v = F.relu(v)
        v = v.view(batch_size, -1)
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


def test_net_shapes():
    model = ResNet()
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

    # To get final policy as (64 x 73), reshape and permute policy output:
    # policy_output shape is (batch, 73, 8, 8)
    # Reshape to (batch, 73, 64) then permute to (batch, 64, 73)
    policy_reshaped = policy_output.reshape(batch_size, 73, 64)
    print()
    print(f"Policy reshaped shape (should be {batch_size} x 73 x 64):", policy_reshaped.shape)

    # Value should be (batch, 1), verify the scalar value
    print()
    print("Value output:", value_output)

# test_net_shapes()




"""################################################################################
                    Section: GameStateNode
################################################################################"""

class GameStateNode:
    def __init__(self, parent=None, board=None, prev_move=0, full_model_input=None, generate_children=False):
        self.parent = parent
        self.children = set()
        self.prev_move = prev_move
        self.moves = game_engine.MoveGenerator()
        self.prior = 0
        self.value_sum = 0
        self.n_visits = 0
        self.full_model_input = full_model_input

        if parent is None:
            self.board = game_engine.BoardState()
        else:
            self.board = board
        if generate_children:
            self.generate_children()
        else:
            self.state = None

    def generate_children(self):
        if self.board.halfmove == 100:
            self.state = DRAW
            return

        for move in self.moves.get_pl_move_list(self.board):
            new_board = self.board.copy()
            if new_board.make(move):
                new_node = GameStateNode(self, new_board, move)
                self.children.add(new_node)
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

    def print_pl_moves(self):
        self.moves.print_pl_moves()

    def new_node(self, move):
        new_board = self.board.copy()
        if new_board.make(move):
            return GameStateNode(self, new_board, move)
            # TODO: need to add the prior
        else:
            return None

    # TODO: the mcts functions are expecting the actual model input - this is partial
    def get_partial_model_input(self):
        return torch.Tensor(self.board.get_partial_model_input()).view(1,-1,8,8)
    
    def get_full_model_input(self):
        if self.full_model_input is None:
            self.full_model_input = torch.cat(self.parent.full_model_input[0, 14:-7, :, :], self.get_partial_model_input())
        return self.full_model_input

    def print(self):
        print("GameStateNode: \n")
        self.board.print()
        print(f"prev_move: {self.prev_move}")

    def make_root(self):
        self.parent = None
        self.value_sum = 0
        self.n_visits = 0

# moves.generate_pl_moves(board)
# moves.print_pl_moves()


# struct GameStateNodeAlphazero {
#     GameStateNodeAlphazero(GameStateNodeAlphazero* parent = nullptr, U32 prev_move = 0):
#         parent(parent),
#         prev_move(prev_move),
#         n_children(0)
#     {
#         if (parent) { // new potential child node
#             this->board_state = parent->board_state;
#             this->valid = BoardState::make(&this->board_state, prev_move);
#         }
#         else { // start position
#             this->board_state = BoardState();
#             this->valid = true;
#             this->prior = 0;
#         }
#     }
#
#     // reset the mcts variables for so we can reuse this node as a root node
#     static void make_root(GameStateNodeAlphazero* node) {
#         node->prior = 0;
#         node->parent = nullptr;
#         // node->prev_move = 0;
#     }
#
#     static void add_pl_child(GameStateNodeAlphazero* node, GameStateNodeAlphazero* child) {
#         if (child->valid) {
#             node->children[node->n_children++] = child;
#             assert (node->n_children < max_n_children); // todo: move this to expand function in python
#         }
#     }
#
# };









"""################################################################################
                    Section: Monte Carlo Tree Search (MCTS)
################################################################################"""

mcts_n_sims = 800

# TODO: convert connect4->chess
def MCTS(root_node: GameStateNode, model: ResNet):
    """
        Performs monte carlo tree search from a parent GameStateNode.
        The child of that parent with the most visits is selected and returned.
        -----------------------------
        INPUTS
        *start node        GameStateNode     represents the gamestate to begin MCTS at    *changes state in this function
        model              torch.nn          outputs p and v heads
        -----------------------------
        SIDE EFFECTS
        - updates the attributes of start_node and its children
        - use n_visits of each child to choose a move
    """

    for n in range(mcts_n_sims):
        curr_node = root_node
        while True:
            if curr_node.n_visits == 0 or curr_node.state in (WHITE_WIN, BLACK_WIN, DRAW): # only rollout for leaf nodes
# BOOKMARK - changed everything above this in this function
                rollout(curr_node, model)
                break
            else:
                if curr_node.n_visits == 1: # only expand if need to
                    expand(curr_node, model)
                curr_node = select_child(curr_node)
                assert curr_node is not None

    return

# todo: this runs, need to test correctness
def rollout(curr_node: GameStateNode, model: ResNet) -> int:
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
        _, leaf_node_value_estimate = model(leaf_node.get_full_model_input())
        leaf_node_value_estimate = leaf_node_value_estimate.item()

    # backprop leaf_node_value_estimate to all parent nodes
    while True:
        curr_node.value_sum += leaf_node_value_estimate
        curr_node.n_visits += 1
        curr_node = curr_node.parent
        if curr_node is None:
            break

    return leaf_node_value_estimate


# TODO: convert connect4->chess
def expand(curr_node, model):

    action_probs,_ = model(curr_node.get_full_model_input())

    for move in curr_node.valid_moves():
        prior = action_probs[0][move].item()
        new_node = curr_node.new_node(move, prior)
        curr_node.children.add(new_node)

# TODO: convert connect4->chess
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

    # todo: confirm dirichlet noise is working
    # todo: make sure policy output sums to 1 for the valid moves
    noise = dirichlet([alpha_dirichlet for _ in range(7)], 1)[0]
    for i, child in enumerate(parent_node.children):
        if child.n_visits == 0:
            explore = inf
            return child

        d = noise[i]
        p = child.prior
        p_prime = p*x_dirichlet + (1-x_dirichlet)*d
        explore = p_prime * ucb_exploration_constant * sqrt(parent_node.n_visits - 1)/(child.n_visits+1)

        exploit = parent_node.turn * child.value_sum/child.n_visits

        UCB1 = exploit + explore
        if UCB1 > best_UCB1:
            best_UCB1 = UCB1
            best_child = child

    return best_child

# TODO: convert connect4->chess
def choose_move(start_node, model, greedy) -> int:
    """
        Uses monte carlo tree search and value/policy networks to choose a move.
        ------------------------
        INPUTS
        start_node*    GameStateNode       state of the game    *changes state inside this function
        model          ResNet              outputs the policy and value heads
        greedy         bool                if greedy select move with most visits, else use visit distribution
        ------------------------
        OUTPUTS
        move           policy_output       tuple in range (73, 8, 8) -> need to convert with temporary cache
        ------------------------
        SIDE EFFECTS
        - updates n_visits and other attributes of start_node and its children
    """

# BOOKMARK - changed everything above this in this function
    MCTS(start_node, model)

    # todo: make sure this works
    if greedy:
        most_visits = 0
        most_visited_move = None
        for child in start_node.children:
            if child.n_visits > most_visits:
                most_visited_move = child.prev_move
                most_visits = child.n_visits
        return most_visited_move
    else:
        visits = [0 for _ in range(7)]
        for child in start_node.children:
            visits[child.prev_move] = child.n_visits
        distribution = [v**(1/exploration_temperature) for v in visits]
        s = sum(distribution)
        distribution = [v/s for v in distribution]
        return random.choices(range(7), distribution)[0]
#


"""################################################################################
                            Section: Training
################################################################################"""

# TODO: convert connect4->chess
def self_play_one_game(model: ResNet):
    """
        INPUTS
            model          torch.nn        full res net (p and v heads)
        -------------------------
        OUTPUTS
            input data     torch.Tensor    (Nx8x8) where N is (8+halfturns)*14 + 7 -> index like [(8+i)*14:(8+i)*14+7, :, :]
            policy data    torch.Tensor    (Mx73x8x8) where M is halfturns         -> index like [i, :, :]
            result         float           game outcome (+1 white won, -1 black won)
    """

    curr_node = GameStateNode()
    input_data = torch.zeros((feature_channels,8,8))
    input_data[-21:, :, :] = curr_node.get_partial_model_input()
    curr_node.full_model_input = input_data

    policy_data = torch.Tensor([])

    while True:
        # BOOKMARK - changed everything above this in this function
        move = choose_move(curr_node, model, greedy=False)
        policy_datum = torch.zeros((1,7))
        for child in curr_node.children:
            i = child.prev_move
            assert i in range(7)
            ap = child.n_visits / (curr_node.n_visits-1)
            policy_datum[0][i] = ap
        policy_data = torch.cat((policy_data, policy_datum))
        
        # todo: why are all the priors 1
        prior = policy_datum[0][move].item()
        curr_node = curr_node.new_node(move, prior)
        if curr_node.state in (X,O,DRAW):
            result = curr_node.state
            break

    return input_data, policy_data, result


