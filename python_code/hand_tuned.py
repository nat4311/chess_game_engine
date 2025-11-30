from utils import pretty_time_elapsed, pretty_datetime
from constants import WHITE, BLACK, WHITE_WIN, BLACK_WIN, DRAW, NOTOVER
import game_engine

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
