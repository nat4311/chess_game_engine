from utils import pretty_time_elapsed, pretty_datetime
from constants import WHITE, BLACK, WHITE_WIN, BLACK_WIN, DRAW, NOTOVER
import game_engine

"""################################################################################
                    Section: GameStateNode
################################################################################"""

class GameStateNode:
    def __init__(self, parent=None, board=None):
        self.parent = parent
        self.children = dict() # indexed by U32_move
        self.moves = game_engine.MoveGenerator()
        self.state = None

        if board is None:
            self.board = game_engine.BoardState()
        else:
            self.board = board

    def generate_children(self):
        if self.board.halfmove == 100:
            self.state = DRAW
            return

        pl_move_list = self.moves.get_pl_move_list(self.board)
        for U32_move in pl_move_list:
            new_board = self.board.copy()
            if new_board.make(U32_move):
                new_node = GameStateNode(parent=self, board=new_board)
                self.children[U32_move] = new_node

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

    def eval(self):
        assert self.state is not None

        return self.board.material_score() + 

    def print(self):
        print("GameStateNode: \n")
        self.board.print()
        print(f"prev_move: {self.prev_move}")
