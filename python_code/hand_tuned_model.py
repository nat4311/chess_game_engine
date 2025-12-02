import game_engine
import time
from math import inf
from utils import pretty_time_elapsed, pretty_datetime
from constants import WHITE, BLACK, WHITE_WIN, BLACK_WIN, DRAW, NOTOVER

"""################################################################################
                    Section: GameStateNode
################################################################################"""

class GameStateNode:
    def __init__(self, parent=None, board=None, prev_move = None, fen=None):
        self.parent = parent
        self.children = dict() # indexed by U32_move
        self.moves = game_engine.MoveGenerator()
        self.state = None
        self.prev_move = prev_move

        if board is None:
            self.board = game_engine.BoardState()
            b = game_engine.BoardState()
            if fen is not None:
                self.board.load(fen)
        else:
            self.board = board
            assert fen is None

    def generate_children(self):
        if self.board.halfmove == 100:
            self.state = DRAW
            return

        pl_move_list = self.moves.get_pl_move_list(self.board)
        for U32_move in pl_move_list:
            new_board = self.board.copy()
            if new_board.make(U32_move):
                new_node = GameStateNode(parent=self, board=new_board, prev_move=U32_move)
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
        if self.state is None:
            self.generate_children()

        if self.state == DRAW:
            return 0
        if self.state == WHITE_WIN:
            return 1000
        if self.state == BLACK_WIN:
            return -1000

        # material_score = self.board.material_score()
        # dp_score = self.board.doubled_pawn_score()
        # mobility_score = len(self.children)
        # enemy_mobility_score = self.board.enemy_mobility_score()
        # print(f"{material_score = }")
        # print(f"{dp_score = }")
        # print(f"{mobility_score = }")
        # print(f"{enemy_mobility_score = }")

        return self.board.material_score() + .5*self.board.doubled_pawn_score() + .1*(len(self.children) - self.board.enemy_mobility_score())

    def print(self):
        print("GameStateNode: \n")
        self.board.print()


    def minimax(self, depth):
        if self.state is None:
            self.generate_children()

        maximizing_player = (self.board.turn == WHITE)
        best_child, _ = self._minimax(self, depth, maximizing_player=maximizing_player)
        assert best_child is not None
        return best_child

    def _minimax(self, node, depth, alpha=-inf, beta=inf, maximizing_player=True):
        if node.state is None:
            node.generate_children()

        if depth == 0 or node.state in {DRAW, WHITE_WIN, BLACK_WIN}:
            return node, node.eval()

        if maximizing_player:
            max_eval = -inf
            best_child = None
            for child in node.children.values():
                _, eval_child = self._minimax(child, depth - 1, alpha, beta, False)
                if eval_child > max_eval:
                    max_eval = eval_child
                    best_child = child
                alpha = max(alpha, eval_child)
                if beta <= alpha:
                    break
            return best_child, max_eval
        else:
            min_eval = inf
            best_child = None
            for child in node.children.values():
                _, eval_child = self._minimax(child, depth - 1, alpha, beta, True)
                if eval_child < min_eval:
                    min_eval = eval_child
                    best_child = child
                beta = min(beta, eval_child)
                if beta <= alpha:
                    break
            return best_child, min_eval



if __name__ == "__main__":
    # fen = "rnbqkbnr/p2ppppp/p7/p7/8/8/PPPPqPPP/RNBQKBNR w KQkq - 0 1"
    # game = GameStateNode(fen=fen)
    # game.print()
    # game.board.enemy_mobility_score()
    # game.eval()

    ###########    single position minimax
    game = GameStateNode()
    t0 = time.time()
    game.minimax(5).print()
    print(pretty_time_elapsed(t0, time.time()))

    ########    play self with minimax
    # game = GameStateNode()
    # while True:
    #     game.print()
    #     best_child = game.minimax(10)
    #     game = best_child
    #     time.sleep(1)
