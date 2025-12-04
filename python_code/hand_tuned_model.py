import game_engine
import cProfile
import pstats
import time
import timeit
from math import inf
from utils import pretty_time_elapsed, pretty_datetime
from constants import *

class GameStateNode0:
    """
    saves children as part of each node, uses way too much RAM

    --------------------------------------------------------------------------------

    single thread timings

    minimax timings from start position, no move ordering
    1 3.437511622905731e-05
    2 0.009956791065633297
    3 0.022661556489765644
    4 1.4706027386710048
    5 1.5414285976439714
    6 dnf (RAM overflow)

    after moving count_legal_moves and legal move generation (for minimax) to c++
    1 3.6990270018577576e-05
    2 0.010196661576628685
    3 0.02238989993929863
    4 1.4012887263670564
    5 1.4680421631783247
    6 dnf (RAM overflow)

    """
    def __init__(self, parent=None, board=None, prev_move = None, fen=None):
        self.parent = parent
        self.children = dict() # indexed by U32_move
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

        pl_move_list = self.board.get_pl_move_list()
        for U32_move in pl_move_list:
            new_board = self.board.copy()
            if new_board.make(U32_move):
                new_node = GameStateNode0(parent=self, board=new_board, prev_move=U32_move)
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
            return 10000 - self.board.turn_no
        if self.state == BLACK_WIN:
            return -10000 + self.board.turn_no

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
        print("GameStateNode0: \n")
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


class GameStateNode1:
    """
    dont store children dict

    --------------------------------------------------------------------------------

    single thread minimax timings from start position

    in python, no move ordering
    depth 1, time = 0.0006825476884841919 s
    depth 2, time = 0.007834275253117085 s
    depth 3, time = 0.017839105799794197 s
    depth 4, time = 0.764815291389823 s
    depth 5, time = 0.8085004044696689 s
    depth 6, time = 162.43249557446688 s
    depth 7, time = 69.25481910258532 s
    depth 8, dnf

    in python, after moving count_legal_moves and legal move generation (for minimax) to c++
    depth 1, time = 8.558016270399094e-05 s
    depth 2, time = 0.0007701413705945015 s
    depth 3, time = 0.0020072944462299347 s
    depth 4, time = 0.07913886103779078 s
    depth 5, time = 0.08957733120769262 s
    depth 6, time = 15.086321806535125 s
    depth 7, time = 7.631460613571107 s
    depth 8, dnf

    in c++, no move ordering
    depth 1, time = 1.7e-05 s
    depth 2, time = 0.000131 s
    depth 3, time = 0.000345 s
    depth 4, time = 0.019428 s
    depth 5, time = 0.022782 s
    depth 6, time = 4.34886 s
    depth 7, time = 1.94325 s
    depth 8, time = 433.817 s
    depth 9, dnf

    in python, move ordering with capture score
    depth 1, time = 0.0001286221668124199
    depth 2, time = 0.0005760975182056427
    depth 3, time = 0.008842403069138527
    depth 4, time = 0.032893081195652485
    depth 5, time = 0.37780505511909723
    depth 6, time = 2.4619059208780527
    depth 7, time = 26.00939543824643

    in c++, move ordering with capture score
    depth 1, time = 1.9e-05 s
    depth 2, time = 7.6e-05 s
    depth 3, time = 0.00112 s
    depth 4, time = 0.006624 s
    depth 5, time = 0.072729 s
    depth 6, time = 0.582312 s
    depth 7, time = 4.93559 s
    depth 8, time = 35.927 s
    depth 9, time = 191.175 s
    depth 10, time = 1529.02 s

    c++, open mp
    depth 1, time = 0.000461 s
    depth 2, time = 8.4e-05 s
    depth 3, time = 0.000601 s
    depth 4, time = 0.002885 s
    depth 5, time = 0.021282 s
    depth 6, time = 0.10138 s
    depth 7, time = 1.39301 s
    depth 8, time = 7.34563 s
    depth 9, time = 70.1452 s

    """
    def __init__(self, board=None, prev_move = None, fen=None):
        self.state = None
        self.prev_move = prev_move

        if board is None:
            self.board = game_engine.BoardState()
            if fen is not None:
                if fen[-1]!="\n":
                    fen += "\n"
                self.board.load(fen)
        else:
            self.board = board
            assert fen is None

    def count_legal_moves(self):
        n_legal_moves = self.board.get_l_move_count()
        self.state = self.board.get_state()

        return n_legal_moves

    def eval(self):
        mobility_score = self.count_legal_moves()

        if self.state == DRAW:
            return 0
        elif self.state == WHITE_WIN:
            return 10000 - self.board.turn_no
        elif self.state == BLACK_WIN:
            return -10000 + self.board.turn_no

        # material_score = self.board.material_score()
        # dp_score = self.board.doubled_pawn_score()
        # mobility_score = len(self.children)
        # enemy_mobility_score = self.board.enemy_mobility_score()
        # print(f"{material_score = }")
        # print(f"{dp_score = }")
        # print(f"{mobility_score = }")
        # print(f"{enemy_mobility_score = }")

        return self.board.material_score() + .5*self.board.doubled_pawn_score() + .1*(mobility_score - self.board.enemy_mobility_score())

    def print(self):
        print("GameStateNode1: \n")
        self.board.print()


    def minimax(self, depth):
        maximizing_player = (self.board.turn == WHITE)
        best_child, _ = self._minimax(self, depth, maximizing_player=maximizing_player)
        return best_child

    def _minimax(self, node, depth, alpha=-inf, beta=inf, maximizing_player=True):
        if node.state is None:
            node.count_legal_moves()
            node.board.sort_l_moves()
        if depth == 0 or node.state in {DRAW, WHITE_WIN, BLACK_WIN}:
            return node, node.eval()

        if maximizing_player:
            max_eval = -inf
            best_child = None
            for U32_move in node.board.get_l_move_list():
                new_board = node.board.copy()
                if not new_board.make(U32_move):
                    raise Exception("l_move failed")
                child = GameStateNode1(board=new_board, prev_move=U32_move)
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
            for U32_move in node.board.get_l_move_list():
                new_board = node.board.copy()
                if not new_board.make(U32_move):
                    raise Exception("l_move failed")
                child = GameStateNode1(board=new_board, prev_move=U32_move)
                _, eval_child = self._minimax(child, depth - 1, alpha, beta, True)
                if eval_child < min_eval:
                    min_eval = eval_child
                    best_child = child
                beta = min(beta, eval_child)
                if beta <= alpha:
                    break
            return best_child, min_eval

def test_minimax1_timings():
    game = GameStateNode1()
    game.minimax(1)

    for i in range(1,9):
        f = lambda: game.minimax(i)
        t = timeit.timeit(f, number=1)
        print(f"depth {i}, time = {t}")

# game = GameStateNode1(fen="7k/bpp3pp/3pq1pp/8/8/5P2/PPPPQPPP/RNB1KBNR w KQ - 0 1\n")
# game.minimax(1)
def main():
    test_minimax1_timings()
    # for i in range(1,7):
    #     print(i)
    #     game.minimax(i)

    # while True:
    #     game.print()
    #     if game.state in (DRAW, WHITE_WIN, BLACK_WIN):
    #         if game.state == DRAW:
    #             print("draw")
    #         if game.state == WHITE_WIN:
    #             print("white wins")
    #         if game.state == BLACK_WIN:
    #             print("black wins")
    #         break
    #
    #     print("----------------------")
    #     t0 = time.time()
    #     best_child = game.minimax(6)
    #     print("time to choose move:", pretty_time_elapsed(t0, time.time()))
    #     game = best_child



if __name__ == "__main__":
    t0 = time.time()
    with cProfile.Profile() as profile:
        main()
    print(pretty_time_elapsed(t0, time.time()))

    results = pstats.Stats(profile)
    results.sort_stats('tottime').print_stats(10)
    results.dump_stats("results.prof")
