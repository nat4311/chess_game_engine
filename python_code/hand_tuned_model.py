import game_engine
from utils import pretty_time_elapsed, pretty_datetime
from constants import WHITE, BLACK, WHITE_WIN, BLACK_WIN, DRAW, NOTOVER

"""################################################################################
                    Section: GameStateNode
################################################################################"""

class GameStateNode:
    def __init__(self, parent=None, board=None, fen=None):
        self.parent = parent
        self.children = dict() # indexed by U32_move
        self.moves = game_engine.MoveGenerator()
        self.state = None

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

        material_score = self.board.material_score()
        dp_score = self.board.doubled_pawn_score()
        mobility_score = len(self.children)
        enemy_mobility_score = self.board.enemy_mobility_score()
        print(f"{material_score = }")
        print(f"{dp_score = }")
        print(f"{mobility_score = }")
        print(f"{enemy_mobility_score = }")

        return self.board.material_score() + .5*self.board.doubled_pawn_score() + .1*(len(self.children) - self.board.enemy_mobility_score())

    def print(self):
        print("GameStateNode: \n")
        self.board.print()


if __name__ == "__main__":
    fen = "rnbqkbnr/p2ppppp/p7/p7/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    game = GameStateNode(fen=fen)
    game.print()
    game.generate_children()
    print(game.eval())
