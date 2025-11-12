import game_engine

board = game_engine.BoardState()
moves = game_engine.MoveGenerator()
board.reset()

moves.generate_pl_moves(board)
moves.print_pl_moves()
# m = moves.get_pl_move_list(board)
# for move in m:
#     game_engine.print_move(move, False)

board.print()
