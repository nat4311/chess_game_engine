"""
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --log-file=valgrind.log python3 python_code/memory_profiling.py
"""

import game_engine
from alphazero import GameStateNode

game = GameStateNode()



