# overall
- use stockfish to train against instead of self play (bec it might take too long)
- baseline engine
    - create hand tuned eval function 
    - minimax and alpha-beta pruning to select moves
- stockfish style engine
    - use the same resnet
    - minimax and alpha-beta pruning to select moves
- optimization
    - profiling
    - GPU and CUDA?



-   A  B  C  D  E  F  G  H   -

8   0  1  2  3  so on .  .   8 
7   .  .  .  .  .  .  .  .   7 
6   .  .  x  .  x  .  .  .   6 
5   .  x  .  .  .  x  .  .   5 
4   .  .  .  ♞  .  .  .  .   4
3   .  x  .  .  .  x  .  .   3
2   .  .  x  .  x  .  .  .   2
1   .  .  .  .  .  .  .  .   1

-   A  B  C  D  E  F  G  H   -

int square = 35
U64 knight_attacks[35] = 0x0014220022140000
U64 white_knight_positions






# bug
new GameStateNode1 failed to create: make(prev_move) failedprev_board:
    A  B  C  D  E  F  G  H

8   ♖  ♘  ♗  .  ♕  ♗  ♘  ♖   8
7   ♙  ♙  ♙  .  ♟  ♔  ♙  .   7
6   .  .  .  .  .  .  .  .   6
5   .  .  .  .  .  ♙  .  ♙   5
4   .  .  .  .  .  .  .  ♟   4
3   .  .  .  .  .  .  .  .   3
2   ♟  ♟  ♟  ♟  .  ♟  ♟  .   2
1   ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜   1

    A  B  C  D  E  F  G  H

               turn: white
    castling_rights: KQ--
       enpassant_sq: none
           halfmove: 1

            turn_no: 6

occupancies both
    A  B  C  D  E  F  G  H

8   1  1  1  .  1  1  1  1   8
7   1  1  1  .  1  1  1  .   7
6   .  .  .  .  .  .  .  .   6
5   .  .  .  .  .  1  .  1   5
4   .  .  .  .  .  .  .  1   4
3   .  .  .  .  .  .  .  .   3
2   1  1  1  1  .  1  1  .   2
1   1  1  1  1  1  1  1  1   1

    A  B  C  D  E  F  G  H
occupancies white
    A  B  C  D  E  F  G  H

8   .  .  .  .  .  .  .  .   8
7   .  .  .  .  1  .  .  .   7
6   .  .  .  .  .  .  .  .   6
5   .  .  .  .  .  .  .  .   5
4   .  .  .  .  .  .  .  1   4
3   .  .  .  .  .  .  .  .   3
2   1  1  1  1  .  1  1  .   2
1   1  1  1  1  1  1  1  1   1

    A  B  C  D  E  F  G  H
occupancies black
    A  B  C  D  E  F  G  H

8   1  1  1  .  1  1  1  1   8
7   1  1  1  .  .  1  1  .   7
6   .  .  .  .  .  .  .  .   6
5   .  .  .  .  .  1  .  1   5
4   .  .  .  .  .  .  .  .   4
3   .  .  .  .  .  .  .  .   3
2   .  .  .  .  .  .  .  .   2
1   .  .  .  .  .  .  .  .   1

    A  B  C  D  E  F  G  H
prev_move:
       print_move: 1366294860
       piece type: P
        source_sq: e7
        target_sq: f8
   promotion_type: Q
        promotion: 1
 double_pawn_push: 0
          capture: 1
enpassant_capture: 0
  castle_kingside: 0
 castle_queenside: 0

terminate called after throwing an instance of 'int'
make: *** [makefile:20: hand] Aborted (core dumped
