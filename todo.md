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


# bug
new GameStateNode1 failed to create: make(prev_move) failedprev_board:
    A  B  C  D  E  F  G  H

8   .  .  .  .  .  .  .  .   8
7   ♙  ♖  ♟  ♚  .  ♙  ♔  .   7
6   .  .  .  .  .  .  .  .   6
5   .  .  .  .  ♖  .  .  ♙   5
4   .  ♗  .  .  .  .  .  .   4
3   .  .  .  .  .  .  .  .   3
2   .  ♟  ♟  ♟  .  .  .  ♟   2
1   .  ♜  ♝  .  .  ♝  .  .   1

    A  B  C  D  E  F  G  H

               turn: white
    castling_rights: ----
       enpassant_sq: none
           halfmove: 3

            turn_no: 28

occupancies both
    A  B  C  D  E  F  G  H

8   .  .  .  .  .  .  .  .   8
7   1  1  1  1  .  1  1  .   7
6   .  .  .  .  .  .  .  .   6
5   .  .  .  .  1  .  .  1   5
4   .  1  .  .  .  .  .  .   4
3   .  .  .  .  .  .  .  .   3
2   .  1  1  1  .  .  .  1   2
1   .  1  1  .  .  1  .  .   1

    A  B  C  D  E  F  G  H
occupancies white
    A  B  C  D  E  F  G  H

8   .  .  .  .  .  .  .  .   8
7   .  .  1  1  .  .  .  .   7
6   .  .  .  .  .  .  .  .   6
5   .  .  .  .  .  .  .  .   5
4   .  .  .  .  .  .  .  .   4
3   .  .  .  .  .  .  .  .   3
2   .  1  1  1  .  .  .  1   2
1   .  1  1  .  .  1  .  .   1

    A  B  C  D  E  F  G  H
occupancies black
    A  B  C  D  E  F  G  H

8   .  .  .  .  .  .  .  .   8
7   1  1  .  .  .  1  1  .   7
6   .  .  .  .  .  .  .  .   6
5   .  .  .  .  1  .  .  1   5
4   .  1  .  .  .  .  .  .   4
3   .  .  .  .  .  .  .  .   3
2   .  .  .  .  .  .  .  .   2
1   .  .  .  .  .  .  .  .   1

    A  B  C  D  E  F  G  H
prev_move:
       print_move: 1342177418
       piece type: P
        source_sq: c7
        target_sq: c8
   promotion_type: Q
        promotion: 1
 double_pawn_push: 0
          capture: 0
enpassant_capture: 0
  castle_kingside: 0
 castle_queenside: 0

terminate called after throwing an instance of 'int'
Aborted (core dumped)g
