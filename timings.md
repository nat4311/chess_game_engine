# --------------------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                                   #
#                                                      PERFT TIMINGS                                                                #
#                                                                                                                                   #
# --------------------------------------------------------------------------------------------------------------------------------- #

perft position 1

    A  B  C  D  E  F  G  H

8   ♖  ♘  ♗  ♕  ♔  ♗  ♘  ♖   8
7   ♙  ♙  ♙  ♙  ♙  ♙  ♙  ♙   7
6   .  .  .  .  .  .  .  .   6
5   .  .  .  .  .  .  .  .   5
4   .  .  .  .  .  .  .  .   4
3   .  .  .  .  .  .  .  .   3
2   ♟  ♟  ♟  ♟  ♟  ♟  ♟  ♟   2
1   ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜   1

    A  B  C  D  E  F  G  H

               turn: white
    castling_rights: KQkq
       enpassant_sq: none
           halfmove: 0

            turn_no: 1

no compiler optimization
depth 1 time: 3 us
depth 2 time: 21 us
depth 3 time: 445 us
depth 4 time: 9 ms
depth 5 time: 242 ms
depth 6 time: 5 s
depth 7 time: 159 s

with -O3
depth 1 time: 3 us
depth 2 time: 14 us
depth 3 time: 278 us
depth 4 time: 6 ms
depth 5 time: 153 ms
depth 6 time: 3 s
depth 7 time: 99 s

