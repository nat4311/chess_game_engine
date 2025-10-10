#include "engine.h"
#include <stdio.h>

void print_bitboard(U64 bitboard) {
    printf("    A  B  C  D  E  F  G  H\n\n");
    for (int y=0; y<8; y++) {
        printf("%d   ", 8-y);
        for (int x=0; x<8; x++) {
            U64 sq = 1ULL << (8*y + x);
            printf("%d  ", (bitboard & sq)? 1 : 0);
        }
        printf(" %d\n", 8-y);
    }
    printf("\n    A  B  C  D  E  F  G  H\n");
}

class BoardState {
public:
    U64 bitboards[12];

    BoardState() {
        reset_board();
    }

    void reset_board() {
        bitboards[BLACK_PAWN] = a7 | b7 | c7 | d7 | e7 | f7 | g7 | h7;
        bitboards[BLACK_ROOK] = a8 | h8;
        bitboards[BLACK_KNIGHT] = b8 | g8;
        bitboards[BLACK_BISHOP] = c8 | f8;
        bitboards[BLACK_QUEEN] = d8;
        bitboards[BLACK_KING] = e8;

        bitboards[WHITE_PAWN] = a2 | b2 | c2 | d2 | e2 | f2 | g2 | h2;
        bitboards[WHITE_ROOK] = a1 | h1;
        bitboards[WHITE_KNIGHT] = b1 | g1;
        bitboards[WHITE_BISHOP] = c1 | f1;
        bitboards[WHITE_QUEEN] = d1;
        bitboards[WHITE_KING] = e1;
    }

    void print_board() {
        printf("    A  B  C  D  E  F  G  H\n\n");
        for (int y=0; y<8; y++) {
            printf("%d   ", 8-y);
            for (int x=0; x<8; x++) {
                U64 sq = 1ULL << (8*y + x);
                int piece_found = -1;
                for (int piece_type=0; piece_type<12; piece_type++){
                    if (bitboards[piece_type] & sq) {
                        piece_found = piece_type;
                        break;
                    }
                }
                printf("%s  ", (piece_found == -1) ? "." : unicode_pieces[piece_found]);
            }
            printf(" %d\n", 8-y);
        }
        printf("\n    A  B  C  D  E  F  G  H\n");
    }

};

int main() {
    BoardState board;
    board.print_board();

    return 0;
}
