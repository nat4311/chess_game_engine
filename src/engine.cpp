/* references:
 * https://www.youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "engine.h"
#include "attacks.cpp"

class BoardState {
public:
    int turn;
    int castling_rights;
    int enpassant_sq;
    int halfmove;         // for 50 move rule - 100 halfmoves without a pawn move or capture is a draw
    U64 bitboards[12];
    U64 occupancies[3];

    BoardState() {
        reset_board();
    }

    void reset_board() {
        turn = WHITE;
        castling_rights = WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE | BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE;
        enpassant_sq = -1;
        halfmove = 0;

        bitboards[BLACK_PAWN] = A7 | B7 | C7 | D7 | E7 | F7 | G7 | H7;
        bitboards[BLACK_ROOK] = A8 | H8;
        bitboards[BLACK_KNIGHT] = B8 | G8;
        bitboards[BLACK_BISHOP] = C8 | F8;
        bitboards[BLACK_QUEEN] = D8;
        bitboards[BLACK_KING] = E8;

        bitboards[WHITE_PAWN] = A2 | B2 | C2 | D2 | E2 | F2 | G2 | H2;
        bitboards[WHITE_ROOK] = A1 | H1;
        bitboards[WHITE_KNIGHT] = B1 | G1;
        bitboards[WHITE_BISHOP] = C1 | F1;
        bitboards[WHITE_QUEEN] = D1;
        bitboards[WHITE_KING] = E1;

        occupancies[WHITE] = A1|B1|C1|D1|E1|F1|G1|H1|A2|B2|C2|D2|E2|F2|G2|H2;
        occupancies[BLACK] = A7|B7|C7|D7|E7|F7|G7|H7|A8|B8|C8|D8|E8|F8|G8|H8;
        occupancies[BOTH] = A1|B1|C1|D1|E1|F1|G1|H1|A2|B2|C2|D2|E2|F2|G2|H2|A7|B7|C7|D7|E7|F7|G7|H7|A8|B8|C8|D8|E8|F8|G8|H8;
    }

    void print_board() {
        printf("==============================\n\n");
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
        printf("\n    A  B  C  D  E  F  G  H\n\n");
        printf("           turn: %s\n", turn == WHITE ? "white" : "black");
        printf("castling_rights: %c%c%c%c\n",
               castling_rights & WHITE_CASTLE_KINGSIDE ? 'K' : '-',
               castling_rights & WHITE_CASTLE_QUEENSIDE ? 'Q' : '-',
               castling_rights & BLACK_CASTLE_KINGSIDE ? 'k' : '-',
               castling_rights & BLACK_CASTLE_QUEENSIDE ? 'q' : '-');
        printf("   enpassant_sq: %s\n", enpassant_sq == -1 ? "none" : sq_str[enpassant_sq]);
        printf("       halfmove: %d\n\n", halfmove);
    }

};

void init_engine() {
    init_pawn_attacks();
    init_knight_attacks();
    init_king_attacks();
    init_bishop_attacks();
}

int main() {
    init_engine();
    BoardState board;
    board.print_board();

    return 0;
}
