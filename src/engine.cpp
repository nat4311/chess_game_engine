/* references:
 * https://www.youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef ENGINE_CPP
#define ENGINE_CPP

#include "engine.h"
#include "attacks.cpp"

/*////////////////////////////////////////////////////////////////////////////////
                               Section: BoardState
/*////////////////////////////////////////////////////////////////////////////////

class BoardState;
void reset_BoardState(BoardState* board_state);

class BoardState {
public:
    int turn;
    int castling_rights;
    int enpassant_sq;
    int halfmove;         // for 50 move rule - 100 halfmoves without a pawn move or capture is a draw
    U64 bitboards[12];
    U64 occupancies[3];

    BoardState() {
        reset_BoardState(this);
    }

};

void reset_BoardState(BoardState* board_state) {
    board_state->turn = WHITE;
    board_state->castling_rights = WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE | BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE;
    board_state->enpassant_sq = -1;
    board_state->halfmove = 0;

    board_state->bitboards[BLACK_PAWN] = A7 | B7 | C7 | D7 | E7 | F7 | G7 | H7;
    board_state->bitboards[BLACK_ROOK] = A8 | H8;
    board_state->bitboards[BLACK_KNIGHT] = B8 | G8;
    board_state->bitboards[BLACK_BISHOP] = C8 | F8;
    board_state->bitboards[BLACK_QUEEN] = D8;
    board_state->bitboards[BLACK_KING] = E8;

    board_state->bitboards[WHITE_PAWN] = A2 | B2 | C2 | D2 | E2 | F2 | G2 | H2;
    board_state->bitboards[WHITE_ROOK] = A1 | H1;
    board_state->bitboards[WHITE_KNIGHT] = B1 | G1;
    board_state->bitboards[WHITE_BISHOP] = C1 | F1;
    board_state->bitboards[WHITE_QUEEN] = D1;
    board_state->bitboards[WHITE_KING] = E1;

    board_state->occupancies[WHITE] = A1|B1|C1|D1|E1|F1|G1|H1|A2|B2|C2|D2|E2|F2|G2|H2;
    board_state->occupancies[BLACK] = A7|B7|C7|D7|E7|F7|G7|H7|A8|B8|C8|D8|E8|F8|G8|H8;
    board_state->occupancies[BOTH] = A1|B1|C1|D1|E1|F1|G1|H1|A2|B2|C2|D2|E2|F2|G2|H2|A7|B7|C7|D7|E7|F7|G7|H7|A8|B8|C8|D8|E8|F8|G8|H8;
}

void print_BoardState(BoardState* board_state) {
    printf("==============================\n\n");
    printf("    A  B  C  D  E  F  G  H\n\n");
    for (int y=0; y<8; y++) {
        printf("%d   ", 8-y);
        for (int x=0; x<8; x++) {
            U64 sq = 1ULL << (8*y + x);
            int piece_found = -1;
            for (int piece_type=0; piece_type<12; piece_type++){
                if (board_state->bitboards[piece_type] & sq) {
                    piece_found = piece_type;
                    break;
                }
            }
            printf("%s  ", (piece_found == -1) ? "." : unicode_pieces[piece_found]);
        }
        printf(" %d\n", 8-y);
    }
    printf("\n    A  B  C  D  E  F  G  H\n\n");
    printf("           turn: %s\n", board_state->turn == WHITE ? "white" : "black");
    printf("castling_rights: %c%c%c%c\n",
           board_state->castling_rights & WHITE_CASTLE_KINGSIDE ? 'K' : '-',
           board_state->castling_rights & WHITE_CASTLE_QUEENSIDE ? 'Q' : '-',
           board_state->castling_rights & BLACK_CASTLE_KINGSIDE ? 'k' : '-',
           board_state->castling_rights & BLACK_CASTLE_QUEENSIDE ? 'q' : '-');
    printf("   enpassant_sq: %s\n", board_state->enpassant_sq == -1 ? "none" : sq_str[board_state->enpassant_sq]);
    printf("       halfmove: %d\n\n", board_state->halfmove);
}

/*////////////////////////////////////////////////////////////////////////////////
                             Section: init and main
/*////////////////////////////////////////////////////////////////////////////////

void init_engine() {
    init_pawn_attacks();
    init_knight_attacks();
    init_king_attacks();
    init_bishop_attacks();
    init_rook_attacks();
}

int main() {
    init_engine();
    BoardState board;
    board.halfmove = 100;
    print_BoardState(&board);

    reset_BoardState(&board);
    print_BoardState(&board);

    return 0;
}

#endif
