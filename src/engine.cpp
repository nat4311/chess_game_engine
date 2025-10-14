/* references:
 * https://www.youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "engine.h"
#include "attacks.h"
#include <assert.h>

/*////////////////////////////////////////////////////////////////////////////////
                               Section: BoardState
/*////////////////////////////////////////////////////////////////////////////////

struct BoardState {
    // side (WHITE or BLACK)
    int turn;

    // WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE | BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE
    int castling_rights;

    // 0-63 (64 for no_sq)
    int enpassant_sq;

    // for 50 move rule - 100 halfmoves without a pawn move or capture is a draw
    int halfmove;

    // index: piece type (WHITE_PAWN, WHITE_KNIGHT, ... , BLACK_KING)
    U64 bitboards[12];

    // index: side (WHITE, BLACK, BOTH)
    U64 occupancies[3];

    BoardState() {
        reset(this);
    }

    static void reset(BoardState* board) {
        board->turn = WHITE;
        board->castling_rights = WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE | BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE;
        board->enpassant_sq = no_sq;
        board->halfmove = 0;

        board->bitboards[BLACK_PAWN] = rank_7;
        board->bitboards[BLACK_ROOK] = A8 | H8;
        board->bitboards[BLACK_KNIGHT] = B8 | G8;
        board->bitboards[BLACK_BISHOP] = C8 | F8;
        board->bitboards[BLACK_QUEEN] = D8;
        board->bitboards[BLACK_KING] = E8;

        board->bitboards[WHITE_PAWN] = rank_2;
        board->bitboards[WHITE_ROOK] = A1 | H1;
        board->bitboards[WHITE_KNIGHT] = B1 | G1;
        board->bitboards[WHITE_BISHOP] = C1 | F1;
        board->bitboards[WHITE_QUEEN] = D1;
        board->bitboards[WHITE_KING] = E1;

        board->occupancies[WHITE] = rank_1 | rank_2;
        board->occupancies[BLACK] = rank_7 | rank_8;
        board->occupancies[BOTH] = rank_1 | rank_2 | rank_7 | rank_8;
    }

    static void print(BoardState* board) {
        printf("    A  B  C  D  E  F  G  H\n\n");
        for (int y=0; y<8; y++) {
            printf("%d   ", 8-y);
            for (int x=0; x<8; x++) {
                U64 sq = 1ULL << (8*y + x);
                int piece_found = -1;
                for (int piece_type=0; piece_type<12; piece_type++){
                    if (board->bitboards[piece_type] & sq) {
                        piece_found = piece_type;
                        break;
                    }
                }
                printf("%s  ", (piece_found == -1) ? "." : unicode_pieces[piece_found]);
            }
            printf(" %d\n", 8-y);
        }
        printf("\n    A  B  C  D  E  F  G  H\n\n");
        printf("           turn: %s\n", board->turn == WHITE ? "white" : "black");
        printf("castling_rights: %c%c%c%c\n",
               board->castling_rights & WHITE_CASTLE_KINGSIDE ? 'K' : '-',
               board->castling_rights & WHITE_CASTLE_QUEENSIDE ? 'Q' : '-',
               board->castling_rights & BLACK_CASTLE_KINGSIDE ? 'k' : '-',
               board->castling_rights & BLACK_CASTLE_QUEENSIDE ? 'q' : '-');
        printf("   enpassant_sq: %s\n", board->enpassant_sq == no_sq ? "none" : sq_str[board->enpassant_sq]);
        printf("       halfmove: %d\n\n", board->halfmove);
    }
};

/*////////////////////////////////////////////////////////////////////////////////
                             Section: moves
/*////////////////////////////////////////////////////////////////////////////////

/* Inputs:
   source_sq            6 bits    0-63 (a8-h1).
   target_sq            6 bits    0-63 (a8-h1).
   piece_type           4 bits    0-11 (WHITE_PAWN, ..., BLACK_KING).
   promotion_type       2 bits    0-3 (KNIGHT_PROMOTION, ..., QUEEN_PROMOTION).
   promotion            1 bit     0-1 (true or false).
   double_pawn_push     1 bit     0-1 (true or false).
   capture              1 bit     0-1 (true or false).
   enpassant_capture    1 bit     0-1 (true or false).
 */
inline U32 encode_move(
    int source_sq,
    int target_sq,
    int piece_type,
    int promotion_type,
    int promotion,
    int double_pawn_push,
    int capture,
    int enpassant_capture
    ) {
    return source_sq|(target_sq<<6)|(piece_type<<12)|(promotion_type<<16)|(promotion<<18)|(double_pawn_push<<19)|(capture<<20)|(enpassant_capture<<21);
}

struct MoveGenerator{
    constexpr static int max_move_index = 256;
    U32 move_list[max_move_index];
    int moves_found;

    void generate_pseudo_legal_moves(BoardState* board) {
        moves_found = 0;
        int source_sq;
        int target_sq;
        U64 source_sq_bit;
        U64 target_sq_bit;

        if (board->turn == WHITE) {
            // pawn moves TODO: test
            U64 pawns = board->bitboards[WHITE_PAWN];
            U64 pawn_blockers = board->occupancies[BOTH];
            while (pawns) {
                source_sq = lsb_scan(pawns);
                pop_lsb(pawns);
                source_sq_bit = sq_bit[source_sq];
                U64 pawn_attacks = get_pawn_attacks(WHITE, source_sq);

                printf("source_sq = %d\n", source_sq);

                if (source_sq_bit & rank_2) { // pawn move from start square
                    printf("start sq\n");
                    if (~(source_sq_bit>>8 & pawn_blockers)) { // single push
                        target_sq = source_sq - 8;
                        move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 0, 0);

                        if (~(source_sq_bit>>16 & pawn_blockers)) { // double push
                            target_sq = source_sq - 16;
                            move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 1, 0, 0);
                        }
                    }

                    while (pawn_attacks) { // normal captures
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[BLACK] & target_sq_bit) {
                            move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 1, 0);
                        }
                    }
                }
                else if (source_sq_bit & rank_7) { // pawn move to promotion square
                    printf("rank 7\n");
                    if (~(source_sq_bit>>8 & pawn_blockers)) { // single push
                        target_sq = source_sq - 8;
                        move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, KNIGHT_PROMOTION, 1, 0, 0, 0);
                        move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, BISHOP_PROMOTION, 1, 0, 0, 0);
                        move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, ROOK_PROMOTION, 1, 0, 0, 0);
                        move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, QUEEN_PROMOTION, 1, 0, 0, 0);
                    }
                    while (pawn_attacks) { // normal captures
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[BLACK] & target_sq_bit) {
                            move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, KNIGHT_PROMOTION, 1, 0, 1, 0);
                            move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, BISHOP_PROMOTION, 1, 0, 1, 0);
                            move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, ROOK_PROMOTION, 1, 0, 1, 0);
                            move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, QUEEN_PROMOTION, 1, 0, 1, 0);
                        }
                    }
                }
                else { // pawn move from not start square and not to promotion
                    printf("rank 3-6\n");
                    if (~(source_sq_bit>>8 & pawn_blockers)) { // single push
                        target_sq = source_sq - 8;
                        move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 0, 0);
                    }

                    while (pawn_attacks) {
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[BLACK] & target_sq_bit) { // normal captures
                            move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 1, 0);
                        }
                        else if (sq_bit[board->enpassant_sq] & target_sq_bit) { // enpassant capture
                            move_list[moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 1, 1);
                        }
                    }
                }
            }

            // TODO: knight moves
            // TODO: bishop moves
            // TODO: rook moves
            // TODO: queen moves
            // TODO: king moves
        }
        else { // turn == BLACK
            // TODO: all
        }

        assert (moves_found <= max_move_index);
    }
};


// TODO: perft test


/*////////////////////////////////////////////////////////////////////////////////
                             Section: init and main
/*////////////////////////////////////////////////////////////////////////////////

void init_engine() {
    init_attacks();
}

int main() {
    init_engine();
    BoardState board;

    MoveGenerator moves;
    moves.generate_pseudo_legal_moves(&board);
    printf("%d\n", moves.moves_found);

    return 0;
}
