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

    // 0-63, 64 for no_sq
    int enpassant_sq;

    // for 50 move rule - 100 halfmoves without a pawn move or capture is a draw
    int halfmove;

    // index: piece type (WHITE_PAWN, WHITE_KNIGHT, ... , BLACK_KING)
    U64 bitboards[12];

    // index: side (WHITE, BLACK, BOTH)
    U64 occupancies[3];

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

    // example:
    // rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1 .
    //                                     pieces turn cast ep hm fm
    static void load(BoardState* board, char* fen_str) {
        int i_fen_str = 0;
        char c;
        int x, y, sq;

        // pieces
        for (int i=0; i<12; i++) { board->bitboards[i] = {0}; }
        for (int i=0; i<3; i++) { board->occupancies[i] = {0}; }
        x = 0;
        y = 0;
        while (1) {
            c = fen_str[i_fen_str++];
            if (c>='1' && c<='8') {
                x += c - '0';
            }
            else if (c == '/') {
                y++;
                x = 0;
            }
            else if (c == ' ') {
                break;
            }
            else {
                sq = x + 8*y;
                x++;
                board->occupancies[BOTH] |= sq_bit[sq];
                     if (c == 'P') { board->bitboards[WHITE_PAWN]   |= sq_bit[sq]; board->occupancies[WHITE] |= sq_bit[sq]; }
                else if (c == 'N') { board->bitboards[WHITE_KNIGHT] |= sq_bit[sq]; board->occupancies[WHITE] |= sq_bit[sq]; }
                else if (c == 'B') { board->bitboards[WHITE_BISHOP] |= sq_bit[sq]; board->occupancies[WHITE] |= sq_bit[sq]; }
                else if (c == 'R') { board->bitboards[WHITE_ROOK]   |= sq_bit[sq]; board->occupancies[WHITE] |= sq_bit[sq]; }
                else if (c == 'Q') { board->bitboards[WHITE_QUEEN]  |= sq_bit[sq]; board->occupancies[WHITE] |= sq_bit[sq]; }
                else if (c == 'K') { board->bitboards[WHITE_KING]   |= sq_bit[sq]; board->occupancies[WHITE] |= sq_bit[sq]; }
                else if (c == 'p') { board->bitboards[BLACK_PAWN]   |= sq_bit[sq]; board->occupancies[BLACK] |= sq_bit[sq]; }
                else if (c == 'n') { board->bitboards[BLACK_KNIGHT] |= sq_bit[sq]; board->occupancies[BLACK] |= sq_bit[sq]; }
                else if (c == 'b') { board->bitboards[BLACK_BISHOP] |= sq_bit[sq]; board->occupancies[BLACK] |= sq_bit[sq]; }
                else if (c == 'r') { board->bitboards[BLACK_ROOK]   |= sq_bit[sq]; board->occupancies[BLACK] |= sq_bit[sq]; }
                else if (c == 'q') { board->bitboards[BLACK_QUEEN]  |= sq_bit[sq]; board->occupancies[BLACK] |= sq_bit[sq]; }
                else if (c == 'k') { board->bitboards[BLACK_KING]   |= sq_bit[sq]; board->occupancies[BLACK] |= sq_bit[sq]; }
                else { printf("invalid piece type: %c\n", c); assert(0); }
            }
        }

        // turn
        c = fen_str[i_fen_str++];
        if (c == 'w') {
            board->turn = WHITE;
        }
        else if (c == 'b') {
            board->turn = BLACK;
        }
        else {
            printf("invalid turn: %c\n", c);
            assert(0);
        }
        i_fen_str++;

        // castling rights
        board->castling_rights = 0;
        c = fen_str[i_fen_str++];
        if (c == '-') {
            i_fen_str++;
        }
        else {
            if (c == 'K') {
                board->castling_rights |= WHITE_CASTLE_KINGSIDE;
                c = fen_str[i_fen_str++];
            }
            if (c == 'Q') {
                board->castling_rights |= WHITE_CASTLE_QUEENSIDE;
                c = fen_str[i_fen_str++];
            }
            if (c == 'k') {
                board->castling_rights |= BLACK_CASTLE_KINGSIDE;
                c = fen_str[i_fen_str++];
            }
            if (c == 'q') {
                board->castling_rights |= BLACK_CASTLE_QUEENSIDE;
                i_fen_str++;
            }
        }

        // en passant
        c = fen_str[i_fen_str++];
        if (c == '-') {
            board->enpassant_sq = no_sq;
        }
        else {
            x = c - 'a';
            c = fen_str[i_fen_str++];
            y = '8' - c;
            sq = x + 8*y;
            board->enpassant_sq = sq;
        }
        i_fen_str++;

        // halfmove
        board->halfmove = 0;
        while (1) {
            c = fen_str[i_fen_str++];
            if (c == ' ') {
                break;
            }
            board->halfmove = board->halfmove * 10 + (c - '0');
        }

        // fullmove - don't care
        
        return;
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

    // TODO: implement this
    // TODO: add move/capture stacks for unmake
    static void make(BoardState* board, U32 move) {
        // encode_move(source_sq, target_sq, piece_type, promotion_type, promotion, double_pawn_push, capture, enpassant_capture)
        int source_sq = decode_move_source_sq(move);
        int target_sq = decode_move_target_sq(move);
        int piece_type = decode_move_piece_type(move);
        int promotion = decode_move_promotion(move);
        int capture = decode_move_capture(move);
    }
    
    // TODO: implement this
    static void unmake(BoardState* board) {

    }
};

/*////////////////////////////////////////////////////////////////////////////////
                             Section: move generator
/*////////////////////////////////////////////////////////////////////////////////

struct MoveGenerator{
    constexpr static int max_pl_move_index = 256;
    U32 pl_move_list[max_pl_move_index];
    int pl_moves_found;

    constexpr static int max_move_index = 256;
    U32 move_list[max_move_index];
    int moves_found;

    // generate pseudo-legal moves.
    // moves stored in pl_move_list.
    // max index pl_moves_found.
    void generate_pl_moves(BoardState* board) {
        pl_moves_found = 0;
        int source_sq;
        int target_sq;
        U64 source_sq_bit;
        U64 target_sq_bit;

        if (board->turn == WHITE) {

            // pawn moves
            U64 pawns = board->bitboards[WHITE_PAWN];
            U64 pawn_blockers = board->occupancies[BOTH];
            while (pawns) {
                source_sq = lsb_scan(pawns);
                pop_lsb(pawns);
                source_sq_bit = sq_bit[source_sq];
                U64 pawn_attacks = get_pawn_attacks(WHITE, source_sq);
                
                // pawn move from start square
                if (source_sq_bit & rank_2) {
                    if (!(source_sq_bit>>8 & pawn_blockers)) { // single push
                        target_sq = source_sq - 8;
                        // printf("%s%s\n", sq_str[source_sq], sq_str[target_sq]);
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 0, 0, 0, 0);

                        if (!(source_sq_bit>>16 & pawn_blockers)) { // double push
                            target_sq = source_sq - 16;
                            // printf("%s%s\n", sq_str[source_sq], sq_str[target_sq]);
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 1, 0, 0, 0, 0);
                        }
                    }

                    // normal captures
                    while (pawn_attacks) {
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[BLACK] & target_sq_bit) {
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 1, 0, 0, 0);
                        }
                    }
                }
                // pawn move to promotion square
                else if (source_sq_bit & rank_7) {
                    if (!(source_sq_bit>>8 & pawn_blockers)) { // single push
                        target_sq = source_sq - 8;
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, KNIGHT_PROMOTION, 1, 0, 0, 0, 0, 0);
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, BISHOP_PROMOTION, 1, 0, 0, 0, 0, 0);
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, ROOK_PROMOTION, 1, 0, 0, 0, 0, 0);
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, QUEEN_PROMOTION, 1, 0, 0, 0, 0, 0);
                    }
                    while (pawn_attacks) { // normal captures
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[BLACK] & target_sq_bit) {
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, KNIGHT_PROMOTION, 1, 0, 1, 0, 0, 0);
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, BISHOP_PROMOTION, 1, 0, 1, 0, 0, 0);
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, ROOK_PROMOTION, 1, 0, 1, 0, 0, 0);
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, QUEEN_PROMOTION, 1, 0, 1, 0, 0, 0);
                        }
                    }
                }
                else { // pawn move from not start square and not to promotion
                    if (!(source_sq_bit>>8 & pawn_blockers)) { // single push
                        target_sq = source_sq - 8;
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 0, 0, 0, 0);
                    }

                    while (pawn_attacks) {
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[BLACK] & target_sq_bit) { // normal captures
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 1, 0, 0, 0);
                        }
                        else if (sq_bit[board->enpassant_sq] & target_sq_bit) { // enpassant capture
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 1, 1, 0, 0);
                        }
                    }
                }
            }

            // knight moves
            U64 knights = board->bitboards[WHITE_KNIGHT];
            while (knights) {
                source_sq = lsb_scan(knights);
                pop_lsb(knights);
                source_sq_bit = sq_bit[source_sq];
                U64 knight_attacks = get_knight_attacks(source_sq);
                while (knight_attacks) {
                    target_sq = lsb_scan(knight_attacks);
                    pop_lsb(knight_attacks);
                    target_sq_bit = sq_bit[target_sq];
                    if (board->occupancies[WHITE] & target_sq_bit) {
                        continue;
                    }
                    int capture = (board->occupancies[BLACK] & target_sq_bit) ? 1 : 0;
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_KNIGHT, 0, 0, 0, capture, 0, 0, 0);
                }
            }

            // bishop moves
            U64 bishops = board->bitboards[WHITE_BISHOP];
            while (bishops) {
                source_sq = lsb_scan(bishops);
                pop_lsb(bishops);
                source_sq_bit = sq_bit[source_sq];
                U64 bishop_attacks = get_bishop_attacks(source_sq, board->occupancies[BOTH]);
                while (bishop_attacks) {
                    target_sq = lsb_scan(bishop_attacks);
                    pop_lsb(bishop_attacks);
                    target_sq_bit = sq_bit[target_sq];
                    if (target_sq_bit & board->occupancies[WHITE]) {
                        continue;
                    }
                    int capture = (board->occupancies[BLACK] & target_sq_bit) ? 1 : 0;
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_BISHOP, 0, 0, 0, capture, 0, 0, 0);
                }
            }

            // rook moves
            U64 rooks = board->bitboards[WHITE_ROOK];
            while (rooks) {
                source_sq = lsb_scan(rooks);
                pop_lsb(rooks);
                source_sq_bit = sq_bit[source_sq];
                U64 rook_attacks = get_rook_attacks(source_sq, board->occupancies[BOTH]);
                while (rook_attacks) {
                    target_sq = lsb_scan(rook_attacks);
                    pop_lsb(rook_attacks);
                    target_sq_bit = sq_bit[target_sq];
                    if (target_sq_bit & board->occupancies[WHITE]) {
                        continue;
                    }
                    int capture = (board->occupancies[BLACK] & target_sq_bit) ? 1 : 0;
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_ROOK, 0, 0, 0, capture, 0, 0, 0);
                }
            }

            // queen moves
            U64 queens = board->bitboards[WHITE_QUEEN];
            while (queens) {
                source_sq = lsb_scan(queens);
                pop_lsb(queens);
                source_sq_bit = sq_bit[source_sq];
                U64 queen_attacks = get_queen_attacks(source_sq, board->occupancies[BOTH]);
                while (queen_attacks) {
                    target_sq = lsb_scan(queen_attacks);
                    pop_lsb(queen_attacks);
                    target_sq_bit = sq_bit[target_sq];
                    if (target_sq_bit & board->occupancies[WHITE]) {
                        continue;
                    }
                    int capture = (board->occupancies[BLACK] & target_sq_bit) ? 1 : 0;
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_QUEEN, 0, 0, 0, capture, 0, 0, 0);
                }
            }
            
            // king moves
            source_sq = lsb_scan(board->bitboards[WHITE_KING]);
            source_sq_bit = sq_bit[source_sq];
            U64 king_attacks = get_king_attacks(source_sq);
            while (king_attacks) {
                target_sq = lsb_scan(king_attacks);
                pop_lsb(king_attacks);
                target_sq_bit = sq_bit[target_sq];
                if (target_sq_bit & board->occupancies[WHITE]) {
                    continue;
                }
                int capture = (board->occupancies[BLACK] & target_sq_bit) ? 1 : 0;
                pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_KING, 0, 0, 0, capture, 0, 0, 0);
            }
            //castling
            if (board->castling_rights & WHITE_CASTLE_KINGSIDE) {
                if (!(board->occupancies[BOTH] & (F1|G1))) {
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, g1, WHITE_KING, 0, 0, 0, 0, 0, 1, 0);
                }
            }
            if (board->castling_rights & WHITE_CASTLE_QUEENSIDE) {
                if (!(board->occupancies[BOTH] & (B1|C1|D1))) {
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, c1, WHITE_KING, 0, 0, 0, 0, 0, 0, 1);
                }
            }

        }
        else { // turn == BLACK
            // pawn moves
            U64 pawns = board->bitboards[BLACK_PAWN];
            U64 pawn_blockers = board->occupancies[BOTH];
            while (pawns) {
                source_sq = lsb_scan(pawns);
                pop_lsb(pawns);
                source_sq_bit = sq_bit[source_sq];
                U64 pawn_attacks = get_pawn_attacks(BLACK, source_sq);
                
                // pawn move from start square
                if (source_sq_bit & rank_2) {
                    if (!(source_sq_bit<<8 & pawn_blockers)) { // single push
                        target_sq = source_sq + 8;
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 0, 0, 0, 0, 0);

                        if (!(source_sq_bit<<16 & pawn_blockers)) { // double push
                            target_sq = source_sq + 16;
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 1, 0, 0, 0, 0);
                        }
                    }

                    // normal captures
                    while (pawn_attacks) {
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[WHITE] & target_sq_bit) {
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 0, 1, 0, 0, 0);
                        }
                    }
                }
                // pawn move to promotion square
                else if (source_sq_bit & rank_2) {
                    if (!(source_sq_bit<<8 & pawn_blockers)) { // single push
                        target_sq = source_sq + 8;
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, KNIGHT_PROMOTION, 1, 0, 0, 0, 0, 0);
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, BISHOP_PROMOTION, 1, 0, 0, 0, 0, 0);
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, ROOK_PROMOTION, 1, 0, 0, 0, 0, 0);
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, QUEEN_PROMOTION, 1, 0, 0, 0, 0, 0);
                    }
                    while (pawn_attacks) { // normal captures
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[WHITE] & target_sq_bit) {
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, KNIGHT_PROMOTION, 1, 0, 1, 0, 0, 0);
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, BISHOP_PROMOTION, 1, 0, 1, 0, 0, 0);
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, ROOK_PROMOTION, 1, 0, 1, 0, 0, 0);
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, QUEEN_PROMOTION, 1, 0, 1, 0, 0, 0);
                        }
                    }
                }
                else { // pawn move from not start square and not to promotion
                    if (!(source_sq_bit<<8 & pawn_blockers)) { // single push
                        target_sq = source_sq + 8;
                        pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 0, 0, 0, 0, 0);
                    }

                    while (pawn_attacks) {
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[WHITE] & target_sq_bit) { // normal captures
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 0, 1, 0, 0, 0);
                        }
                        else if (sq_bit[board->enpassant_sq] & target_sq_bit) { // enpassant capture
                            pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 0, 1, 1, 0, 0);
                        }
                    }
                }
            }

            // knight moves
            U64 knights = board->bitboards[BLACK_KNIGHT];
            while (knights) {
                source_sq = lsb_scan(knights);
                pop_lsb(knights);
                source_sq_bit = sq_bit[source_sq];
                U64 knight_attacks = get_knight_attacks(source_sq);
                while (knight_attacks) {
                    target_sq = lsb_scan(knight_attacks);
                    pop_lsb(knight_attacks);
                    target_sq_bit = sq_bit[target_sq];
                    if (target_sq_bit & board->occupancies[BLACK]) {
                        continue;
                    }
                    int capture = (board->occupancies[WHITE] & target_sq_bit) ? 1 : 0;
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_KNIGHT, 0, 0, 0, capture, 0, 0, 0);
                }
            }

            // bishop moves
            U64 bishops = board->bitboards[BLACK_BISHOP];
            while (bishops) {
                source_sq = lsb_scan(bishops);
                pop_lsb(bishops);
                source_sq_bit = sq_bit[source_sq];
                U64 bishop_attacks = get_bishop_attacks(source_sq, board->occupancies[BOTH]);
                while (bishop_attacks) {
                    target_sq = lsb_scan(bishop_attacks);
                    pop_lsb(bishop_attacks);
                    target_sq_bit = sq_bit[target_sq];
                    if (target_sq_bit & board->occupancies[BLACK]) {
                        continue;
                    }
                    int capture = (board->occupancies[WHITE] & target_sq_bit) ? 1 : 0;
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_BISHOP, 0, 0, 0, capture, 0, 0, 0);
                }
            }

            // rook moves
            U64 rooks = board->bitboards[BLACK_ROOK];
            while (rooks) {
                source_sq = lsb_scan(rooks);
                pop_lsb(rooks);
                source_sq_bit = sq_bit[source_sq];
                U64 rook_attacks = get_rook_attacks(source_sq, board->occupancies[BOTH]);
                while (rook_attacks) {
                    target_sq = lsb_scan(rook_attacks);
                    pop_lsb(rook_attacks);
                    target_sq_bit = sq_bit[target_sq];
                    if (target_sq_bit & board->occupancies[BLACK]) {
                        continue;
                    }
                    int capture = (board->occupancies[WHITE] & target_sq_bit) ? 1 : 0;
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_ROOK, 0, 0, 0, capture, 0, 0, 0);
                }
            }

            // queen moves
            U64 queens = board->bitboards[BLACK_QUEEN];
            while (queens) {
                source_sq = lsb_scan(queens);
                pop_lsb(queens);
                source_sq_bit = sq_bit[source_sq];
                U64 queen_attacks = get_queen_attacks(source_sq, board->occupancies[BOTH]);
                while (queen_attacks) {
                    target_sq = lsb_scan(queen_attacks);
                    pop_lsb(queen_attacks);
                    target_sq_bit = sq_bit[target_sq];
                    if (target_sq_bit & board->occupancies[BLACK]) {
                        continue;
                    }
                    int capture = (board->occupancies[WHITE] & target_sq_bit) ? 1 : 0;
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_QUEEN, 0, 0, 0, capture, 0, 0, 0);
                }
            }
            
            // king moves
            source_sq = lsb_scan(board->bitboards[BLACK_KING]);
            source_sq_bit = sq_bit[source_sq];
            U64 king_attacks = get_king_attacks(source_sq);
            while (king_attacks) {
                target_sq = lsb_scan(king_attacks);
                pop_lsb(king_attacks);
                target_sq_bit = sq_bit[target_sq];
                if (target_sq_bit & board->occupancies[BLACK]) {
                    continue;
                }
                int capture = (board->occupancies[WHITE] & target_sq_bit) ? 1 : 0;
                pl_move_list[pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_KING, 0, 0, 0, capture, 0, 0, 0);
            }
            //castling
            if (board->castling_rights & BLACK_CASTLE_KINGSIDE) {
                if (!(board->occupancies[BOTH] & (F8|G8))) {
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, g8, BLACK_KING, 0, 0, 0, 0, 0, 1, 0);
                }
            }
            if (board->castling_rights & BLACK_CASTLE_QUEENSIDE) {
                if (!(board->occupancies[BOTH] & (B8|C8|D8))) {
                    pl_move_list[pl_moves_found++] = encode_move(source_sq, c8, BLACK_KING, 0, 0, 0, 0, 0, 0, 1);
                }
            }
        }

        assert (pl_moves_found <= max_pl_move_index);
    }

    // TODO: implement and test with perft test
    // generate legal moves
    void generate_moves(BoardState* board) {
        generate_pl_moves(board);
    }

    void print_pl_moves(int piece_type) {
        printf("             dcekq\n");
        for (int i=0; i<pl_moves_found; i++) {
            U32 move = pl_move_list[i];
            if (piece_type == NO_PIECE || decode_move_piece_type(move) == piece_type) {
                print_move(pl_move_list[i]);
            }
        }
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
    BoardState::reset(&board);

    // char start_fen[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    char fen1[] = "rnbqkbnr/pppppppp/8/8/8/1p6/4pPPP/R3KN1R w KQkq a3 0 1";
    BoardState::load(&board, fen1);
    BoardState::print(&board);
    // board.enpassant_sq = b6;

    MoveGenerator moves;
    moves.generate_pl_moves(&board);
    moves.print_pl_moves(WHITE_KNIGHT);

    return 0;
}
