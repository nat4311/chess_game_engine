/* references:
 * https://www.youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "engine.h"
#include "attacks.h"
#include <assert.h>
#include <unistd.h>

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
                else {
                    printf("invalid piece type: %c\n", c); assert(0);
                    throw std::runtime_error("invalid piece type: " + std::to_string(c) + "\n");
                }
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
        printf("               turn: %s\n", board->turn == WHITE ? "white" : "black");
        printf("    castling_rights: %c%c%c%c\n",
               board->castling_rights & WHITE_CASTLE_KINGSIDE ? 'K' : '-',
               board->castling_rights & WHITE_CASTLE_QUEENSIDE ? 'Q' : '-',
               board->castling_rights & BLACK_CASTLE_KINGSIDE ? 'k' : '-',
               board->castling_rights & BLACK_CASTLE_QUEENSIDE ? 'q' : '-');
        printf("       enpassant_sq: %s\n", board->enpassant_sq == no_sq ? "none" : sq_str[board->enpassant_sq]);
        printf("           halfmove: %d\n\n", board->halfmove);
    }

    // attempt to make a pl move.
    // returns 0 if move is illegal due to checks, 1 if legal.
    static bool make(BoardState* board, U32 move) {
        int source_sq = decode_move_source_sq(move);
        int target_sq = decode_move_target_sq(move);
        int moving_piece_type = decode_move_piece_type(move);
        int promotion = decode_move_promotion(move);
        int capture = decode_move_capture(move);
        int enpassant_capture = decode_move_enpassant_capture(move);
        int castle_kingside = decode_move_castle_kingside(move);
        int castle_queenside = decode_move_castle_queenside(move);
        int double_pawn_push = decode_move_double_pawn_push(move);
        int captured_piece_type = NO_PIECE;
        int king_sq_after_move = no_sq;

        // update the pieces (unless trying to castle out of or through check)
        if (castle_kingside) {
            if (board->turn == WHITE) {
                if (sq_is_attacked(e1, BLACK, board) || sq_is_attacked(f1, BLACK, board)) { return 0; }

                board->bitboards[WHITE_KING] ^= (E1|G1);
                board->bitboards[WHITE_ROOK] ^= (H1|F1);
                board->occupancies[WHITE] ^= (E1|F1|G1|H1);
                board->occupancies[BOTH] ^= (E1|F1|G1|H1);
                king_sq_after_move = g1;
            }
            else { // board->turn == BLACK 
                if (sq_is_attacked(e8, WHITE, board) || sq_is_attacked(f8, WHITE, board)) { return 0; }

                board->bitboards[BLACK_KING] ^= (E8|G8);
                board->bitboards[BLACK_ROOK] ^= (H8|F8);
                board->occupancies[BLACK] ^= (E8|F8|G8|H8);
                board->occupancies[BOTH] ^= (E8|F8|G8|H8);
                king_sq_after_move = g8;
            }
        }
        else if (castle_queenside) {
            if (board->turn == WHITE) {
                if (sq_is_attacked(e1, BLACK, board) || sq_is_attacked(d1, BLACK, board)) { return 0; }

                board->bitboards[WHITE_KING] ^= (E1|C1);
                board->bitboards[WHITE_ROOK] ^= (A1|D1);
                board->occupancies[WHITE] ^= (A1|C1|D1|E1);
                board->occupancies[BOTH] ^= (A1|C1|D1|E1);
                king_sq_after_move = c1;
            }
            else { // board->turn == BLACK
                if (sq_is_attacked(e8, WHITE, board) || sq_is_attacked(d8, WHITE, board)) { return 0; }

                board->bitboards[BLACK_KING] ^= (E8|C8);
                board->bitboards[BLACK_ROOK] ^= (A8|D8);
                board->occupancies[BLACK] ^= (A8|C8|D8|E8);
                board->occupancies[BOTH] ^= (A8|C8|D8|E8);
                king_sq_after_move = c8;
            }
        }
        else { // non-castling move
            U64 source_sq_bit = sq_bit[source_sq];
            U64 target_sq_bit = sq_bit[target_sq];
            U64 source_and_target_sq_bits = source_sq_bit | target_sq_bit;

            if (enpassant_capture) {
                board->occupancies[BOTH] ^= sq_bit[source_sq];
                int capture_sq_bit = sq_bit[target_sq];
                if (board->turn == WHITE) {
                    capture_sq_bit <<= 8;
                    board->bitboards[BLACK_PAWN] ^= capture_sq_bit;
                    board->occupancies[BLACK] ^= capture_sq_bit;
                }
                else {
                    capture_sq_bit >>= 8;
                    board->bitboards[WHITE_PAWN] ^= capture_sq_bit;
                    board->occupancies[WHITE] ^= capture_sq_bit;
                }
            }
            else if (capture) {
                if (board->turn == WHITE) {
                    if (board->bitboards[BLACK_PAWN] & target_sq_bit) {
                        captured_piece_type = BLACK_PAWN;
                        goto CAPTURED_PIECE_FOUND;
                    }
                    if (board->bitboards[BLACK_KNIGHT] & target_sq_bit) {
                        captured_piece_type = BLACK_KNIGHT;
                        goto CAPTURED_PIECE_FOUND;
                    }
                    if (board->bitboards[BLACK_BISHOP] & target_sq_bit) {
                        captured_piece_type = BLACK_BISHOP;
                        goto CAPTURED_PIECE_FOUND;
                    }
                    if (board->bitboards[BLACK_ROOK] & target_sq_bit) {
                        captured_piece_type = BLACK_ROOK;
                        goto CAPTURED_PIECE_FOUND;
                    }
                    if (board->bitboards[BLACK_QUEEN] & target_sq_bit) {
                        captured_piece_type = BLACK_QUEEN;
                        goto CAPTURED_PIECE_FOUND;
                    }
                    BoardState::print(board);
                    print_move(move, 1);
                    throw std::runtime_error("move has capture flag set but no captured_piece was found\n");
                }
                else { // board->turn == BLACK
                    if (board->bitboards[WHITE_PAWN] & target_sq_bit) {
                        captured_piece_type = WHITE_PAWN;
                        goto CAPTURED_PIECE_FOUND;
                    }
                    if (board->bitboards[WHITE_KNIGHT] & target_sq_bit) {
                        captured_piece_type = WHITE_KNIGHT;
                        goto CAPTURED_PIECE_FOUND;
                    }
                    if (board->bitboards[WHITE_BISHOP] & target_sq_bit) {
                        captured_piece_type = WHITE_BISHOP;
                        goto CAPTURED_PIECE_FOUND;
                    }
                    if (board->bitboards[WHITE_ROOK] & target_sq_bit) {
                        captured_piece_type = WHITE_ROOK;
                        goto CAPTURED_PIECE_FOUND;
                    }
                    if (board->bitboards[WHITE_QUEEN] & target_sq_bit) {
                        captured_piece_type = WHITE_QUEEN;
                        goto CAPTURED_PIECE_FOUND;
                    }
                    BoardState::print(board);
                    print_move(move, 1);
                    throw std::runtime_error("move has capture flag set but no captured_piece was found\n");
                }

                CAPTURED_PIECE_FOUND:
                board->occupancies[BOTH] ^= source_sq_bit;
                board->bitboards[captured_piece_type] ^= target_sq_bit;
                board->occupancies[!board->turn] ^= target_sq_bit;
            }
            else { // no capture
                board->occupancies[BOTH] ^= source_and_target_sq_bits;
            }

            board->bitboards[moving_piece_type] ^= source_and_target_sq_bits;
            board->occupancies[board->turn] ^= source_and_target_sq_bits;
            king_sq_after_move = lsb_scan(board->turn==WHITE? board->bitboards[WHITE_KING] : board->bitboards[BLACK_KING]);
        }
        // check for checks after piece update
        if (sq_is_attacked(king_sq_after_move, !board->turn, board)) {
            unmake(source_sq, target_sq, moving_piece_type, enpassant_capture, captured_piece_type, castle_kingside, castle_queenside, board);
            return 0;
        }

        // update turn
        board->turn = !board->turn;
        // update castling rights
        if (moving_piece_type == WHITE_KING) { board->castling_rights &= (BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE); }
        else if (moving_piece_type == BLACK_KING) { board->castling_rights &= (WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE); }
        else if (moving_piece_type == WHITE_ROOK) {
            if (source_sq == a1) { board->castling_rights &= (WHITE_CASTLE_KINGSIDE|BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE); }
            else if (source_sq == h1) { board->castling_rights &= (WHITE_CASTLE_QUEENSIDE|BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE); }
        }
        else if (moving_piece_type == BLACK_ROOK) {
            if (source_sq == a8) { board->castling_rights &= (BLACK_CASTLE_KINGSIDE|WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE); }
            else if (source_sq == h8) { board->castling_rights &= (BLACK_CASTLE_QUEENSIDE|WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE); }
        }
        // update enpassant_sq
        if (double_pawn_push) {
            if (moving_piece_type == WHITE_PAWN) {
                board->enpassant_sq = target_sq+8;
            }
            else {
                assert(moving_piece_type == BLACK_PAWN);
                board->enpassant_sq = target_sq-8;
            }
        }
        else {
            board->enpassant_sq = no_sq;
        }
        // update halfmove
        if (capture || moving_piece_type == WHITE_PAWN || moving_piece_type == BLACK_PAWN) { board->halfmove = 0; }
        else { board->halfmove++; }

        return 1;
    }

    // for checking for checks
    static bool sq_is_attacked(int sq, int by_side, BoardState* board) {
        assert (sq >= 0 && sq <= 63);
        if (by_side == WHITE) {
            if (get_pawn_attacks(BLACK, sq) & board->bitboards[WHITE_PAWN]) { return 1; }
            if (get_knight_attacks(sq) & board->bitboards[WHITE_KNIGHT]) { return 1; }
            if (get_bishop_attacks(sq, board->occupancies[BOTH]) & board->bitboards[WHITE_BISHOP]) { return 1; }
            if (get_rook_attacks(sq, board->occupancies[BOTH]) & board->bitboards[WHITE_ROOK]) { return 1; }
            if (get_queen_attacks(sq, board->occupancies[BOTH]) & board->bitboards[WHITE_QUEEN]) { return 1; }
            if (get_king_attacks(sq) & board->bitboards[WHITE_KING]) { return 1; }
        }
        else { // by_side == BLACK
            if (get_pawn_attacks(WHITE, sq) & board->bitboards[BLACK_PAWN]) { return 1; }
            if (get_knight_attacks(sq) & board->bitboards[BLACK_KNIGHT]) { return 1; }
            if (get_bishop_attacks(sq, board->occupancies[BOTH]) & board->bitboards[BLACK_BISHOP]) { return 1; }
            if (get_rook_attacks(sq, board->occupancies[BOTH]) & board->bitboards[BLACK_ROOK]) { return 1; }
            if (get_queen_attacks(sq, board->occupancies[BOTH]) & board->bitboards[BLACK_QUEEN]) { return 1; }
            if (get_king_attacks(sq) & board->bitboards[BLACK_KING]) { return 1; }
        }
        return 0;
    }
    
    // assumes that only the pieces have been moved, board state variables have not updated yet (turn, ep, castling, halfmove, etc)
    static void unmake(int source_sq, int target_sq, int moved_piece_type, int enpassant_capture, int captured_piece_type, int castle_kingside, int castle_queenside, BoardState* board) {
        if (castle_kingside) {
            if (board->turn == WHITE) {
                board->bitboards[WHITE_KING] ^= (E1|G1);
                board->bitboards[WHITE_ROOK] ^= (H1|F1);
                board->occupancies[WHITE] ^= (E1|F1|G1|H1);
                board->occupancies[BOTH] ^= (E1|F1|G1|H1);
            }
            else { // board->turn == BLACK
                board->bitboards[BLACK_KING] ^= (E8|G8);
                board->bitboards[BLACK_ROOK] ^= (H8|F8);
                board->occupancies[BLACK] ^= (E8|F8|G8|H8);
                board->occupancies[BOTH] ^= (E8|F8|G8|H8);
            }
        }
        else if (castle_queenside) {
            if (board->turn == WHITE) {
                board->bitboards[WHITE_KING] ^= (E1|C1);
                board->bitboards[WHITE_ROOK] ^= (A1|D1);
                board->occupancies[WHITE] ^= (A1|C1|D1|E1);
                board->occupancies[BOTH] ^= (A1|C1|D1|E1);
            }
            else { // board->turn == BLACK
                board->bitboards[BLACK_KING] ^= (E8|C8);
                board->bitboards[BLACK_ROOK] ^= (A8|D8);
                board->occupancies[BLACK] ^= (A8|C8|D8|E8);
                board->occupancies[BOTH] ^= (A8|C8|D8|E8);
            }
        }
        else { // normal move
            U64 source_and_target_sq_bits = sq_bit[source_sq] | sq_bit[target_sq];
            board->bitboards[moved_piece_type] ^= source_and_target_sq_bits;
            board->occupancies[board->turn] ^= source_and_target_sq_bits;
            if (captured_piece_type == NO_PIECE) {
                board->occupancies[BOTH] ^= source_and_target_sq_bits;
            }
            else if (enpassant_capture) {
                int capture_sq_bit = sq_bit[target_sq];
                if (board->turn == WHITE) {
                    assert (captured_piece_type == BLACK_PAWN);
                    capture_sq_bit <<= 8;
                    board->occupancies[BOTH] ^= sq_bit[source_sq];
                    board->bitboards[BLACK_PAWN] ^= capture_sq_bit;
                    board->occupancies[BLACK] ^= capture_sq_bit;
                }
                else {
                    assert (captured_piece_type == WHITE_PAWN);
                    capture_sq_bit >>= 8;
                    board->occupancies[BOTH] ^= sq_bit[source_sq];
                    board->bitboards[WHITE_PAWN] ^= capture_sq_bit;
                    board->occupancies[WHITE] ^= capture_sq_bit;
                }
            }
            else {
                assert (captured_piece_type >= WHITE_PAWN && captured_piece_type <= BLACK_KING);
                board->occupancies[BOTH] ^= sq_bit[source_sq];
                board->bitboards[captured_piece_type] ^= sq_bit[target_sq];
                board->occupancies[!board->turn] ^= sq_bit[target_sq];
            }
        }
    }
};

/*////////////////////////////////////////////////////////////////////////////////
                             Section: move generator
/*////////////////////////////////////////////////////////////////////////////////

struct MoveGenerator{
    constexpr static int max_pl_move_index = 256;
    U32 pl_move_list[max_pl_move_index];
    int pl_moves_found;

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
                if (source_sq_bit & rank_7) {
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

    int count_l_moves(BoardState* board) {
        int l_moves_found = 0;
        generate_pl_moves(board);
        for (int move_index=0; move_index<pl_moves_found; move_index++) {
            BoardState board_copy = *board;
            if (BoardState::make(&board_copy, pl_move_list[move_index])) {
                l_moves_found++;
            }
        }
        return l_moves_found;
    }

    void print_pl_moves(int piece_type) {
        printf("PL MOVES     dcekq\n------------------\n");
        for (int i=0; i<pl_moves_found; i++) {
            U32 move = pl_move_list[i];
            if (piece_type == NO_PIECE || decode_move_piece_type(move) == piece_type) {
                print_move(pl_move_list[i], 0);
            }
        }
    }
};

/*////////////////////////////////////////////////////////////////////////////////
                             Section: perft testing
/*////////////////////////////////////////////////////////////////////////////////

char perft_initial_position[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// TODO: test this with chessprogramming.org/Perft_Results
// TODO:: ensure you don't need to check for mate 
U64 perft(BoardState *board, int depth) {
    MoveGenerator moves;
    if (depth == 1) {
        return moves.count_l_moves(board);
    }
    else {
        U64 nodes = 0;
        moves.generate_pl_moves(board);
        for (int move_index=0; move_index<moves.pl_moves_found; move_index++) {
            BoardState board_copy = *board;
            if (BoardState::make(&board_copy, moves.pl_move_list[move_index])) {
                nodes += perft(&board_copy, depth-1);
            }
        }
        return nodes;
    }
}

void perft_test(char start_fen[], int depth) {
    // initialize
    auto t0 = timestamp();
    BoardState board;
    MoveGenerator moves;
    BoardState::load(&board, start_fen);

    // generate nodes
    U64 nodes = perft(&board, depth);

    // print timing info
    auto t1 = timestamp();
    U64 time_elapsed;
    std::string time_elapsed_str;
    if (delta_timestamp_s(t0,t1) > 0) { time_elapsed = delta_timestamp_s(t0, t1); time_elapsed_str = " s\n"; }
    else if (delta_timestamp_ms(t0,t1) > 0) { time_elapsed = delta_timestamp_ms(t0, t1); time_elapsed_str = " ms\n"; }
    else { time_elapsed = delta_timestamp_us(t0, t1); time_elapsed_str = " us\n"; }
    std::cout
        << "============================\n"
        << "perft_test complete\n"
        << "nodes: " << nodes << "\n"
        << "time: " << time_elapsed << time_elapsed_str
        << "depth: " << depth << "\n";
    return;
}

/*////////////////////////////////////////////////////////////////////////////////
                             Section: init and main
/*////////////////////////////////////////////////////////////////////////////////

void manual_move_check() {
    BoardState board;
    MoveGenerator moves;

    // char fen1[] = "rnbqk2r/p1pppppp/2B5/Pp6/8/8/8/R3K2R b KQkq b6 0 1";
    char fen1[] = "rnbq1bnr/ppp1pppp/2kp4/8/P7/2R5/1PPPPPPP/1NBQKBNR b KQ - 3 1";
    BoardState::load(&board, fen1);
    std::cout << "==============================\n" << "start: \n\n";
    BoardState::print(&board);

    U64 sleep_time = .5*1000000ULL;
    moves.generate_pl_moves(&board);
    for (int i=0; i<moves.pl_moves_found; i++) {
        U32 move = moves.pl_move_list[i];
        if (decode_move_piece_type(move) == BLACK_PAWN) {
            BoardState board_copy = board;
            if(!BoardState::make(&board_copy, move)) {
                continue;
            }
            usleep(sleep_time);
            std::cout << "==============================\n" << move << "\n\n";
            BoardState::print(&board_copy);
        }
    }
    usleep(sleep_time);
    std::cout << "==============================\n" << "start again" << "\n\n";
    BoardState::print(&board);
}

void init_engine() {
    init_attacks();
}

int main() {
    init_engine();

    // perft test debugging
    // depth 6: I get 119059985
    //      should be 119060324
    for (int depth = 6; depth <= 7; depth++) {
        perft_test(perft_initial_position, depth);
    }

    // manual_move_check();

    return 0;
}
