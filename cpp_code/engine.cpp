/* references:
 * https://www.youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "engine.h"
#include "attacks.h"
#include <assert.h>
#include <unistd.h>
#include <iostream>

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

    // the standard turn number minus one
    int turn_no;

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
        board->turn_no = 0;

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

    static BoardState copy(BoardState* board) {
        return *board;
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

        // turn_no
        board->turn_no = 0;
        while (1) {
            c = fen_str[i_fen_str++];
            if (c == '\n') {
                break;
            }
            board->turn_no = board->turn_no * 10 + (c - '0');
        }
        board->turn_no--;
        
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
        printf("            turn_no: %d\n\n", board->turn_no + 1);
    }

    // attempt to make a pl move.
    // returns 0 if move is illegal due to checks, 1 if legal.
    static bool make(BoardState* board, U32 move, bool unmake_move_flag=false) {
        int source_sq = decode_move_source_sq(move);
        int target_sq = decode_move_target_sq(move);
        int moving_piece_type = decode_move_piece_type(move);
        int promotion = decode_move_promotion(move);
        int promotion_piece_type = decode_move_promotion_piece_type(move);
        int capture = decode_move_capture(move);
        int enpassant_capture = decode_move_enpassant_capture(move);
        int castle_kingside = decode_move_castle_kingside(move);
        int castle_queenside = decode_move_castle_queenside(move);
        int double_pawn_push = decode_move_double_pawn_push(move);
        int captured_piece_type = NO_PIECE;
        int king_sq_after_move = no_sq;

        ////////////// update the pieces (unless trying to castle out of or through check)
        if (castle_kingside) {
            if (board->turn == WHITE) {
                if (sq_is_attacked(board, e1, BLACK) || sq_is_attacked(board, f1, BLACK)) { return 0; }

                board->bitboards[WHITE_KING] ^= (E1|G1);
                board->bitboards[WHITE_ROOK] ^= (H1|F1);
                board->occupancies[WHITE] ^= (E1|F1|G1|H1);
                board->occupancies[BOTH] ^= (E1|F1|G1|H1);
                king_sq_after_move = g1;
            }
            else { // board->turn == BLACK 
                if (sq_is_attacked(board, e8, WHITE) || sq_is_attacked(board, f8, WHITE)) { return 0; }

                board->bitboards[BLACK_KING] ^= (E8|G8);
                board->bitboards[BLACK_ROOK] ^= (H8|F8);
                board->occupancies[BLACK] ^= (E8|F8|G8|H8);
                board->occupancies[BOTH] ^= (E8|F8|G8|H8);
                king_sq_after_move = g8;
            }
        }
        else if (castle_queenside) {
            if (board->turn == WHITE) {
                if (sq_is_attacked(board, e1, BLACK) || sq_is_attacked(board, d1, BLACK)) { return 0; }

                board->bitboards[WHITE_KING] ^= (E1|C1);
                board->bitboards[WHITE_ROOK] ^= (A1|D1);
                board->occupancies[WHITE] ^= (A1|C1|D1|E1);
                board->occupancies[BOTH] ^= (A1|C1|D1|E1);
                king_sq_after_move = c1;
            }
            else { // board->turn == BLACK
                if (sq_is_attacked(board, e8, WHITE) || sq_is_attacked(board, d8, WHITE)) { return 0; }

                board->bitboards[BLACK_KING] ^= (E8|C8);
                board->bitboards[BLACK_ROOK] ^= (A8|D8);
                board->occupancies[BLACK] ^= (A8|C8|D8|E8);
                board->occupancies[BOTH] ^= (A8|C8|D8|E8);
                king_sq_after_move = c8;
            }
        }
        else { // not castling or promotion move
            U64 source_sq_bit = sq_bit[source_sq];
            U64 target_sq_bit = sq_bit[target_sq];
            U64 source_and_target_sq_bits = source_sq_bit | target_sq_bit;

            if (enpassant_capture) {
                board->occupancies[BOTH] ^= source_and_target_sq_bits;
                if (board->turn == WHITE) {
                    U64 capture_sq_bit = target_sq_bit << 8;
                    board->bitboards[BLACK_PAWN] ^= capture_sq_bit;
                    board->occupancies[BLACK] ^= capture_sq_bit;
                    board->occupancies[BOTH] ^= capture_sq_bit;
                    captured_piece_type = BLACK_PAWN;
                }
                else {
                    U64 capture_sq_bit = target_sq_bit >> 8;
                    board->bitboards[WHITE_PAWN] ^= capture_sq_bit;
                    board->occupancies[WHITE] ^= capture_sq_bit;
                    board->occupancies[BOTH] ^= capture_sq_bit;
                    captured_piece_type = WHITE_PAWN;
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
                    if (board->bitboards[BLACK_KING] & target_sq_bit) {
                        captured_piece_type = BLACK_KING;
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
                    if (board->bitboards[WHITE_KING] & target_sq_bit) {
                        captured_piece_type = WHITE_KING;
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

            if (promotion) {
                // if (
                //     promotion_piece_type != WHITE_QUEEN &&
                //     promotion_piece_type != WHITE_ROOK &&
                //     promotion_piece_type != WHITE_KNIGHT &&
                //     promotion_piece_type != WHITE_BISHOP &&
                //     promotion_piece_type != BLACK_QUEEN &&
                //     promotion_piece_type != BLACK_ROOK &&
                //     promotion_piece_type != BLACK_KNIGHT &&
                //     promotion_piece_type != BLACK_BISHOP
                // ) {
                //     std::cout << "promotion_piece_type invalid: " << promotion_piece_type << "\n";
                //     print_move(move, true);
                //     std::cout << (source_sq_bit & rank_2) << " debug\n";
                //     throw std::runtime_error("invalid promotion_piece_type\n");
                // }
                board->bitboards[moving_piece_type] ^= source_sq_bit;
                board->occupancies[board->turn] ^= source_and_target_sq_bits;
                board->bitboards[promotion_piece_type] ^= target_sq_bit;
            }
            else {
                board->bitboards[moving_piece_type] ^= source_and_target_sq_bits;
                board->occupancies[board->turn] ^= source_and_target_sq_bits;
            }

            king_sq_after_move = lsb_scan(board->turn==WHITE? board->bitboards[WHITE_KING] : board->bitboards[BLACK_KING]);
        }
        /////////////// check for checks after piece update
        if (sq_is_attacked(board, king_sq_after_move, !board->turn)) {
            unmake(board, source_sq, target_sq, moving_piece_type, enpassant_capture, captured_piece_type, castle_kingside, castle_queenside, promotion, promotion_piece_type);
            return 0;
        }

        ////////////// unmake the move immediately (ie: if only checking legality)
        if (unmake_move_flag) {
            // std::cout << "------------------------" << std::endl;
            // std::cout << "before unmake" << std::endl;
            // BoardState::print(board);
            unmake(board, source_sq, target_sq, moving_piece_type, enpassant_capture, captured_piece_type, castle_kingside, castle_queenside, promotion, promotion_piece_type);
            // std::cout << "after unmake" << std::endl;
            // BoardState::print(board);
            // std::cout << "------------------------" << std::endl;
            // throw std::runtime_error("halting for debug");
            return 1;
        }


        /////////////// board state updates
        if (board->turn == BLACK) {
            board->turn_no++;
            board->turn = WHITE;
        }
        else {
            board->turn = BLACK;
        }

        // update castling rights (moving a king/rook)
        if (moving_piece_type == WHITE_KING) {
            board->castling_rights &= (BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE);
        }
        else if (moving_piece_type == BLACK_KING) {
            board->castling_rights &= (WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE);
        }
        else if (moving_piece_type == WHITE_ROOK) {
            if (source_sq == a1) { board->castling_rights &= (WHITE_CASTLE_KINGSIDE|BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE); }
            else if (source_sq == h1) { board->castling_rights &= (WHITE_CASTLE_QUEENSIDE|BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE); }
        }
        else if (moving_piece_type == BLACK_ROOK) {
            if (source_sq == a8) { board->castling_rights &= (BLACK_CASTLE_KINGSIDE|WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE); }
            else if (source_sq == h8) { board->castling_rights &= (BLACK_CASTLE_QUEENSIDE|WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE); }
        }
        // update castling rights (capturing a rook)
        if (captured_piece_type == WHITE_ROOK) {
            if (target_sq == a1) {
                board->castling_rights &= (BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE | WHITE_CASTLE_KINGSIDE);
            }
            else if (target_sq == h1) {
                board->castling_rights &= (BLACK_CASTLE_KINGSIDE | BLACK_CASTLE_QUEENSIDE | WHITE_CASTLE_QUEENSIDE);
            }
        }
        else if (captured_piece_type == BLACK_ROOK) {
            if (target_sq == a8) {
                board->castling_rights &= (WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE | BLACK_CASTLE_KINGSIDE);
            }
            else if (target_sq == h8) {
                board->castling_rights &= (WHITE_CASTLE_KINGSIDE | WHITE_CASTLE_QUEENSIDE | BLACK_CASTLE_QUEENSIDE);
            }
        }
        
        // update enpassant_sq
        if (double_pawn_push) {
            if (moving_piece_type == WHITE_PAWN) {
                board->enpassant_sq = target_sq+8;
            }
            else if (moving_piece_type == BLACK_PAWN) {
                board->enpassant_sq = target_sq-8;
            }
            else {
                std::cout << "double_pawn_push invalid moving_piece_type: " << moving_piece_type << "\n";
                print(board);
                print_move(move, true);
                throw std::runtime_error("double_pawn_push error");
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
    static bool sq_is_attacked(BoardState* board, int sq, int by_side) {
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

    static bool king_is_attacked(BoardState* board) {
        int king_sq = lsb_scan(board->turn==WHITE? board->bitboards[WHITE_KING] : board->bitboards[BLACK_KING]);
        if (sq_is_attacked(board, king_sq, !board->turn)) {
            return true;
        }
        else {
            return false;
        }
    }
    
    // assumes that only the pieces have been moved, board state variables have not updated yet (turn, ep, castling, halfmove, etc)
    static void unmake(BoardState* board, int source_sq, int target_sq, int moved_piece_type, int enpassant_capture, int captured_piece_type, int castle_kingside, int castle_queenside, int promotion, int promotion_piece_type) {
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
        else if (promotion) {
            U64 source_sq_bit = sq_bit[source_sq];
            U64 target_sq_bit = sq_bit[target_sq];
            U64 source_and_target_sq_bits = source_sq_bit | target_sq_bit;
            board->bitboards[moved_piece_type] ^= source_sq_bit;
            board->bitboards[promotion_piece_type] ^= target_sq_bit;
            board->bitboards[board->turn] ^= source_and_target_sq_bits;
            if (captured_piece_type == NO_PIECE) {
                board->bitboards[BOTH] ^= source_and_target_sq_bits;
            }
            else {
                board->bitboards[captured_piece_type] ^= target_sq_bit;
                board->bitboards[!board->turn] ^= target_sq_bit;
                board->bitboards[BOTH] ^= source_sq_bit;
            }
        }
        else { // not castling or promotion move
            U64 source_sq_bit = sq_bit[source_sq];
            U64 target_sq_bit = sq_bit[target_sq];
            U64 source_and_target_sq_bits = source_sq_bit | target_sq_bit;
            board->bitboards[moved_piece_type] ^= source_and_target_sq_bits;
            board->occupancies[board->turn] ^= source_and_target_sq_bits;
            if (captured_piece_type == NO_PIECE) {
                board->occupancies[BOTH] ^= source_and_target_sq_bits;
            }
            else if (enpassant_capture) {
                int capture_sq_bit = target_sq_bit;
                if (board->turn == WHITE) {
                    assert (captured_piece_type == BLACK_PAWN);
                    capture_sq_bit <<= 8;
                    board->occupancies[BOTH] ^= source_sq_bit;
                    board->bitboards[BLACK_PAWN] ^= capture_sq_bit;
                    board->occupancies[BLACK] ^= capture_sq_bit;
                }
                else {
                    assert (captured_piece_type == WHITE_PAWN);
                    capture_sq_bit >>= 8;
                    board->occupancies[BOTH] ^= source_sq_bit;
                    board->bitboards[WHITE_PAWN] ^= capture_sq_bit;
                    board->occupancies[WHITE] ^= capture_sq_bit;
                }
            }
            else {
                assert (captured_piece_type >= WHITE_PAWN && captured_piece_type <= BLACK_KING);
                board->occupancies[BOTH] ^= source_sq_bit;
                board->bitboards[captured_piece_type] ^= target_sq_bit;
                board->occupancies[!board->turn] ^= target_sq_bit;
            }
        }
    }

    static bool get_castle_K(BoardState* board) {
        return board->castling_rights & WHITE_CASTLE_KINGSIDE;
    }
    static bool get_castle_Q(BoardState* board) {
        return board->castling_rights & WHITE_CASTLE_QUEENSIDE;
    }
    static bool get_castle_k(BoardState* board) {
        return board->castling_rights & BLACK_CASTLE_KINGSIDE;
    }
    static bool get_castle_q(BoardState* board) {
        return board->castling_rights & BLACK_CASTLE_QUEENSIDE;
    }
};

/*////////////////////////////////////////////////////////////////////////////////
                             Section: move generator
/*////////////////////////////////////////////////////////////////////////////////

// generates moves and stores them inside struct
struct MoveGenerator{
    constexpr static int max_pl_move_index = 256;
    U32 pl_move_list[max_pl_move_index];
    int pl_moves_found;

    // generate pseudo-legal moves.
    // moves stored in pl_move_list.
    // max index pl_moves_found.
    static void generate_pl_moves(MoveGenerator* moves, BoardState* board) {
        moves->pl_moves_found = 0;
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
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 0, 0, 0, 0);

                        if (!(source_sq_bit>>16 & pawn_blockers)) { // double push
                            target_sq = source_sq - 16;
                            // printf("%s%s\n", sq_str[source_sq], sq_str[target_sq]);
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 1, 0, 0, 0, 0);
                        }
                    }

                    // normal captures
                    while (pawn_attacks) {
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[BLACK] & target_sq_bit) {
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 1, 0, 0, 0);
                        }
                    }
                }
                // pawn move to promotion square
                else if (source_sq_bit & rank_7) {
                    if (!(source_sq_bit>>8 & pawn_blockers)) { // single push
                        target_sq = source_sq - 8;
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, WHITE_KNIGHT, 1, 0, 0, 0, 0, 0);
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, WHITE_BISHOP, 1, 0, 0, 0, 0, 0);
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, WHITE_ROOK, 1, 0, 0, 0, 0, 0);
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, WHITE_QUEEN, 1, 0, 0, 0, 0, 0);
                    }
                    while (pawn_attacks) { // normal captures
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[BLACK] & target_sq_bit) {
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, WHITE_KNIGHT, 1, 0, 1, 0, 0, 0);
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, WHITE_BISHOP, 1, 0, 1, 0, 0, 0);
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, WHITE_ROOK, 1, 0, 1, 0, 0, 0);
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, WHITE_QUEEN, 1, 0, 1, 0, 0, 0);
                        }
                    }
                }
                else { // pawn move from not start square and not to promotion
                    if (!(source_sq_bit>>8 & pawn_blockers)) { // single push
                        target_sq = source_sq - 8;
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 0, 0, 0, 0);
                    }

                    while (pawn_attacks) {
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[BLACK] & target_sq_bit) { // normal captures
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 1, 0, 0, 0);
                        }
                        else if (sq_bit[board->enpassant_sq] & target_sq_bit) { // enpassant capture
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_PAWN, 0, 0, 0, 1, 1, 0, 0);
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
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_KNIGHT, 0, 0, 0, capture, 0, 0, 0);
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
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_BISHOP, 0, 0, 0, capture, 0, 0, 0);
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
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_ROOK, 0, 0, 0, capture, 0, 0, 0);
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
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_QUEEN, 0, 0, 0, capture, 0, 0, 0);
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
                moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, WHITE_KING, 0, 0, 0, capture, 0, 0, 0);
            }
            //castling
            if (board->castling_rights & WHITE_CASTLE_KINGSIDE) {
                if (!(board->occupancies[BOTH] & (F1|G1))) {
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, g1, WHITE_KING, 0, 0, 0, 0, 0, 1, 0);
                }
            }
            if (board->castling_rights & WHITE_CASTLE_QUEENSIDE) {
                if (!(board->occupancies[BOTH] & (B1|C1|D1))) {
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, c1, WHITE_KING, 0, 0, 0, 0, 0, 0, 1);
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
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 0, 0, 0, 0, 0);

                        if (!(source_sq_bit<<16 & pawn_blockers)) { // double push
                            target_sq = source_sq + 16;
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 1, 0, 0, 0, 0);
                        }
                    }

                    // normal captures
                    while (pawn_attacks) {
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[WHITE] & target_sq_bit) {
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 0, 1, 0, 0, 0);
                        }
                    }
                }
                // pawn move to promotion square
                else if (source_sq_bit & rank_2) {
                    if (!(source_sq_bit<<8 & pawn_blockers)) { // single push
                        target_sq = source_sq + 8;
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, BLACK_KNIGHT, 1, 0, 0, 0, 0, 0);
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, BLACK_BISHOP, 1, 0, 0, 0, 0, 0);
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, BLACK_ROOK, 1, 0, 0, 0, 0, 0);
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, BLACK_QUEEN, 1, 0, 0, 0, 0, 0);
                    }
                    while (pawn_attacks) { // normal captures
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[WHITE] & target_sq_bit) {
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, BLACK_KNIGHT, 1, 0, 1, 0, 0, 0);
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, BLACK_BISHOP, 1, 0, 1, 0, 0, 0);
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, BLACK_ROOK, 1, 0, 1, 0, 0, 0);
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, BLACK_QUEEN, 1, 0, 1, 0, 0, 0);
                        }
                    }
                }
                else { // pawn move from not start square and not to promotion
                    if (!(source_sq_bit<<8 & pawn_blockers)) { // single push
                        target_sq = source_sq + 8;
                        moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 0, 0, 0, 0, 0);
                    }

                    while (pawn_attacks) {
                        target_sq = lsb_scan(pawn_attacks);
                        pop_lsb(pawn_attacks);
                        target_sq_bit = sq_bit[target_sq];
                        if (board->occupancies[WHITE] & target_sq_bit) { // normal captures
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 0, 1, 0, 0, 0);
                        }
                        else if (sq_bit[board->enpassant_sq] & target_sq_bit) { // enpassant capture
                            moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_PAWN, 0, 0, 0, 1, 1, 0, 0);
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
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_KNIGHT, 0, 0, 0, capture, 0, 0, 0);
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
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_BISHOP, 0, 0, 0, capture, 0, 0, 0);
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
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_ROOK, 0, 0, 0, capture, 0, 0, 0);
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
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_QUEEN, 0, 0, 0, capture, 0, 0, 0);
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
                moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, target_sq, BLACK_KING, 0, 0, 0, capture, 0, 0, 0);
            }
            //castling
            if (board->castling_rights & BLACK_CASTLE_KINGSIDE) {
                if (!(board->occupancies[BOTH] & (F8|G8))) {
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, g8, BLACK_KING, 0, 0, 0, 0, 0, 1, 0);
                }
            }
            if (board->castling_rights & BLACK_CASTLE_QUEENSIDE) {
                if (!(board->occupancies[BOTH] & (B8|C8|D8))) {
                    moves->pl_move_list[moves->pl_moves_found++] = encode_move(source_sq, c8, BLACK_KING, 0, 0, 0, 0, 0, 0, 1);
                }
            }
        }

        assert (moves->pl_moves_found <= max_pl_move_index);
    }

    // int count_l_moves(BoardState* board) {
    //     int l_moves_found = 0;
    //     generate_pl_moves(board);
    //     for (int move_index=0; move_index<pl_moves_found; move_index++) {
    //         BoardState board_copy = *board;
    //         if (BoardState::make(&board_copy, pl_move_list[move_index])) {
    //             l_moves_found++;
    //         }
    //     }
    //     return l_moves_found;
    // }

    void print_pl_moves(int piece_type = NO_PIECE) {
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
                             Section: GameStateNodes
/*////////////////////////////////////////////////////////////////////////////////

// struct GameStateNodeAlphazero {
//     BoardState board_state;
//     MoveGenerator move_generator;
//     GameStateNodeAlphazero* parent;
//     U32 prev_move;
//     static const int max_n_children = 256; // maximum number of pseudo legal moves in a position probably
//     GameStateNodeAlphazero* children[max_n_children];
//     int n_children; // n_children-1 is max index of children array
//     float prior; // prior probability of selecting this node in mcts
//     bool valid; // is the node a legal position
//
//     GameStateNodeAlphazero(GameStateNodeAlphazero* parent = nullptr, U32 prev_move = 0):
//         parent(parent),
//         prev_move(prev_move),
//         n_children(0)
//     {
//         if (parent) { // new potential child node
//             this->board_state = parent->board_state;
//             this->valid = BoardState::make(&this->board_state, prev_move);
//         }
//         else { // start position
//             this->board_state = BoardState();
//             this->valid = true;
//             this->prior = 0;
//         }
//     }
//
//     // reset the mcts variables for so we can reuse this node as a root node
//     static void make_root(GameStateNodeAlphazero* node) {
//         node->prior = 0;
//         node->parent = nullptr;
//         // node->prev_move = 0;
//     }
//
//     static void add_pl_child(GameStateNodeAlphazero* node, GameStateNodeAlphazero* child) {
//         if (child->valid) {
//             node->children[node->n_children++] = child;
//             assert (node->n_children < max_n_children); // todo: move this to expand function in python
//         }
//     }
//
// };

/*////////////////////////////////////////////////////////////////////////////////
                             Section: perft testing
/*////////////////////////////////////////////////////////////////////////////////

// from https://www.chessprogramming.org/Perft_Results
char perft_position_1[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n";
char perft_position_2[] = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\n";
char perft_position_3[] = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1 \n";
char perft_position_4a[] = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1\n";
char perft_position_4b[] = "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1\n";
char perft_position_5[] = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8\n";

// nodes, captures, ep, castles, promotions
U64 perft_position_1_results[][5] = {
    20,         0,         0,      0,      0,
    400,        0,         0,      0,      0,
    8902,       34,        0,      0,      0,
    197281,     1576,      0,      0,      0,
    4865609,    82719,     258,    0,      0,
    119060324,  2812008,   5248,   0,      0,
    3195901860, 108329926, 319617, 883453, 0,
    0
};
U64 perft_position_2_results[][5] = {
    48,         8,          0,       2,         0,
    2039,       351,        1,       91,        0,
    97862,      17102,      45,      3162,      0,
    4085603,    757163,     1929,    128013,    15172,
    193690690,  35043416,   73365,   4993637,   8392,
    8031647685, 1558445089, 3577504, 184513607, 56627920,
    0
};
U64 perft_position_3_results[][5] = {
    14,         1,          0,       0, 0,
    191,        14,         0,       0, 0,
    2812,       209,        2,       0, 0,
    43238,      3348,       123,     0, 0,
    674624,     52051,      1165,    0, 0,
    11030083,   940350,     33325,   0, 7552,
    178633661,  14519036,   294874,  0, 140024,
    3009794393, 267586558,  8009239, 0, 6578076,
    0
};
U64 perft_position_4_results[][5] = {
    6,         0,         0,     0,        0,
    264,       87,        0,     6,        48,
    9467,      1021,      4,     0,        120,
    422333,    131393,    0,     7795,     60032,
    15833292,  2046173,   6512,  0,        329464,
    706045033, 210369132, 212,   10882006, 81102984,
    0
};
U64 perft_position_5_results[][5] = {
    44,       0, 0, 0, 0,
    1486,     0, 0, 0, 0,
    62379,    0, 0, 0, 0,
    2103487,  0, 0, 0, 0,
    89941194, 0, 0, 0, 0,
    0
};

void manual_move_check(char fen[], int piece_type, float sleep_time_s) {
    BoardState board;
    MoveGenerator moves;

    BoardState::load(&board, fen);
    std::cout << "==============================\n" << "start: \n\n";
    BoardState::print(&board);

    U64 sleep_time_us = sleep_time_s*1000000ULL;
    moves.generate_pl_moves(&moves, &board);
    std::cout << "pl moves found: " << moves.pl_moves_found << std::endl;
    for (int i=0; i<moves.pl_moves_found; i++) {
        U32 move = moves.pl_move_list[i];
        if (piece_type == NO_PIECE || decode_move_piece_type(move) == piece_type) {
            BoardState board_copy = board;
            std::cout << "==================\n";
            if(!BoardState::make(&board_copy, move)) {
                std::cout << "move: " << move << " is illegal\n";
                continue;
            }
            usleep(sleep_time_us);
            std::cout << "move: " << move;
            std::cout << "\n";
            BoardState::print(&board_copy);
        }
    }
    usleep(sleep_time_us);
    std::cout << "==============================\n" << "start again" << "\n\n";
    BoardState::print(&board);
}

struct PerftResults {
    U64 nodes = 0;
    U64 captures = 0;
    U64 enpassants = 0;
    U64 castles = 0;
    U64 promotions = 0;
    U64 prev_checkmates = 0;
    U64 pawn_moves = 0;
    U64 knight_moves = 0;
    U64 bishop_moves = 0;
    U64 rook_moves = 0;
    U64 queen_moves = 0;
    U64 king_moves = 0;
};

void perft(PerftResults* results, BoardState *board, int depth, bool include_piece_types) {
    MoveGenerator moves;
    if (depth == 1) {
        int l_moves = 0;
        moves.generate_pl_moves(&moves, board);

        for (int move_index=0; move_index<moves.pl_moves_found; move_index++) {
            BoardState board_copy = *board;
            U32 move = moves.pl_move_list[move_index];
            if (BoardState::make(&board_copy, move)) {
                l_moves++;

                if (decode_move_capture(move)) {
                    results->captures++;
                }
                if (decode_move_enpassant_capture(move)){
                    results->enpassants++;
                }
                if (decode_move_castle_kingside(move) || decode_move_castle_queenside(move)){
                    results->castles++;
                }
                if (decode_move_promotion(move)) {
                    results->promotions++;
                }
                if (include_piece_types) {
                    int piece_type = decode_move_piece_type(move);
                    if (piece_type == WHITE_PAWN || piece_type == BLACK_PAWN) {
                        results->pawn_moves++;
                    }
                    else if (piece_type == WHITE_KNIGHT || piece_type == BLACK_KNIGHT) {
                        results->knight_moves++;
                    }
                    else if (piece_type == WHITE_BISHOP || piece_type == WHITE_BISHOP) {
                        results->bishop_moves++;
                    }
                    else if (piece_type == WHITE_ROOK || piece_type == BLACK_ROOK) {
                        results->rook_moves++;
                    }
                    else if (piece_type == WHITE_QUEEN || piece_type == BLACK_QUEEN) {
                        results->queen_moves++;
                    }
                    else if (piece_type == WHITE_KING || piece_type == BLACK_KING) {
                        results->king_moves++;
                    }
                }
            }
        }

        if (l_moves == 0) {
            results->prev_checkmates++;
        }
        else {
            results->nodes += l_moves;
        }
    }
    else {
        moves.generate_pl_moves(&moves, board);
        for (int move_index=0; move_index<moves.pl_moves_found; move_index++) {
            BoardState board_copy = *board;
            if (BoardState::make(&board_copy, moves.pl_move_list[move_index])) {
                perft(results, &board_copy, depth-1, include_piece_types);
            }
        }
    }
}

void perft_test(char start_fen[], int depth, bool include_piece_types) {
    // initialize
    auto t0 = timestamp();
    BoardState board;
    BoardState::load(&board, start_fen);
    std::cout << "============================\n";
    std::cout << "initial board state\n\n";
    BoardState::print(&board);

    // generate nodes
    PerftResults results;
    perft(&results, &board, depth, include_piece_types);

    // print timing info
    auto t1 = timestamp();
    U64 time_elapsed;
    std::string time_elapsed_str;
    if (delta_timestamp_s(t0,t1) > 0) { time_elapsed = delta_timestamp_s(t0, t1); time_elapsed_str = " s\n"; }
    else if (delta_timestamp_ms(t0,t1) > 0) { time_elapsed = delta_timestamp_ms(t0, t1); time_elapsed_str = " ms\n"; }
    else { time_elapsed = delta_timestamp_us(t0, t1); time_elapsed_str = " us\n"; }
    std::cout
        << "--------------------------\n"
        << "perft_test complete\n"
        << "depth: " << depth << "\n\n"
        << "nodes: " << results.nodes << "\n"
        << "captures: " << results.captures << "\n"
        << "enpassants: " << results.enpassants << "\n"
        << "castles: " << results.castles << "\n"
        << "promotions: " << results.promotions << "\n"
        << "checkmates (depth-1): " << results.prev_checkmates << "\n";
    if (include_piece_types) {
        std::cout
            << "pawn moves: " << results.pawn_moves << "\n"
            << "knight moves: " << results.knight_moves << "\n"
            << "bishop moves: " << results.bishop_moves << "\n"
            << "rook moves: " << results.rook_moves << "\n"
            << "queen moves: " << results.queen_moves << "\n"
            << "king moves: " << results.king_moves << "\n";
    }

    std::cout << "\ntime: " << time_elapsed << time_elapsed_str;
    return;
}

bool perft_suite_single_position(char perft_position[], U64 perft_position_results[][5], bool nodes_only, bool slow_test, bool include_piece_types) {
    bool any_fail = false;
    BoardState board;

    BoardState::load(&board, perft_position);
    BoardState::print(&board);
    for (int depth=1; depth<=10; depth++) {
        if ((!slow_test && perft_position_results[depth-1][0] > 100000000) || perft_position_results[depth-1][0] == 0) {
            break;
        }
        std::cout << "----------------------------\n";

        // init
        auto t0 = timestamp();
        bool fail = false;
        BoardState::load(&board, perft_position);

        // generate nodes
        PerftResults results;
        perft(&results, &board, depth, include_piece_types);

        // compare to known results
        if (perft_position_results[depth-1][0] != results.nodes) {
            fail = true;
            std::cout
                << "    nodes: " << results.nodes << "\n"
                << "should be: " << perft_position_results[depth-1][0] << "\n";
        }
        if (!nodes_only && perft_position_results[depth-1][1] != results.captures) {
            fail = true;
            std::cout
                << " captures: " << results.captures << "\n"
                << "should be: " << perft_position_results[depth-1][1] << "\n";
        }
        if (!nodes_only && perft_position_results[depth-1][2] != results.enpassants) {
            fail = true;
            std::cout
                << "enpassants: " << results.enpassants << "\n"
                << " should be: " << perft_position_results[depth-1][2] << "\n";
        }
        if (!nodes_only && perft_position_results[depth-1][3] != results.castles) {
            fail = true;
            std::cout
                << "  castles: " << results.castles << "\n"
                << "should be: " << perft_position_results[depth-1][3] << "\n";
        }
        if (!nodes_only && perft_position_results[depth-1][4] != results.promotions) {
            fail = true;
            std::cout
                << "promotions: " << results.promotions << "\n"
                << " should be: " << perft_position_results[depth-1][4] << "\n";
        }

        any_fail |= fail;

        auto t1 = timestamp();
        U64 time_elapsed;
        std::string time_elapsed_str;
        if (delta_timestamp_s(t0,t1) > 0) { time_elapsed = delta_timestamp_s(t0, t1); time_elapsed_str = " s\n"; }
        else if (delta_timestamp_ms(t0,t1) > 0) { time_elapsed = delta_timestamp_ms(t0, t1); time_elapsed_str = " ms\n"; }
        else { time_elapsed = delta_timestamp_us(t0, t1); time_elapsed_str = " us\n"; }

        std::cout
            << (fail? "\n" : "") << "depth " << depth << (fail? ": FAIL" : ": PASS")
            << "\ntime: " << time_elapsed << time_elapsed_str;
    }

    return any_fail;
}

void perft_suite(bool slow_test) {
    auto t0 = timestamp();
    bool fail = false;

    std::cout << "======================================\n";
    std::cout << "perft position 1\n\n";
    fail |= perft_suite_single_position(perft_position_1, perft_position_1_results, false, slow_test, false);

    std::cout << "======================================\n";
    std::cout << "perft position 2\n\n";
    fail |= perft_suite_single_position(perft_position_2, perft_position_2_results, false, slow_test, false);

    std::cout << "======================================\n";
    std::cout << "perft position 3\n\n";
    fail |= perft_suite_single_position(perft_position_3, perft_position_3_results, false, slow_test, false);

    std::cout << "======================================\n";
    std::cout << "perft position 4a\n\n";
    fail |= perft_suite_single_position(perft_position_4a, perft_position_4_results, false, slow_test, false);

    std::cout << "======================================\n";
    std::cout << "perft position 4b\n\n";
    fail |= perft_suite_single_position(perft_position_4b, perft_position_4_results, false, slow_test, false);

    std::cout << "======================================\n";
    std::cout << "perft position 5\n\n";
    fail |= perft_suite_single_position(perft_position_5, perft_position_5_results, true, slow_test, false);

    auto t1 = timestamp();
    std::cout << "======================================\n\n\n";
    std::cout << "perft_suite " << (fail? "FAIL\n" : "PASS\n");
    std::cout << "time: " << delta_timestamp_s(t0,t1) << " s\n";

}

/*////////////////////////////////////////////////////////////////////////////////
                             Section: init and main
/*////////////////////////////////////////////////////////////////////////////////

void unit_tests() {
    perft_suite(true);
}

void init_engine() {
    std::cout << "Initializing engine..." << std::endl;

    auto t0 = timestamp();
    init_attacks();
    auto t1 = timestamp();
    std::cout << "    attacks initialized in " << delta_timestamp_ms(t0, t1) << " ms\n\n";
}

int main() {
    init_engine();
    // unit_tests();
    int source_sq = a7;
    int target_sq = a5;
    int piece_type = BLACK_PAWN;
    int promotion_piece_type = WHITE_PAWN;
    int promotion = 0;
    int double_pawn_push = 1;
    int capture = 0;
    int enpassant_capture = 0;
    int castle_kingside = 0;
    int castle_queenside = 0;
    U32 move = encode_move(
        source_sq,
        target_sq,
        piece_type,
        promotion_piece_type,
        promotion,
        double_pawn_push,
        capture,
        enpassant_capture,
        castle_kingside,
        castle_queenside
    );
    char fen[] = "rnbqkbnr/p1pppppp/8/1P6/8/8/1PPPPPPP/RNBQKBNR b KQkq - 0 3\n";

    BoardState board;
    board.load(&board, fen);
    // BoardState::print(&board);
    BoardState::make(&board, move);
    // BoardState::print(&board);
    U32 move2 = encode_move(b5, a6, WHITE_PAWN, WHITE_PAWN, 0, 0, 1, 1, 0, 0); 
    BoardState::print(&board);
    board.make(&board, move2, true);
    BoardState::print(&board);

    return 0;
}
