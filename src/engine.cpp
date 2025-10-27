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
        int promotion_piece_type = decode_move_promotion_type(move);
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
        else { // not castling or promotion move
            U64 source_sq_bit = sq_bit[source_sq];
            U64 target_sq_bit = sq_bit[target_sq];
            U64 source_and_target_sq_bits = source_sq_bit | target_sq_bit;

            if (promotion) {
                board->bitboards[moving_piece_type] ^= source_sq_bit;
                board->occupancies[board->turn] ^= source_and_target_sq_bits;
                board->bitboards[promotion_piece_type] ^= target_sq_bit;
            }
            else {
                board->bitboards[moving_piece_type] ^= source_and_target_sq_bits;
                board->occupancies[board->turn] ^= source_and_target_sq_bits;
            }

            if (enpassant_capture) {
                board->occupancies[BOTH] ^= source_sq_bit;
                if (board->turn == WHITE) {
                    U64 capture_sq_bit = target_sq_bit << 8;
                    board->bitboards[BLACK_PAWN] ^= capture_sq_bit;
                    board->occupancies[BLACK] ^= capture_sq_bit;
                    board->occupancies[BOTH] ^= capture_sq_bit;
                }
                else {
                    U64 capture_sq_bit = target_sq_bit >> 8;
                    board->bitboards[WHITE_PAWN] ^= capture_sq_bit;
                    board->occupancies[WHITE] ^= capture_sq_bit;
                    board->occupancies[BOTH] ^= capture_sq_bit;
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

            king_sq_after_move = lsb_scan(board->turn==WHITE? board->bitboards[WHITE_KING] : board->bitboards[BLACK_KING]);
        }
        // check for checks after piece update
        if (sq_is_attacked(king_sq_after_move, !board->turn, board)) {
            unmake(source_sq, target_sq, moving_piece_type, enpassant_capture, captured_piece_type, castle_kingside, castle_queenside, promotion_piece_type, board);
            return 0;
        }

        // std::cout << "debug:\n";
        // // if (get_rook_attacks(sq, board->occupancies[BOTH]) & board->bitboards[WHITE_ROOK]) { return 1; }
        // print_bitboard(board->occupancies[BOTH]);
        // // print_bitboard(get_rook_attacks(king_sq_after_move, board->occupancies[BOTH] & board->bitboards[WHITE_ROOK]));

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
    static void unmake(int source_sq, int target_sq, int moved_piece_type, int enpassant_capture, int captured_piece_type, int castle_kingside, int castle_queenside, int promotion_piece_type, BoardState* board) {
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
        else if (promotion_piece_type != NO_PIECE) {
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

void manual_move_check(char fen[], int piece_type, float sleep_time_s) {
    BoardState board;
    MoveGenerator moves;

    BoardState::load(&board, fen);
    std::cout << "==============================\n" << "start: \n\n";
    BoardState::print(&board);

    U64 sleep_time_us = sleep_time_s*1000000ULL;
    moves.generate_pl_moves(&board);
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
};

void perft(PerftResults* results, BoardState *board, int depth) {
    MoveGenerator moves;
    if (depth == 1) {
        int l_moves = 0;
        moves.generate_pl_moves(board);

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
        moves.generate_pl_moves(board);
        for (int move_index=0; move_index<moves.pl_moves_found; move_index++) {
            BoardState board_copy = *board;
            if (BoardState::make(&board_copy, moves.pl_move_list[move_index])) {
                perft(results, &board_copy, depth-1);
            }
        }
    }
}

void perft_test(char start_fen[], int depth) {
    // initialize
    auto t0 = timestamp();
    BoardState board;
    MoveGenerator moves;
    BoardState::load(&board, start_fen);
    std::cout << "============================\n";
    std::cout << "initial board state\n\n";
    BoardState::print(&board);

    // generate nodes
    PerftResults results;
    perft(&results, &board, depth);

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
        << "depth: " << depth << "\n\n"
        << "nodes: " << results.nodes << "\n"
        << "captures: " << results.captures << "\n"
        << "enpassants: " << results.enpassants << "\n"
        << "castles: " << results.castles << "\n"
        << "promotions: " << results.promotions << "\n"
        << "checkmates (depth-1): " << results.prev_checkmates << "\n"
        << "\ntime: " << time_elapsed << time_elapsed_str;
    return;
}

// from https://www.chessprogramming.org/Perft_Results
char perft_position_1[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
char perft_position_2[] = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
char perft_position_3[] = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1 ";
char perft_position_4a[] = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
char perft_position_4b[] = "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1";

// nodes, captures, ep, castles, promotions
U64 perft_position_1_results[][5] = {
    20,         0,         0,      0,      0,
    400,        0,         0,      0,      0,
    8902,       34,        0,      0,      0,
    197281,     1576,      0,      0,      0,
    4865609,    82719,     258,    0,      0,
    119060324,  2812008,   5248,   0,      0,
    3195901860, 108329926, 319617, 883453, 0
};
U64 perft_position_2_results[][5] = {
    48,         8,          0,       2,         0,
    2039,       351,        1,       91,        0,
    97862,      17102,      45,      3162,      0,
    4085603,    757163,     1929,    128013,    15172,
    193690690,  35043416,   73365,   4993637,   8392,
    8031647685, 1558445089, 3577504, 184513607, 56627920
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
};
U64 perft_position_4_results[][5] = {
    6,         0,         0,     0,        0,
    264,       87,        0,     6,        48,
    9467,      1021,      4,     0,        120,
    422333,    131393,    0,     7795,     60032,
    15833292,  2046173,   6512,  0,        329464,
    706045033, 210369132, 212,   10882006, 81102984
};

bool perft_suite_single_position(char perft_position[], U64 perft_position_results[][5], bool slow_test) {
    bool suite_fail = false;
    BoardState board;
    MoveGenerator moves;
    int max_depth;

    // position 3
    BoardState::load(&board, perft_position);
    BoardState::print(&board);
    for (int depth=1; depth<=10; depth++) {
        if (!slow_test && perft_position_results[depth-1][0] > 100000000) { break; }
        std::cout << "----------------------------\n";

        // init
        auto t0 = timestamp();
        bool fail = false;
        BoardState::load(&board, perft_position);

        // generate nodes
        PerftResults results;
        perft(&results, &board, depth);

        // compare to known results
        if (perft_position_results[depth-1][0] != results.nodes) {
            fail = true;
            std::cout
                << "    nodes: " << results.nodes << "\n"
                << "should be: " << perft_position_results[depth-1][0] << "\n";
        }
        if (perft_position_results[depth-1][1] != results.captures) {
            fail = true;
            std::cout
                << " captures: " << results.captures << "\n"
                << "should be: " << perft_position_results[depth-1][1] << "\n";
        }
        if (perft_position_results[depth-1][2] != results.enpassants) {
            fail = true;
            std::cout
                << "enpassants: " << results.enpassants << "\n"
                << " should be: " << perft_position_results[depth-1][2] << "\n";
        }
        if (perft_position_results[depth-1][3] != results.castles) {
            fail = true;
            std::cout
                << "  castles: " << results.castles << "\n"
                << "should be: " << perft_position_results[depth-1][3] << "\n";
        }
        if (perft_position_results[depth-1][4] != results.promotions) {
            fail = true;
            std::cout
                << "promotions: " << results.promotions << "\n"
                << " should be: " << perft_position_results[depth-1][4] << "\n";
        }

        suite_fail |= fail;

        auto t1 = timestamp();
        U64 time_elapsed;
        std::string time_elapsed_str;
        if (delta_timestamp_s(t0,t1) > 0) { time_elapsed = delta_timestamp_s(t0, t1); time_elapsed_str = " s\n"; }
        else if (delta_timestamp_ms(t0,t1) > 0) { time_elapsed = delta_timestamp_ms(t0, t1); time_elapsed_str = " ms\n"; }
        else { time_elapsed = delta_timestamp_us(t0, t1); time_elapsed_str = " us\n"; }

        std::cout
            << "depth " << depth << (fail? ": FAIL" : ": PASS")
            << "\ntime: " << time_elapsed << time_elapsed_str;
    }

    return suite_fail;
}

void perft_suite(bool slow_test) {
    std::cout << "======================================\n";
    std::cout << "perft position 1\n\n";
    perft_suite_single_position(perft_position_1, perft_position_1_results, slow_test);

    std::cout << "======================================\n";
    std::cout << "perft position 2\n\n";
    perft_suite_single_position(perft_position_2, perft_position_2_results, slow_test);

    std::cout << "======================================\n";
    std::cout << "perft position 3\n\n";
    perft_suite_single_position(perft_position_3, perft_position_3_results, slow_test);

    std::cout << "======================================\n";
    std::cout << "perft position 4a\n\n";
    perft_suite_single_position(perft_position_4a, perft_position_4_results, slow_test);

    std::cout << "======================================\n";
    std::cout << "perft position 4b\n\n";
    perft_suite_single_position(perft_position_4b, perft_position_4_results, slow_test);
}


/*////////////////////////////////////////////////////////////////////////////////
                             Section: init and main
/*////////////////////////////////////////////////////////////////////////////////

void init_engine() {
    std::cout << "--------------------------\n";

    auto t0 = timestamp();
    init_attacks();
    auto t1 = timestamp();
    std::cout << "attacks initialized in " << delta_timestamp_ms(t0, t1) << " ms\n";

    std::cout << "\n";
}

int main() {
    init_engine();

    // TODO: debug perft results
    perft_suite(false);

    ////////////////////    debug single position
    // char fen[] = "8/2p5/3p4/KP5r/1R2Pp1k/8/411P1/8 b - e3 0 1 ";
    // manual_move_check(fen, BLACK_PAWN, .7);


    ////////////////////    perft test debugging
    //                       me
    //                       right ans

    // nodes at depth 5: 4865596 -> not enough captures
    //                   4865609
    // perft_test(perft_position_1, 5);

    // depth 4: too many nodes, captures, castles, (ep and promos good)
    // perft_test(perft_position_2, 4);

    // depth 5 too many nodes
    // perft_test(perft_position_3, 2);



    return 0;
}
