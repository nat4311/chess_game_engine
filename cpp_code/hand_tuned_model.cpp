#define OVERRIDE_ENGINE_CPP_MAIN
#include "engine.cpp"
#include <limits>
#include <unistd.h>

float inf = std::numeric_limits<double>::infinity();

/*/////////////////////////////////////////////////////////////////////////////
                          Section: Helper funcs
/*/////////////////////////////////////////////////////////////////////////////

#define gameover(game) (game.board.state == DRAW || game.board.state == WHITE_WIN || game.board.state == BLACK_WIN)

/*
         source_sq            6 bits (0-5)      0-63     (a8-h1)
         target_sq            6 bits (6-11)     0-63     (a8-h1)
         piece_type           4 bits (12-15)    0-11     (WHITE_PAWN, ..., BLACK_KING)
         double_pawn_push     1 bit  (16)       0-1      (true or false)
         enpassant_capture    1 bit  (17)       0-1      (true or false)
         castle_kingside      1 bit  (18)       0-1      (true or false)
         castle_queenside     1 bit  (19)       0-1      (true or false)
         capture              1 bit  (20)       0-1      (true or false)
(NEW)    capture_score        4 bits (21-25)    0-16     (capturing_piece_score - captured_piece_score)
         promotion_type       4 bits (27-30)    1-4,7-10 (WHITE_KNIGHT, ..., WHITE_QUEEN, BLACK_KNIGHT, ..., BLACK_QUEEN)
         promotion            1 bit  (31)       0-1      (true or false)
 */
void encode_move_capture_score(U32* move, BoardState *board) {
    if (!decode_move_capture(*move)) { return; }

    int piece_type = decode_move_piece_type(*move);
    int target_sq = decode_move_target_sq(*move);
    U64 target_sq_bit = sq_bit[target_sq];
    int captured_piece_type;

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
            BoardState::print(board);
            print_move(*move, 1);
            throw std::runtime_error("tried to capture king\n");
        }
        BoardState::print(board);
        print_move(*move, 1);
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
            BoardState::print(board);
            print_move(*move, 1);
            throw std::runtime_error("tried to capture king\n");
        }
        BoardState::print(board);
        print_move(*move, 1);
        throw std::runtime_error("move has capture flag set but no captured_piece was found\n");
    }
    CAPTURED_PIECE_FOUND:

    int capture_score = piece_score[piece_type] - piece_score[captured_piece_type] + 8;
    assert (capture_score>=0 && capture_score<=16);
    (*move) |= (capture_score<<21);
    return;
}

/*/////////////////////////////////////////////////////////////////////////////
                          Section: GameStateNode and minimax
/*/////////////////////////////////////////////////////////////////////////////

struct GameStateNode1 {
    U32 prev_move;
    BoardState board;

    GameStateNode1(BoardState* prev_board = NULL, U32 prev_move = 0, const char* fen="") {
        this->prev_move = prev_move;
        if (prev_board == NULL) {
            board = BoardState();
            if (fen[0] != '\0') {
                BoardState::load(&board, fen);
            }
        }
        else {
            board = BoardState::copy(prev_board);
            assert(BoardState::make(&board, prev_move));
        }
    }

    float count_legal_moves() {
        BoardState::generate_l_moves(&board);
        float n_legal_moves = (float)board.l.moves_found;

        return n_legal_moves;
    }

    float eval() {
        int mobility_score = count_legal_moves();

        if (board.state == DRAW) {
            return 0.0;
        }
        else if (board.state == WHITE_WIN) {
            return 10000.0 - (float)board.turn_no;
        }
        else if (board.state == BLACK_WIN) {
            return -10000.0 + (float)board.turn_no;
        }

        return
            (float)BoardState::material_score(&board) +
            0.5*(float)BoardState::doubled_pawn_score(&board) +
            0.1*(float)(mobility_score - BoardState::enemy_mobility_score(&board));
    }
};

struct minimax_result {
    GameStateNode1 best_child;
    float child_eval;
};

minimax_result _minimax(GameStateNode1* node, int depth, bool maximizing_player=true, float alpha=-inf, float beta=inf) {
    BoardState::generate_l_moves(&(node->board));
    if (depth==0 || node->board.state==DRAW || node->board.state==WHITE_WIN || node->board.state==BLACK_WIN) {
        minimax_result res = {*node, node->eval()};
        return res;
    }

    minimax_result best_result;
    if (maximizing_player) {
        best_result.child_eval = -inf;
        for (int i = 0; i < node->board.l.moves_found; i++) {
            GameStateNode1 child = GameStateNode1(&node->board, node->board.l.move_list[i]);
            auto res = _minimax(&child, depth-1, false, alpha, beta);
            if (res.child_eval > best_result.child_eval) {
                best_result.child_eval = res.child_eval;
                best_result.best_child = child;
            }
            if (res.child_eval > alpha) {
                alpha = res.child_eval;
            }
            alpha = std::max(alpha, res.child_eval);
            if (beta <= alpha) {
                break;
            }
        }
        assert (best_result.child_eval != -inf);
    }
    else {
        best_result.child_eval = inf;
        for (int i = 0; i < node->board.l.moves_found; i++) {
            GameStateNode1 child = GameStateNode1(&node->board, node->board.l.move_list[i]);
            auto res = _minimax(&child, depth-1, true, alpha, beta);
            if (res.child_eval < best_result.child_eval) {
                best_result.child_eval = res.child_eval;
                best_result.best_child = child;
            }
            beta = std::min(beta, res.child_eval);
            if (beta <= alpha) {
                break;
            }
        }
        assert (best_result.child_eval != inf);
    }
    return best_result;
}

minimax_result minimax(GameStateNode1* root, int depth) {
    bool maximizing_player = (root->board.turn == WHITE);
    minimax_result best = _minimax(root, depth, maximizing_player);
    return best;
}

/*/////////////////////////////////////////////////////////////////////////////
                          Section: Unit tests
/*/////////////////////////////////////////////////////////////////////////////

void test_minimax_timings(int max_depth = 8) {
    GameStateNode1 game;

    for (int i = 1; i<=max_depth; i++) {
        auto t0 = timestamp();
        minimax(&game, i);
        auto t1 = timestamp();
        std::cout << "depth " << i << ", time = " << delta_timestamp_us(t0,t1)/1000000.0 << " s\n";
    }
}

void test_minimax_checkmates() {
    bool pass1 = false;
    bool pass2 = false;

    const char* fen1 = "rnb1kbnr/ppppqppp/8/8/8/7P/PPPPB1PP/RNB4K b kq - 0 1\n";
    GameStateNode1 game1(NULL, 0, fen1);
    for (int i=1; i<=10; i++) {
        auto r = minimax(&game1, 6);
        game1 = r.best_child;
        BoardState::print(&game1.board);
        if (gameover(game1)) {
            pass1 = (i==3);
            break;
        }
    }

    const char* fen2 = "nb5k/ppppn1pp/6pr/8/5b2/8/PPPPQPPP/RNB1KBNR w KQ - 0 1\n";
    GameStateNode1 game2(NULL, 0, fen2);
    for (int i=1; i<=10; i++) {
        auto r = minimax(&game2, 6);
        game2 = r.best_child;
        BoardState::print(&game2.board);
        if (gameover(game2)) {
            pass2 = (i==5);
            break;
        }
    }

    if (pass1 && pass2) {
        std::cout << "pass\n";
    }
    else {
        std::cout << "fail\n";
    }
}

/*/////////////////////////////////////////////////////////////////////////////
                          Section: Main
/*/////////////////////////////////////////////////////////////////////////////

int main() {
    init_engine();
    std::cout << "starting hand_tuned_model.cpp\n";

    test_minimax_timings();

    return 0;
}
