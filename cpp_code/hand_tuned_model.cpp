#define OVERRIDE_ENGINE_CPP_MAIN
#include "engine.cpp"
#include <limits>
#include <unistd.h>

float inf = std::numeric_limits<double>::infinity();

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

#define gameover(game) (game.board.state == DRAW || game.board.state == WHITE_WIN || game.board.state == BLACK_WIN)

int main() {
    init_engine();
    std::cout << "starting hand_tuned_model.cpp\n";

    GameStateNode1 game;

    //TODO: measure timings from start pos
    for (int i = 1; i<=10; i++) {
        auto t0 = timestamp();
        minimax(&game, i);
        auto t1 = timestamp();
        std::cout << "depth " << i << ", time = " << delta_timestamp_us(t0,t1)/1000000.0 << " s\n";
    }
    // for (int i=1; i<=10; i++) {
    //     // std::cout << "i: " << i << std::endl;
    //     auto r = minimax(&game, 6);
    //     game = r.best_child;
    //     BoardState::print(&game.board);
    //     if (gameover(game)) {
    //         break;
    //     }
    // }

    return 0;
}
