#define OVERRIDE_ENGINE_CPP_MAIN
#include "engine.cpp"
#include <limits>

float inf = std::numeric_limits<double>::infinity();

struct GameStateNode1 {
    // NULL_STATE, BLACK_WIN, DRAW, WHITE_WIN, NOTOVER
    int state;

    U32 prev_move;
    BoardState board;

    GameStateNode1(BoardState* prev_board = NULL, U32 prev_move = 0, char* fen="") {
        state = NULL_STATE;
        this->prev_move = prev_move;
        if (prev_board == NULL) {
            board = BoardState();
            if (fen != "") {
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
        state = BoardState::get_state(&board);

        return n_legal_moves;
    }

    float eval() {
        int mobility_score = count_legal_moves();

        if (state == DRAW) {
            return 0.0;
        }
        else if (state == WHITE_WIN) {
            return 10000.0 - (float)board.turn_no;
        }
        else if (state == BLACK_WIN) {
            return -10000.0 + (float)board.turn_no;
        }

        return
            (float)BoardState::material_score(&board) +
            0.5*(float)BoardState::doubled_pawn_score(&board) +
            0.1*(float)(mobility_score - BoardState::enemy_mobility_score(&board));
    }
};

typedef struct minimax_result {
    GameStateNode1 best_child;
    float child_eval;
};

minimax_result _minimax(GameStateNode1* node, int depth, bool maximizing_player=true, float alpha=-inf, float beta=inf) {
    minimax_result tmp_result;
    minimax_result best_result;

    if (node->state == NULL_STATE) {
        BoardState::generate_l_moves(&node->board);
    }
    if (depth==0 || node->state==DRAW || node->state==WHITE_WIN || node->state==BLACK_WIN) {
        tmp_result.best_child = *node;
        tmp_result.child_eval = node->eval();
    }

    if (maximizing_player) {
        float max_eval = -inf;
        for (int i = 0; i < node->board.l.moves_found; i++) {
            GameStateNode1 child = GameStateNode1(&node->board, node->board.l.move_list[i]);
            tmp_result = _minimax(&child, depth-1, alpha, beta, false);
            if (tmp_result.child_eval > max_eval) {
                max_eval = tmp_result.child_eval;
                best_result = tmp_result;
            }
            if (tmp_result.child_eval > alpha) {
                alpha = tmp_result.child_eval;
            }
            if (beta <= alpha) {
                break;
            }
        }
        return best_result;
    }
    else {
        float min_eval = inf;
        for (int i = 0; i < node->board.l.moves_found; i++) {
            GameStateNode1 child = GameStateNode1(&node->board, node->board.l.move_list[i]);
            tmp_result = _minimax(&child, depth-1, alpha, beta, false);
            if (tmp_result.child_eval < min_eval) {
                min_eval = tmp_result.child_eval;
                best_result = tmp_result;
            }
            if (tmp_result.child_eval < beta) {
                beta = tmp_result.child_eval;
            }
            if (beta <= alpha) {
                break;
            }
        }
        return best_result;
    }
}

minimax_result minimax(GameStateNode1* root, int depth) {
    bool maximizing_player = (root->board.turn == WHITE);
    minimax_result best = _minimax(root, depth, maximizing_player);
    return best;
}

int main() {
    init_engine();
    std::cout << "starting hand_tuned_model.cpp\n";

    return 0;
}


