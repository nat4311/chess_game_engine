#define OVERRIDE_ENGINE_CPP_MAIN
#include "engine.cpp"
#include <limits>

float inf = std::numeric_limits<double>::infinity();

struct GameStateNode1 {
    // NULL_STATE, BLACK_WIN, DRAW, WHITE_WIN, NOTOVER
    int state;

    U32 prev_move;
    BoardState board;

    GameStateNode1(BoardState* prev_board = NULL, U32 prev_move = 0, const char* fen="") {
        state = NULL_STATE;
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

struct minimax_result {
    GameStateNode1 best_child;
    float child_eval;
};

minimax_result _minimax(GameStateNode1* node, int depth, bool maximizing_player=true, float alpha=-inf, float beta=inf) {
    if (node->state == NULL_STATE) {
        BoardState::generate_l_moves(&node->board);
    }
    if (depth==0 || node->state==DRAW || node->state==WHITE_WIN || node->state==BLACK_WIN) {
        minimax_result res = {*node, node->eval()};
        return res;
    }

    minimax_result best_result;
    if (maximizing_player) {
        best_result.child_eval = -inf;
        for (int i = 0; i < node->board.l.moves_found; i++) {
            GameStateNode1 child = GameStateNode1(&node->board, node->board.l.move_list[i]);
            auto res = _minimax(&child, depth-1, alpha, beta, false);
            if (res.child_eval > best_result.child_eval) {
                best_result.child_eval = res.child_eval;
                best_result.best_child = child;
            }
            if (res.child_eval > alpha) {
                alpha = res.child_eval;
            }
            if (beta <= alpha) {
                break;
            }
        }
    }
    else {
        best_result.child_eval = inf;
        for (int i = 0; i < node->board.l.moves_found; i++) {
            GameStateNode1 child = GameStateNode1(&node->board, node->board.l.move_list[i]);
            auto res = _minimax(&child, depth-1, alpha, beta, true);
            if (res.child_eval < best_result.child_eval) {
                best_result.child_eval = res.child_eval;
                best_result.best_child = child;
            }
            if (res.child_eval < beta) {
                beta = res.child_eval;
            }
            if (beta <= alpha) {
                break;
            }
        }
    }
    return best_result;
}

minimax_result minimax(GameStateNode1* root, int depth) {
    bool maximizing_player = (root->board.turn == WHITE);
    minimax_result best = _minimax(root, depth, maximizing_player);
    return best;
}

int main() {
    init_engine();
    std::cout << "starting hand_tuned_model.cpp\n";

    GameStateNode1 game;
    BoardState::print(&game.board);
    // minimax(&game, 1)

    for (int i=1; i<=10; i++) {
        std::cout << "i: " << i << std::endl;
        auto r = minimax(&game, i);
        BoardState::print(&r.best_child.board);
    }

    return 0;
}


