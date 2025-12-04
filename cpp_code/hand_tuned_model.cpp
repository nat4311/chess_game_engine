#define OVERRIDE_ENGINE_CPP_MAIN
#include "engine.cpp"
#include <limits>
#include <unistd.h>
#include <omp.h>

#define gameover(game) (game.board.state == DRAW || game.board.state == WHITE_WIN || game.board.state == BLACK_WIN)
float inf = std::numeric_limits<double>::infinity();

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
            // TODO: remove this debug stuff
            if (!BoardState::make(&board, prev_move)) {
                std::cout << "new GameStateNode1 failed to create: make(prev_move) failed";

                std::cout << "prev_board:\n";
                BoardState::print(prev_board);

                std::cout << "occupancies both\n";
                print_bitboard(prev_board->occupancies[BOTH]);

                std::cout << "occupancies white\n";
                print_bitboard(prev_board->occupancies[WHITE]);

                std::cout << "occupancies black\n";
                print_bitboard(prev_board->occupancies[BLACK]);

                std::cout << "prev_move:\n";
                print_move(prev_move, true);

                throw(1);
            }
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

struct minimax_result1 {
    GameStateNode1 node;
    float eval;
};

minimax_result1 _minimax1(GameStateNode1* node, int depth, bool maximizing_player=true, float alpha=-inf, float beta=inf) {
    BoardState::generate_l_moves(&node->board);
    BoardState::sort_l_moves(&node->board);
    if (depth==0 || node->board.state==DRAW || node->board.state==WHITE_WIN || node->board.state==BLACK_WIN) {
        minimax_result1 res = {*node, node->eval()};
        return res;
    }

    minimax_result1 best_result;
    if (maximizing_player) {
        best_result.eval = -inf;
        for (int i = 0; i < node->board.l.moves_found; i++) {
            GameStateNode1 child = GameStateNode1(&node->board, node->board.l.move_list[i], "");
            auto res = _minimax1(&child, depth-1, false, alpha, beta);
            if (res.eval > best_result.eval) {
                best_result.eval = res.eval;
                best_result.node = child;
            }
            if (res.eval > alpha) {
                alpha = res.eval;
            }
            alpha = std::max(alpha, res.eval);
            if (beta <= alpha) {
                break;
            }
        }
        assert (best_result.eval != -inf);
    }
    else {
        best_result.eval = inf;
        for (int i = 0; i < node->board.l.moves_found; i++) {
            GameStateNode1 child = GameStateNode1(&node->board, node->board.l.move_list[i], "");
            auto res = _minimax1(&child, depth-1, true, alpha, beta);
            if (res.eval < best_result.eval) {
                best_result.eval = res.eval;
                best_result.node = child;
            }
            beta = std::min(beta, res.eval);
            if (beta <= alpha) {
                break;
            }
        }
        assert (best_result.eval != inf);
    }
    return best_result;
}

minimax_result1 minimax1(GameStateNode1* root, int depth) {
    bool maximizing_player = (root->board.turn == WHITE);
    minimax_result1 best = _minimax1(root, depth, maximizing_player);
    return best;
}

minimax_result1 minimax1_omp(GameStateNode1* root, int depth) {
    bool maximizing_player = (root->board.turn == WHITE);
    minimax_result1 best;
    BoardState::generate_l_moves(&root->board);
    int n = root->board.l.moves_found;
    minimax_result1 results[n];

    #pragma omp parallel for num_threads(16) schedule(dynamic)
    for (int i = 0; i < n; i++) {
        U32 move = root->board.l.move_list[i];
        GameStateNode1 child = GameStateNode1(&root->board, move, "");
        // store result in array
        minimax_result1 child_result = minimax1(&child, depth-1);
        results[i].node = child;
        results[i].eval = child_result.eval;
    }

    // loop through array and pick best
    if (maximizing_player) {
        float best_eval = -inf;
        for (int i = 0; i < n; i++) {
            if (results[i].eval > best_eval) {
                best_eval = results[i].eval;
                best = results[i];
            }
        }
    }
    else {
        float best_eval = inf;
        for (int i = 0; i < n; i++) {
            if (results[i].eval < best_eval) {
                best_eval = results[i].eval;
                best = results[i];
            }
        }
    }

    return best;
}

/*/////////////////////////////////////////////////////////////////////////////
                          Section: Unit tests
/*/////////////////////////////////////////////////////////////////////////////

void test_minimax1_timings(int max_depth = 8) {
    GameStateNode1 game;

    for (int i = 1; i<=max_depth; i++) {
        auto t0 = timestamp();
        minimax1(&game, i);
        auto t1 = timestamp();
        std::cout << "depth " << i << ", time = " << delta_timestamp_us(t0,t1)/1000000.0 << " s\n";
    }
}

void test_minimax1_checkmates() {
    bool pass1 = false;
    bool pass2 = false;

    const char* fen1 = "rnb1kbnr/ppppqppp/8/8/8/7P/PPPPB1PP/RNB4K b kq - 0 1\n";
    GameStateNode1 game1(NULL, 0, fen1);
    for (int i=1; i<=10; i++) {
        auto r = minimax1(&game1, 6);
        game1 = r.node;
        BoardState::print(&game1.board);
        if (gameover(game1)) { pass1 = (i==3); break; }
    }

    const char* fen2 = "nb5k/ppppn1pp/6pr/8/5b2/8/PPPPQPPP/RNB1KBNR w KQ - 0 1\n";
    GameStateNode1 game2(NULL, 0, fen2);
    for (int i=1; i<=10; i++) {
        auto r = minimax1(&game2, 6);
        game2 = r.node;
        BoardState::print(&game2.board);
        if (gameover(game2)) { pass2 = (i==5); break; }
    }

    if (pass1 && pass2) { std::cout << "pass\n"; }
    else { std::cout << "fail\n"; }
}

void test_minimax1_omp_timings(int max_depth = 12) {
    GameStateNode1 game;

    for (int i = 1; i<=max_depth; i++) {
        auto t0 = timestamp();
        minimax1_omp(&game, i);
        auto t1 = timestamp();
        std::cout << "depth " << i << ", time = " << delta_timestamp_us(t0,t1)/1000000.0 << " s\n";
    }
}

void test_minimax1_omp_checkmates() {
    bool pass1 = false;
    bool pass2 = false;

    const char* fen1 = "rnb1kbnr/ppppqppp/8/8/8/7P/PPPPB1PP/RNB4K b kq - 0 1\n";
    GameStateNode1 game1(NULL, 0, fen1);
    BoardState::print(&game1.board);
    for (int i=1; i<=10; i++) {
        auto r = minimax1_omp(&game1, 6);
        game1 = r.node;
        BoardState::print(&game1.board);
        if (gameover(game1)) { pass1 = (i==3); break; }
    }

    std::cout << "====================\n";

    const char* fen2 = "nb5k/ppppn1pp/6pr/8/5b2/8/PPPPQPPP/RNB1KBNR w KQ - 0 1\n";
    GameStateNode1 game2(NULL, 0, fen2);
    BoardState::print(&game2.board);
    for (int i=1; i<=10; i++) {
        auto r = minimax1_omp(&game2, 6);
        game2 = r.node;
        BoardState::print(&game2.board);
        if (gameover(game2)) { pass2 = (i==5); break; }
    }

    if (pass1 && pass2) { std::cout << "pass\n"; }
    else { std::cout << "fail\n"; }
}

/*/////////////////////////////////////////////////////////////////////////////
                          Section: Main
/*/////////////////////////////////////////////////////////////////////////////

int main() {
    init_engine();
    std::cout << "starting hand_tuned_model.cpp\n";

    test_minimax1_omp_timings();
    // test_minimax1_omp_checkmates();

    // DEBUG 2
    // GameStateNode1 root;
    // U32 move = 67892;
    // // GameStateNode1 child = GameStateNode1(&root.board, move, "", &root);
    // GameStateNode1 child = GameStateNode1(&root.board, move);
    // BoardState::print(&child.board);
    // // minimax1(&child, 8);
    //
    // move = 30150;
    // // GameStateNode1 child2 = GameStateNode1(&child.board, move, "", &child);
    // GameStateNode1 child2 = GameStateNode1(&child.board, move);
    // BoardState::print(&child2.board);
    // // minimax1(&child2, 7);
    //
    // move = 1828;
    // // GameStateNode1 child3 = GameStateNode1(&child2.board, move, "", &child2);
    // GameStateNode1 child3 = GameStateNode1(&child2.board, move);
    // BoardState::print(&child3.board);
    // // minimax1(&child3, 6);
    //
    // move = 91851;
    // // GameStateNode1 child4 = GameStateNode1(&child3.board, move, "", &child3);
    // GameStateNode1 child4 = GameStateNode1(&child3.board, move);
    // BoardState::print(&child4.board);
    // std::cout << "before\n";
    // print_bitboard(child4.board.occupancies[BOTH]);
    // // minimax1(&child4, 5);
    // BoardState::generate_l_moves(&child4.board);
    // // BoardState::sort_l_moves(&child4.board);
    // // for (int i = 0; i < child4.board.l.moves_found; i++) {
    // //     GameStateNode1 child = GameStateNode1(&child4.board, child4.board.l.move_list[i], "", &child4);
    // // }
    // std::cout << "after\n";
    // print_bitboard(child4.board.occupancies[BOTH]);

    // END DEBUG 2

    // /////////////// DEBUG
    // GameStateNode1 game;
    // BoardState::print(&game.board);
    // print_bitboard(game.board.occupancies[BOTH]);
    //
    // U32 move = 67892;
    // print_move(move, true);
    // std::cout << "\n\n";
    // game = GameStateNode1(&game.board, move);
    // BoardState::print(&game.board);
    // print_bitboard(game.board.occupancies[BOTH]);
    //
    // move = 30150;
    // print_move(move, true);
    // std::cout << "\n\n";
    // game = GameStateNode1(&game.board, move);
    // BoardState::print(&game.board);
    // print_bitboard(game.board.occupancies[BOTH]);
    //
    // move = 1828;
    // print_move(move, true);
    // std::cout << "\n\n";
    // game = GameStateNode1(&game.board, move);
    // BoardState::print(&game.board);
    // print_bitboard(game.board.occupancies[BOTH]);
    //
    // move = 91851;
    // print_move(move, true);
    // std::cout << "\n\n";
    // game = GameStateNode1(&game.board, move);
    // BoardState::print(&game.board);
    // print_bitboard(game.board.occupancies[BOTH]);
    // /////////////// END DEBUG



    // test_minimax1_omp_checkmates();

    // const char* fen = "rnb1kbnr/ppp1pppp/8/8/8/8/qPPPPPPP/RNBQKBNR w KQkq - 0 1\n";
    // BoardState board;
    // BoardState::load(&board, fen);
    // BoardState::print_l_moves(&board);
    // sort_l_moves(&board);
    // BoardState::print_l_moves(&board);

    return 0;
}
