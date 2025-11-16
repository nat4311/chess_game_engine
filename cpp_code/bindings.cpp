#include "engine.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// 73 move types.
// queen type move (8 dir x 7 dist = 56).
// knight move (8 dir).
// pawn underpromotion (3 dir x 3 promotion_piece_type = 9).
// 56 + 8 + 9 = 73
int policy_move_index_0(U32 move) {
    int source_sq = decode_move_source_sq(move);
    int target_sq = decode_move_target_sq(move);
    int piece_type = decode_move_piece_type(move);
    int promotion = decode_move_promotion(move);
    int promotion_piece_type = decode_move_promotion_piece_type(move);
    int dir = target_sq - source_sq;

    if (piece_type == WHITE_KNIGHT || piece_type == BLACK_KNIGHT) {
        if (dir == -15) {
            return 56;
        }
        else if (dir == -6) {
            return 57;
        }
        else if (dir == 10) {
            return 58;
        }
        else if (dir == 17) {
            return 59;
        }
        else if (dir == 15) {
            return 60;
        }
        else if (dir == 6) {
            return 61;
        }
        else if (dir == -10) {
            return 62;
        }
        else if (dir == -17) {
            return 63;
        }
        else {
            throw std::runtime_error("policy_move_index_0() error: invalid knight move direction");
        }
    }

    if ((piece_type == WHITE_PAWN || piece_type == BLACK_PAWN) && promotion && (promotion_piece_type!=WHITE_QUEEN) && (promotion_piece_type!=BLACK_QUEEN)) {
        if (promotion_piece_type == WHITE_KNIGHT) {
            if (dir == -9) { return 64; }
            if (dir == -8) { return 65; }
            if (dir == -7) { return 66; }
        }
        else if (promotion_piece_type == BLACK_KNIGHT) {
            if (dir == 7) { return 64; }
            if (dir == 8) { return 65; }
            if (dir == 9) { return 66; }
        }
        else if (promotion_piece_type == WHITE_BISHOP) {
            if (dir == -9) { return 67; }
            if (dir == -8) { return 68; }
            if (dir == -7) { return 69; }
        }
        else if (promotion_piece_type == BLACK_BISHOP) {
            if (dir == 7) { return 67; }
            if (dir == 8) { return 68; }
            if (dir == 9) { return 69; }
        }
        else if (promotion_piece_type == WHITE_ROOK) {
            if (dir == -9) { return 70; }
            if (dir == -8) { return 71; }
            if (dir == -7) { return 72; }
        }
        else if (promotion_piece_type == BLACK_ROOK) {
            if (dir == 7) { return 70; }
            if (dir == 8) { return 71; }
            if (dir == 9) { return 72; }
        }
        else {
            throw std::runtime_error("policy_move_index_0() error: invalid promotion_piece_type");
        }
    }

    // queen type moves
    if (dir == -8)  { return 0; }
    if (dir == -16) { return 1; }
    if (dir == -24) { return 2; }
    if (dir == -32) { return 3; }
    if (dir == -40) { return 4; }
    if (dir == -48) { return 5; }
    if (dir == -56) { return 6; }

    if (dir == -7)  { return 7; }
    if (dir == -14) { return 8; }
    if (dir == -21) { return 9; }
    if (dir == -28) { return 10; }
    if (dir == -35) { return 11; }
    if (dir == -42) { return 12; }
    if (dir == -49) { return 13; }

    if (dir == 1)   { return 14; }
    if (dir == 2)   { return 15; }
    if (dir == 3)   { return 16; }
    if (dir == 4)   { return 17; }
    if (dir == 5)   { return 18; }
    if (dir == 6)   { return 19; }
    if (dir == 7)   { return 20; }

    if (dir == 9)   { return 21; }
    if (dir == 18)  { return 22; }
    if (dir == 27)  { return 23; }
    if (dir == 36)  { return 24; }
    if (dir == 45)  { return 25; }
    if (dir == 54)  { return 26; }
    if (dir == 63)  { return 27; }

    if (dir == 8)   { return 28; }
    if (dir == 16)  { return 29; }
    if (dir == 24)  { return 30; }
    if (dir == 32)  { return 31; }
    if (dir == 40)  { return 32; }
    if (dir == 48)  { return 33; }
    if (dir == 56)  { return 34; }

    if (dir == 7)   { return 35; }
    if (dir == 14)  { return 36; }
    if (dir == 21)  { return 37; }
    if (dir == 28)  { return 38; }
    if (dir == 35)  { return 39; }
    if (dir == 42)  { return 40; }
    if (dir == 49)  { return 41; }

    if (dir == -1)  { return 42; }
    if (dir == -2)  { return 43; }
    if (dir == -3)  { return 44; }
    if (dir == -4)  { return 45; }
    if (dir == -5)  { return 46; }
    if (dir == -6)  { return 47; }
    if (dir == -7)  { return 48; }

    if (dir == -9)  { return 49; }
    if (dir == -18) { return 50; }
    if (dir == -27) { return 51; }
    if (dir == -36) { return 52; }
    if (dir == -45) { return 53; }
    if (dir == -54) { return 54; }
    if (dir == -63) { return 55; }

    throw std::runtime_error("policy_move_index_0() error: did not find an index");
}

// 64 source squares
int policy_move_index_1(U32 move) {
    return decode_move_source_sq(move);
}

py::array_t<U32> get_pl_move_list(MoveGenerator &self, BoardState* board) {
    self.generate_pl_moves(&self, board);
    size_t size = self.pl_moves_found;
    const U32* data_ptr = self.pl_move_list;
    return py::array_t<U32>(size, data_ptr);
}

py::array_t<bool> get_bitboards(BoardState &self) {
    // piece_type, sq
    bool data[12][64];
    for (int piece=0; piece<=11; piece++) {
        for (int sq = 0; sq <= 63; sq++) {
            data[piece][sq] = 1 & (self.bitboards[piece] >> sq);
        }
    }
    return py::array_t<bool>({12, 64}, &data[0][0]);
}

py::array_t<U8> get_partial_model_input(BoardState* board) {
    py::array_t<U8> arr({21, 8, 8});
    auto buf = arr.mutable_unchecked<3>();

    // piece bitboards
    for (int piece_type = 0; piece_type < 12; piece_type++) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                int sq = y * 8 + x;
                buf(piece_type, y, x) = (board->bitboards[piece_type] >> sq) & 1;
            }
        }
    }

    // todo: repetitions
    U8* ptr = static_cast<U8*>(arr.request().ptr);
    std::memset(ptr + 12*64, 0, 2*64*sizeof(U8));

    // turn - White or Black
    std::memset(ptr + 14*64, board->turn, 64*sizeof(U8));

    // turn_no - capped at 255
    std::memset(ptr + 15*64, (board->turn_no > 255? 255 : board->turn_no), 64*sizeof(U8));

    bool castle_K = board->castling_rights & WHITE_CASTLE_KINGSIDE;
    bool castle_Q = board->castling_rights & WHITE_CASTLE_QUEENSIDE;
    bool castle_k = board->castling_rights & BLACK_CASTLE_KINGSIDE;
    bool castle_q = board->castling_rights & BLACK_CASTLE_QUEENSIDE;

    // castling
    std::memset(ptr + 16*64, board->castling_rights & castle_K, 64*sizeof(U8));
    std::memset(ptr + 17*64, board->castling_rights & castle_Q, 64*sizeof(U8));
    std::memset(ptr + 18*64, board->castling_rights & castle_k, 64*sizeof(U8));
    std::memset(ptr + 19*64, board->castling_rights & castle_q, 64*sizeof(U8));

    // halfmove
    std::memset(ptr + 20*64, board->halfmove, 64*sizeof(U8));

    return arr;
}

PYBIND11_MODULE(game_engine, m, py::mod_gil_not_used()) {
    m.doc() = "chess game engine module with pybind11"; // optional module docstring

    m.def("init_engine", &init_engine, "initialize the game engine");
    m.def("unit_tests", &unit_tests, "run the unit tests");
    m.def("print_move", &print_move, py::arg("U32_move"), py::arg("bool_verbose"), "print a move");
    m.def("policy_move_index_0", &policy_move_index_0, "get the 73 index");
    m.def("policy_move_index_1", &policy_move_index_1, "get the 64 index");

    py::class_<BoardState>(m, "BoardState")
        .def(py::init<>())
        .def("print", &BoardState::print, "print the board state to terminal")
        .def("reset", &BoardState::reset, "reset the board state to start position")
        .def("load", &BoardState::load, "load a fen string", py::arg("fen_str"))
        .def("copy", &BoardState::copy, "copy by value of a board state")
        .def("make", &BoardState::make, "make a move")
        .def("king_is_attacked", &BoardState::king_is_attacked, "returns True if king is in check")
        .def("get_bitboards", &get_bitboards, "get all 12 piece bitboards as 8x8x12 bool array")
        .def("get_partial_model_input", &get_partial_model_input, "get partial model input as 21x8x8 U8 array")
        .def_readonly("halfmove", &BoardState::halfmove)
        .def_readonly("turn_no", &BoardState::turn_no)
        .def("__repr__", [](const BoardState &a){ return "<BoardState object>"; } );

    py::class_<MoveGenerator>(m, "MoveGenerator")
        .def(py::init<>())
        .def("generate_pl_moves", &MoveGenerator::generate_pl_moves, "generate pseudolegal moves -> retrieve the moves with this.get_pl_move_list")
        .def("get_pl_move_list", &get_pl_move_list, "array of pseudo legal moves")
        .def("print_pl_moves", &MoveGenerator::print_pl_moves, py::arg("piece_type") = 12, "print the pseudo legal moves for a specific piece")
        .def("__repr__", [](const MoveGenerator &a){ return "<MoveGenerator object>"; } );

    // py::class_<GameStateNodeAlphazero>(m, "GameStateNodeAlphazero")
    //     .def(py::init<GameStateNodeAlphazero*, U32>(),
    //          py::arg("parent") = nullptr,
    //          py::arg("prev_move") = 0)
    //     .def("make_root", &GameStateNodeAlphazero::make_root, "make this node a root node for mcts (resets mcts variables and sets parent->NULL)")
    //     .def("add_pl_child", &GameStateNodeAlphazero::add_pl_child, "attempt to add a child node to the children array (won't do it if the move was illegal)")
    //     .def_readonly("board_state", &GameStateNodeAlphazero::board_state)
    //     .def_readonly("move_generator", &GameStateNodeAlphazero::move_generator)
    //     .def_readonly("parent", &GameStateNodeAlphazero::parent)
    //     .def_readonly("prev_move", &GameStateNodeAlphazero::prev_move)
    //     .def_readonly_static("max_n_children", &GameStateNodeAlphazero::max_n_children)
    //     .def_readonly("children", &GameStateNodeAlphazero::children)
    //     .def_readonly("n_children", &GameStateNodeAlphazero::n_children)
    //     .def_readwrite("prior", &GameStateNodeAlphazero::prior)
    //     .def_readonly("valid", &GameStateNodeAlphazero::valid)
    //     .def("__repr__", [](const GameStateNodeAlphazero &a){ return "<GameStateNodeAlphazero object>"; } );

    init_engine();
}
