#include "engine.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

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

py::array_t<U8> get_model_input(BoardState* board) {
    py::array_t<U8> arr({21, 8, 8});
    auto buf = arr.mutable_unchecked<3>();

    // piece bitboards
    for (int piece_type = 0; piece_type < 12; piece_type++) {
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                int sq = y * 8 + x;
                buf(piece_type, x, y) = (board->bitboards[piece_type] >> sq) & 1;
            }
        }
    }

    // todo: repetitions
    U8* ptr = static_cast<U8*>(arr.request().ptr);
    std::memset(ptr + 12*64, 0, 2*64*sizeof(U8));

    // turn
    std::memset(ptr + 14*64, board->turn, 64*sizeof(U8));

    // todo: total_moves
    std::memset(ptr + 15*64, board->turn, 64*sizeof(U8));

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

    // todo: make, unmake
    py::class_<BoardState>(m, "BoardState")
        .def(py::init<>())
        .def("print", &BoardState::print, "print the board state to terminal")
        .def("reset", &BoardState::reset, "reset the board state to start position")
        .def("load", &BoardState::load, "load a fen string", py::arg("fen_str"))
        .def("copy", &BoardState::copy, "copy by value of a board state")
        .def("make", &BoardState::make, "make a move")
        .def("get_bitboards", &get_bitboards, "get all 12 piece bitboards as 8x8x12 bool array")
        .def("get_model_input", &get_model_input, "get partial model input as 21x8x8 U8 array")
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
