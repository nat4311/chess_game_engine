#include "engine.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

py::array_t<U32> get_pl_move_list(MoveGenerator &self, BoardState* board) {
    self.generate_pl_moves(board);
    size_t size = self.pl_moves_found;
    const U32* data_ptr = self.pl_move_list;
    return py::array_t<U32>(
        size,
        data_ptr
    );
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
        .def("__repr__", [](const BoardState &a){ return "<BoardState object>"; } );

    py::class_<MoveGenerator>(m, "MoveGenerator")
        .def(py::init<>())
        .def("generate_pl_moves", &MoveGenerator::generate_pl_moves, "generate pseudolegal moves -> retrieve the moves with this.get_pl_move_list")
        .def("get_pl_move_list", &get_pl_move_list, "array of pseudo legal moves")
        .def("print_pl_moves", &MoveGenerator::print_pl_moves, py::arg("piece_type") = 12, "print the pseudo legal moves for a specific piece")
        .def("__repr__", [](const MoveGenerator &a){ return "<MoveGenerator object>"; } );

    init_engine();
}
