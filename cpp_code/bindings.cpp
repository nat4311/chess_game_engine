#define BINDINGS_CPP
#include "engine.cpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

// TODO: DO THE PYBIND CODE
PYBIND11_MODULE(game_engine, m, py::mod_gil_not_used()) {
    m.doc() = "chess game engine module with pybind11"; // optional module docstring

    m.def("init_engine", &init_engine, "initialize the game engine");
}

