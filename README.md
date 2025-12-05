########################################################################################################################
                                                Important Files
########################################################################################################################

python_code

    hand_tuned_model.py
        this was the python version of my baseline chess engine (hand crafted evaluation + minimax search)

    alphazero.py
        This is where I tried making the alphazero style chess engine (resnet + monte carlo tree search)
        running this file runs self play training
        I was planning to reuse this same neural network for the stockfish style engine (neural network + minimax search)

    alphazero2.py
        I also tried having stockfish generate training data to see if the network would improve this way
    
    game_engine.cpython-313-x86_64-linux-gnu.so*
        this is my move generator c++ module exposed with pybind11

    model_evaluation.py
        this is where I tested my models against stockfish

    
cpp_code

    hand_tuned_model.cpp
        this was the c++ version of my baseline chess engine (hand crafted evaluation + minimax search)

    bindings.cpp
        pybind11 module generation code
    
    engine.cpp
        this was the main file for testing my move generation - it contains the important classes I exposed to python
