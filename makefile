all:
	g++ -DMAIN -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) cpp_code/bindings.cpp cpp_code/attacks.cpp -o python_code/game_engine$$(python3 -m pybind11 --extension-suffix)

debug:
	g++ -DMAIN -g -O0 -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes)  cpp_code/bindings.cpp cpp_code/attacks.cpp -o python_code/game_engine$$(python3 -m pybind11 --extension-suffix)
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --log-file=valgrind.log python3 python_code/alphazero.py

test:
	# g++ -DMAIN -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) cpp_code/bindings.cpp cpp_code/attacks.cpp -o python_code/game_engine$$(python3 -m pybind11 --extension-suffix)
	g++ -DMAIN cpp_code/engine.cpp cpp_code/attacks.cpp -o build/engine_test
	./build/engine_test
	# cd python_code && python3 test.py

unit_test:
	g++ -DMAIN cpp_code/engine.cpp cpp_code/attacks.cpp -o build/engine_test
	./build/engine_test
