all:
	# g++ -DMAIN -O3 cpp_code/engine.cpp cpp_code/attacks.cpp -o build/engine_test
	g++ -DMAIN -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) cpp_code/engine.cpp cpp_code/attacks.cpp -o python_code/game_engine$$(python3 -m pybind11 --extension-suffix)

debug:
	g++ -DMAIN -g cpp_code/engine.cpp cpp_code/attacks.cpp -o build/engine_test

test:
	g++ -DMAIN -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes)  cpp_code/bindings.cpp cpp_code/engine.cpp cpp_code/attacks.cpp -o python_code/game_engine$$(python3 -m pybind11 --extension-suffix)
	cd python_code && python3 test.py

unit_test:
	g++ -DMAIN cpp_code/engine.cpp cpp_code/attacks.cpp -o build/engine_test
	./build/engine_test
