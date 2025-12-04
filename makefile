all:
	g++ -DMAIN -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) cpp_code/bindings.cpp cpp_code/attacks.cpp -o python_code/game_engine$$(python3 -m pybind11 --extension-suffix)

debug:
	g++ -DMAIN -g -O0 -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes)  cpp_code/bindings.cpp cpp_code/attacks.cpp -o python_code/game_engine$$(python3 -m pybind11 --extension-suffix)
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --log-file=valgrind.log python3 python_code/alphazero.py

test:
	# g++ -DMAIN -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) cpp_code/bindings.cpp cpp_code/attacks.cpp -o python_code/game_engine$$(python3 -m pybind11 --extension-suffix)
	# cd python_code && python3 test.py
	g++ -DMAIN -O3 -Wall cpp_code/engine.cpp cpp_code/attacks.cpp -o build/engine_test
	./build/engine_test

engine_test:
	g++ -O3 -Wall -DMAIN cpp_code/engine.cpp cpp_code/attacks.cpp -o build/engine_test
	./build/engine_test

hand_test:
	g++ -DMAIN -fopenmp -O3 -Wall cpp_code/hand_tuned_model.cpp cpp_code/attacks.cpp -o build/hand_tuned_model
	./build/hand_tuned_model

profile:
	g++ -DMAIN -g -O0 -Wall cpp_code/hand_tuned_model.cpp cpp_code/attacks.cpp -o build/hand_tuned_model
	perf stat -d -d ./build/hand_tuned_model > profiling_results/perf.txt 2>&1
	perf record ./build/hand_tuned_model
	perf report --stdio >> profiling_results/perf.txt

	g++ -DMAIN -g -O3 -Wall cpp_code/hand_tuned_model.cpp cpp_code/attacks.cpp -o build/hand_tuned_model
	perf stat -d -d ./build/hand_tuned_model > profiling_results/perf_optimized.txt 2>&1
	perf record ./build/hand_tuned_model
	perf report --stdio >> profiling_results/perf_optimized.txt

