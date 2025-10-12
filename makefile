all:
	g++ -DMAIN -O3 src/engine.cpp src/attacks.cpp -o build/engine_test

debug:
	g++ -DMAIN -g src/engine.cpp src/attacks.cpp -o build/engine_test

test:
	g++ -DMAIN src/engine.cpp src/attacks.cpp -o build/engine_test
	./build/engine_test
