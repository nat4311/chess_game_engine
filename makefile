all:
	gcc -O3 src/engine.cpp -o build/engine

debug:
	gcc -g src/engine.cpp -o build/engine_debug

test:
	gcc src/engine.cpp -o build/engine_test
	./build/engine_test
