.DEFAULT_GOAL := all

clean:
	rm 0_hello_world
	rm 1_add
	rm 2_add_parallel
	rm 3_add_parallel_threads

all:
	nvcc 0_hello_world.cu -o 0_hello_world
	nvcc 1_add.cu -o 1_add
	nvcc 2_add_parallel.cu -o 2_add_parallel
	nvcc 3_add_parallel_threads.cu -o 3_add_parallel_threads

