clean:
	rm *.out

test:
	gcc bench_single.c utils.c operation.c -o tester_single.out -lopenblas -lm -g
	gcc bench_multi.c utils.c operation.c -o tester_multi.out -lopenblas -lm -g