CC=gcc
CFLAGS=-fPIC -shared

libconvertDimension.so: convertDimension.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm libconvertDimension.so

run_docker:
	docker run --oom-kill-disable  --rm -v "$$(pwd)":/work docker-nvidia-test 

run_docker_shell:
	docker run -it --rm -v "$$(pwd)":/work docker-nvidia-test /bin/bash

build_docker:
	docker build -t docker-nvidia-test .
