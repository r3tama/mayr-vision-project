CC=gcc
CFLAGS=-fPIC -shared

libconvertDimension.so: convertDimension.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm libconvertDimension.so
