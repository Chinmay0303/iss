# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -O3
LDFLAGS = -lm

# OpenMP flags
OMPFLAGS = -fopenmp

# Source files
SEQ_SRC = kmeans_sequential.c
PAR_SRC = kmeans_parallel.c

# Executables
SEQ_EXE = kmeans_sequential
PAR_EXE = kmeans_parallel

.PHONY: all clean

all: $(SEQ_EXE) $(PAR_EXE)

# Rule to build the sequential executable
$(SEQ_EXE): $(SEQ_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to build the parallel executable
$(PAR_EXE): $(PAR_SRC)
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS) $(OMPFLAGS)

# Rule to clean up generated files
clean:
	rm -f $(SEQ_EXE) $(PAR_EXE) *.o *.log *.csv