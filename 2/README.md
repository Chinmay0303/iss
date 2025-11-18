# NLPKKT240 Sparse Matrix Data

## Overview
This directory contains the NLPKKT240 sparse matrix and corresponding vector for educational purposes in HPC and parallel computing courses.

## Files
- `nlpkkt240_matrix.bin` - Binary CSR format sparse matrix (28M×28M, ~401M non-zeros)
- `nlpkkt240_vector.bin` - Binary format input vector
- `sparse_matrix.h` - Header file with CSR structure and function declarations
- `sparse_matrix.cpp` - Implementation of matrix/vector loading functions
- `main.cpp` - Example serial sparse matrix-vector multiplication
- `Makefile` - Build system with instructions

## Matrix Information
- **Name**: NLPKKT240 (from SuiteSparse Collection)
- **Size**: 27,993,600 × 27,993,600
- **Non-zeros**: ~401 million
- **Memory**: ~12 GB
- **Type**: Optimization problem matrix

## Quick Start

### 1. Copy files to your working directory:
```bash
cp /scratch/public/sparse_matrix_data/* .
```

### 2. Build and run the example:
```bash
make
make test
```

### 3. Get help with build options:
```bash
make help
```



## File Format Details

### Matrix Format (CSR - Compressed Sparse Row)
Binary file structure:
1. `int n` - Matrix dimension
2. `int nnz` - Number of non-zeros
3. `int row_ptr[n+1]` - Row pointers
4. `int col_idx[nnz]` - Column indices
5. `double values[nnz]` - Matrix values

### Vector Format
Binary file structure:
1. `int n` - Vector size
2. `double values[n]` - Vector data

## Notes
- Files are read-only - copy to your directory to modify
- Use SLURM for parallel jobs on the cluster
- Monitor memory usage - this is a large matrix
- Consider using multiple nodes for MPI experiments