#include "sparse_matrix.h"
#include <iostream>
#include <vector>
#include <chrono>

// Simple serial sparse matrix-vector multiplication
void spmv_serial(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < A.n; i++) {
        y[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            y[i] += A.values[j] * x[A.col_idx[j]];
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

int main() {
    std::cout << "=== NLPKKT240 Sparse Matrix-Vector Multiplication Example ===" << std::endl;
    
    // File paths - read from shared directory (not copied locally)
    const std::string matrix_file = "/scratch/public/sparse_matrix_data/nlpkkt240_matrix.bin";
    const std::string vector_file = "/scratch/public/sparse_matrix_data/nlpkkt240_vector.bin";
    
    // Load matrix and vector
    CSRMatrix A;
    std::vector<double> x, y;
    
    std::cout << "\nLoading matrix from: " << matrix_file << std::endl;
    if (!load_matrix(matrix_file, A)) {
        std::cerr << "Failed to load matrix." << std::endl;
        std::cerr << "Make sure you have read access to /scratch/public/sparse_matrix_data/" << std::endl;
        return -1;
    }
    
    std::cout << "Loading vector from: " << vector_file << std::endl;
    if (!load_vector(vector_file, x)) {
        std::cerr << "Failed to load vector." << std::endl;
        return -1;
    }
    
    // Print matrix information
    print_matrix_info(A);
    
    // Prepare result vector
    y.resize(A.n, 0.0);
    
    // Perform serial SpMV
    std::cout << "\nPerforming serial sparse matrix-vector multiplication..." << std::endl;
    spmv_serial(A, x, y);
    
    // Verify result (check a few values)
    std::cout << "\nSample results:" << std::endl;
    std::cout << "y[0] = " << y[0] << std::endl;
    std::cout << "y[" << A.n/2 << "] = " << y[A.n/2] << std::endl;
    std::cout << "y[" << A.n-1 << "] = " << y[A.n-1] << std::endl;
    
    return 0;
}
