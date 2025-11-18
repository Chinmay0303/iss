#include <iostream>
#include <vector>
#include <numeric>
#include <mpi.h>
#include "sparse_matrix.h" // This file defines CSRMatrix struct

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // These pointers will hold the full matrix data on rank 0 only
    int N;
    long long nnz;
    double *val = nullptr;
    int *col_ind = nullptr;
    long long *row_ptr = nullptr;
    double *x = nullptr;
    double *y = nullptr;

    // These vectors are used by rank 0 to plan the data distribution
    std::vector<int> sendcounts_nnz, displs_nnz;
    std::vector<int> row_counts, row_displs;
    std::vector<int> row_ptr_counts, row_ptr_displs;
    
    double total_start_time, total_end_time;

    if (world_rank == 0) {
        // --- FIX STARTS HERE ---
        
        // 1. Create the objects that the load functions expect.
        CSRMatrix matrix;
        std::vector<double> x_vec;

        // 2. Call the functions correctly.
        if (!load_matrix("data.mtx", matrix)) {
            std::cerr << "Failed to load matrix." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!load_vector("data.vec", x_vec)) {
            std::cerr << "Failed to load vector." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 3. Extract the raw pointers and sizes from the objects for MPI.
        N = matrix.num_rows;
        nnz = matrix.num_non_zeros;
        val = matrix.val;
        col_ind = matrix.col_ind;
        row_ptr = matrix.row_ptr;
        x = x_vec.data(); // Get the raw pointer from the vector

        // --- FIX ENDS HERE ---

        y = new double[N]();

        // Calculate the load-balanced distribution for parallel runs
        if (world_size > 1) {
            // ... the rest of the planning logic is correct ...
// ...existing code...
            sendcounts_nnz.resize(world_size);
            displs_nnz.resize(world_size);
            row_counts.resize(world_size);
            row_displs.resize(world_size);
            row_ptr_counts.resize(world_size);
            row_ptr_displs.resize(world_size);

            long long nnz_per_process = nnz / world_size;
            int current_process = 0;
            long long current_process_nnz = 0;
            int start_row = 0;

            displs_nnz[0] = 0;
            row_displs[0] = 0;
            row_ptr_displs[0] = 0;

            for (int i = 0; i < N; ++i) {
                long long row_nnz = row_ptr[i+1] - row_ptr[i];
                current_process_nnz += row_nnz;

                if (current_process_nnz >= nnz_per_process && current_process < world_size - 1) {
                    int end_row = i + 1;
                    row_counts[current_process] = end_row - start_row;
                    sendcounts_nnz[current_process] = row_ptr[end_row] - displs_nnz[current_process];
                    row_ptr_counts[current_process] = row_counts[current_process] + 1;
                    
                    current_process++;
                    start_row = end_row;
                    displs_nnz[current_process] = row_ptr[start_row];
                    row_displs[current_process] = start_row;
                    row_ptr_displs[current_process] = start_row;
                    current_process_nnz = 0;
                }
            }
            row_counts[world_size - 1] = N - start_row;
            sendcounts_nnz[world_size - 1] = nnz - displs_nnz[world_size - 1];
            row_ptr_counts[world_size - 1] = row_counts[world_size - 1] + 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
// ...existing code...
// ... The rest of the file remains the same, but the cleanup needs adjustment ...
// ...
    if (world_size == 64) {
// ...existing code...
        }
    }

    // --- CLEANUP FIX ---
    // The CSRMatrix struct and std::vector manage their own memory.
    // We only need to delete what we explicitly allocated with 'new'.
    if (world_rank == 0) {
        // The memory for val, col_ind, row_ptr is managed by the CSRMatrix object
        // which will be destroyed automatically when it goes out of scope.
        // The memory for x is managed by the x_vec vector.
        // We only need to delete y.
        delete[] y;
    }
    
    // y_local was allocated by each process.
    delete[] y_local;

    // x was only allocated by non-root processes.
    if (world_rank != 0) {
        delete[] x;
    }

    // These local buffers were allocated by all processes in parallel runs.
    if (world_size > 1) {
        delete[] local_val;
        delete[] local_col_ind;
        delete[] local_row_ptr;
    }

    MPI_Finalize();
    return 0;
}