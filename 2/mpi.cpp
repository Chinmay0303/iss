#include <iostream>
#include <vector>
#include <numeric>
#include <mpi.h>
#include "load.h"

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
        // Use the provided functions to load data
        load_matrix("data.mtx", &N, &nnz, &val, &col_ind, &row_ptr);
        load_vector("data.vec", N, &x);
        y = new double[N]();

        // Calculate the load-balanced distribution for parallel runs
        if (world_size > 1) {
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
    total_start_time = MPI_Wtime();

    int my_row_count;
    int my_nnz_count;

    if (world_size > 1) {
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(row_counts.data(), 1, MPI_INT, &my_row_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (world_rank != 0) x = new double[N];
        MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(sendcounts_nnz.data(), 1, MPI_INT, &my_nnz_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        my_row_count = N;
        my_nnz_count = nnz;
    }

    double* local_val = new double[my_nnz_count];
    int* local_col_ind = new int[my_nnz_count];
    long long* local_row_ptr = new long long[my_row_count + 1];

    if (world_size > 1) {
        MPI_Scatterv(val, sendcounts_nnz.data(), displs_nnz.data(), MPI_DOUBLE, local_val, my_nnz_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(col_ind, sendcounts_nnz.data(), displs_nnz.data(), MPI_INT, local_col_ind, my_nnz_count, MPI_INT, 0, MPI_COMM_WORLD);
        int my_row_ptr_count;
        MPI_Scatter(row_ptr_counts.data(), 1, MPI_INT, &my_row_ptr_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(row_ptr, row_ptr_counts.data(), row_ptr_displs.data(), MPI_LONG_LONG, local_row_ptr, my_row_ptr_count, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    } else {
        local_val = val;
        local_col_ind = col_ind;
        local_row_ptr = row_ptr;
    }
    
    if (world_size > 1) {
        long long offset = local_row_ptr[0];
        for (int i = 0; i <= my_row_count; ++i) {
            local_row_ptr[i] -= offset;
        }
    }

    double* y_local = new double[my_row_count]();
    double compute_start_time = MPI_Wtime();
    for (int i = 0; i < my_row_count; ++i) {
        double sum = 0.0;
        for (long long j = local_row_ptr[i]; j < local_row_ptr[i+1]; ++j) {
            sum += local_val[j] * x[local_col_ind[j]];
        }
        y_local[i] = sum;
    }
    double compute_end_time = MPI_Wtime();
    double my_compute_time = compute_end_time - compute_start_time;

    if (world_size > 1) {
        MPI_Gatherv(y_local, my_row_count, MPI_DOUBLE, y, row_counts.data(), row_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        y = y_local;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    total_end_time = MPI_Wtime();

    if (world_rank == 0) {
        std::cout << "Total execution time: " << total_end_time - total_start_time << " seconds." << std::endl;
    }

    double max_compute_time, min_compute_time;
    MPI_Reduce(&my_compute_time, &max_compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_compute_time, &min_compute_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        double load_imbalance = (min_compute_time > 0) ? (max_compute_time - min_compute_time) / min_compute_time : 0.0;
        std::cout << "Max compute time: " << max_compute_time << std::endl;
        std::cout << "Min compute time: " << min_compute_time << std::endl;
        std::cout << "Load Imbalance: " << load_imbalance << std::endl;
    }

    if (world_size == 64) {
        std::vector<double> all_compute_times(world_size);
        MPI_Gather(&my_compute_time, 1, MPI_DOUBLE, all_compute_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (world_rank == 0) {
            std::cout << "--- Computation times for 64 processes ---" << std::endl;
            for (int i = 0; i < world_size; ++i) {
                std::cout << "Process " << i << ": " << all_compute_times[i] << " s" << std::endl;
            }
            std::cout << "------------------------------------------" << std::endl;
        }
    }

    if (world_size == 1) { y = nullptr; }
    if (world_rank == 0) {
        if (world_size > 1) delete[] val;
        if (world_size == 1) {
             col_ind = nullptr;
             row_ptr = nullptr;
        } else {
             delete[] col_ind;
             delete[] row_ptr;
        }
        delete[] y;
    }
    delete[] x;
    delete[] y_local;
    if (world_size > 1) {
        delete[] local_val;
        delete[] local_col_ind;
        delete[] local_row_ptr;
    }

    MPI_Finalize();
    return 0;
}