#!/bin/bash

# --- SLURM Batch Script for MPI SpMV ---

#SBATCH --job-name=spmv-mpi-test    # Job name for easy identification
#SBATCH --output=spmv-output-%j.log # Single log file for all script output
#SBATCH --error=spmv-error-%j.log   # Single error file
#SBATCH --nodes=4                   # Request 4 nodes to accommodate up to 128 processes
#SBATCH --ntasks-per-node=32        # Request 32 tasks (cores) per node (4*32=128 total cores)
#SBATCH --time=00:30:00             # Set a 30-minute time limit for the entire job

# --- Configuration ---
EXECUTABLE="./main" # The name of the executable created by 'make'
RESULTS_CSV="spmv_results.csv"
COMPUTE_TIMES_64_CSV="compute_times_64_process.csv"
NUM_RUNS=5
PROCESS_COUNTS=(1 8 16 32 64 128)

# --- Script Start ---

# Change to the directory where the job was submitted from
cd $SLURM_SUBMIT_DIR

# Load the necessary MPI module (this command might vary on your cluster)
module load mpi

echo "================================================="
echo "   MPI SpMV Experiments                         "
echo "Nodes allocated: $SLURM_JOB_NODELIST"
echo "================================================="

echo "Step 1: Running setup script to create symlinks and Makefile..."
bash /scratch/public/sparse_matrix_data/setup.sh
echo "Setup complete."
echo "----------------------------------------"

echo "Step 2: Compiling the source code with 'make'..."
# The 'make' command will now use the corrected Makefile to compile mpi.cpp
make
echo "Compilation successful. Executable '$EXECUTABLE' is ready."
echo "----------------------------------------"

# Create the main CSV file and write the header
echo "Step 3: Preparing the output file '$RESULTS_CSV'..."
echo "Processes,Run,TotalTime,MaxComputeTime,MinComputeTime,LoadImbalance" > "$RESULTS_CSV"
echo "----------------------------------------"

# --- Run Experiments ---
echo "Step 4: Running all experiments..."
for processes in "${PROCESS_COUNTS[@]}"; do
    echo ""
    echo "================================================="
    echo " Running with P = $processes MPI processes ($NUM_RUNS runs)"
    echo "================================================="

    for i in $(seq 1 $NUM_RUNS); do
        echo "--- Run $i for P=$processes ---"
        
        output=$(srun -n "$processes" "$EXECUTABLE")
        
        total_time=$(echo "$output" | grep "Total execution time:" | awk '{print $4}')
        max_time=$(echo "$output" | grep "Max compute time:" | awk '{print $4}')
        min_time=$(echo "$output" | grep "Min compute time:" | awk '{print $4}')
        imbalance=$(echo "$output" | grep "Load Imbalance:" | awk '{print $3}')
        
        echo "$processes,$i,$total_time,$max_time,$min_time,$imbalance" >> "$RESULTS_CSV"

        if [ "$processes" -eq 64 ] && [ "$i" -eq 1 ]; then
            echo "Process,ComputeTime" > "$COMPUTE_TIMES_64_CSV"
            echo "$output" | grep "Process " | awk '{print $2 "," $3}' | sed 's/://' >> "$COMPUTE_TIMES_64_CSV"
            echo "    -> Saved detailed 64-process compute times to '$COMPUTE_TIMES_64_CSV'."
        fi
    done
done

echo "================================================="
echo " All experiments completed successfully! "
echo " Results are in '$RESULTS_CSV' "
echo "================================================="    