#!/bin/bash

# --- SLURM Batch Script for K-Means ---

#SBATCH --job-name=kmeans-test         # Job name
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=32             # Request 32 CPUs for the job
#SBATCH --output=kmeans-output-%j.log  # Standard output log
#SBATCH --error=kmeans-error-%j.log    # Standard error log

# --- Configuration ---
SEQ_EXE="./kmeans_sequential"
PAR_EXE="./kmeans_parallel"
DATA_FILE="points.txt"
OUTPUT_CSV="results.csv"
NUM_RUNS=5
THREADS=(4 8 16 32)
SCHEDULES=("static" "dynamic")

# --- Script Start ---

# Change to the directory where the job was submitted from
# This ensures it runs in /iss/1 if you submit from there
cd $SLURM_SUBMIT_DIR

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Step 1: Compiling the source code..."
make clean
make
echo "Compilation successful."
echo "----------------------------------------"

# Check if the data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file '$DATA_FILE' not found."
    exit 1
fi

# Create the CSV file and write the header
echo "Step 2: Preparing the output file '$OUTPUT_CSV'..."
echo "Type,Threads,Schedule,Run,Time(s)" > "$OUTPUT_CSV"
echo "----------------------------------------"

# --- Run Sequential Experiment ---
echo "Step 3: Running Sequential tests..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  - Run $i/$NUM_RUNS..."
    time_output=$( { $SEQ_EXE $DATA_FILE > /dev/null; } 2>&1 )
    execution_time=$(echo "$time_output" | awk '{print $4}')
    echo "Sequential,1,N/A,$i,$execution_time" >> "$OUTPUT_CSV"
done
echo "Sequential tests complete."
echo "----------------------------------------"

# --- Run Parallel Experiments ---
echo "Step 4: Running Parallel tests..."
for schedule in "${SCHEDULES[@]}"; do
    for threads in "${THREADS[@]}"; do
        echo "  - Running test: $threads threads, schedule($schedule)..."
        for i in $(seq 1 $NUM_RUNS); do
            echo "    - Run $i/$NUM_RUNS..."
            time_output=$( { $PAR_EXE $DATA_FILE "$threads" "$schedule" > /dev/null; } 2>&1 )
            execution_time=$(echo "$time_output" | awk '{print $7}')
            echo "Parallel,$threads,$schedule,$i,$execution_time" >> "$OUTPUT_CSV"
        done
    done
done
echo "Parallel tests complete."
echo "----------------------------------------"

echo "All experiments finished successfully!"
echo "Results have been saved to '$OUTPUT_CSV'."
echo "Job log is in kmeans-output-${SLURM_JOB_ID}.log"
