#!/bin/bash

# --- Configuration ---
SEQ_EXE="./kmeans_sequential.exe"
PAR_EXE="./kmeans_parallel.exe"
DATA_FILE="points.txt"
OUTPUT_CSV="results.csv"
NUM_RUNS=5
THREADS=(4 8 16 32)
SCHEDULES=("static" "dynamic")

# --- Script Start ---

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
    echo "Please download the data file and place it in this directory."
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
    # Run the command, redirect stdout to /dev/null, and capture stderr
    time_output=$( { $SEQ_EXE $DATA_FILE > /dev/null; } 2>&1 )
    # Extract the time value (the 4th word in the output string)
    execution_time=$(echo "$time_output" | awk '{print $4}')
    # Append the result to the CSV
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
            # Run the command, redirect stdout to /dev/null, and capture stderr
            time_output=$( { $PAR_EXE $DATA_FILE "$threads" "$schedule" > /dev/null; } 2>&1 )
            # Extract the time value (the 7th word in the output string)
            execution_time=$(echo "$time_output" | awk '{print $7}')
            # Append the result to the CSV
            echo "Parallel,$threads,$schedule,$i,$execution_time" >> "$OUTPUT_CSV"
        done
    done
done
echo "Parallel tests complete."
echo "----------------------------------------"

echo "All experiments finished successfully!"
echo "Results have been saved to '$OUTPUT_CSV'."