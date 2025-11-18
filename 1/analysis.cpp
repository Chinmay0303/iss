#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <vector>
#include <string>

// Define constants for the K-Means algorithm
#define K 20
#define MAX_ITERATIONS 100
#define NUM_RUNS 5

// Represents a 2D point with an assigned cluster
struct Point {
    double x, y;
    int cluster;
};

// --- Function Declarations ---

// Sequential K-Means Implementation
double run_sequential(const std::vector<Point>& initial_points);

// Parallel K-Means Implementation
double run_parallel(const std::vector<Point>& initial_points, int num_threads, const std::string& schedule_type);

#endif // KMEANS_HPP
