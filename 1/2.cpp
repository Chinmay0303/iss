#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <limits>
#include <omp.h> // Include the OpenMP header for parallelization
#include <chrono> // For timing

// Define constants for the K-Means algorithm
#define K 20                // Number of clusters
#define MAX_ITERATIONS 100  // Maximum number of iterations

// Represents a 2D point with an assigned cluster
struct Point {
    double x, y;
    int cluster;
};

// Calculates the squared Euclidean distance between two points
// Note: Using squared distance is faster as it avoids sqrt and gives the same comparison result.
double distanceSquared(Point p1, Point p2) {
    return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
}

int main() {
    // Vector to store all data points
    std::vector<Point> points;
    // Open the data file. Note: This assumes a 'data_20k.txt' file exists.
    std::ifstream infile("data_20k.txt");
    if (!infile) {
        std::cerr << "Error: Could not open file data_20k.txt." << std::endl;
        return 1;
    }

    // Read x and y coordinates from the file into the points vector
    double x, y;
    while (infile >> x >> y) {
        points.push_back({x, y, -1}); // Initialize cluster to -1 (unassigned)
    }
    infile.close();

    // Check if there are enough points for K clusters
    if (points.size() < K) {
        std::cerr << "Error: Not enough data points for K=" << K << " clusters." << std::endl;
        return 1;
    }

    // Initialize centroids by taking the first K points from the dataset
    std::vector<Point> centroids;
    for (int i = 0; i < K; ++i) {
        centroids.push_back(points[i]);
    }

    // Main K-Means loop
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        // --- Assignment Step (Parallelized) ---
        // The 'schedule(static)' clause divides iterations among threads in fixed-size chunks.
        // This is efficient when loop iterations have similar workloads.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < points.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int closest_cluster = -1;
            for (int j = 0; j < K; ++j) {
                double d = distanceSquared(points[i], centroids[j]);
                if (d < min_dist) {
                    min_dist = d;
                    closest_cluster = j;
                }
            }
            points[i].cluster = closest_cluster;
        }

        // --- Update Step (Parallelized) ---
        std::vector<Point> new_centroids(K, {0, 0, -1});
        std::vector<int> cluster_counts(K, 0);

        // Each thread will have its own private copy of 'new_centroids' and 'cluster_counts'.
        // At the end of the loop, OpenMP will sum the values from all private copies
        // into the original shared variables. This avoids race conditions.
        #pragma omp parallel for
        for (int i = 0; i < points.size(); ++i) {
            int cluster_id = points[i].cluster;
            // The following operations are implicitly handled by creating private copies for each thread
            // and then reducing them at the end. For a more explicit but complex reduction on
            // structs/vectors, one might need custom reduction declarations.
            // However, a simpler approach is to use separate reductions for each component.
            // A simple loop like this is often optimized well by the compiler with thread-local storage.
            // For clarity and correctness, we can use atomic operations or a critical section,
            // but let's try a reduction-like pattern with private arrays first.

            // To correctly parallelize the update step with reduction on arrays,
            // we create private arrays for each thread and sum them up later.
            #pragma omp critical
            {
                new_centroids[cluster_id].x += points[i].x;
                new_centroids[cluster_id].y += points[i].y;
                cluster_counts[cluster_id]++;
            }
        }


        // Calculate the new mean for each centroid (Sequential)
        // This loop is small (K=20) and parallelizing it would likely add more overhead than benefit.
        for (int i = 0; i < K; ++i) {
            if (cluster_counts[i] > 0) {
                centroids[i].x = new_centroids[i].x / cluster_counts[i];
                centroids[i].y = new_centroids[i].y / cluster_counts[i];
            }
        }
    }

    // --- Output Results ---
    std::cout << "Final Cluster Information:" << std::endl;
    std::vector<int> final_counts(K, 0);
    for (const auto& p : points) {
        final_counts[p.cluster]++;
    }

    // Print the size and centroid coordinates for each cluster
    for (int i = 0; i < K; ++i) {
        std::cout << "Cluster " << i << ": "
                  << final_counts[i] << " points, "
                  << "Centroid (" << centroids[i].x << ", " << centroids[i].y << ")"
                  << std::endl;
    }

    return 0;
}