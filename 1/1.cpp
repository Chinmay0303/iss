#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <limits>

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
    // Open the data file
    std::ifstream infile("data_20k.txt");
    if (!infile) {
        std::cerr << "Error: Could not open file." << std::endl;
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
        // --- Assignment Step ---
        // Assign each point to the nearest centroid
        for (auto& p : points) {
            double min_dist = std::numeric_limits<double>::max();
            int closest_cluster = -1;
            for (int i = 0; i < K; ++i) {
                double d = distanceSquared(p, centroids[i]);
                if (d < min_dist) {
                    min_dist = d;
                    closest_cluster = i;
                }
            }
            p.cluster = closest_cluster;
        }

        // --- Update Step ---
        // Recalculate centroids based on the mean of points in each cluster
        std::vector<Point> new_centroids(K, {0, 0, -1});
        std::vector<int> cluster_counts(K, 0);

        // Sum up coordinates and count points for each cluster
        for (const auto& p : points) {
            new_centroids[p.cluster].x += p.x;
            new_centroids[p.cluster].y += p.y;
            cluster_counts[p.cluster]++;
        }

        // Calculate the new mean for each centroid
        for (int i = 0; i < K; ++i) {
            if (cluster_counts[i] > 0) { // Avoid division by zero for empty clusters
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
