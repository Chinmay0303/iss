#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define K 20
#define MAX_ITERATIONS 1000

typedef struct {
    double x, y;
} Point;

// A high-resolution timer for Windows
#if defined(_WIN32)
#include <Windows.h>
double get_time() {
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
double get_time() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
#endif

// Function to calculate squared Euclidean distance (more efficient)
double euclidean_distance_sq(Point p1, Point p2) {
    return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <data_file>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    int num_points = 0;
    double temp_x, temp_y;
    while (fscanf(file, "%lf %lf", &temp_x, &temp_y) == 2) {
        num_points++;
    }
    rewind(file);

    Point *points = (Point *)malloc(num_points * sizeof(Point));
    if (!points) {
        fprintf(stderr, "Failed to allocate memory for points\n");
        fclose(file);
        return 1;
    }

    for (int i = 0; i < num_points; i++) {
        fscanf(file, "%lf %lf", &points[i].x, &points[i].y);
    }
    fclose(file);

    Point centroids[K];
    int *cluster_assignments = (int *)calloc(num_points, sizeof(int));
    if (!cluster_assignments) {
        fprintf(stderr, "Failed to allocate memory for assignments\n");
        free(points);
        return 1;
    }

    // 1. Initialization: Use a more robust method to avoid duplicate centroids
    srand(42); // Use a fixed seed for reproducible results
    int *used_indices = (int *)calloc(num_points, sizeof(int));
    for (int i = 0; i < K; i++) {
        int rand_index;
        do {
            rand_index = rand() % num_points;
        } while (used_indices[rand_index]);
        centroids[i] = points[rand_index];
        used_indices[rand_index] = 1;
    }
    free(used_indices);

    double start_time = get_time();

    int iterations = 0;
    int changed = 1;
    while (changed && iterations < MAX_ITERATIONS) {
        changed = 0;

        // 2. Assignment Step
        for (int i = 0; i < num_points; i++) {
            double min_dist_sq = DBL_MAX;
            int best_cluster = -1;
            for (int j = 0; j < K; j++) {
                double dist_sq = euclidean_distance_sq(points[i], centroids[j]);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_cluster = j;
                }
            }
            if (cluster_assignments[i] != best_cluster) {
                cluster_assignments[i] = best_cluster;
                changed = 1;
            }
        }

        // 3. Update Step
        Point new_centroids[K] = {0};
        int cluster_counts[K] = {0};

        for (int i = 0; i < num_points; i++) {
            int cluster_idx = cluster_assignments[i];
            new_centroids[cluster_idx].x += points[i].x;
            new_centroids[cluster_idx].y += points[i].y;
            cluster_counts[cluster_idx]++;
        }

        for (int i = 0; i < K; i++) {
            if (cluster_counts[i] > 0) {
                centroids[i].x = new_centroids[i].x / cluster_counts[i];
                centroids[i].y = new_centroids[i].y / cluster_counts[i];
            } else {
                // Handle empty clusters by re-initializing the centroid
                int rand_index = rand() % num_points;
                centroids[i] = points[rand_index];
            }
        }
        iterations++;
    }

    double end_time = get_time();
    double elapsed_time = end_time - start_time;

    int final_cluster_counts[K] = {0};
    for(int i = 0; i < num_points; i++) {
        final_cluster_counts[cluster_assignments[i]]++;
    }

    for (int i = 0; i < K; i++) {
        printf("%d %d %f %f\n", i, final_cluster_counts[i], centroids[i].x, centroids[i].y);
    }

    fprintf(stderr, "Sequential execution time: %f seconds\n", elapsed_time);

    free(points);
    free(cluster_assignments);

    return 0;
}