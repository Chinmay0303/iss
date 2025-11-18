#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define K 20
#define MAX_ITERATIONS 1000

typedef struct {
    double x, y;
} Point;

// Function to calculate squared Euclidean distance (more efficient)
double euclidean_distance_sq(Point p1, Point p2) {
    return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <data_file> <num_threads> <schedule_type (static/dynamic)>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[2]);
    char* schedule_type = argv[3];

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

    omp_set_num_threads(num_threads);

    Point centroids[K];
    int *cluster_assignments = (int *)calloc(num_points, sizeof(int));
    if (!cluster_assignments) {
        fprintf(stderr, "Failed to allocate memory for assignments\n");
        free(points);
        return 1;
    }

    // 1. Initialization: Use a more robust method with a fixed seed
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

    double start_time = omp_get_wtime();

    int iterations = 0;
    int changed_global;
    do {
        changed_global = 0;

        // --- MODIFICATION HERE ---
        // Define a chunk size for dynamic scheduling to reduce overhead
        int chunk_size = (num_points > 10000) ? 1000 : 100; 

        // 2. Assignment Step (Parallelized)
        // The 'schedule' clause is now inside the pragma.
        // For this problem, static is best. Dynamic is included for the experiment.
        if (strcmp(schedule_type, "static") == 0) {
            #pragma omp parallel for schedule(static) reduction(+:changed_global)
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
                    changed_global += 1;
                }
            }
        } else { // dynamic
            #pragma omp parallel for schedule(dynamic, chunk_size) reduction(+:changed_global)
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
                    changed_global += 1;
                }
            }
        }

        // 3. Update Step (Parallelized)
        Point new_centroids[K] = {0};
        int cluster_counts[K] = {0};

        #pragma omp parallel
        {
            Point local_new_centroids[K] = {0};
            int local_cluster_counts[K] = {0};

            #pragma omp for schedule(runtime) nowait
            for (int i = 0; i < num_points; i++) {
                int cluster_idx = cluster_assignments[i];
                local_new_centroids[cluster_idx].x += points[i].x;
                local_new_centroids[cluster_idx].y += points[i].y;
                local_cluster_counts[cluster_idx]++;
            }

            #pragma omp critical
            {
                for (int i = 0; i < K; i++) {
                    new_centroids[i].x += local_new_centroids[i].x;
                    new_centroids[i].y += local_new_centroids[i].y;
                    cluster_counts[i] += local_cluster_counts[i];
                }
            }
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
    } while (changed_global > 0 && iterations < MAX_ITERATIONS);

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    int final_cluster_counts[K] = {0};
    for(int i = 0; i < num_points; i++) {
        final_cluster_counts[cluster_assignments[i]]++;
    }

    for (int i = 0; i < K; i++) {
        printf("%d %d %f %f\n", i, final_cluster_counts[i], centroids[i].x, centroids[i].y);
    }

    fprintf(stderr, "Parallel execution time (%d threads, %s): %f seconds\n", num_threads, schedule_type, elapsed_time);

    free(points);
    free(cluster_assignments);

    return 0;
}