#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

// Compile with:    g++ gs.cpp -o gs -fopenmp
// Run with:        ./gs

// Algorithm to match the Gauss-Seidel Lab
void gauss_seidel(const int n, const double * A, const double * b, double * x, int max_iter, double & error, double & residual) {
    double x_last[n], b_pred[n];
    for (int i = 0; i < n; ++i) {
        b_pred[i] = 0;
        x[i] = 1;
        x_last[i] = 0;
    }

    for (int k=0; k < max_iter; k++) {
        // Copy x into x_last
        for (int i=0; i < n; i++) { x_last[i] = x[i]; }

        for (int i=0; i<n; i++) {
            double sum_term = 0;
            for (int j=0; j < n; j++) {
                if (j < i) {
                    sum_term += *(A + i*n + j) * x[j];
                } else if (j > i) {
                    sum_term += *(A + i*n + j) * x_last[j];
                }
            }
            sum_term = b[i] - sum_term;
            x[i] = (1.0/(*(A + i*n + i)))*sum_term;
        }
    }
    // Find error between current and previous iteration
    error = 0;
    for (int i=0; i < n; i++) {
        error += std::pow((x[i] - x_last[i]), 2);
    }
    error = sqrt(error);

    residual = 0;
    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++) {
            b_pred[i] += (*(A + i*n + j)) * x[j];
        }
        residual += std::pow((b[i] - b_pred[i]), 2);
    }
    residual = sqrt(residual);
}

// Multithreaded Algorithm to match the Gauss-Seidel Lab
void gauss_seidel_omp(const int n, const double * A, const double * b, double * x, int max_iter, double & error, double & residual) {
    double x_last[n], b_pred[n];
    for (int i = 0; i < n; ++i) {
        b_pred[i] = 0;
        x[i] = 1;
        x_last[i] = 0;
    }

    for (int k=0; k < max_iter; k++) {
        // Copy x into x_last
        #pragma omp parallel for 
        for (int i=0; i < n; i++) { x_last[i] = x[i]; }
 
        // #pragma omp parallel for
        for (int i=0; i<n; i++) {
            double sum_term = 0;
            for (int j=0; j < n; j++) {

                if (j < i) {
                    sum_term += *(A + i*n + j) * x[j];  // read x[j]
                } else if (j > i) {
                    sum_term += *(A + i*n + j) * x_last[j];
                }
            }
            sum_term = b[i] - sum_term;
            x[i] = (1.0/(*(A + i*n + i)))*sum_term; // write x[j]
        }
    }
    // Find error between current and previous iteration
    error = 0;
    #pragma omp parallel for 
    for (int i=0; i < n; i++) {
        error += std::pow((x[i] - x_last[i]), 2);
    }
    error = sqrt(error);

    residual = 0;
    #pragma omp parallel for 
    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++) {
            b_pred[i] += (*(A + i*n + j)) * x[j];
        }
        residual += std::pow((b[i] - b_pred[i]), 2);
    }
    residual = sqrt(residual);
}


int main() {
    // Initialize varaibles
    const int n = 4;
    const double A[n][n] = {
        {4, 3, 2, 1}, 
        {-1, 2, 0, .5},
        {0, 3, 6, 2},
        {1, 1.3, 7, 10}
    };
    const double b[n] = {59.2, 47, 12, 22};
    const int max_iter = 5;

    double b_pred[n], x[n];
    double error = 0;
    double residual = 0;

    // cout << "######## BEGIN GAUSS SEIDEL (VANILLA) ########" << endl;
    auto gs_start = chrono::high_resolution_clock::now();
    gauss_seidel(n, (double*)A, b, x, max_iter, error, residual);
    auto gs_end = chrono::high_resolution_clock::now();
    auto gs_duration = chrono::duration_cast<chrono::microseconds>(gs_end - gs_start);
    // cout << "######## END GAUSS SEIDEL (VANILLA) ########" << endl;
    
    // Outputs
    cout << endl << "Gauss-Seidel took " << gs_duration.count() << " microseconds" << endl;
    cout << "x: [ \t";
    for (int i=0; i<n; i++) {
        cout << x[i] << " \t";
    } 
    cout << "]" << endl;
    cout << "Error between the last two iterations (2 norm): " << error << endl;
    cout << "Residual (2 norm): " << residual << endl << endl;

    cout << "Now let\'s try with multiple treads~" << endl;
    printf("This region is using %d threads\n", omp_get_num_threads()); 
    printf("Inside a parallel region, we can run at most %d threads\n", omp_get_max_threads()); 
    printf("~~~ Enter parallel region ~~~\n");
    #pragma omp parallel
    {
        // I'll use printf because its output will not be mixed with concurrent printf calls
        // When using cout with "<<", these calls may get mixed between threads
        if (omp_get_thread_num() == 0) {
            printf("This region is using %d threads\n", omp_get_num_threads()); 
        }
        printf("Hi, I\'m thread number %d\n", omp_get_thread_num());
    }
    printf("~~~ Exit parallel region ~~~\n\n");

    // cout << endl << "######## BEGIN GAUSS SEIDEL (OMP) ########" << endl;
    auto gs_omp_start = chrono::high_resolution_clock::now();
    gauss_seidel_omp(n, (double*)A, b, x, max_iter, error, residual);
    auto gs_omp_end = chrono::high_resolution_clock::now();
    auto gs_omp_duration = chrono::duration_cast<chrono::microseconds>(gs_omp_end - gs_omp_start);
    // cout << "######## END GAUSS SEIDEL (OMP) ########" << endl;

    // Outputs
    cout << "Gauss-Seidel with OMP took " << gs_omp_duration.count() << " microseconds" << endl;
    cout << "x: [ \t";
    for (int i=0; i<n; i++) {
        cout << x[i] << " \t";
    } 
    cout << "]" << endl;
    cout << "Error between the last two iterations (2 norm): " << error << endl;
    cout << "Residual (2 norm): " << residual << endl;

    return 0;
} 