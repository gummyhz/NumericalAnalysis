#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "../read_txt_to_vec.cpp"

using namespace std;

// g++ gs.cpp -o gs

const int n = 2;

// Algorithm to match the Gauss-Seidel Lab
void gauss_seidel(int n, const double * A, const double * b, double * x, int max_iter, double & error, double & residual) {
    double x_last[n], b_pred[n];
    for (int i = 0; i < n; ++i) {
        b_pred[i] = 1;
        x[i] = 1;
        x_last[i] = 0;
    }

    for (int k=0; k < max_iter; k++) {
        // Copy x into x_last
        for (int i=0; i < n; i++) { x_last[i] = x[i]; }

        for (int i=0; i<n; i++) {
            // cout << "--- i: " << i << endl;
            double sum_term = 0;
            for (int j=0; j < n; j++) {
                // cout << "i: " << i << " j: " << j << endl;
                if (j < i) {
                    sum_term += *(A + i*n + j) * x[j];
                    // cout << "   sum_term += " << *(A + i*n + j) * x[j] << endl;
                } else if (j > i) {
                    sum_term += *(A + i*n + j) * x_last[j];
                    // cout << "   sum_term += " << *(A + i*n + j) * x_last[j] << endl;
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
        b_pred[i] = 0;
        for (int j=0; j < n; j++) {
            b_pred[i] += (*(A + i*n + j)) * x[j];
        }
        // cout << "A*x[" << i << "]: " << b_pred[i] << " \tb[" << i << "]: " << b[i] << endl;
        residual += std::pow((b[i] - b_pred[i]), 2);
    }
    residual = sqrt(residual);
}

// Multithreaded Algorithm to match the Gauss-Seidel Lab
void gauss_seidel_omp(int n, const double * A, const double * b, double * x, int max_iter, double & error, double & residual) {
    double x_last[n], b_pred[n];
    for (int i = 0; i < n; ++i) {
        b_pred[i] = 1;
        x[i] = 1;
        x_last[i] = 0;
    }

    for (int k=0; k < max_iter; k++) {
        // Copy x into x_last
        #pragma omp parallel for
        for (int i=0; i < n; i++) { x_last[i] = x[i]; }

        #pragma omp parallel for
        for (int i=0; i<n; i++) {
            double sum_term = 0;
            // cout << "--- i: " << i << endl;
            for (int j=0; j < n; j++) {
                // cout << "i: " << i << " j: " << j << endl;
                if (j < i) {
                    sum_term += *(A + i*n + j) * x[j];
                    // cout << "   sum_term += " << *(A + i*n + j) * x[j] << endl;
                } else if (j > i) {
                    sum_term += *(A + i*n + j) * x_last[j];
                    // cout << "   sum_term += " << *(A + i*n + j) * x_last[j] << endl;
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
        b_pred[i] = 0;
        for (int j=0; j < n; j++) {
            b_pred[i] += (*(A + i*n + j)) * x[j];
            // cout << A[i][j] * x[j] << endl;

        }
        // cout << "A*x[" << i << "]: " << b_pred[i] << " \tb[" << i << "]: " << b[i] << endl;
        residual += std::pow((b[i] - b_pred[i]), 2);
    }
    residual = sqrt(residual);
}


int main() {
    // Initialize varaibles
    const double A[n][n] = {
        {4, 3}, 
        {-1, 2}
    };
    const double b[n] = {59.2, 47};
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
    cout << "Residual (2 norm): " << residual << endl;

    // cout << endl << "######## BEGIN GAUSS SEIDEL (OMP) ########" << endl;
    auto gs_omp_start = chrono::high_resolution_clock::now();
    gauss_seidel_omp(n, (double*)A, b, x, max_iter, error, residual);
    auto gs_omp_end = chrono::high_resolution_clock::now();
    auto gs_omp_duration = chrono::duration_cast<chrono::microseconds>(gs_omp_end - gs_omp_start);
    // cout << "######## END GAUSS SEIDEL (OMP) ########" << endl;

    // Outputs
    cout << endl << "Gauss-Seidel with OMP took " << gs_omp_duration.count() << " microseconds" << endl;
    cout << "x: [ \t";
    for (int i=0; i<n; i++) {
        cout << x[i] << " \t";
    } 
    cout << "]" << endl;
    cout << "Error between the last two iterations (2 norm): " << error << endl;
    cout << "Residual (2 norm): " << residual << endl;

    return 0;
} 