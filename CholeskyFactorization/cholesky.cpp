#include <stdio.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <cmath>
#include <random>

using namespace std;

// Compile with:    g++ cholesky.cpp -o cholesky -fopenmp
// Run with:        ./cholesky


void print_matrix(const double * A, const int n) {
    for (int i=0; i<n; i++) {
        cout << "[  ";
        for (int j=0; j < n; j++) {
            printf("%.3f\t", *(A + i*n + j));
        }
        cout << "]" << endl;
    } 
}

bool is_symmetric(const double * A, const int n) {
    for (int i=0; i<n; i++) {
        for (int j=0; j < n; j++) {
            if (*(A + i*n + j) != *(A + j*n + i)) {
                return false;
            }
        }
    } 
    return true;
}

// C = A * B
void mult_mat(const int n, const double * A, const double * B, double * C) {
    int nrowsA = n;
    int ncolsA = n; // ncolsA same as nrowsL
    int nrowsB = n;
    int ncolsB = n;
    for (int i=0; i<nrowsA; i++) {
        for (int j=0; j<ncolsB; j++) {
            *(C + i*ncolsA + j) = 0;
            for (int k=0; k<nrowsB; k++) {
                *(C + i*ncolsA + j) += ( *(A + i*ncolsA + k) * *(B + k*ncolsB + j ) );
            }
        }
    }
}

// nxn matrices assumed
// C = A - B
void sub_mat(const int n, const double * A, const double * B, double * C) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            *(C + i*n + j) = *(A + i*n + j) - *(B + i*n + j);
        }
    }
}

double frob_norm(const int n, const double * A, const double * A_approx) {
    double sum = 0.0;
    double diff = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            diff = *(A + i*n + j) - *(A_approx + i*n + j);
            sum += diff * diff;
        }
    }
    return sqrt(sum);
}

// Lstar = L^T
void get_trans(const int n, const double * L, double * Lstar) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            *(Lstar + j*n + i) = *(L + i*n + j);
        }
    }
}

// Algorithm from lab; Algorithm 6.6 / page 423
void cholesky(const int n, const double * A, double * L) {
    if (!is_symmetric((double*)A, n)) {
        cout << "A is not symmetric!" << endl;
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            *(L + i*n + j) = 0; // initialize L to all 0
        }
    }

    // STEP 1
    if (*A < 0) {
        cout << "Square root of a negative number! *A = " << *A << endl;
    } else {
        *L = sqrt(*A); // set L_11 to square root of A_11
    }
    
    // STEP 2
    for (int i=1; i<n; i++) {
        if ((*(L)) == 0) {
            cout << "Divide by zero!" << endl;
        } else {
            *(L + i*n) = (*(A + i*n)) / (*(L)); // fill first column of L
        }
    }

    // STEP 3
    for (int i=1; i<n-1; i++) {

        // STEP 4
        *(L + i*n + i) = *(A + i*n + i);
        for (int k=0; k<i; k++) {
            *(L + i*n + i) -= ((*(L + i*n + k)) * (*(L + i*n + k)));
        }
        *(L + i*n + i) = sqrt(*(L + i*n + i));

        // STEP 5
        for (int j=i+1; j<n; j++) {
            *(L + j*n + i) = *(A + j*n + i);
            for (int k=0; k<i; k++) {
                // cout << "Ljk = " << (*(L + j*n + k)) << " * " << (*(L + i*n + k))  << "  = " << (*(L + j*n + k)) * (*(L + i*n + k))  << endl;
                *(L + j*n + i) -= ( (*(L + j*n + k)) * (*(L + i*n + k)) );
                // pow(((*(L + j*n + k))), 2);
            }
            if (*(L + i*n + i) == 0) {
                cout << "Divide by zero!" << endl;
            } else {
                *(L + j*n + i) /= *(L + i*n + i);
            }
        }
    }

    // STEP 6
    *(L + (n-1)*n + (n-1)) = *(A + (n-1)*n + (n-1));
    for (int k=0; k<n-1; k++) {
        *(L + (n-1)*n + (n-1)) -= ((*(L + (n-1)*n + k)) * (*(L + (n-1)*n + k)));
    }
    if (*(L + (n-1)*n + (n-1)) < 0) {
        cout << "Square root of a negative number! L[" << (n-1) << "][" << (n-1) << "] = " << *(L + (n-1)*n + (n-1)) << endl;
    } else {
        *(L + (n-1)*n + (n-1)) = sqrt(*(L + (n-1)*n + (n-1)));
    }

}

// Algorithm from lab; Algorithm 6.6 / page 423
void parallel_cholesky(const int n, const double * A, double * L) {
    if (!is_symmetric((double*)A, n)) {
        cout << "A is not symmetric!" << endl;
    }
    #pragma omp parallel for collapse(2)
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            *(L + i*n + j) = 0; // initialize L to all 0
        }
    }

    // STEP 1
    if (*A < 0) {
        cout << "Square root of a negative number! *A = " << *A << endl;
    } else {
        *L = sqrt(*A); // set L_11 to square root of A_11
    }
    
    // STEP 2
    #pragma omp parallel for
    for (int i=1; i<n; i++) {
        if ((*(L)) == 0) {
            cout << "Divide by zero!" << endl;
        } else {
            *(L + i*n) = (*(A + i*n)) / (*(L)); // fill first column of L
        }
    }

    // STEP 3
    for (int i=1; i<n-1; i++) {

        // STEP 4
        double sum4 = 0;
        #pragma omp parallel for reduction(-:sum4)
        for (int k=0; k<i; k++) {
            sum4 += ((*(L + i*n + k)) * (*(L + i*n + k)));
        }
        *(L + i*n + i) = sqrt(*(A + i*n + i) - sum4);

        // STEP 5
        #pragma omp parallel for 
        for (int j=i+1; j<n; j++) {
            *(L + j*n + i) = *(A + j*n + i);
            for (int k=0; k<i; k++) {
                *(L + j*n + i) -= ( (*(L + j*n + k)) * (*(L + i*n + k)) );
            }
            if (*(L + i*n + i) == 0) {
                cout << "Divide by zero!" << endl;
            } else {
                *(L + j*n + i) /= *(L + i*n + i);
            }
        }
    }

    // STEP 6
    // *(L + (n-1)*n + (n-1)) = *(A + (n-1)*n + (n-1));
    double sum6 = 0;
    // #pragma omp parallel for reduction(-:sum6)
    for (int k=0; k<n-1; k++) {
        sum6 += ((*(L + (n-1)*n + k)) * (*(L + (n-1)*n + k)));
    }
    *(L + (n-1)*n + (n-1)) = sqrt(*(A + (n-1)*n + (n-1)) - sum6);
}


int main() {
    // Declare varaibles
    const int n = 1000;
    cout << "n = " << n << endl;
    double * B = new double[n*n];
    double * A = new double[n*n];
    double * L = new double[n*n];
    double * Lstar = new double[n*n];
    double * A_approx = new double[n*n];


    // Generate random matrix B
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[i*n + j] = dis(gen);
        }
    }

    // Compute A = B^T * B (always positive definite)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                A[i*n + j] += B[k*n + i] * B[k*n + j];
            }
        }
    }
    
    auto start = chrono::high_resolution_clock::now();
    cholesky(n, A, L);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << endl << "Cholesky took " << duration.count() << " microseconds" << endl;
    get_trans(n, L, Lstar);
    mult_mat(n, L, Lstar, A_approx);
    double chol_error_norm = frob_norm(n, A, A_approx);
    cout << "Frobenius norm of error between LL* and A = " << chol_error_norm << endl;
    // cout << "L: " << endl;
    // print_matrix((double*)L, n);
    // cout << endl;
    // cout << "L*: " << endl;
    // print_matrix((double*)Lstar, n);
    // cout << endl;
    // cout << "LL*: " << endl;
    // print_matrix((double*)A_approx, n);

  
    // cout << "Now let\'s try with multiple treads~" << endl << endl;

    auto p_start = chrono::high_resolution_clock::now();
    parallel_cholesky(n, A, L);
    auto p_end = chrono::high_resolution_clock::now();
    auto p_duration = chrono::duration_cast<chrono::microseconds>(p_end - p_start);
    cout << endl << "Parallel cholesky took " << p_duration.count() << " microseconds" << endl;
    get_trans(n, L, Lstar);
    mult_mat(n, L, Lstar, A_approx);
    double p_chol_error_norm = frob_norm(n, A, A_approx);
    cout << "Frobenius norm of error between LL* and A = " << p_chol_error_norm << endl;
    // cout << "L: " << endl;
    // print_matrix((double*)L, n);
    // cout << endl;
    // cout << "L*: " << endl;
    // print_matrix((double*)Lstar, n);
    // cout << endl;
    // cout << "LL*: " << endl;
    // print_matrix((double*)A_approx, n);

    delete[] B;
    delete[] A;
    delete[] L;
    delete[] Lstar;
    delete[] A_approx;
 
    return 0;
} 