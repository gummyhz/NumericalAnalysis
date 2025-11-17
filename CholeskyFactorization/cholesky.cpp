#include <stdio.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <cmath>

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

// Algorithm from lab; Algorithm 6.6 / page 423
int cholesky(const int n, const double * A, double * L) {
    if (!is_symmetric((double*)A, n)) {
        cout << "A is not symmetric!" << endl;
        return 1;
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
                return 1;
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
                cout << "Ljk = " << (*(L + j*n + k)) << " * " << (*(L + i*n + k))  << "  = " << (*(L + j*n + k)) * (*(L + i*n + k))  << endl;
                *(L + j*n + i) -= ( (*(L + j*n + k)) * (*(L + i*n + k)) );
                std::pow(((*(L + j*n + k))), 2);
            }
            if (*(L + i*n + i) == 0) {
                cout << "Divide by zero!" << endl;
                return 1;
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
        return 1;
    } else {
        *(L + (n-1)*n + (n-1)) = sqrt(*(L + (n-1)*n + (n-1)));
    }
    
    return 0;
}


int main() {
    // Initialize varaibles
    const int n = 3;
    const double A[n][n] = {
        {2, -1, 0}, 
        {-1, 2, -1},
        {0, -1, 2}
    };
    double L[n][n];

    cholesky(n, (double*)A, (double*)L);
    print_matrix((double*)L, n);

  
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
    printf("This region is using %d threads\n\n", omp_get_num_threads()); 

    return 0;
} 