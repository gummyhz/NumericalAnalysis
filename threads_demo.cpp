#include <stdio.h>
#include <omp.h>
using namespace std;

// Compile with:    g++ threads_demo.cpp -o threads_demo -fopenmp
// Run with:        ./threads_demo

int main() {
    printf("\nThis region is using %d threads\n\n", omp_get_num_threads()); 
    printf("Inside a parallel region, we can run at most %d threads\n", omp_get_max_threads()); 
    printf("~~~ Enter parallel region ~~~\n");
    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            printf("This region is using %d threads\n", omp_get_num_threads()); 
        }
        printf("Hi, I\'m thread number %d\n", omp_get_thread_num());
    }
    printf("~~~ Exit parallel region ~~~\n\n");
    printf("This region is using %d threads\n\n", omp_get_num_threads()); 
    return 0;
} 