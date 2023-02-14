const char* dgemm_desc = "Simple blocked dgemm.";

#include <immintrin.h>
#include <stdio.h>
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 24
#endif
#define min(a, b) (((a) < (b)) ? (a) : (b))

void foo(double s, double* a, double* c, int n);

void foo2(double s, double *a, double *c, int n);

void foo3(double s, double *a, double *c, int n);

void square_dgemm(int n, double* A, double* B, double* C) {
    // For each row i of A
    for (int j = 0; j < n; ++j)
    {
        for (int k = 0; k < n; ++k)
        {
            //foo(B[k + j * n], A + k * n, C + j * n, n);
            int largestMul = n - (n % 8);
            int diff = n - largestMul;
            foo3(B[k + j * n], A + k * n, C + j * n, largestMul);
            foo(B[k + j * n], A + k * n + largestMul, C + j * n + largestMul, diff);
        }
    }
}

void foo3(double s, double *A, double *C, int n)
{
    // element wise multiply a and s
    // then add result back to c
    for (int i = 0; i < n/8; i++)
    {
        // Declare
        double A0, A1, A2, A3, A4, A5, A6, A7;
        double C0, C1, C2, C3, C4, C5, C6, C7;

        // Load
        A0 = A[i * 8 + 0], A1 = A[i * 8 + 1], A2 = A[i * 8 + 2], A3 = A[i * 8 + 3], A4 = A[i * 8 + 4], A5 = A[i * 8 + 5], A6 = A[i * 8 + 6], A7 = A[i * 8 + 7];
        C0 = C[i * 8 + 0], C1 = C[i * 8 + 1], C2 = C[i * 8 + 2], C3 = C[i * 8 + 3], C4 = C[i * 8 + 4], C5 = C[i * 8 + 5], C6 = C[i * 8 + 6], C7 = C[i * 8 + 7];

        // Compute
        C0 = s * A0 + C0, C1 = s * A1 + C1, C2 = s * A2 + C2, C3 = s * A3 + C3, C4 = s * A4 + C4, C5 = s * A5 + C5, C6 = s * A6 + C6, C7 = s * A7 + C7;

        // Store
        C[i * 8 + 0] = C0, C[i * 8 + 1] = C1, C[i * 8 + 2] = C2, C[i * 8 + 3] = C3, C[i * 8 + 4] = C4, C[i * 8 + 5] = C5, C[i * 8 + 6] = C6, C[i * 8 + 7] = C7;

    }
}

void foo2(double s, double *a, double *c, int n)
{
    __m256d sv = _mm256_set1_pd(s);
    for (int i = 0; i < n/4; i++)
    {
        __m256d cv = _mm256_loadu_pd(&c[4 * i]);
        __m256d av = _mm256_loadu_pd(&a[4 * i]);
        __m256d t1 = _mm256_mul_pd(sv, av);
        cv = _mm256_add_pd(t1, cv);
        _mm256_storeu_pd(&c[4 * i], cv);
    }
}

void foo(double s, double* a, double* c, int n)
{
    for (int i = 0; i < n; ++i)
    {
        c[i] = s * a[i] + c[i];
    }
}