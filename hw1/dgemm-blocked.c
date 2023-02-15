const char* dgemm_desc = "Simple blocked dgemm.";

#include <immintrin.h>
#include <stdio.h>
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 24
#endif
#include <prfchwintrin.h>
#define min(a, b) (((a) < (b)) ? (a) : (b))

void foo(double s, double* a, double* c, int n);

void foo2(double s, double *a, double *c, int n);

void foo3(double s, double *a, double *c, int n);

static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C);

void square_dgemm_blocked(int lda, double* A, double* B, double* C) {
    // do inner products of sequences of block and store back to results
    for (int j = 0; j < lda; j+= BLOCK_SIZE)
    {
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            for (int i = 0; i < lda; i += BLOCK_SIZE)
            {
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // unpack matrix A (memory aligned)
    // unpack matrix B (memory aligned)
    double* A2 = _mm_malloc(M * K * sizeof(double), 64);
    double* B2 = _mm_malloc(K * N * sizeof(double), 64);

    for (int j = 0; j < K; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            A2[i + j * M] = A[j * lda + i];
        }
    }

    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < K; ++i)
        {
            B2[i + j * K] = B[j * lda + i];
        }
    }

    // within the block, use inner product to compute and store the result
    for (int j = 0; j < N; ++j)
    {
        for (int k = 0; k < K; ++k)
        {
            //foo(B2[k + j * K], A2 + k * M, C + j * lda, M);
            int largestMul = M - (M % 8);
            int diff = M - largestMul;
            foo3(B[k + j * n], A + k * n, C + j * n, largestMul);
            //foo2(B2[k + j * K], A2+ k * M, C + j * n, largestMul);
            foo(B2[k + j * K], A2 + k * M + largestMul, C + j * n + largestMul, diff);

        }
    }
}

void square_dgemm(int n, double* A, double* B, double* C) {
    
    // allocate new spaces to make A and B memory aligned
    double* A2 = _mm_malloc(n * n * sizeof(double), 64);
    double* B2 = _mm_malloc(n * n * sizeof(double), 64);

    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            A2[i + j * n] = A[j * n + i];
        }
    }

    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            B2[i + j * n] = B[j * n + i];
        }
    }

    for (int j = 0; j < n; ++j)
    {
        for (int k = 0; k < n; ++k)
        {
            // use inner product to compute the result of the entire array
            //foo(B[k + j * n], A + k * n, C + j * n, n);
            int largestMul = n - (n % 4);
            int diff = n - largestMul;
            //foo3(B[k + j * n], A + k * n, C + j * n, largestMul);
            foo2(B[k + j * n], A + k * n, C + j * n, largestMul);
            foo(B[k + j * n], A + k * n + largestMul, C + j * n + largestMul, diff);

            //foo(B2[k + j * n], A2 + k * n, C + j * n, n);
        }
    }
}

void foo3(double s, double *A, double *C, int n)
{
    // size-8 microkernel for broadcasting and element wise computation
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
    // size-4 microkernel for broadcasting and element wise computation
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
    // direct broadcasting and element wise computation
    for (int i = 0; i < n; ++i)
    {
        c[i] = s * a[i] + c[i];
    }
}