const char* dgemm_desc = "Simple blocked dgemm.";
#pragma GCC optimize ("peel-loops")

#include <stdio.h>
#include <immintrin.h>
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 28
#define BLOCK_SIZE2 18
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

// int micro_kernel(double* A, double* B, double* C)
// {
//     // Declare
//     __m256d Ar;
//     __m256d Br;
//     int c;

//     // Load
//     Ar = __mm256_load_pd(A);
//     Br = __mm256_load_pd(B);
    

//     // Compute
//     c = __mm256_dp_ps(Ar, Br, 1);

//     // Store
//     return c;
// }

static void do_block2(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // for small A block and small B block, multiply to small C block
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            // C[i][j]
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // for small A block and small B block, multiply to small C block
    // For each row i of A
    for (int i = 0; i < M; i += BLOCK_SIZE2)
    {
        for (int j = 0; j < N; j += BLOCK_SIZE2)
        {
            for (int k = 0; k < K; k += BLOCK_SIZE2)
            {
                int M2 = min(BLOCK_SIZE2, M - i);
                int N2 = min(BLOCK_SIZE2, N - j);
                int K2 = min(BLOCK_SIZE2, K - k);
                do_block2(lda, M2, N2, K2, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
     // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // block A for as many times, and block B for as many times
                // for each block combination along (length/block size), need ot perform do block
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}
