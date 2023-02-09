const char* dgemm_desc = "Simple blocked dgemm.";

#include <immintrin.h>
#include <stdio.h>
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 24
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

void micro_kernel(double* A, double* B, double* C)
{
    __m256d Ar;
    __m256d Br;
    __m256d Cr;
    __m256d Dr;

    // Load
    Ar = _mm256_set1_pd(A[0]);
    Br = _mm256_load_pd(B);
    Cr = _mm256_load_pd(C);

    // Compute
    Dr = _mm256_mul_pd(Ar, Br);
    Dr = _mm256_add_pd(Dr, Cr);

    // Store
    _mm256_store_pd(C, Dr);

}
/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    for (int k = 0; k < K; ++k)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; j+=4)
            {
                //printf("k: %d, i: %d, j: %d", k, i, j);
                // A[i][k] one element, B[k][j], C[i][j]
                //micro_kernel(A + i + k * lda, B + k + j * lda, C + i + j * lda);
                double cij = C[i + j * lda];
                cij += A[i + k * lda] * B[k + j * lda];
                C[i + j * lda] = cij;
            }
        }
    }
    // For each row i of A
    // for (int i = 0; i < M; ++i) {
    //     // For each column j of B
    //     for (int j = 0; j < N; ++j) {
    //         // Compute C(i,j)
    //         double cij = C[i + j * lda];
    //         for (int k = 0; k < K; ++k) {
    //             cij += A[i + k * lda] * B[k + j * lda];
    //         }
    //         C[i + j * lda] = cij;
    //     }
    // }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    printf("Here");
    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
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