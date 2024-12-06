#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

void MatrixInit(float *M, int n, int p) {
    // Initialisation du générateur de nombres aléatoires
    srand(time(NULL));

    // Parcours de chaque élément de la matrice
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            // Générer une valeur aléatoire entre -1 et 1
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%5.1f ", M[i * p + j]);
        }
        printf("\n");
    }
}


void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    // Calcul de l'index des threads
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Vérifier si l'index est valide (dans les limites de la matrice)
    if (row < n && col < p) {
        Mout[row * p + col] = M1[row * p + col] + M2[row * p + col];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    // Initialiser la matrice de sortie à 0
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mout[i * n + j] = 0.0f;
        }
    }

    // Multiplication classique (3 boucles imbriquées)
    for (int i = 0; i < n; i++) { // Parcourir les lignes de M1
        for (int j = 0; j < n; j++) { // Parcourir les colonnes de M2
            for (int k = 0; k < n; k++) { // Parcourir les colonnes de M1 et les lignes de M2
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    // Calcul de l'index des threads (ligne et colonne)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Vérifier si l'index est dans les limites de la matrice
    if (row < n && col < n) {
        float sum = 0.0f;

        // Multiplication ligne-colonne
        for (int k = 0; k < n; k++) {
            sum += M1[row * n + k] * M2[k * n + col];
        }

        // Stocker le résultat dans Mout
        Mout[row * n + col] = sum;
    }
}

int main() {
    int n = 1000, p = 1000; // Dimensions des matrices

    // Allocation mémoire sur le CPU
    float *M1, *M2, *Mout_cpu, *Mout_gpu;
    M1 = (float*)malloc(n * p * sizeof(float));
    M2 = (float*)malloc(n * p * sizeof(float));
    Mout_cpu = (float*)malloc(n * p * sizeof(float));
    Mout_gpu = (float*)malloc(n * p * sizeof(float));

    // Initialisation des matrices
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    // Affichage des matrices initiales
    printf("Matrice M1 :\n");
    //MatrixPrint(M1, n, p);
    printf("\nMatrice M2 :\n");
    //MatrixPrint(M2, n, p);

    //////////////////////
    // ADDITION - CPU   //
    //////////////////////
    clock_t start_cpu_add = clock();
    MatrixAdd(M1, M2, Mout_cpu, n, p);
    clock_t end_cpu_add = clock();
    printf("\nRésultat addition CPU :\n");
    //MatrixPrint(Mout_cpu, n, p);
    printf("Temps addition CPU : %.4f secondes\n", (double)(end_cpu_add - start_cpu_add) / CLOCKS_PER_SEC);

    //////////////////////
    // ADDITION - GPU   //
    //////////////////////
    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void**)&d_M1, n * p * sizeof(float));
    cudaMalloc((void**)&d_M2, n * p * sizeof(float));
    cudaMalloc((void**)&d_Mout, n * p * sizeof(float));

    cudaMemcpy(d_M1, M1, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, n * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    cudaMatrixAdd<<<gridDim, blockDim>>>(d_M1, d_M2, d_Mout, n, p);

    cudaMemcpy(Mout_gpu, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nRésultat addition GPU :\n");
    //MatrixPrint(Mout_gpu, n, p);


    //////////////////////////
    // MULTIPLICATION - CPU //
    //////////////////////////
    clock_t start_cpu_mult = clock();
    MatrixMult(M1, M2, Mout_cpu, n);
    clock_t end_cpu_mult = clock();
    printf("\nRésultat multiplication CPU :\n");
    //MatrixPrint(Mout_cpu, n, n);
    printf("Temps multiplication CPU : %.4f secondes\n", (double)(end_cpu_mult - start_cpu_mult) / CLOCKS_PER_SEC);

    //////////////////////////
    // MULTIPLICATION - GPU //
    //////////////////////////
    cudaMatrixMult<<<gridDim, blockDim>>>(d_M1, d_M2, d_Mout, n);

    cudaMemcpy(Mout_gpu, d_Mout, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nRésultat multiplication GPU :\n");
    //MatrixPrint(Mout_gpu, n, n);


    // Libération mémoire
    free(M1);
    free(M2);
    free(Mout_cpu);
    free(Mout_gpu);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    return 0;
}