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
            printf("%f ", M[i * p + j]);
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
    int n = 3; // Taille de la matrice
    float *M1, *M2, *Mout, *d_M1, *d_M2, *d_Mout;

    // Allocation de mémoire sur l'hôte (CPU)
    M1 = (float*)malloc(n * n * sizeof(float));
    M2 = (float*)malloc(n * n * sizeof(float));
    Mout = (float*)malloc(n * n * sizeof(float));

    // Initialisation des matrices
    for (int i = 0; i < n * n; i++) {
        M1[i] = 1.0f; // Remplir M1 avec 1
        M2[i] = 2.0f; // Remplir M2 avec 2
    }

    // Allocation de mémoire sur le GPU
    cudaMalloc(&d_M1, n * n * sizeof(float));
    cudaMalloc(&d_M2, n * n * sizeof(float));
    cudaMalloc(&d_Mout, n * n * sizeof(float));

    // Copier les matrices de l'hôte (CPU) vers le GPU
    cudaMemcpy(d_M1, M1, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Configurer les dimensions des blocs et des grilles
    dim3 blockDim(16, 16); // Taille d'un bloc (16 x 16 threads)
    dim3 gridDim((n + 15) / 16, (n + 15) / 16); // Taille de la grille

    // Lancer le kernel pour effectuer la multiplication
    cudaMatrixMult<<<gridDim, blockDim>>>(d_M1, d_M2, d_Mout, n);

    // Copier le résultat du GPU vers l'hôte
    cudaMemcpy(Mout, d_Mout, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Afficher la matrice résultante
    printf("Matrice résultante (M1 * M2) :\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", Mout[i * n + j]);
        }
        printf("\n");
    }

    // Libérer la mémoire
    free(M1);
    free(M2);
    free(Mout);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    return 0;
}