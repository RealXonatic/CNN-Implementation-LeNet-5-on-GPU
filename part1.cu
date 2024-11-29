#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){

}

void MatrixMult(float *M1, float *M2, float *Mout, int n){

}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n){

}

int main(int argc, char *argv[]){

    int n = 3, p = 3;  // Taille de la matrice
    float *M1 = (float*)malloc(n * p * sizeof(float));
    float *M2 = (float*)malloc(n * p * sizeof(float));
    float *Mout = (float*)malloc(n * p * sizeof(float));

    // Initialisation des matrices
    printf("Matrice M1 :\n");
    MatrixInit(M1, n, p);
    MatrixPrint(M1, n, p);

    printf("\nMatrice M2 :\n");
    MatrixInit(M2, n, p);
    MatrixPrint(M2, n, p);

    // Addition des matrices
    MatrixAdd(M1, M2, Mout, n, p);

    // Affichage du résultat
    printf("\nMatrice résultante (M1 + M2) :\n");
    MatrixPrint(Mout, n, p);

    // Libération de la mémoire
    free(M1);
    free(M2);
    free(Mout);

    return 0;
}
