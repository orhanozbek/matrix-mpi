/* filename: matMultiplyWithMPI.cpp * parallel matrix multiplication with MPI
 * C(m,n) = A(m,p) * B(p,n)
 * input: three parameters - m, p, n * @copyright: fengfu-chris
 */
#include<iostream>
#include<mpi.h>
#include<math.h>
#include<stdlib.h>void initMatrixWithRV(float *A, int rows, int cols);void matMultiplyWithSingleThread(float *A, float *B, float *matResult, int m, int p, int n);

int main(int argc, char** argv)
{
    int m = atoi(argv[1]);
    int p = atoi(argv[2]);
    int n = atoi(argv[3]);
    
    float *A, *B, *C;
    float *bA, *bC;  

    int myrank, numprocs;

    MPI_Status status;
  
    MPI_Init(&argc, &argv);  // 并行开始
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs); 
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
    
    int bm = m / numprocs;

    bA = new float[bm * p];
    B  = new float[p * n];
    bC = new float[bm * n];

    if(myrank == 0){
        A = new float[m * p];
        C = new float[m * n];
        
        initMatrixWithRV(A, m, p);
        initMatrixWithRV(B, p, n);
    }
    MPI_Barrier(MPI_COMM_WORLD);
　　/* step 1: 数据分配 */
    MPI_Scatter(A, bm * p, MPI_FLOAT, bA, bm *p, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, p * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
　　/* step 2: 并行计算C的各个分块 */
    matMultiplyWithSingleThread(bA, B, bC, bm, p, n);
    
    MPI_Barrier(MPI_COMM_WORLD);
    　　/* step 3: 汇总结果 */
    MPI_Gather(bC, bm * n, MPI_FLOAT, C, bm * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  　　/* step 3-1: 解决历史遗留问题（多余的分块） */
    int remainRowsStartId = bm * numprocs;
    if(myrank == 0 && remainRowsStartId < m){
        int remainRows = m - remainRowsStartId;
        matMultiplyWithSingleThread(A + remainRowsStartId * p, B, C + remainRowsStartId * n, remainRows, p, n);
    }
  
    delete[] bA;
    delete[] B;
    delete[] bC;
    
    if(myrank == 0){
        delete[] A;
        delete[] C;
    }
    
    MPI_Finalize(); // 并行结束

    return 0;
}

void initMatrixWithRV(float *A, int rows, int cols)
{
    srand((unsigned)time(NULL));
    for(int i = 0; i < rows*cols; i++){
        A[i] = (float)rand() / RAND_MAX;
    }
}
void matMultiplyWithSingleThread(float *A, float *B, float *matResult, int m, int p, int n)
{
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            float temp = 0;
            for(int k=0; k<p; k++){
                temp += A[i*p+k] * B[k*n + j];
            }
            matResult[i*n+j] = temp;
        }
    }
}