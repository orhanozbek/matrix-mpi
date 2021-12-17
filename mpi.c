/* filename: matMultiplyWithMPI_updated.cpp
 * parallel matrix multiplication with MPI: updated
 * C(m,n) = A(m,p) * B(p,n)
 * input: three parameters - m, p, n
 * @copyright: fengfu-chris
 */
#include<iostream>
#include<mpi.h>
#include<math.h>
#include<stdlib.h>

void initMatrixWithRV(float *A, int rows, int cols);
void copyMatrix(float *A, float *A_copy, int rows, int cols);
// A: m*p, B: p*n  ！！！ note that B is stored by column first
void matMultiplyWithTransposedB(float *A, float *B, float *matResult, int m, int n, int p);

int main(int argc, char** argv)
{
　 int m = atoi(argv[1]);
　 int n = atoi(argv[2]);
　 int p = atoi(argv[3]);
    
   float *A, *B, *C;
   float *bA, *bB_send, *bB_recv, *bC, *bC_send;
　 int myrank, numprocs;

    MPI_Status status;
  
    MPI_Init(&argc, &argv);  // 并行开始
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs); 
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
 　　
　  int bm = m / numprocs;
    int bn = n / numprocs;

    bA = new float[bm * p];
    bB_send = new float[bn * p];
    bB_recv = new float[bn * p];
    bC = new float[bm * bn];
    bC_send = new float[bm * n];
    
    if(myrank == 0){
        A = new float[m * p];
        B = new float[n * p];
        C = new float[m * n];
        
        initMatrixWithRV(A, m, p);
        initMatrixWithRV(B, n, p);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(A, bm * p, MPI_FLOAT, bA, bm * p, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, bn * p, MPI_FLOAT, bB_recv, bn * p, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int sendTo = (myrank + 1) % numprocs;
    int recvFrom = (myrank - 1 + numprocs) % numprocs;
        
    int circle = 0;  
    do{
        matMultiplyWithTransposedB(bA, bB_recv, bC, bm, bn, p);
        int blocks_col = (myrank - circle + numprocs) % numprocs;
        for(int i=0; i<bm; i++){
            for(int j=0; j<bn; j++){
                bC_send[i*n + blocks_col*bn + j] = bC[i*bn + j];
            }
        }

        if(myrank % 2 == 0){
            copyMatrix(bB_recv, bB_send, bn, p);
            MPI_Ssend(bB_send, bn*p, MPI_FLOAT, sendTo, circle, MPI_COMM_WORLD);
            MPI_Recv(bB_recv, bn*p, MPI_FLOAT, recvFrom, circle, MPI_COMM_WORLD, &status);
        }else{
            MPI_Recv(bB_recv, bn*p, MPI_FLOAT, recvFrom, circle, MPI_COMM_WORLD, &status); 
            MPI_Ssend(bB_send, bn*p, MPI_FLOAT, sendTo, circle, MPI_COMM_WORLD);
            copyMatrix(bB_recv, bB_send, bn, p);    
        }
        
        circle++;
    }while(circle < numprocs);

　 MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(bC_send, bm * n, MPI_FLOAT, C, bm * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if(myrank == 0){
        int remainAStartId = bm * numprocs;
        int remainBStartId = bn * numprocs;
        
        for(int i=remainAStartId; i<m; i++){
            for(int j=0; j<n; j++){
                float temp=0;
                for(int k=0; k<p; k++){
                    temp += A[i*p + k] * B[j*p +k];
                }
                C[i*p + j] = temp;
            }
        }
        
        for(int i=0; i<remainAStartId; i++){
            for(int j=remainBStartId; j<n; j++){
                float temp = 0;
                for(int k=0; k<p; k++){
                    temp += A[i*p + k] * B[j*p +k];
                }
                C[i*p + j] = temp;
            }
        }
    }
    
    delete[] bA;
    delete[] bB_send;
    delete[] bB_recv;
    delete[] bC;
    delete[] bC_send;
    
    if(myrank == 0){
        delete[] A;
        delete[] B;
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
void copyMatrix(float *A, float *A_copy, int rows, int cols)
{
    for(int i=0; i<rows*cols; i++){
        A_copy[i] = A[i];
    }
}

void matMultiplyWithTransposedB(float *A, float *B, float *matResult, int m, int p, int n)
{
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            float temp = 0;
            for(int k=0; k<p; k++){
                temp += A[i*p+k] * B[j*p+k];
            }
            matResult[i*n+j] = temp;
        }
    }
}