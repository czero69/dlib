#include <iostream>
#include <cuda.h>
#include <driver_types.h>
#include "myCuda_tensorToDets.h"
using namespace std;

inline void gpuAssert(cudaError_t code, char * file, int line, bool Abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
        if (Abort) exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifdef DLIB_USE_CUDA

__device__ MyRectAndOffset dev_rects_data[MAX_RECTS];
__device__ int dev_rects_count = 0;

__device__ int my_push_back(MyRectAndOffset & mt) {
  int insert_pt = atomicAdd(&dev_rects_count, 1);
  if (insert_pt < MAX_RECTS){
    dev_rects_data[insert_pt] = mt;
    return insert_pt;}
  else return -1;}

int *a, *b;  // host data
int *c, *c2;  // results

__global__ void thresholdDets_useBB(const float * data, const long int nr_m_nc, const size_t det_win_size, float scoreThreshold)
{
    for(int k = 0; k < det_win_size; k++)
        for(int i = blockIdx.x * blockDim.x + threadIdx.x + k*nr_m_nc;
            i < nr_m_nc + k*nr_m_nc;
            i += blockDim.x * gridDim.x)
        {
            // get score
            float score = data[i];
            if( score > scoreThreshold)
            {
                  MyRectAndOffset rect;
                  rect.score = score;
                  rect.i = i;
                  rect.k = k;
                  const int offset = (det_win_size +  k*3) * nr_m_nc; // here it is k*4 = k = k*3, 4 is countfrom({dx,dy,dw,dh})
                  rect.dx = data[i + offset];
                  rect.dy = data[i + offset + nr_m_nc];
                  rect.dw = data[i + offset + 2*nr_m_nc];
                  rect.dh = data[i + offset + 3*nr_m_nc];
                  my_push_back(rect);
            }
         }
}

__global__ void thresholdDets_noBB(const float * data, const long int nr_m_nc, const size_t det_win_size, float scoreThreshold)
{
    for(int k = 0; k < det_win_size; k++)
        for(int i = blockIdx.x * blockDim.x + threadIdx.x + k*nr_m_nc;
            i < nr_m_nc + k*nr_m_nc;
            i += blockDim.x * gridDim.x)
        {
            // get score
            float score = data[i];
            if( score > scoreThreshold)
            {
                  MyRectAndOffset rect;
                  rect.score = score;
                  rect.i = i;
                  rect.k = k;
                  // no bb ..//
                  my_push_back(rect);
            }
         }
}

__global__ void Find()
{

    if(threadIdx.x < 10) //Simulate a found occurrence
    {
        MyRectAndOffset a;
        a.dx = 1;
        a.dy = 1;
        a.dw = 1;
        a.dh = 1;
        my_push_back(a);
    }
}

__global__ void vecAdd(int *A,int *B,int *C,int N)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   C[i] = A[i] + B[i];
}

void vecAdd_h(int *A1,int *B1, int *C1, int N)
{
   for(int i=0;i<N;i++)
      C1[i] = A1[i] * B1[i];
}


extern "C"
namespace dlib{
namespace cuda{

int get_rects_from_device2host(std::vector<MyRectAndOffset> & mr, const float* dev_data, const long int nr_m_nc, const size_t det_win_size, double scoreThreshold, bool useBB){

    //Find<<< 2, 256 >>>();
    // dev_rects_count = 0;
    int * _dev_rects_count;
    if(useBB)
        thresholdDets_useBB<<< MAX_RECTS/256, 256 >>>(dev_data, nr_m_nc, det_win_size, float(scoreThreshold));
    else {
        thresholdDets_noBB<<< MAX_RECTS/256, 256 >>>(dev_data, nr_m_nc, det_win_size, float(scoreThreshold));
    }
    int dsize;
    gpuErrchk(cudaMemcpyFromSymbol(&dsize, dev_rects_count, sizeof(int)));
    if (dsize >= MAX_RECTS) {printf("too many rects in get_rects_from_device2host\n"); return 1;}
    std::vector<MyRectAndOffset> results(dsize);
    gpuErrchk(cudaMemcpyFromSymbol(&(results[0]), dev_rects_data, dsize*sizeof(MyRectAndOffset)));
    // cudaMemset(dev_rects_data, 0, dsize*sizeof(MyRectAndOffset)); // @try cudaMemSetAsync
    // cudaMemset(&dev_rects_count, 0, sizeof(int));
    // cudaMemset(d_a, 0, num*sizeof(mystruct));
    gpuErrchk( cudaGetSymbolAddress((void **)&_dev_rects_count, dev_rects_count) );
    gpuErrchk( cudaMemset(_dev_rects_count,0,sizeof(int)));
    mr = results;

    return 0;
}

int set_flag_cudaDeviceSchedule(cudaDeviceScheduleMode::Choice scheduleChoice){

    if(scheduleChoice == cudaDeviceScheduleMode::Choice::Spinning)
    {
        gpuErrchk(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    }
    else if(scheduleChoice == cudaDeviceScheduleMode::Choice::Yield)
    {
        gpuErrchk(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    }
    else if(scheduleChoice == cudaDeviceScheduleMode::Choice::Blocking)
    {
        gpuErrchk(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    }
    else
        gpuErrchk(cudaSetDeviceFlags(cudaDeviceScheduleYield));

    return 0;
}

int runTest()
{
   std::cout << "runTest runTest runTest" << std::endl;
   printf("Begin \n");
   int n=10000000;
   int nBytes = n*sizeof(int);
   int block_size, block_no;
   a = (int *)malloc(nBytes);
   b = (int *)malloc(nBytes);
   c = (int *)malloc(nBytes);
   c2 = (int *)malloc(nBytes);
   int *a_d,*b_d,*c_d;
   block_size=1024;
   block_no = n/block_size;
   dim3 dimBlock(block_size,1,1);
   dim3 dimGrid(block_no,1,1);
   for(int i=0;i<n;i++)
      a[i]=i,b[i]=i;
   printf("Allocating device memory on host..\n");
   cudaMalloc((void **)&a_d,n*sizeof(int));
   cudaMalloc((void **)&b_d,n*sizeof(int));
   cudaMalloc((void **)&c_d,n*sizeof(int));
   printf("Copying to device..\n");
   cudaMemcpy(a_d,a,n*sizeof(int),cudaMemcpyHostToDevice);
   cudaMemcpy(b_d,b,n*sizeof(int),cudaMemcpyHostToDevice);
   clock_t start_d=clock();
   printf("Doing GPU Vector add\n");
   std::cout << "block_no x block_size: " << block_no << ", " << block_size << std::endl;
   vecAdd<<<block_no,block_size>>>(a_d,b_d,c_d,n);
   cudaThreadSynchronize();
   clock_t end_d = clock();
   clock_t start_h = clock();
   printf("Doing CPU Vector add\n");
   vecAdd_h(a,b,c2,n);
   clock_t end_h = clock();
   double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
   double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;
   cudaMemcpy(c,c_d,n*sizeof(int),cudaMemcpyDeviceToHost);
   printf("%d %f %f\n",n,time_d,time_h);
   cudaFree(a_d);
   cudaFree(b_d);
   cudaFree(c_d);
   return 0;
}
}
}

# endif
