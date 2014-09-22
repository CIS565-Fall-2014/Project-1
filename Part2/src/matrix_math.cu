#include <cstdio>
#include <cmath>

#define DIM 5
#define COUNT (DIM * DIM)

cudaEvent_t ev0;
cudaEvent_t ev1;

/******** Helpers ************************************************************/

#define CHECK_ERROR(msg) (checkCUDAError((msg), __LINE__))

/// Check for CUDA errors. From Part1.
void checkCUDAError(const char *msg, int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "line %d: CUDA error: %s - %s\n",
                line, msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


/******** GPU implementations ************************************************/

/// Add two matrices.
__global__ void mat_add(float *dst, float *A, float *B)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > DIM || y > DIM) { return; }
    int i = y * DIM + x;

    dst[i] = A[i] + B[i];
}

/// Subtract two matrices.
__global__ void mat_sub(float *dst, float *A, float *B)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > DIM || y > DIM) { return; }
    int i = y * DIM + x;

    dst[i] = A[i] - B[i];
}

/// Multiply two matrices.
__global__ void mat_mul(float *dst, float *A, float *B)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > DIM || y > DIM) { return; }
    int i = y * DIM + x;

    float sum = 0;
    for (int j = 0; j < DIM; ++j) {
        float Aj = A[y * DIM + j];
        float Bj = B[j * DIM + x];
        sum += Aj * Bj;
    }

    dst[i] = sum;
}


/******** CPU implementations ************************************************/

void mat_add_cpu(float *dst, float *A, float *B)
{
    for (int i = 0; i < COUNT; ++i) {
        dst[i] = A[i] + B[i];
    }
}

void mat_sub_cpu(float *dst, float *A, float *B)
{
    for (int i = 0; i < COUNT; ++i) {
        dst[i] = A[i] - B[i];
    }
}

void mat_mul_cpu(float *dst, float *A, float *B)
{
    for (int x = 0; x < DIM; ++x) {
        for (int y = 0; y < DIM; ++y) {
            float sum = 0;
            for (int j = 0; j < DIM; ++j) {
                float a = A[y * DIM + j];
                float b = B[j * DIM + x];
                sum += a * b;
            }
            dst[y * DIM + x] = sum;
        }
    }
}


/******** Main functions *****************************************************/

float *hst_A, *hst_C, *hst_D;
float *dev_A, *dev_C;

bool test_impl(
    void (*gpu)(float *, float *, float *),
    void (*cpu)(float *, float *, float *))
{
    // Calculate grid/block sizes
    dim3 dimGrid(1, 1);
    dim3 dimBlock(DIM, DIM);

    // Run on GPU
    cudaMemcpy(dev_A, hst_A, COUNT * sizeof(*dev_A), cudaMemcpyHostToDevice);
    CHECK_ERROR("memcpy");
    (*gpu)<<<dimGrid, dimBlock>>>(dev_C, dev_A, dev_A);
    CHECK_ERROR("function");
    cudaMemcpy(hst_C, dev_C, COUNT * sizeof(*hst_C), cudaMemcpyDeviceToHost);
    CHECK_ERROR("memcpy");

    // Run on CPU
    (*cpu)(hst_D, hst_A, hst_A);

    // Check results
    for (int i = 0; i < COUNT; ++i) {
        if (hst_C[i] != hst_D[i]) {
            return false;
        }
    }
    return true;
}

float perf_test_one(dim3 grid, dim3 block, int iters,
        void (*f)(float *, float *, float *))
{
    // cudaEvent stuff reference: http://choorucode.com/2011/04/28/cuda-timer/
    float t = 0;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(ev0, 0);
        (*f)<<<grid, block>>>(dev_C, dev_A, dev_A);
        cudaEventRecord(ev1, 0);
        cudaMemcpy(hst_C, dev_C, COUNT * sizeof(*hst_C), cudaMemcpyDeviceToHost);

        cudaEventSynchronize(ev1);
        float timeValue;
        cudaEventElapsedTime(&timeValue, ev0, ev1);
        t += timeValue;
    }
    return t;
}

void perf_test(const char *name, void (*f)(float *, float *, float *))
{
    const int iters = 10000;
    for (int g = 1; g <= DIM; ++g) {
        for (int b = 1; b <= DIM; ++b) {
            float t = perf_test_one(g, b, iters, f);
            printf("%s %d %d %f\n", name, g, b, t);
        }
    }
}

int main()
{
    // Allocate matrices
    hst_A = (float *) malloc(COUNT * sizeof(*hst_A));
    hst_C = (float *) malloc(COUNT * sizeof(*hst_C));
    hst_D = (float *) malloc(COUNT * sizeof(*hst_D));
    cudaMalloc(&dev_A, COUNT * sizeof(*dev_A));
    cudaMalloc(&dev_C, COUNT * sizeof(*dev_C));
    CHECK_ERROR("malloc");

    // Initialize host/device matrices
    for (int i = 0; i < COUNT; ++i) {
        hst_A[i] = (float) i;
    }
    cudaMemcpy(dev_A, hst_A, COUNT * sizeof(*dev_A), cudaMemcpyHostToDevice);

    // Test GPU implementations
    printf("%s  mat_add\n", test_impl(mat_add, mat_add_cpu) ? "pass" : "FAIL");
    printf("%s  mat_sub\n", test_impl(mat_sub, mat_sub_cpu) ? "pass" : "FAIL");
    printf("%s  mat_mul\n", test_impl(mat_mul, mat_mul_cpu) ? "pass" : "FAIL");

    // Performance timing tests
    // cudaEvent stuff reference: http://choorucode.com/2011/04/28/cuda-timer/
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    perf_test("mat_add", mat_add);
    perf_test("mat_sub", mat_sub);
    perf_test("mat_mul", mat_mul);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    return 0;
}
