#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <chrono>
#include <thread>
#include <vector>

#define PRINT 1

int SZ = 1000000;  // Larger size for performance comparison
int *v1, *v2, *result_opencl, *result_threaded;

// OpenCL variables
cl_mem bufV1, bufV2, bufResult;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_event event = NULL;
int err;

// Function declarations
cl_device_id create_device();
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
void setup_kernel_memory();
void copy_kernel_args();
void free_memory();
void init_vectors(int size);
void print_vectors(int *A, int *B, int *C, int size);
void vector_add_threaded(int num_threads);
void vector_add_worker(int start, int end);

int main(int argc, char **argv)
{
    if (argc > 1)
        SZ = atoi(argv[1]);

    init_vectors(SZ);
    
    printf("Vector Addition Performance Comparison\n");
    printf("Vector Size: %d\n", SZ);
    printf("========================================\n");

    // OpenCL Implementation
    auto start_opencl = std::chrono::high_resolution_clock::now();
    
    size_t global[1] = {(size_t)SZ};
    
    setup_openCL_device_context_queue_kernel((char *)"./vector_add.cl", (char *)"vector_add");
    setup_kernel_memory();
    copy_kernel_args();
    
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, SZ * sizeof(int), &result_opencl[0], 0, NULL, NULL);
    
    auto end_opencl = std::chrono::high_resolution_clock::now();
    auto duration_opencl = std::chrono::duration_cast<std::chrono::microseconds>(end_opencl - start_opencl);
    
    printf("OpenCL Execution Time: %ld microseconds\n", duration_opencl.count());
    
    // Multi-threaded Implementation
    auto start_threaded = std::chrono::high_resolution_clock::now();
    
    vector_add_threaded(std::thread::hardware_concurrency());
    
    auto end_threaded = std::chrono::high_resolution_clock::now();
    auto duration_threaded = std::chrono::duration_cast<std::chrono::microseconds>(end_threaded - start_threaded);
    
    printf("Multi-threaded Execution Time: %ld microseconds\n", duration_threaded.count());
    
    // Performance comparison
    double speedup = (double)duration_threaded.count() / duration_opencl.count();
    printf("OpenCL Speedup: %.2fx\n", speedup);
    
    // Verify results match
    bool results_match = true;
    for (int i = 0; i < SZ && results_match; i++) {
        if (result_opencl[i] != result_threaded[i]) {
            results_match = false;
        }
    }
    printf("Results Match: %s\n", results_match ? "Yes" : "No");
    
    if (SZ <= 20) {
        printf("\nSample Results:\n");
        print_vectors(v1, v2, result_opencl, SZ);
    }
    
    free_memory();
    return 0;
}

void init_vectors(int size)
{
    v1 = (int *)malloc(sizeof(int) * size);
    v2 = (int *)malloc(sizeof(int) * size);
    result_opencl = (int *)malloc(sizeof(int) * size);
    result_threaded = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++) {
        v1[i] = rand() % 100;
        v2[i] = rand() % 100;
        result_opencl[i] = 0;
        result_threaded[i] = 0;
    }
}

void print_vectors(int *A, int *B, int *C, int size)
{
    if (PRINT == 0) return;
    
    printf("V1: ");
    for (int i = 0; i < size && i < 10; i++) {
        printf("%d ", A[i]);
    }
    printf("\nV2: ");
    for (int i = 0; i < size && i < 10; i++) {
        printf("%d ", B[i]);
    }
    printf("\nResult: ");
    for (int i = 0; i < size && i < 10; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");
}

void vector_add_threaded(int num_threads)
{
    std::vector<std::thread> threads;
    int chunk_size = SZ / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? SZ : (i + 1) * chunk_size;
        threads.emplace_back(vector_add_worker, start, end);
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

void vector_add_worker(int start, int end)
{
    for (int i = start; i < end; i++) {
        result_threaded[i] = v1[i] + v2[i];
    }
}

void free_memory()
{
    clReleaseMemObject(bufV1);
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufResult);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    
    free(v1);
    free(v2);
    free(result_opencl);
    free(result_threaded);
}

void copy_kernel_args()
{
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufResult);
}

void setup_kernel_memory()
{
    bufV1 = clCreateBuffer(context, CL_MEM_READ_ONLY, SZ * sizeof(int), NULL, NULL);
    bufV2 = clCreateBuffer(context, CL_MEM_READ_ONLY, SZ * sizeof(int), NULL, NULL);
    bufResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SZ * sizeof(int), NULL, NULL);
    
    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname)
{
    device_id = create_device();
    cl_int err;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device_id, filename);

    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    }

    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0) {
        perror("Couldn't create a kernel");
        exit(1);
    }
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    program_handle = fopen(filename, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

cl_device_id create_device() {
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) {
        perror("Couldn't identify a platform");
        exit(1);
    } 

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if(err == CL_DEVICE_NOT_FOUND) {
        printf("GPU not found, using CPU\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if(err < 0) {
        perror("Couldn't access any devices");
        exit(1);   
    }

    return dev;
}