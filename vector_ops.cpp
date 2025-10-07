// SIT315 Seminar 8 deliverable copy
// Source copied from vector_opencl/vector_ops.cpp at submission time
// For build/run instructions, see ../How_To_Run.md

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <chrono>
#include <thread>
#include <vector>

#define PRINT 1

int SZ = 8;
int *v;

// Device buffer that holds the vector data on the OpenCL device (GPU/CPU)
cl_mem bufV;

// Additional device buffers for vector addition (A + B -> C)
cl_mem bufA, bufB, bufC;

// OpenCL device handle (represents a single compute device selected from a platform)
cl_device_id device_id;
// OpenCL context (lifecycle container that owns memory objects, programs, and command queues)
cl_context context;
// Compiled OpenCL program object created from kernel source
cl_program program;
// Handle to an OpenCL kernel function within the program (here: square_magnitude)
cl_kernel kernel;
// Command queue used by the host to enqueue work and memory commands to the device
cl_command_queue queue;
// Optional event used to wait/profile enqueued operations
cl_event event = NULL;

int err;

// Select and return an available OpenCL device (prefer GPU, fallback to CPU)
cl_device_id create_device();
// Create the OpenCL context, build program from file, create command queue and requested kernel
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);
// Read kernel source from file, create program object and build it for the device
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
// Allocate device buffers needed by the current kernel and upload input data
void setup_kernel_memory();
// Bind kernel parameters (set kernel arguments before launching the kernel)
void copy_kernel_args();
// Release all allocated host/device resources
void free_memory();

// Helpers for vector addition and timing
void setup_vector_add_memory(const int *A, const int *B, int size);
void set_args_vector_add(int size);
void release_vector_add_memory();
void vector_add_multithread(const int *A, const int *B, int *C, int size, unsigned numThreads);

void init(int *&A, int size);
void print(int *A, int size);

int main(int argc, char **argv)
{
	if (argc > 1)
		SZ = atoi(argv[1]);

	init(v, SZ);


	// Global ND-range size for a 1D kernel launch (number of work-items equals vector length)
	size_t global[1] = {(size_t)SZ};

	//initial vector
	print(v, SZ);

	setup_openCL_device_context_queue_kernel((char *)"./vector_ops.cl", (char *)"square_magnitude");

	setup_kernel_memory();
	copy_kernel_args();

	// Enqueue an N-D range kernel for execution.
	// Args: command queue, kernel handle, work-dim (1D here), global_work_offset (NULL),
	//       global_work_size (array of size work-dim), local_work_size (NULL = runtime decides),
	//       num_events_in_wait_list, event_wait_list, event (receives completion event).
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
	clWaitForEvents(1, &event);

	// Read back results from device buffer to host memory (blocking read)
	// Args: queue, buffer, blocking_read, offset, size_in_bytes, host_ptr, wait_list_count, wait_list, return_event
	clEnqueueReadBuffer(queue, bufV, CL_TRUE, 0, SZ * sizeof(int), &v[0], 0, NULL, NULL);
	if (event) { clReleaseEvent(event); event = NULL; }

	//result vector
	print(v, SZ);

	// ------------------------------------------
	// Parallel Vector Addition (OpenCL vs threads)
	// ------------------------------------------
	int *A = NULL, *B = NULL, *C = NULL, *C_mt = NULL;
	init(A, SZ);
	init(B, SZ);
	C = (int *)malloc(sizeof(int) * SZ);
	C_mt = (int *)malloc(sizeof(int) * SZ);

	// Create an additional kernel for vector addition from the same program
	cl_kernel kernel_add = clCreateKernel(program, "vector_add", &err);
	if (err < 0) {
		perror("Couldn't create vector_add kernel");
		exit(1);
	}

	// Prepare device memory for A, B, and C
	setup_vector_add_memory(A, B, SZ);

	// Set kernel arguments
	err = clSetKernelArg(kernel_add, 0, sizeof(int), (void *)&SZ);
	if (err == CL_SUCCESS) err = clSetKernelArg(kernel_add, 1, sizeof(cl_mem), (void *)&bufA);
	if (err == CL_SUCCESS) err = clSetKernelArg(kernel_add, 2, sizeof(cl_mem), (void *)&bufB);
	if (err == CL_SUCCESS) err = clSetKernelArg(kernel_add, 3, sizeof(cl_mem), (void *)&bufC);
	if (err < 0) {
		perror("Couldn't set vector_add kernel args");
		exit(1);
	}

	// Time OpenCL vector addition (host-side timing around enqueue + readback)
	auto t0 = std::chrono::high_resolution_clock::now();
	cl_event add_event = NULL;
	clEnqueueNDRangeKernel(queue, kernel_add, 1, NULL, global, NULL, 0, NULL, &add_event);
	clWaitForEvents(1, &add_event);
	clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, SZ * sizeof(int), &C[0], 0, NULL, NULL);
	if (add_event) { clReleaseEvent(add_event); add_event = NULL; }
	auto t1 = std::chrono::high_resolution_clock::now();
	double opencl_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

	// Time multi-threaded C++ version
	unsigned threads = std::thread::hardware_concurrency();
	if (threads == 0) threads = 4; // default fallback
	auto t2 = std::chrono::high_resolution_clock::now();
	vector_add_multithread(A, B, C_mt, SZ, threads);
	auto t3 = std::chrono::high_resolution_clock::now();
	double mt_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

	// Validate a couple of elements
	bool ok = true;
	for (int i = 0; i < SZ; ++i) {
		if (C[i] != A[i] + B[i]) { ok = false; break; }
	}

	printf("Vector add check: %s\n", ok ? "OK" : "MISMATCH");
	printf("OpenCL add time: %.3f ms\n", opencl_ms);
	printf("Threads add time: %.3f ms (threads=%u)\n", mt_ms, threads);

	// Print a few results
	if (PRINT) {
		printf("A: "); print(A, SZ);
		printf("B: "); print(B, SZ);
		printf("C(OpenCL): "); print(C, SZ);
		printf("C(Threads): "); print(C_mt, SZ);
	}

	// Cleanup resources for vector addition
	clReleaseKernel(kernel_add);
	release_vector_add_memory();
	free(A); free(B); free(C); free(C_mt);

	//frees memory for device, kernel, queue, etc.
	//you will need to modify this to free your own buffers
	free_memory();
}

void init(int *&A, int size)
{
	A = (int *)malloc(sizeof(int) * size);

	for (long i = 0; i < size; i++)
	{
		A[i] = rand() % 100; // any number less than 100
	}
}

void print(int *A, int size)
{
	if (PRINT == 0)
	{
		return;
	}

	if (PRINT == 1 && size > 15)
	{
		for (long i = 0; i < 5; i++)
		{                        //rows
			printf("%d ", A[i]); // print the cell value
		}
		printf(" ..... ");
		for (long i = size - 5; i < size; i++)
		{                        //rows
			printf("%d ", A[i]); // print the cell value
		}
	}
	else
	{
		for (long i = 0; i < size; i++)
		{                        //rows
			printf("%d ", A[i]); // print the cell value
		}
	}
	printf("\n----------------------------\n");
}

void free_memory()
{
	//free the buffers
	clReleaseMemObject(bufV);
	// Note: vector-add buffers are released in release_vector_add_memory()

	//free opencl objects
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);

	free(v);
}


void copy_kernel_args()
{
	// clSetKernelArg binds host-side values/buffers to the kernel's argument list.
	// Args: kernel handle, arg index, arg size in bytes, pointer to value or cl_mem handle
	err = clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);
	if (err == CL_SUCCESS)
		err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV);

	if (err < 0)
	{
		perror("Couldn't create a kernel argument");
		printf("error = %d", err);
		exit(1);
	}
}

void setup_kernel_memory()
{
	// clCreateBuffer allocates a device memory object (buffer) within the context.
	// Args: context, memory flags (e.g., CL_MEM_READ_WRITE, CL_MEM_READ_ONLY), size in bytes,
	//       host pointer (optional), error code out.
	// Flags control access semantics and optional host pointer usage.
	bufV = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, &err);
	if (err < 0) { perror("Couldn't create buffer V"); exit(1);} 

	// Copy matrices to the GPU
	clEnqueueWriteBuffer(queue, bufV, CL_TRUE, 0, SZ * sizeof(int), &v[0], 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname)
{
	device_id = create_device();
	cl_int err;

	// clCreateContext creates an OpenCL context that ties together devices and resources
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	if (err < 0)
	{
		perror("Couldn't create a context");
		exit(1);
	}

	program = build_program(context, device_id, filename);

	// clCreateCommandQueueWithProperties creates a command queue used to submit work to a device.
	// Properties can enable features like profiling; 0 uses defaults.
	queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
	if (err < 0)
	{
		perror("Couldn't create a command queue");
		exit(1);
	};


	kernel = clCreateKernel(program, kernelname, &err);
	if (err < 0)
	{
		perror("Couldn't create a kernel");
		printf("error =%d", err);
		exit(1);
	};
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{

	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;

	/* Read program file and place content into buffer */
	// Open in binary mode to avoid CRLF translation issues on Windows
	program_handle = fopen(filename, "rb");
	if (program_handle == NULL)
	{
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

	//ToDo: Add comment (what is the purpose of clCreateProgramWithSource function? What are its arguments?)
	// Pass NULL for lengths so the implementation uses the null-terminated string
	program = clCreateProgramWithSource(ctx, 1,
										(const char **)&program_buffer, NULL, &err);
	if (err < 0)
	{
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program 

   The fourth parameter accepts options that configure the compilation. 
   These are similar to the flags used by gcc. For example, you can 
   define a macro with the option -DMACRO=VALUE and turn off optimization 
   with -cl-opt-disable.
   */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0)
	{

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
							  0, NULL, &log_size);
		program_log = (char *)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
							  log_size + 1, program_log, NULL);
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

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
	  perror("Couldn't identify a platform");
	  exit(1);
   } 

   // Access a device
   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
	  // CPU
	  printf("GPU not found\n");
	  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
	  perror("Couldn't access any devices");
	  exit(1);   
   }

   return dev;
}

// ----------------------------
// Vector addition support code
// ----------------------------
void setup_vector_add_memory(const int *A, const int *B, int size) {
	cl_int e = 0;
	bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(int), NULL, &e);
	if (e < 0) { perror("Couldn't create buffer A"); exit(1);} 
	bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(int), NULL, &e);
	if (e < 0) { perror("Couldn't create buffer B"); exit(1);} 
	bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(int), NULL, &e);
	if (e < 0) { perror("Couldn't create buffer C"); exit(1);} 

	// Upload inputs
	clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, size * sizeof(int), A, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, size * sizeof(int), B, 0, NULL, NULL);
}

void release_vector_add_memory() {
	if (bufA) clReleaseMemObject(bufA);
	if (bufB) clReleaseMemObject(bufB);
	if (bufC) clReleaseMemObject(bufC);
	bufA = bufB = bufC = NULL;
}

void vector_add_multithread(const int *A, const int *B, int *C, int size, unsigned numThreads) {
	if (numThreads < 1) numThreads = 1;
	std::vector<std::thread> workers;
	workers.reserve(numThreads);
	auto worker = [&](int start, int end){
		for (int i = start; i < end; ++i) {
			C[i] = A[i] + B[i];
		}
	};
	int chunk = (size + (int)numThreads - 1) / (int)numThreads;
	int start = 0;
	for (unsigned t = 0; t < numThreads; ++t) {
		int s = start;
		int e = std::min(size, s + chunk);
		if (s >= e) break;
		workers.emplace_back(worker, s, e);
		start = e;
	}
	for (auto &th : workers) th.join();
}
// Copied deliverable version
// ...existing code...
