// OpenCL kernel for parallel vector addition
// This kernel executes on the OpenCL device with each work-item processing one vector element
__kernel void vector_add(const int size,
                        __global int* v1,
                        __global int* v2,
                        __global int* result) {
    
    // Get the unique global thread ID
    const int globalIndex = get_global_id(0);
    
    // Ensure we don't go out of bounds
    if (globalIndex < size) {
        // Perform vector addition: result[i] = v1[i] + v2[i]
        result[globalIndex] = v1[globalIndex] + v2[globalIndex];
    }
}