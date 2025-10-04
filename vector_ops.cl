// OpenCL kernel function that computes the square of each vector element
// This function executes on the OpenCL device (GPU/CPU) in parallel across multiple processing elements
__kernel void square_magnitude(const int size,
                      __global int* v) {
    
    // Thread identifiers
    const int globalIndex = get_global_id(0);   
 
    //uncomment to see the index each PE works on
    //printf("Kernel process index :(%d)\n ", globalIndex);

    v[globalIndex] = v[globalIndex] * v[globalIndex];
}
