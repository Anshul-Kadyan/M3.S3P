# SIT315 Seminar 8 - OpenCL Programming

## Overview
This directory contains the completed OpenCL programming exercises for SIT315 Seminar 8, including:
1. Enhanced vector operations program with comprehensive comments
2. Vector addition implementation comparing OpenCL vs multi-threaded performance

## Files Description

### Original Enhanced Files
- **vector_ops.cpp** - Original program with added comments explaining OpenCL concepts
- **vector_ops.cl** - OpenCL kernel for computing square magnitude with comments

### New Vector Addition Implementation
- **vector_add.cpp** - Performance comparison program (OpenCL vs multi-threaded)
- **vector_add.cl** - OpenCL kernel for parallel vector addition
- **Makefile** - Build configuration for Unix/Linux systems
- **compile.bat** - Windows compilation script

## Prerequisites
- OpenCL SDK installed
- C++ compiler (g++ recommended)
- OpenCL-compatible device (GPU or CPU)

## Compilation

### Windows
```cmd
compile.bat
```

### Linux/Unix
```bash
make all
```

### Manual Compilation
```bash
g++ -std=c++11 vector_ops.cpp -lOpenCL -o vector_ops
g++ -std=c++11 vector_add.cpp -lOpenCL -o vector_add
```

## Execution

### Vector Operations (Original Program)
```bash
./vector_ops [vector_size]
# Example: ./vector_ops 1000
```

### Vector Addition Comparison
```bash
./vector_add [vector_size]
# Example: ./vector_add 1000000
```

## Expected Output

### Vector Operations
- Displays original vector
- Shows squared magnitude results
- Demonstrates OpenCL kernel execution

### Vector Addition
- Performance timing for both implementations
- Speedup calculation (OpenCL vs multi-threaded)
- Result verification
- Sample output for small vectors

## Performance Notes
- Larger vector sizes (>100,000) show better OpenCL performance
- GPU devices typically outperform CPU for parallel operations
- Memory transfer overhead affects small dataset performance
- Multi-threaded performance scales with CPU core count

## Learning Objectives Achieved
1. ✅ Understanding OpenCL terminology and concepts
2. ✅ Adding comprehensive code comments
3. ✅ Implementing parallel vector addition
4. ✅ Performance comparison between OpenCL and multi-threading
5. ✅ Memory management in OpenCL applications