# Setup Instructions for Running OpenCL Programs

## Current Status
❌ **C++ Compiler Not Found** - No compiler is currently available in the system PATH.

## Required Setup

### 1. Install C++ Compiler
Choose one of these options:

#### Option A: MinGW-w64 (Recommended)
```cmd
# Download from: https://www.mingw-w64.org/downloads/
# Or use package manager like Chocolatey:
choco install mingw
```

#### Option B: Visual Studio Build Tools
```cmd
# Download Visual Studio Community or Build Tools
# Include C++ build tools and Windows SDK
```

### 2. Install OpenCL SDK
```cmd
# Intel OpenCL SDK (supports CPU and Intel GPU)
# Download from: https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html

# Or vendor-specific drivers:
# - NVIDIA: CUDA Toolkit
# - AMD: AMD APP SDK
```

### 3. Verify Installation
```cmd
# Check compiler
g++ --version

# Check OpenCL headers
dir "C:\Program Files (x86)\Intel\OpenCL SDK\include\CL\cl.h"
```

## Quick Setup Commands

### Using Chocolatey (if available)
```cmd
choco install mingw
choco install intel-opencl-sdk
```

### Manual Setup
1. Download MinGW-w64 installer
2. Install to C:\mingw64
3. Add C:\mingw64\bin to system PATH
4. Download and install Intel OpenCL SDK
5. Restart command prompt

## After Setup - Run Programs
```cmd
cd vector_opencl
g++ -std=c++11 vector_ops.cpp -lOpenCL -o vector_ops.exe
g++ -std=c++11 vector_add.cpp -lOpenCL -o vector_add.exe

vector_ops.exe
vector_add.exe 1000000
```