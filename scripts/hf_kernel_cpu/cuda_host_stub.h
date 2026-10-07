/* Minimal host definitions that let kernels/hf_summation.cu compile as plain C++.
 * The kernel body is then executed on the CPU one "thread" at a time. */
#include <math.h>
#define __global__
#define __device__
#define __restrict__
struct HostDim3 { int x, y, z; };
static HostDim3 blockIdx, blockDim, threadIdx;
