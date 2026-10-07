#!/usr/bin/env bash
# Compile kernels/hf_summation.cu as host C++ and execute it on the CPU.
# Checks the kernel source (not just the Rust replica) against nucrust-hf
# without a CUDA toolkit or GPU. Requires a C++17 compiler (c++ / clang++ / g++).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat "$here/cuda_host_stub.h" "$root/kernels/hf_summation.cu" "$here/harness.cpp" > "$tmp/hf_kernel_cpu.cpp"
"${CXX:-c++}" -std=c++17 -O0 -Wall -Wextra -o "$tmp/hf_kernel_cpu" "$tmp/hf_kernel_cpu.cpp"
"$tmp/hf_kernel_cpu"
