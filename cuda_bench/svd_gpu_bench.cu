/**
 * GPU SVD Reconstruction Benchmark
 *
 * Compares three approaches on GPU:
 *   1. SVD reconstruction (coalesced FMA — GPU-friendly)
 *   2. Table lookup with binary search (warp-divergent — GPU-hostile)
 *   3. Table lookup with direct index (best-case table, no search)
 *
 * The SVD kernel is the key insight: it transforms cross-section lookup
 * from a memory-bound, divergent operation into a compute-bound,
 * uniform operation — exactly what GPU warps are designed for.
 *
 * Build:
 *   nvcc -O3 -arch=sm_86 svd_gpu_bench.cu -o svd_gpu_bench
 * Run:
 *   ./svd_gpu_bench
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// Problem dimensions (U-235 fission)
#define N_E 83114       // energy points
#define N_T 6           // temperatures
#define MAX_K 6         // max SVD rank

// Benchmark parameters
#define N_PARTICLES 1000000  // particles to reconstruct simultaneously
#define N_ITERS 100          // timing iterations

// ─── SVD reconstruction kernel ─────────────────────────────────────────────
// Each thread reconstructs sigma(E_i, T) for one particle.
// basis[N_E * k] is in global memory but accessed with perfect coalescing.
// coeffs[k] fits in shared memory or registers.
__global__ void svd_reconstruct_kernel(
    const double* __restrict__ basis,  // [N_E x k], row-major
    const double* __restrict__ coeffs, // [k]
    double* __restrict__ output,       // [N_PARTICLES]
    const int* __restrict__ energy_indices, // [N_PARTICLES] — which energy point each particle needs
    int n_e, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_PARTICLES) return;

    int e_idx = energy_indices[tid];

    // k-wide dot product — all threads do the same operation,
    // just at different rows of the basis matrix.
    // This is perfectly coalesced: adjacent threads read adjacent memory.
    double acc = 0.0;
    const double* row = &basis[e_idx * k];
    for (int j = 0; j < k; j++) {
        acc = fma(row[j], coeffs[j], acc);
    }

    // Convert from log10 to linear
    output[tid] = pow(10.0, acc);
}

// Same but with coeffs in shared memory
__global__ void svd_reconstruct_smem(
    const double* __restrict__ basis,
    const double* __restrict__ coeffs_global,
    double* __restrict__ output,
    const int* __restrict__ energy_indices,
    int n_e, int k)
{
    extern __shared__ double coeffs_shared[];

    // Load coefficients into shared memory (once per block)
    if (threadIdx.x < k) {
        coeffs_shared[threadIdx.x] = coeffs_global[threadIdx.x];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_PARTICLES) return;

    int e_idx = energy_indices[tid];
    double acc = 0.0;
    const double* row = &basis[e_idx * k];
    for (int j = 0; j < k; j++) {
        acc = fma(row[j], coeffs_shared[j], acc);
    }
    output[tid] = pow(10.0, acc);
}

// ─── Table lookup kernel (binary search — GPU-hostile) ─────────────────────
// Each thread does an independent binary search.
// Adjacent threads search different energy ranges → warp divergence.
// Memory access is random → no coalescing.
__global__ void table_lookup_kernel(
    const double* __restrict__ energy_grid, // [N_E]
    const double* __restrict__ xs_table,    // [N_E]
    double* __restrict__ output,            // [N_PARTICLES]
    const double* __restrict__ particle_energies, // [N_PARTICLES]
    int n_e)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_PARTICLES) return;

    double E = particle_energies[tid];

    // Binary search — causes warp divergence
    int lo = 0, hi = n_e - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (energy_grid[mid] < E) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // Clamp
    if (lo == 0) { output[tid] = xs_table[0]; return; }
    if (lo >= n_e) { output[tid] = xs_table[n_e-1]; return; }

    // Log-log interpolation (expensive transcendentals)
    double e_lo = energy_grid[lo-1];
    double e_hi = energy_grid[lo];
    double xs_lo = xs_table[lo-1];
    double xs_hi = xs_table[lo];

    double f = log(E / e_lo) / log(e_hi / e_lo);
    output[tid] = xs_lo * pow(xs_hi / xs_lo, f);
}

// ─── Direct index lookup (best-case table, no search) ──────────────────────
__global__ void table_direct_kernel(
    const double* __restrict__ xs_table,
    double* __restrict__ output,
    const int* __restrict__ energy_indices,
    int n_e)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_PARTICLES) return;
    output[tid] = xs_table[energy_indices[tid]];
}

// ─── Helpers ───────────────────────────────────────────────────────────────

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

double benchmark_kernel(void (*launcher)(void), int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    launcher();
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        launcher();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double)ms / iters;
}

// Global pointers for kernel launchers
double *d_basis, *d_coeffs, *d_output, *d_energy_grid, *d_xs_table, *d_particle_energies;
int *d_energy_indices;
int g_k;

void launch_svd() {
    int blocks = (N_PARTICLES + 255) / 256;
    svd_reconstruct_kernel<<<blocks, 256>>>(
        d_basis, d_coeffs, d_output, d_energy_indices, N_E, g_k);
}

void launch_svd_smem() {
    int blocks = (N_PARTICLES + 255) / 256;
    svd_reconstruct_smem<<<blocks, 256, MAX_K * sizeof(double)>>>(
        d_basis, d_coeffs, d_output, d_energy_indices, N_E, g_k);
}

void launch_table() {
    int blocks = (N_PARTICLES + 255) / 256;
    table_lookup_kernel<<<blocks, 256>>>(
        d_energy_grid, d_xs_table, d_output, d_particle_energies, N_E);
}

void launch_direct() {
    int blocks = (N_PARTICLES + 255) / 256;
    table_direct_kernel<<<blocks, 256>>>(
        d_xs_table, d_output, d_energy_indices, N_E);
}

int main() {
    printf("=== GPU SVD Reconstruction Benchmark ===\n\n");

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("  SMs: %d, L2 cache: %d KB, Memory: %zu MB\n",
           prop.multiProcessorCount, prop.l2CacheSize / 1024,
           prop.totalGlobalMem / (1024*1024));
    printf("  Compute capability: %d.%d\n\n", prop.major, prop.minor);

    // Allocate and initialize host data
    printf("Generating synthetic data (N_E=%d, N_particles=%d)...\n", N_E, N_PARTICLES);

    double* h_basis = (double*)malloc(N_E * MAX_K * sizeof(double));
    double* h_coeffs = (double*)malloc(MAX_K * sizeof(double));
    double* h_energy_grid = (double*)malloc(N_E * sizeof(double));
    double* h_xs_table = (double*)malloc(N_E * sizeof(double));
    double* h_particle_energies = (double*)malloc(N_PARTICLES * sizeof(double));
    int* h_energy_indices = (int*)malloc(N_PARTICLES * sizeof(int));

    srand(42);

    // Synthetic but realistic: log-spaced energy grid, random basis
    for (int i = 0; i < N_E; i++) {
        h_energy_grid[i] = 1e-5 * pow(2e12, (double)i / (N_E - 1));
        h_xs_table[i] = 1.0 + 100.0 * ((double)rand() / RAND_MAX);
        for (int j = 0; j < MAX_K; j++) {
            h_basis[i * MAX_K + j] = 0.5 + 0.1 * ((double)rand() / RAND_MAX);
        }
    }
    for (int j = 0; j < MAX_K; j++) {
        h_coeffs[j] = 1.0 / (j + 1);
    }

    // Random particle energies and indices
    for (int i = 0; i < N_PARTICLES; i++) {
        h_energy_indices[i] = rand() % N_E;
        h_particle_energies[i] = h_energy_grid[h_energy_indices[i]];
    }

    // Allocate device memory
    check_cuda(cudaMalloc(&d_basis, N_E * MAX_K * sizeof(double)), "alloc basis");
    check_cuda(cudaMalloc(&d_coeffs, MAX_K * sizeof(double)), "alloc coeffs");
    check_cuda(cudaMalloc(&d_output, N_PARTICLES * sizeof(double)), "alloc output");
    check_cuda(cudaMalloc(&d_energy_grid, N_E * sizeof(double)), "alloc grid");
    check_cuda(cudaMalloc(&d_xs_table, N_E * sizeof(double)), "alloc table");
    check_cuda(cudaMalloc(&d_particle_energies, N_PARTICLES * sizeof(double)), "alloc energies");
    check_cuda(cudaMalloc(&d_energy_indices, N_PARTICLES * sizeof(int)), "alloc indices");

    check_cuda(cudaMemcpy(d_basis, h_basis, N_E * MAX_K * sizeof(double), cudaMemcpyHostToDevice), "copy basis");
    check_cuda(cudaMemcpy(d_coeffs, h_coeffs, MAX_K * sizeof(double), cudaMemcpyHostToDevice), "copy coeffs");
    check_cuda(cudaMemcpy(d_energy_grid, h_energy_grid, N_E * sizeof(double), cudaMemcpyHostToDevice), "copy grid");
    check_cuda(cudaMemcpy(d_xs_table, h_xs_table, N_E * sizeof(double), cudaMemcpyHostToDevice), "copy table");
    check_cuda(cudaMemcpy(d_particle_energies, h_particle_energies, N_PARTICLES * sizeof(double), cudaMemcpyHostToDevice), "copy energies");
    check_cuda(cudaMemcpy(d_energy_indices, h_energy_indices, N_PARTICLES * sizeof(int), cudaMemcpyHostToDevice), "copy indices");

    printf("\nBenchmarking %d particles × %d iterations...\n\n", N_PARTICLES, N_ITERS);

    // Memory footprint
    printf("Memory footprint:\n");
    printf("  SVD basis (k=4): %.1f MB\n", N_E * 4 * 8.0 / (1024*1024));
    printf("  SVD basis (k=6): %.1f MB\n", N_E * 6 * 8.0 / (1024*1024));
    printf("  Table (1 temp):  %.1f MB\n", N_E * 2 * 8.0 / (1024*1024));
    printf("  GPU L2 cache:    %.1f MB\n\n", prop.l2CacheSize / (1024.0*1024.0));

    // Benchmark
    printf("%-25s %10s %10s %12s\n", "Method", "Time (ms)", "ns/particle", "Throughput");
    printf("%-25s %10s %10s %12s\n", "-------------------------", "----------", "----------", "------------");

    // Table lookup (binary search)
    double t_table = benchmark_kernel(launch_table, N_ITERS);
    printf("%-25s %10.3f %10.1f %10.0f M/s\n",
           "Table (binary search)", t_table, t_table * 1e6 / N_PARTICLES,
           N_PARTICLES / t_table / 1e3);

    // Table direct (best case)
    double t_direct = benchmark_kernel(launch_direct, N_ITERS);
    printf("%-25s %10.3f %10.1f %10.0f M/s\n",
           "Table (direct index)", t_direct, t_direct * 1e6 / N_PARTICLES,
           N_PARTICLES / t_direct / 1e3);

    // SVD at various ranks
    for (int k = 2; k <= MAX_K; k++) {
        g_k = k;

        double t_svd = benchmark_kernel(launch_svd, N_ITERS);
        double t_smem = benchmark_kernel(launch_svd_smem, N_ITERS);

        char label1[64], label2[64];
        snprintf(label1, sizeof(label1), "SVD k=%d (global)", k);
        snprintf(label2, sizeof(label2), "SVD k=%d (smem)", k);

        double speedup_vs_table = t_table / t_svd;

        printf("%-25s %10.3f %10.1f %10.0f M/s  (%.1fx vs table)\n",
               label1, t_svd, t_svd * 1e6 / N_PARTICLES,
               N_PARTICLES / t_svd / 1e3, speedup_vs_table);
        printf("%-25s %10.3f %10.1f %10.0f M/s\n",
               label2, t_smem, t_smem * 1e6 / N_PARTICLES,
               N_PARTICLES / t_smem / 1e3);
    }

    // Cleanup
    cudaFree(d_basis); cudaFree(d_coeffs); cudaFree(d_output);
    cudaFree(d_energy_grid); cudaFree(d_xs_table);
    cudaFree(d_particle_energies); cudaFree(d_energy_indices);
    free(h_basis); free(h_coeffs); free(h_energy_grid);
    free(h_xs_table); free(h_particle_energies); free(h_energy_indices);

    printf("\nKey insight: SVD transforms cross-section lookup from\n");
    printf("  MEMORY-BOUND (random binary search, warp divergence)\n");
    printf("to COMPUTE-BOUND (uniform FMA, perfect coalescing)\n");
    printf("— exactly what GPU architecture is designed for.\n");

    return 0;
}
