//! GPU-accelerated SVD reconstruction via CUDA (cudarc).
//!
//! Provides batch cross-section reconstruction on NVIDIA GPUs.
//! The SVD dot product maps perfectly to GPU execution:
//!   - Sequential basis access -> coalesced memory reads
//!   - Uniform rank-k FMA loop -> no warp divergence
//!   - Coefficients in shared memory -> one load per block
//!
//! Feature-gated: compile with `--features cuda`.

#[cfg(feature = "cuda")]
pub mod cuda {
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc;

    /// CUDA kernel source for SVD reconstruction.
    const SVD_KERNEL_SRC: &str = r#"
extern "C" __global__ void svd_reconstruct(
    const float* __restrict__ basis,
    const double* __restrict__ coeffs,
    double* __restrict__ output,
    const int* __restrict__ energy_indices,
    int n_e, int rank, int n_particles)
{
    extern __shared__ double shared_coeffs[];
    if (threadIdx.x < rank) {
        shared_coeffs[threadIdx.x] = coeffs[threadIdx.x];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    int e_idx = energy_indices[tid];
    if (e_idx < 0 || e_idx >= n_e) return;

    const float* row = &basis[e_idx * rank];
    double acc = 0.0;
    for (int j = 0; j < rank; j++) {
        acc = fma((double)row[j], shared_coeffs[j], acc);
    }
    output[tid] = exp2(acc * 3.321928094887362);
}
"#;

    /// GPU context for SVD reconstruction.
    pub struct GpuSvdContext {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        func: CudaFunction,
    }

    impl GpuSvdContext {
        /// Initialize GPU context and compile SVD kernel.
        pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let ctx = CudaContext::new(0)?;

            // Compile PTX at runtime
            let ptx = nvrtc::compile_ptx(SVD_KERNEL_SRC)?;

            let module = ctx.load_module(ptx)?;
            let func = module.load_function("svd_reconstruct")?;
            let stream = ctx.default_stream();

            println!("  GPU initialized (CUDA)");

            Ok(Self { ctx, stream, func })
        }

        /// Batch SVD reconstruction on GPU.
        ///
        /// Given a basis matrix (f32), coefficients (f64), and energy indices
        /// for N particles, returns N cross-section values (f64, linear scale).
        pub fn reconstruct_batch(
            &self,
            basis: &[f32],
            coeffs: &[f64],
            energy_indices: &[i32],
            n_e: usize,
            rank: usize,
        ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
            let n_particles = energy_indices.len();

            // Upload to GPU
            let d_basis = self.stream.memcpy_stod(basis)?;
            let d_coeffs = self.stream.memcpy_stod(coeffs)?;
            let d_indices = self.stream.memcpy_stod(energy_indices)?;
            let mut d_output: CudaSlice<f64> = self.stream.alloc_zeros(n_particles)?;

            // Launch config
            let block_size = 256_u32;
            let grid_size = (n_particles as u32 + block_size - 1) / block_size;
            let config = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: (rank * std::mem::size_of::<f64>()) as u32,
            };

            let n_e_i32 = n_e as i32;
            let rank_i32 = rank as i32;
            let n_particles_i32 = n_particles as i32;

            // Launch kernel
            unsafe {
                self.stream.launch_builder(&self.func)
                    .arg(&d_basis)
                    .arg(&d_coeffs)
                    .arg(&mut d_output)
                    .arg(&d_indices)
                    .arg(&n_e_i32)
                    .arg(&rank_i32)
                    .arg(&n_particles_i32)
                    .launch(config)?;
            }

            // Download results
            let output = self.stream.memcpy_dtov(&d_output)?;
            Ok(output)
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::GpuSvdContext;
