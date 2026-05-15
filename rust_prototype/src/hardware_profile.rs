//! Process-wide hardware introspection cached at first access.
//!
//! Queries `hardware-query` for RAM, CPU cache hierarchy, AVX
//! feature flags, and GPU VRAM. The cache below answers
//! "what should I size for *this* machine?" without each call site
//! re-querying.
//!
//! Plumbed into:
//! - `nuclide_cache::l1_memory` budget (replaces the 16 GiB
//!   stand-in).
//! - `gpu_transport::nuclide_buffer_cache` budget cross-check.
//! - Future: rayon pool size, SIMD-path gating, NUMA hints.

use std::sync::OnceLock;

/// Binary-prefix conversions. Used widely enough to live here so
/// every cache / budget call site shares one definition.
pub const KIB: usize = 1024;
pub const MIB: usize = 1024 * KIB;
pub const GIB: usize = 1024 * MIB;

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// System RAM, bytes. `0` if the query failed.
    pub total_ram_bytes: u64,
    /// Currently available (not "free") RAM, bytes.
    pub available_ram_bytes: u64,
    /// Physical cores; rayon pool size baseline.
    pub cpu_physical_cores: u32,
    /// Logical (SMT) cores.
    pub cpu_logical_cores: u32,
    /// L1d per core, KB. SIMD-tile sizing.
    pub cpu_l1_kb: u32,
    /// L2 per core, KB. Block-size sizing.
    pub cpu_l2_kb: u32,
    /// L3 shared, KB. Bundle / per-nuclide footprint sanity.
    pub cpu_l3_kb: u32,
    /// `avx`, `avx2`, `fma`, `sse4_2`, `neon` flags. Always lower-case.
    pub cpu_features: Vec<String>,
    /// CUDA-capable GPU memory, bytes. `None` if no CUDA GPU.
    pub gpu_vram_bytes: Option<u64>,
    /// Reported CUDA compute capability (e.g. `"8.6"` for Ampere).
    pub cuda_capability: Option<String>,
}

impl HardwareProfile {
    /// Number of nuclide bundles that fit at `fraction` of RAM
    /// given a typical 1.4 GB per-case bundle (informational only —
    /// the actual cache uses byte budget).
    pub fn estimated_bundle_capacity(&self, fraction: f64, bundle_bytes: usize) -> usize {
        if bundle_bytes == 0 || self.total_ram_bytes == 0 {
            return 0;
        }
        let budget = (self.total_ram_bytes as f64 * fraction) as u64;
        (budget / bundle_bytes as u64) as usize
    }

    /// True when AVX2+FMA is available on x86-64. Gates SIMD
    /// kernel paths.
    pub fn supports_avx2_fma(&self) -> bool {
        self.cpu_features.iter().any(|f| f == "avx2")
            && self.cpu_features.iter().any(|f| f == "fma")
    }

    /// One-line summary; engine binaries print this at startup so
    /// users can see what was detected.
    pub fn one_line_summary(&self) -> String {
        let ram_gb = self.total_ram_bytes as f64 / GIB as f64;
        let gpu = match self.gpu_vram_bytes {
            Some(b) => format!(
                ", GPU {:.1} GB{}",
                b as f64 / GIB as f64,
                self.cuda_capability
                    .as_deref()
                    .map(|s| format!(" (sm_{})", s.replace('.', "")))
                    .unwrap_or_default()
            ),
            None => String::new(),
        };
        format!(
            "RAM {:.1} GB, {}p/{}l cores, L1 {} KB / L2 {} KB / L3 {} KB{}",
            ram_gb,
            self.cpu_physical_cores,
            self.cpu_logical_cores,
            self.cpu_l1_kb,
            self.cpu_l2_kb,
            self.cpu_l3_kb,
            gpu,
        )
    }
}

/// Process-wide cached profile. First call pays the
/// `hardware-query` initialisation cost (~100-300 ms — `wmi` /
/// `sysinfo` query); subsequent calls are a single atomic load.
pub fn hardware_profile() -> &'static HardwareProfile {
    static PROFILE: OnceLock<HardwareProfile> = OnceLock::new();
    PROFILE.get_or_init(query_profile)
}

fn query_profile() -> HardwareProfile {
    use hardware_query::{CPUInfo, GPUInfo, MemoryInfo};

    let mem = MemoryInfo::query().ok();
    let cpu = CPUInfo::query().ok();
    let gpus = GPUInfo::query_all().ok().unwrap_or_default();

    let (total_ram_bytes, available_ram_bytes) = mem
        .as_ref()
        .map(|m| (m.total_mb() * MIB as u64, m.available_mb() * MIB as u64))
        .unwrap_or((0, 0));

    let cpu_physical_cores = cpu.as_ref().map(|c| c.physical_cores()).unwrap_or(0);
    let cpu_logical_cores = cpu.as_ref().map(|c| c.logical_cores()).unwrap_or(0);
    let cpu_l1_kb = cpu.as_ref().map(|c| c.l1_cache_kb()).unwrap_or(0);
    let cpu_l2_kb = cpu.as_ref().map(|c| c.l2_cache_kb()).unwrap_or(0);
    let cpu_l3_kb = cpu.as_ref().map(|c| c.l3_cache_kb()).unwrap_or(0);
    let cpu_features: Vec<String> = cpu
        .as_ref()
        .map(|c| c.features().iter().map(|f| format!("{f:?}").to_lowercase()).collect())
        .unwrap_or_default();

    let cuda_gpu = gpus.iter().find(|g| g.supports_cuda());
    let gpu_vram_bytes = cuda_gpu.map(|g| g.memory_mb() * MIB as u64);
    let cuda_capability = cuda_gpu.and_then(|g| g.cuda_capability().map(String::from));

    HardwareProfile {
        total_ram_bytes,
        available_ram_bytes,
        cpu_physical_cores,
        cpu_logical_cores,
        cpu_l1_kb,
        cpu_l2_kb,
        cpu_l3_kb,
        cpu_features,
        gpu_vram_bytes,
        cuda_capability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `hardware_profile()` is callable + cached across the process.
    /// Returns *some* profile even on minimal CI runners (zeros are
    /// acceptable; we don't fail on missing CPU / GPU data).
    #[test]
    fn profile_initialises() {
        let p1 = hardware_profile();
        let p2 = hardware_profile();
        // Same pointer = OnceLock cached.
        assert!(std::ptr::eq(p1, p2));
        // Print summary so test logs surface what was detected.
        println!("hardware: {}", p1.one_line_summary());
    }

    #[test]
    fn estimated_bundle_capacity_handles_zero() {
        let p = HardwareProfile {
            total_ram_bytes: 0,
            available_ram_bytes: 0,
            cpu_physical_cores: 0,
            cpu_logical_cores: 0,
            cpu_l1_kb: 0,
            cpu_l2_kb: 0,
            cpu_l3_kb: 0,
            cpu_features: vec![],
            gpu_vram_bytes: None,
            cuda_capability: None,
        };
        assert_eq!(p.estimated_bundle_capacity(0.75, 1_400_000_000), 0);
        let p2 = HardwareProfile {
            total_ram_bytes: (16 * GIB) as u64,
            ..p
        };
        // 0.75 × 16 GB / 1.4 GB ≈ 8 bundles.
        let cap = p2.estimated_bundle_capacity(0.75, 1_400_000_000);
        assert!(cap >= 7 && cap <= 9, "got {cap}");
    }
}
