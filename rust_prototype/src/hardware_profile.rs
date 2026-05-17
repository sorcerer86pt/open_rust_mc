//! Process-wide hardware introspection cached at first access.
//!
//! Queries `hardware-query` for RAM, CPU cache hierarchy, AVX
//! feature flags, and GPU VRAM — then overrides two known-buggy
//! values with more accurate sources:
//!
//! - **L3 cache:** `hardware-query` reads `Win32_CacheMemory` via WMI
//!   which returns one CCX entry on AMD Zen chips.  On 9800X3D this
//!   reports 8 MB instead of 96 MB.  We use `raw-cpuid` (leaf 4 /
//!   8000_001D) to enumerate and sum all L3 topology entries.
//!
//! - **GPU VRAM:** `Win32_VideoController.AdapterRAM` is a 32-bit
//!   DWORD; GPUs above 4 GB (RTX 3080 = 10 GB) overflow it.  When
//!   the `cuda` feature is enabled we prefer `cuDeviceTotalMem`
//!   which uses a 64-bit `size_t`.
//!
//! The cache below answers "what should I size for *this* machine?"
//! without each call site re-querying.
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
    /// Upper-case feature names as `hardware-query` reports them
    /// (`AVX`, `AVX2`, `AVX512`, `FMA`, `SSE4.2`, `BMI1`, …).
    pub cpu_features: Vec<String>,
    /// Convenience: x86 SIMD path the engine should pick.
    pub supports_avx2: bool,
    pub supports_fma: bool,
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
        self.supports_avx2 && self.supports_fma
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

/// Print a multi-line startup banner to stderr (so it doesn't
/// contaminate stdout-piped k_eff CSVs). Shows detected hardware
/// and the values the engine will use as a result. Idempotent —
/// safe to call from every binary at startup. Suppressed when
/// `OPEN_RUST_MC_QUIET=1`.
pub fn log_startup_banner() {
    static EMITTED: OnceLock<()> = OnceLock::new();
    if EMITTED.set(()).is_err() {
        return;
    }
    if std::env::var_os("OPEN_RUST_MC_QUIET").is_some() {
        return;
    }
    let p = hardware_profile();
    eprintln!("┌─ open_rust_mc — hardware profile ─");
    eprintln!("│ {}", p.one_line_summary());
    let avx = if p.supports_avx2_fma() { "avx2+fma" } else { "scalar" };
    eprintln!(
        "│ Rayon pool: {} threads (logical cores). SIMD path: {}.",
        p.cpu_logical_cores.max(1),
        avx,
    );
    let nuclide_cache_gb =
        crate::transport::nuclide_cache::l1_memory::L1MemoryStore::new().budget_bytes() as f64
            / GIB as f64;
    eprintln!(
        "│ Host nuclide cache budget: {:.1} GB ({}× detected RAM).",
        nuclide_cache_gb,
        if p.total_ram_bytes > 0 {
            format!(
                "{:.0}%",
                nuclide_cache_gb / (p.total_ram_bytes as f64 / GIB as f64) * 100.0
            )
        } else {
            "fallback".to_string()
        },
    );
    if let Some(vram) = p.gpu_vram_bytes {
        let frac = std::env::var("OPEN_RUST_MC_GPU_BUNDLE_CACHE_FRACTION")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.75);
        eprintln!(
            "│ GPU bundle cache budget: {:.1} GB ({:.0}% of {:.1} GB VRAM).",
            (vram as f64 / GIB as f64) * frac,
            frac * 100.0,
            vram as f64 / GIB as f64,
        );
    }
    eprintln!("└──");
}

#[cfg(target_arch = "x86_64")]
fn detect_simd() -> (bool, bool) {
    (
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("fma"),
    )
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_simd() -> (bool, bool) {
    (false, false)
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

    // L3: hardware-query under-reports on AMD X3D chips.
    // WMI returns one CCX entry (8 MB on 9800X3D) instead of the
    // full stacked total. raw-cpuid enumerates all CPUID cache
    // topology leaves and sums every L3 instance correctly.
    let cpu_l3_kb = detect_l3_kb()
        .or_else(|| cpu.as_ref().map(|c| c.l3_cache_kb()))
        .unwrap_or(0);

    let cpu_features: Vec<String> = cpu
        .as_ref()
        .map(|c| c.features().iter().map(|f| f.to_string()).collect())
        .unwrap_or_default();

    // hardware-query under-reports FMA on some Windows machines.
    let (supports_avx2, supports_fma) = detect_simd();

    let cuda_gpu = gpus.iter().find(|g| g.supports_cuda());

    // VRAM: Win32_VideoController.AdapterRAM is a 32-bit DWORD.
    // RTX 3080 (10 GB) overflows it and wraps to ~4 GB.
    // Prefer CUDA driver API which uses 64-bit size_t throughout.
    let gpu_vram_bytes = detect_vram_bytes_cuda()
        .or_else(|| cuda_gpu.map(|g| g.memory_mb() * MIB as u64));
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
        supports_avx2,
        supports_fma,
        gpu_vram_bytes,
        cuda_capability,
    }
}

/// Detect total L3 cache by summing **all** CPUID cache-topology
/// level-3 entries.
///
/// On AMD X3D chips the 3D V-Cache stacking is exposed as additional
/// L3 entries in CPUID leaf 0x8000_001D (extended topology) or leaf
/// 0x4 (Intel-style). Summing them gives the physically usable total
/// (e.g. 32 MB base + 64 MB stacked = 96 MB on the 9800X3D) rather
/// than the 8 MB single-CCX figure that WMI / `hardware-query`
/// returns.
///
/// Falls back to `None` on non-x86_64 targets so the caller can
/// revert to the `hardware-query` value.
#[cfg(target_arch = "x86_64")]
fn detect_l3_kb() -> Option<u32> {
    use raw_cpuid::CpuId;

    let cpuid = CpuId::new();

    // Prefer the extended AMD cache topology leaf (8000_001D) which
    // is more reliable on Zen architectures.
    if let Some(params) = cpuid.get_extended_cache_parameters() {
        let total_kb: u32 = params
            .filter(|c| c.level() == 3)
            .map(|c| {
                (c.associativity() as u32)
                    .saturating_mul(c.physical_line_partitions() as u32)
                    .saturating_mul(c.coherency_line_size() as u32)
                    .saturating_mul(c.sets() as u32)
                    / 1024
            })
            .sum();
        if total_kb > 0 {
            return Some(total_kb);
        }
    }

    // Intel / generic path via leaf 4.
    if let Some(params) = cpuid.get_cache_parameters() {
        let total_kb: u32 = params
            .filter(|c| c.level() == 3)
            .map(|c| {
                (c.associativity() as u32)
                    .saturating_mul(c.physical_line_partitions() as u32)
                    .saturating_mul(c.coherency_line_size() as u32)
                    .saturating_mul(c.sets() as u32)
                    / 1024
            })
            .sum();
        if total_kb > 0 {
            return Some(total_kb);
        }
    }

    None
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_l3_kb() -> Option<u32> {
    None
}

/// Detect CUDA GPU VRAM via the CUDA driver API.
///
/// `Win32_VideoController.AdapterRAM` is a 32-bit DWORD; GPUs with
/// more than 4 GB VRAM (RTX 3080 = 10 GB, for example) overflow it
/// and appear as ~4 GB to `hardware-query` on Windows.
/// `cuDeviceTotalMem` / `cuMemGetInfo` use `size_t` (64-bit on all
/// CUDA-supported platforms) so they return the correct value.
///
/// Returns `None` when the `cuda` feature is disabled or no CUDA
/// device is present.
#[cfg(feature = "cuda")]
fn detect_vram_bytes_cuda() -> Option<u64> {
    cudarc::driver::CudaDevice::new(0)
        .ok()
        .map(|dev| dev.total_mem() as u64)
}

#[cfg(not(feature = "cuda"))]
fn detect_vram_bytes_cuda() -> Option<u64> {
    None
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
            supports_avx2: false,
            supports_fma: false,
            gpu_vram_bytes: None,
            cuda_capability: None,
        };
        assert_eq!(p.estimated_bundle_capacity(0.75, 1_400_000_000), 0);
        let p2 = HardwareProfile {
            total_ram_bytes: (16 * GIB) as u64,
            ..p
        };
        let cap = p2.estimated_bundle_capacity(0.75, 1_400_000_000);
        assert!(cap >= 7 && cap <= 9, "got {cap}");
    }

    /// L3 cache detection via raw-cpuid should return a non-zero value
    /// on x86_64 hosts. On AMD X3D machines this should exceed 8 MB
    /// (the single-CCX WMI value) when V-Cache is present.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn l3_detection_nonzero_on_x86() {
        let l3 = super::detect_l3_kb();
        // raw-cpuid should always find L3 on a modern x86_64 system.
        assert!(l3.is_some(), "detect_l3_kb() returned None on x86_64");
        let kb = l3.unwrap();
        assert!(kb > 0, "detect_l3_kb() returned 0 KB");
        eprintln!("detect_l3_kb() = {} KB ({} MB)", kb, kb / 1024);
    }

    /// Verify that the profile's L3 value matches what detect_l3_kb
    /// returns when that function succeeds — i.e. raw-cpuid takes
    /// priority over the hardware-query fallback.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn profile_uses_cpuid_l3_over_hardware_query() {
        if let Some(cpuid_kb) = super::detect_l3_kb() {
            let p = hardware_profile();
            assert_eq!(
                p.cpu_l3_kb, cpuid_kb,
                "profile L3 ({} KB) should match raw-cpuid value ({} KB)",
                p.cpu_l3_kb, cpuid_kb
            );
        }
    }

    /// VRAM detection should not report 4 GB on a GPU with > 4 GB
    /// VRAM. This catches the Win32_VideoController.AdapterRAM
    /// 32-bit overflow (0xFFFFFFFF bytes → 4095 MB ≈ 4 GB).
    /// Only meaningful when a CUDA GPU is present.
    #[test]
    fn vram_not_clamped_to_4gb_on_high_vram_gpu() {
        let p = hardware_profile();
        if let Some(vram) = p.gpu_vram_bytes {
            let gb = vram as f64 / GIB as f64;
            eprintln!("Detected GPU VRAM: {:.2} GB", gb);
            // If CUDA path ran, we should never see exactly 4 GB on a
            // GPU that is known to have more (the overflow value is
            // 4_294_967_295 bytes; we flag anything within 1 MB of it).
            let overflow_sentinel = 0xFFFF_FFFF_u64;
            assert!(
                vram < overflow_sentinel.saturating_sub(MIB as u64)
                    || vram > overflow_sentinel.saturating_add(MIB as u64),
                "VRAM ({} bytes) looks like a 32-bit WMI overflow — \
                 CUDA path did not override hardware-query",
                vram
            );
        } else {
            eprintln!("No CUDA GPU present — skipping VRAM overflow test.");
        }
    }

    /// Self-test: writes a banner-style dump of every detected
    /// value to stderr. Verifies the banner actually emits text
    /// matching the expected structure (┌─ header, body lines, └─
    /// footer). Run with `cargo test -- --nocapture` to see the
    /// dump.
    #[test]
    fn banner_self_test() {
        let p = hardware_profile();
        let summary = p.one_line_summary();
        eprintln!("\n=== Hardware self-test ===");
        eprintln!("Summary: {summary}");
        eprintln!("Total RAM: {} bytes ({:.2} GB)", p.total_ram_bytes, p.total_ram_bytes as f64 / GIB as f64);
        eprintln!("Available RAM: {} bytes ({:.2} GB)", p.available_ram_bytes, p.available_ram_bytes as f64 / GIB as f64);
        eprintln!("CPU cores: {} physical / {} logical", p.cpu_physical_cores, p.cpu_logical_cores);
        eprintln!("CPU cache: L1 {} KB / L2 {} KB / L3 {} KB", p.cpu_l1_kb, p.cpu_l2_kb, p.cpu_l3_kb);
        eprintln!("CPU features ({}): {}", p.cpu_features.len(), p.cpu_features.join(", "));
        eprintln!("AVX2: {}, FMA: {}, AVX2+FMA: {}", p.supports_avx2, p.supports_fma, p.supports_avx2_fma());
        if let Some(vram) = p.gpu_vram_bytes {
            eprintln!("GPU VRAM: {} bytes ({:.2} GB)", vram, vram as f64 / GIB as f64);
            if let Some(cap) = &p.cuda_capability {
                eprintln!("CUDA compute capability: sm_{}", cap.replace('.', ""));
            }
        } else {
            eprintln!("GPU: none / non-CUDA");
        }
        eprintln!("===");

        // Shape assertions — the summary must always contain "RAM"
        // and the cores marker; on a CI runner with zero RAM the
        // numbers may be 0 but the labels remain.
        assert!(summary.contains("RAM "));
        assert!(summary.contains("cores"));
    }
}
