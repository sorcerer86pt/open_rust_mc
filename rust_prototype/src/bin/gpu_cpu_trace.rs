//! GPU vs CPU transport trace — step-by-step comparison.
//!
//! Traces N particles for M steps on both GPU and CPU with identical initial
//! conditions and RNG seeds, writes per-step data to CSV for diff analysis.
//!
//! Usage:
//!   cargo run --release --features cuda --bin gpu_cpu_trace -- <data_dir> \
//!     --particles 50 --steps 200

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() { cuda_main::run(); }

#[cfg(feature = "cuda")]
mod cuda_main {
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Arc;
    use clap::Parser;

    use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
    use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
    use open_rust_mc::geometry::{Aabb, Vec3};
    use open_rust_mc::gpu_transport::GpuTransportContext;
    use open_rust_mc::hdf5_reader;
    use open_rust_mc::physics::collision::MicroXs;
    use open_rust_mc::thermal::ThermalScatteringData;
    use open_rust_mc::transport::material::Material;
    use open_rust_mc::transport::rng::Rng;
    use open_rust_mc::transport::simulate::XsProvider;
    use open_rust_mc::transport::xs_provider;

    #[derive(Parser)]
    struct Args {
        data_dir: PathBuf,
        #[arg(short, long, default_value_t = 6)]
        rank: usize,
        #[arg(short, long, default_value_t = 50)]
        particles: usize,
        #[arg(short, long, default_value_t = 200)]
        steps: usize,
    }

    const NUCLIDE_SPECS: &[(&str, f64, f64, usize)] = &[
        ("U235.h5", 233.025, 2.43, 3),
        ("U238.h5", 236.006, 2.49, 3),
        ("O16.h5",  15.858,  0.0,  3),
        ("H1.h5",    0.999,  0.0,  2),
        ("Zr90.h5", 89.132,  0.0,  2),
        ("Zr91.h5", 90.130,  0.0,  2),
        ("Zr92.h5", 91.126,  0.0,  2),
        ("Zr94.h5", 93.120,  0.0,  2),
        ("O16.h5",  15.858,  0.0,  2),
    ];

    fn setup_materials() -> Vec<Material> {
        let mut fuel = Material::new("UO2", 900.0);
        fuel.add_nuclide(0.000719, 0);
        fuel.add_nuclide(0.022482, 1);
        fuel.add_nuclide(0.046402, 2);
        let mut clad = Material::new("Zircaloy", 600.0);
        clad.add_nuclide(0.022932, 4);
        clad.add_nuclide(0.004996, 5);
        clad.add_nuclide(0.007636, 6);
        clad.add_nuclide(0.007740, 7);
        let mut water = Material::new("H2O", 600.0);
        water.add_nuclide(0.049486, 3);
        water.add_nuclide(0.024743, 8);
        vec![fuel, clad, water]
    }

    fn setup_geometry() -> (Vec<Surface>, Vec<Cell>) {
        let fuel_or = 0.4096;
        let clad_ir = 0.4180;
        let clad_or = 0.4750;
        let pitch = 1.2600;
        let half = pitch / 2.0;
        let z_half = half;
        let surfaces = vec![
            Surface::CylinderZ { center_x: 0.0, center_y: 0.0, radius: fuel_or, bc: BoundaryCondition::Transmission },
            Surface::CylinderZ { center_x: 0.0, center_y: 0.0, radius: clad_ir, bc: BoundaryCondition::Transmission },
            Surface::CylinderZ { center_x: 0.0, center_y: 0.0, radius: clad_or, bc: BoundaryCondition::Transmission },
            Surface::PlaneX { x0: -half, bc: BoundaryCondition::Reflective },
            Surface::PlaneX { x0:  half, bc: BoundaryCondition::Reflective },
            Surface::PlaneY { y0: -half, bc: BoundaryCondition::Reflective },
            Surface::PlaneY { y0:  half, bc: BoundaryCondition::Reflective },
            Surface::PlaneZ { z0: -z_half, bc: BoundaryCondition::Reflective },
            Surface::PlaneZ { z0:  z_half, bc: BoundaryCondition::Reflective },
        ];
        let box_aabb = Aabb::new(Vec3::new(-half, -half, -z_half), Vec3::new(half, half, z_half));
        let cells = vec![
            Cell::new(CellId(0), cell::intersect_all(vec![
                cell::inside(0), cell::outside(7), cell::inside(8),
            ]), CellFill::Material(0))
            .with_aabb(Aabb::new(Vec3::new(-fuel_or, -fuel_or, -z_half), Vec3::new(fuel_or, fuel_or, z_half)))
            .with_temperature(900.0),
            Cell::new(CellId(1), cell::intersect_all(vec![
                cell::outside(0), cell::inside(1), cell::outside(7), cell::inside(8),
            ]), CellFill::Void),
            Cell::new(CellId(2), cell::intersect_all(vec![
                cell::outside(1), cell::inside(2), cell::outside(7), cell::inside(8),
            ]), CellFill::Material(1))
            .with_aabb(Aabb::new(Vec3::new(-clad_or, -clad_or, -z_half), Vec3::new(clad_or, clad_or, z_half)))
            .with_temperature(600.0),
            Cell::new(CellId(3), cell::intersect_all(vec![
                cell::outside(2), cell::outside(3), cell::inside(4),
                cell::outside(5), cell::inside(6),
                cell::outside(7), cell::inside(8),
            ]), CellFill::Material(2))
            .with_aabb(box_aabb)
            .with_temperature(600.0),
        ];
        (surfaces, cells)
    }

    fn initial_source(n: usize) -> Vec<(f64, f64, f64, f64)> {
        let fuel_or = 0.4096_f64;
        let half = 0.63_f64;
        let mut rng = Rng::new(42, 0);
        let mut sites = Vec::with_capacity(n);
        while sites.len() < n {
            let x = -fuel_or + rng.uniform() * 2.0 * fuel_or;
            let y = -fuel_or + rng.uniform() * 2.0 * fuel_or;
            let z = -half + rng.uniform() * 2.0 * half;
            if x * x + y * y < fuel_or * fuel_or {
                sites.push((x, y, z, 1.0e6));
            }
        }
        sites
    }

    fn load_thermal(data_dir: &PathBuf) -> Vec<Option<Arc<ThermalScatteringData>>> {
        let h2o_path = data_dir.join("c_H_in_H2O.h5");
        let h2o_thermal: Option<Arc<ThermalScatteringData>> = if h2o_path.exists() {
            hdf5_reader::load_thermal_scattering(&h2o_path).ok().map(Arc::new)
        } else { None };
        let mut thermal: Vec<Option<Arc<ThermalScatteringData>>> = vec![None; NUCLIDE_SPECS.len()];
        if let Some(ref tsl) = h2o_thermal { thermal[3] = Some(Arc::clone(tsl)); }
        thermal
    }

    /// CPU trace of one particle, matching GPU debug_transport_trace exactly.
    /// Returns Vec of rows, each row is TRACE_COLS f64 values.
    fn cpu_trace_particle(
        pid: usize,
        source: &(f64, f64, f64, f64),
        max_steps: usize,
        surfaces: &[Surface],
        cells: &[Cell],
        materials: &[Material],
        xs_provider: &xs_provider::TableXsProvider,
    ) -> Vec<[f64; 17]> {
        use open_rust_mc::geometry;
        use open_rust_mc::transport::particle::Particle;
        use open_rust_mc::physics::collision::InelasticData;

        // Match GPU RNG seeding exactly: pcg_init(seed=42+tid*100000, stream=tid)
        let mut rng = Rng::new(42 + pid as u64 * 100000, pid as u64);
        let mu = 2.0 * rng.uniform() - 1.0;
        let phi = 2.0 * std::f64::consts::PI * rng.uniform();
        let st = (1.0 - mu * mu).sqrt();
        let dir = Vec3::new(st * phi.cos(), st * phi.sin(), mu);

        let cell_idx = geometry::ray::find_cell(
            Vec3::new(source.0, source.1, source.2), surfaces, cells,
        ).unwrap_or(0);
        let mut particle = Particle::new(
            Vec3::new(source.0, source.1, source.2), dir, source.3, cell_idx,
        );

        let mut rows = Vec::new();
        let mut void_crossings = 0_u32;

        for _step in 0..max_steps {
            if !particle.is_alive() { break; }
            let mut row = [0.0_f64; 17];

            let cell = &cells[particle.cell_idx];
            let mat_idx = match cell.fill {
                CellFill::Material(m) => m as usize,
                CellFill::Void => {
                    void_crossings += 1;
                    row[0] = particle.energy;
                    row[1] = particle.pos.x; row[2] = particle.pos.y; row[3] = particle.pos.z;
                    row[4] = particle.cell_idx as f64;
                    row[5] = -1.0;
                    row[9] = 9.0; // void

                    if void_crossings > 100 { row[9] = 8.0; particle.kill(); rows.push(row); break; }

                    let trace = geometry::ray::trace_step(
                        particle.pos, particle.dir, particle.cell_idx, surfaces, cells,
                    );
                    match trace {
                        Some(hit) => {
                            row[8] = hit.distance;
                            let bc = surfaces[hit.surface_idx].boundary_condition();
                            match bc {
                                BoundaryCondition::Vacuum => {
                                    row[9] = 8.0; particle.kill();
                                }
                                BoundaryCondition::Reflective => {
                                    particle.advance(hit.distance);
                                    let n = surfaces[hit.surface_idx].normal_at(particle.pos);
                                    let d = particle.dir;
                                    particle.dir = d - n * (2.0 * d.dot(n));
                                }
                                BoundaryCondition::Transmission => {
                                    let nudge = (hit.distance * 1e-8).max(1e-8);
                                    particle.advance(hit.distance + nudge);
                                    if let Some(next) = hit.next_cell_idx {
                                        particle.cell_idx = next;
                                    } else {
                                        row[9] = 8.0; particle.kill();
                                    }
                                }
                            }
                        }
                        None => { row[9] = 8.0; particle.kill(); }
                    }
                    row[15] = particle.energy;
                    rows.push(row);
                    continue;
                }
                _ => { row[9] = 8.0; particle.kill(); rows.push(row); break; }
            };

            void_crossings = 0;
            let material = &materials[mat_idx];

            row[0] = particle.energy;
            row[1] = particle.pos.x; row[2] = particle.pos.y; row[3] = particle.pos.z;
            row[4] = particle.cell_idx as f64;
            row[5] = mat_idx as f64;

            // XS lookup
            let urr_xi = rng.uniform();
            let n_nuclides = material.nuclides.len();
            let mut micro_xs_arr = [MicroXs::default(); 8];
            let mut micro_totals = [0.0_f64; 8];
            let mut thermal_xs_add = [0.0_f64; 8];

            for (i, nuc) in material.nuclides.iter().enumerate() {
                let mut xs = xs_provider.lookup(nuc.xs_kernel_idx, particle.energy);
                xs_provider.apply_urr(nuc.xs_kernel_idx, &mut xs, particle.energy, urr_xi);

                if let Some(tsl) = xs_provider.thermal_scattering(nuc.xs_kernel_idx) {
                    if particle.energy < tsl.energy_max && particle.energy > 0.0 {
                        let t_idx = tsl.select_temperature(cell.temperature, rng.uniform());
                        let thermal_total = tsl.total_xs(particle.energy, t_idx).max(0.0);
                        if thermal_total > 0.0 {
                            let delta = thermal_total - xs.elastic;
                            xs.total += delta;
                            thermal_xs_add[i] = thermal_total;
                            xs.elastic = 0.0;
                        }
                    }
                }
                micro_totals[i] = xs.total;
                micro_xs_arr[i] = xs;
            }
            let macro_total = material.macro_total(&micro_totals[..n_nuclides]);
            row[6] = macro_total;

            if macro_total <= 0.0 { row[9] = 8.0; particle.kill(); rows.push(row); break; }

            let xi_coll = rng.uniform();
            let d_coll = -xi_coll.ln() / macro_total;
            row[16] = xi_coll;
            row[7] = d_coll;

            let trace = geometry::ray::trace_step(
                particle.pos, particle.dir, particle.cell_idx, surfaces, cells,
            );
            let d_surf = trace.as_ref().map_or(1e20, |h| h.distance);
            row[8] = d_surf;

            if let Some(hit) = trace {
                if hit.distance < d_coll {
                    let bc = surfaces[hit.surface_idx].boundary_condition();
                    match bc {
                        BoundaryCondition::Vacuum => {
                            row[9] = 8.0; particle.advance(hit.distance); particle.kill();
                        }
                        BoundaryCondition::Reflective => {
                            row[9] = 6.0;
                            particle.advance(hit.distance);
                            let n = surfaces[hit.surface_idx].normal_at(particle.pos);
                            let d = particle.dir;
                            particle.dir = d - n * (2.0 * d.dot(n));
                        }
                        BoundaryCondition::Transmission => {
                            row[9] = 7.0;
                            particle.advance(hit.distance + (hit.distance * 1e-8).max(1e-8));
                            if let Some(next) = hit.next_cell_idx {
                                particle.cell_idx = next;
                            } else {
                                row[9] = 8.0; particle.kill();
                            }
                        }
                    }
                    row[15] = particle.energy;
                    rows.push(row);
                    continue;
                }
            }

            // Collision
            particle.advance(d_coll);
            let nuc_idx = material.sample_nuclide(
                &micro_totals[..n_nuclides], macro_total, rng.uniform(),
            );
            let xs_kernel_idx = material.nuclides[nuc_idx].xs_kernel_idx;
            let xs = &micro_xs_arr[nuc_idx];
            let a = xs.awr;
            row[10] = xs_kernel_idx as f64;
            row[11] = xs.elastic;
            row[12] = xs.inelastic;
            row[13] = xs.fission;
            row[14] = xs.capture;

            // Reaction sampling
            let xi = rng.uniform() * xs.total;
            let mut cum = 0.0;

            let use_thermal = thermal_xs_add[nuc_idx] > 0.0;
            if use_thermal {
                if xi < thermal_xs_add[nuc_idx] {
                    row[9] = 0.0; // thermal elastic / S(a,b)
                    let tsl = xs_provider.thermal_scattering(xs_kernel_idx).unwrap();
                    let t_idx = tsl.select_temperature(cell.temperature, rng.uniform());
                    let (e_out, mu) = tsl.sample(particle.energy, t_idx, &mut rng);
                    particle.energy = e_out;
                    let phi = 2.0 * std::f64::consts::PI * rng.uniform();
                    let sin_mu = (1.0 - mu * mu).max(0.0).sqrt();
                    let d = particle.dir;
                    let w2 = d.z * d.z;
                    if w2 < 0.999 {
                        let inv_sq = 1.0 / (1.0 - w2).sqrt();
                        particle.dir = Vec3::new(
                            mu * d.x + sin_mu * (d.x * d.z * phi.cos() - d.y * phi.sin()) * inv_sq,
                            mu * d.y + sin_mu * (d.y * d.z * phi.cos() + d.x * phi.sin()) * inv_sq,
                            mu * d.z - sin_mu * (1.0 - w2).sqrt() * phi.cos(),
                        );
                    } else {
                        let sign = if d.z > 0.0 { 1.0 } else { -1.0 };
                        particle.dir = Vec3::new(sin_mu * phi.cos(), sin_mu * phi.sin() * sign, mu * sign);
                    }
                } else {
                    // Non-thermal: simplified — treat as capture for trace purposes
                    // (fission/inelastic would need full collision processing)
                    row[9] = 5.0;
                    particle.kill();
                }
            } else {
                cum += xs.elastic;
                if xi < cum {
                    row[9] = 0.0; // elastic
                    // Free-gas or two-body elastic
                    let cell_kT = cell.temperature * 8.617333262e-5;
                    if particle.energy < 400.0 * cell_kT {
                        // Simplified free-gas: use same kinematics as GPU
                        let sigma = (cell_kT / a).sqrt();
                        let v_n = (2.0 * particle.energy).sqrt();
                        let u1 = rng.uniform(); let u2 = rng.uniform();
                        let r_bm = sigma * (-2.0 * u1.max(1e-30).ln()).sqrt();
                        let th = 2.0 * std::f64::consts::PI * u2;
                        let vtx = r_bm * th.cos(); let vty = r_bm * th.sin();
                        let u1 = rng.uniform(); let u2 = rng.uniform();
                        let r_bm = sigma * (-2.0 * u1.max(1e-30).ln()).sqrt();
                        let th = 2.0 * std::f64::consts::PI * u2;
                        let vtz = r_bm * th.cos();
                        let d = particle.dir;
                        let vnx = d.x * v_n; let vny = d.y * v_n; let vnz = d.z * v_n;
                        let vrx = vnx - vtx; let vry = vny - vty; let vrz = vnz - vtz;
                        let vr = (vrx*vrx + vry*vry + vrz*vrz).sqrt().max(1e-20);
                        let ia1 = 1.0 / (1.0 + a);
                        let vcx = (vnx + a*vtx)*ia1; let vcy = (vny + a*vty)*ia1; let vcz = (vnz + a*vtz)*ia1;
                        let vcn = vr * a * ia1;
                        let e_rel = 0.5 * (a / (a + 1.0)) * vr * vr;
                        // Angular dist at relative energy
                        let ang = xs_provider.elastic_angular_dist(xs_kernel_idx);
                        let mu_cm = if let Some(ref ad) = ang {
                            ad.sample_mu(e_rel, &mut rng)
                        } else {
                            2.0 * rng.uniform() - 1.0
                        };
                        let phi = 2.0 * std::f64::consts::PI * rng.uniform();
                        let st = (1.0_f64 - mu_cm*mu_cm).max(0.0).sqrt();
                        let vrh = Vec3::new(vrx/vr, vry/vr, vrz/vr);
                        let (p2, q) = if vrh.z.abs() < 0.999 {
                            let ip = 1.0 / (1.0 - vrh.z*vrh.z).sqrt();
                            let p = Vec3::new(-vrh.y*ip, vrh.x*ip, 0.0);
                            let q = Vec3::new(vrh.y*p.z - vrh.z*p.y, vrh.z*p.x - vrh.x*p.z, vrh.x*p.y - vrh.y*p.x);
                            (p, q)
                        } else {
                            let ip = 1.0 / (1.0 - vrh.x*vrh.x).sqrt();
                            let p = Vec3::new(0.0, -vrh.z*ip, vrh.y*ip);
                            let q = Vec3::new(vrh.y*p.z - vrh.z*p.y, vrh.z*p.x - vrh.x*p.z, vrh.x*p.y - vrh.y*p.x);
                            (p, q)
                        };
                        let sx = mu_cm*vrh.x + st*(phi.cos()*p2.x + phi.sin()*q.x);
                        let sy = mu_cm*vrh.y + st*(phi.cos()*p2.y + phi.sin()*q.y);
                        let sz = mu_cm*vrh.z + st*(phi.cos()*p2.z + phi.sin()*q.z);
                        let vox = vcx + vcn*sx; let voy = vcy + vcn*sy; let voz = vcz + vcn*sz;
                        let vo = (vox*vox + voy*voy + voz*voz).sqrt();
                        particle.energy = (0.5 * vo * vo).max(1e-11);
                        if vo > 1e-20 {
                            particle.dir = Vec3::new(vox/vo, voy/vo, voz/vo);
                        }
                    } else {
                        // Two-body elastic
                        let ang = xs_provider.elastic_angular_dist(xs_kernel_idx);
                        let mu_cm = if let Some(ref ad) = ang {
                            ad.sample_mu(particle.energy, &mut rng)
                        } else {
                            2.0 * rng.uniform() - 1.0
                        };
                        let alpha = ((a-1.0)/(a+1.0)).powi(2);
                        particle.energy = (particle.energy * (1.0+alpha+(1.0-alpha)*mu_cm)/2.0).max(1e-11);
                        let mu_lab = (1.0 + a*mu_cm) / (1.0 + a*a + 2.0*a*mu_cm).sqrt();
                        let phi = 2.0 * std::f64::consts::PI * rng.uniform();
                        let sin_mu = (1.0 - mu_lab*mu_lab).max(0.0).sqrt();
                        let d = particle.dir;
                        let w2 = d.z * d.z;
                        if w2 < 0.999 {
                            let inv_sq = 1.0 / (1.0 - w2).sqrt();
                            particle.dir = Vec3::new(
                                mu_lab*d.x + sin_mu*(d.x*d.z*phi.cos() - d.y*phi.sin())*inv_sq,
                                mu_lab*d.y + sin_mu*(d.y*d.z*phi.cos() + d.x*phi.sin())*inv_sq,
                                mu_lab*d.z - sin_mu*(1.0-w2).sqrt()*phi.cos(),
                            );
                        } else {
                            let sign = if d.z > 0.0 { 1.0 } else { -1.0 };
                            particle.dir = Vec3::new(sin_mu*phi.cos(), sin_mu*phi.sin()*sign, mu_lab*sign);
                        }
                    }
                } else {
                    cum += xs.inelastic;
                    if xi < cum {
                        row[9] = 1.0; // inelastic (simplified)
                        particle.energy = (particle.energy * 0.5).max(1e-5);
                        let mu = 2.0*rng.uniform()-1.0;
                        let phi = 2.0 * std::f64::consts::PI * rng.uniform();
                        let sin_mu = (1.0 - mu*mu).max(0.0).sqrt();
                        let d = particle.dir;
                        let w2 = d.z * d.z;
                        if w2 < 0.999 {
                            let inv_sq = 1.0 / (1.0 - w2).sqrt();
                            particle.dir = Vec3::new(
                                mu*d.x + sin_mu*(d.x*d.z*phi.cos() - d.y*phi.sin())*inv_sq,
                                mu*d.y + sin_mu*(d.y*d.z*phi.cos() + d.x*phi.sin())*inv_sq,
                                mu*d.z - sin_mu*(1.0-w2).sqrt()*phi.cos(),
                            );
                        } else {
                            let sign = if d.z > 0.0 { 1.0 } else { -1.0 };
                            particle.dir = Vec3::new(sin_mu*phi.cos(), sin_mu*phi.sin()*sign, mu*sign);
                        }
                    } else {
                        cum += xs.n2n;
                        if xi < cum { row[9] = 2.0; } // simplified
                        else {
                            cum += xs.n3n;
                            if xi < cum { row[9] = 3.0; }
                            else {
                                cum += xs.fission;
                                if xi < cum { row[9] = 4.0; particle.kill(); }
                                else { row[9] = 5.0; particle.kill(); } // capture
                            }
                        }
                    }
                }
            }

            row[15] = particle.energy;
            rows.push(row);
        }
        rows
    }

    fn write_trace_csv(filename: &str, header: &str, traces: &[(usize, Vec<[f64; 17]>)]) {
        let mut f = std::fs::File::create(filename).expect("create csv");
        writeln!(f, "{header}").unwrap();
        for (pid, rows) in traces {
            for (s, row) in rows.iter().enumerate() {
                write!(f, "{pid},{s}").unwrap();
                for v in row { write!(f, ",{:.8e}", v).unwrap(); }
                writeln!(f).unwrap();
            }
        }
    }

    pub fn run() {
        let args = Args::parse();
        let n = args.particles;
        let max_steps = args.steps;

        println!("=== GPU vs CPU Transport Trace ===");
        println!("Particles: {n}, Steps: {max_steps}");

        // Load nuclear data — Table mode (matches GPU pointwise)
        println!("\n── Loading nuclear data (Table mode) ──");
        let mut svd_kernels = Vec::new();
        let mut table_nuclides = Vec::new();
        for &(filename, awr, nu_bar, nuc_temp_idx) in NUCLIDE_SPECS {
            let path = args.data_dir.join(filename);
            svd_kernels.push(xs_provider::load_nuclide(&path, args.rank, nuc_temp_idx, awr, nu_bar));
            table_nuclides.push(xs_provider::load_nuclide_table(&path, nuc_temp_idx, awr, nu_bar));
        }
        let thermal = load_thermal(&args.data_dir);
        let provider = xs_provider::TableXsProvider { nuclides: table_nuclides, thermal: thermal.clone() };
        let svd_thermal = thermal;

        let materials = setup_materials();
        let (surfaces, cells) = setup_geometry();
        let source = initial_source(n);

        // GPU trace
        println!("\n── GPU trace ──");
        let gpu = GpuTransportContext::new().expect("GPU init");
        let nuc_data = gpu.upload_nuclide_data(&svd_kernels, args.rank).expect("upload nuc");
        let awrs: Vec<f64> = NUCLIDE_SPECS.iter().map(|s| s.1).collect();
        let nu_bars: Vec<f64> = NUCLIDE_SPECS.iter().map(|s| s.2).collect();
        let mat_data = gpu.upload_material_data(&materials, &awrs, &nu_bars).expect("upload mat");
        let h2o_path = args.data_dir.join("c_H_in_H2O.h5");
        let sab_data = if h2o_path.exists() {
            match hdf5_reader::load_thermal_scattering(&h2o_path) {
                Ok(tsl) => { let t = tsl.select_temperature(600.0, 0.5); gpu.upload_sab_data(&tsl, t).expect("sab") }
                Err(_) => gpu.upload_sab_data_empty().expect("sab"),
            }
        } else { gpu.upload_sab_data_empty().expect("sab") };
        let wmp_data = gpu.upload_wmp_data_empty(NUCLIDE_SPECS.len()).expect("wmp empty");

        let gpu_trace = gpu.run_debug_trace(&source, &nuc_data, &mat_data, &sab_data, &wmp_data, max_steps as u32, 0).expect("GPU trace");

        let header = "particle,step,energy,pos_x,pos_y,pos_z,cell,material,macro_total,d_coll,d_surf,event,hit_nuc,micro_el,micro_inel,micro_fis,micro_cap,out_energy,rng_xi";
        // Write GPU CSV
        {
            let cols = 17;
            let mut f = std::fs::File::create("gpu_trace.csv").expect("create");
            writeln!(f, "{header}").unwrap();
            for pid in 0..n {
                let steps = gpu_trace.step_counts[pid] as usize;
                for s in 0..steps {
                    let base = pid * max_steps * cols + s * cols;
                    write!(f, "{pid},{s}").unwrap();
                    for c in 0..cols { write!(f, ",{:.8e}", gpu_trace.data[base + c]).unwrap(); }
                    writeln!(f).unwrap();
                }
            }
        }
        println!("  Wrote gpu_trace.csv");

        // CPU trace
        println!("\n── CPU trace ──");
        let mut cpu_traces = Vec::new();
        for pid in 0..n {
            let rows = cpu_trace_particle(pid, &source[pid], max_steps, &surfaces, &cells, &materials, &provider);
            cpu_traces.push((pid, rows));
        }
        write_trace_csv("cpu_trace.csv", header, &cpu_traces);
        println!("  Wrote cpu_trace.csv");

        // Compare first divergence per particle
        println!("\n── Comparing traces ──");
        let cols = 17;
        let mut n_match = 0;
        let mut n_diverge = 0;
        for pid in 0..n.min(20) {
            let gpu_steps = gpu_trace.step_counts[pid] as usize;
            let cpu_rows = &cpu_traces[pid].1;
            let min_steps = gpu_steps.min(cpu_rows.len());

            let mut first_diff_step = None;
            for s in 0..min_steps {
                let gb = pid * max_steps * cols + s * cols;
                let gpu_e = gpu_trace.data[gb]; // energy
                let cpu_e = cpu_rows[s][0];
                let gpu_ev = gpu_trace.data[gb + 9] as i32;
                let cpu_ev = cpu_rows[s][9] as i32;
                let gpu_mt = gpu_trace.data[gb + 6]; // macro_total
                let cpu_mt = cpu_rows[s][6];

                let e_diff = if cpu_e > 0.0 { ((gpu_e - cpu_e) / cpu_e).abs() } else { 0.0 };
                let mt_diff = if cpu_mt > 0.0 { ((gpu_mt - cpu_mt) / cpu_mt).abs() } else { 0.0 };

                if gpu_ev != cpu_ev || e_diff > 0.01 || mt_diff > 0.01 {
                    first_diff_step = Some(s);
                    break;
                }
            }

            if let Some(s) = first_diff_step {
                let gb = pid * max_steps * cols + s * cols;
                println!("  P{pid:>3} diverges at step {s}: GPU E={:.4e} ev={} mt={:.4e} | CPU E={:.4e} ev={} mt={:.4e}",
                    gpu_trace.data[gb], gpu_trace.data[gb+9] as i32,
                    gpu_trace.data[gb+6],
                    cpu_rows[s][0], cpu_rows[s][9] as i32,
                    cpu_rows[s][6],
                );
                // Print detailed trace for first diverging particle
                if n_diverge == 0 {
                    let start = if s > 3 { s - 3 } else { 0 };
                    let end = (s + 2).min(min_steps);
                    println!("    ── Detail for P{pid} steps {start}..{end} ──");
                    println!("    step |  E(GPU)      E(CPU)     | cell  mat | d_coll(GPU)  d_coll(CPU) | d_surf(GPU) d_surf(CPU) | ev(G) ev(C) | mt(GPU)    mt(CPU)    | E_out(GPU)  E_out(CPU)");
                    for st in start..end {
                        let g = pid * max_steps * cols + st * cols;
                        let c = &cpu_rows[st];
                        let mark = if st == s { " <<<" } else { "" };
                        println!("    {:>4} | {:11.4e} {:11.4e} | {:>4} {:>4} | {:11.5e} {:11.5e} | {:11.5e} {:11.5e} | {:>5} {:>5} | {:10.6e} {:10.6e} | {:11.4e} {:11.4e}{}",
                            st,
                            gpu_trace.data[g], c[0],         // energy
                            gpu_trace.data[g+4] as i32, c[4] as i32, // cell
                            gpu_trace.data[g+7], c[7],       // d_coll
                            gpu_trace.data[g+8], c[8],       // d_surf
                            gpu_trace.data[g+9] as i32, c[9] as i32, // event
                            gpu_trace.data[g+6], c[6],       // macro_total
                            gpu_trace.data[g+15], c[15],     // E_out
                            mark,
                        );
                    }
                }
                n_diverge += 1;
            } else {
                n_match += 1;
            }
        }
        println!("\n  Match: {n_match}, Diverge: {n_diverge} (of {} checked)", n.min(20));
    }
}
