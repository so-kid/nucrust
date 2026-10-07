//! Criterion benchmarks for Coulomb wave functions, one group per computation
//! regime (Steed / 1F1 series / rho shift) plus multi-l and batch calls.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nucrust_special::{coulomb_wave, coulomb_wave_batch_simd};

fn bench_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("coulomb_wave");
    let cases: &[(&str, f64, f64, u32, u32)] = &[
        ("oscillatory_l0", 1.0, 10.0, 0, 1),
        ("oscillatory_l0-20", 2.0, 30.0, 0, 21),
        ("near_turning_point", 5.0, 10.0, 0, 1),
        ("forbidden_l0", 10.0, 3.0, 0, 1),
        ("forbidden_l0-10", 10.0, 3.0, 0, 11),
        ("forbidden_deep", 30.0, 10.0, 0, 1),
        ("small_rho", 1.0, 0.1, 0, 1),
        ("large_eta_rho_shift", 100.0, 170.0, 0, 1),
    ];
    for &(name, eta, rho, l_min, n_l) in cases {
        group.bench_function(name, |b| {
            b.iter(|| coulomb_wave(black_box(eta), black_box(rho), l_min, n_l).unwrap())
        });
    }
    group.finish();
}

fn bench_batch(c: &mut Criterion) {
    // 256 points spread over the oscillatory and forbidden regions.
    let etas: Vec<f64> = (0..256).map(|i| 0.5 + (i % 16) as f64).collect();
    let rhos: Vec<f64> = (0..256).map(|i| 0.6 + (i / 16) as f64 * 1.5).collect();
    c.bench_function("coulomb_wave_batch_simd/256x(l=0..4)", |b| {
        b.iter(|| coulomb_wave_batch_simd(black_box(&etas), black_box(&rhos), 0, 5).unwrap())
    });
}

criterion_group!(benches, bench_single, bench_batch);
criterion_main!(benches);
