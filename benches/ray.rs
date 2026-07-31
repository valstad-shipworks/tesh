use criterion::{Criterion, criterion_group, criterion_main};
use glam::Vec3;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use std::hint::black_box;

use tesh::primitives::{icosphere, torus};
use tesh::{Mesh, TriMesh};

const N_RAYS: usize = 256;

fn rand_unit(rng: &mut SmallRng) -> Vec3 {
    loop {
        let v = Vec3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );
        let len_sq = v.length_squared();
        if len_sq > 1e-4 {
            return v / len_sq.sqrt();
        }
    }
}

/// Rays fired inward from a shell around the mesh — nearly all hit.
fn inward_rays(rng: &mut SmallRng, radius: f32) -> Vec<(Vec3, Vec3)> {
    (0..N_RAYS)
        .map(|_| {
            let dir = rand_unit(rng);
            (-dir * radius, dir)
        })
        .collect()
}

/// Rays passing well clear of the mesh — every one misses.
fn missing_rays(rng: &mut SmallRng, radius: f32) -> Vec<(Vec3, Vec3)> {
    (0..N_RAYS)
        .map(|_| {
            let dir = rand_unit(rng);
            let side = dir.any_orthogonal_vector() * (radius * 3.0);
            (-dir * radius + side, dir)
        })
        .collect()
}

/// Rays from random points in random directions: a mix of hits, misses and inside starts.
fn scattered_rays(rng: &mut SmallRng, radius: f32) -> Vec<(Vec3, Vec3)> {
    (0..N_RAYS)
        .map(|_| {
            let origin = Vec3::new(
                rng.random_range(-radius..radius),
                rng.random_range(-radius..radius),
                rng.random_range(-radius..radius),
            );
            (origin, rand_unit(rng))
        })
        .collect()
}

/// A camera-like fan from one eye point across the mesh: neighbouring rays share most of
/// their path through the tree.
fn coherent_rays(radius: f32) -> Vec<(Vec3, Vec3)> {
    let eye = Vec3::new(0.0, 0.0, -radius);
    let w = (N_RAYS as f32).sqrt() as usize;
    (0..N_RAYS)
        .map(|i| {
            let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / w as f32);
            let target = Vec3::new(x * 2.4 - 1.2, y * 2.4 - 1.2, 0.0);
            (eye, (target - eye).normalize())
        })
        .collect()
}

fn bench_set(c: &mut Criterion, tag: &str, mesh: &TriMesh, radius: f32) {
    let mut rng = SmallRng::seed_from_u64(0x2E5A);
    let sets = [
        ("hit", inward_rays(&mut rng, radius)),
        ("miss", missing_rays(&mut rng, radius)),
        ("scattered", scattered_rays(&mut rng, radius)),
        ("coherent", coherent_rays(radius)),
    ];
    // Warm the lazily built BVH so the first sample does not pay for it.
    let _ = mesh.ray_intersect(Vec3::splat(-radius), Vec3::ONE.normalize());

    for (name, rays) in &sets {
        c.bench_function(&format!("ray_{tag}_{name}"), |b| {
            b.iter(|| {
                let mut acc = 0.0f32;
                for &(origin, dir) in rays {
                    if let Some(h) = mesh.ray_intersect(black_box(origin), black_box(dir)) {
                        acc += h.t + h.face as f32;
                    }
                }
                acc
            })
        });
    }

    c.bench_function(&format!("ray_{tag}_batch"), |b| {
        b.iter(|| black_box(mesh).ray_intersects(black_box(&sets[2].1)))
    });
}

fn bench_icosphere(c: &mut Criterion) {
    let mesh = icosphere(1.0, 5);
    assert_eq!(mesh.element_count(), 20480);
    bench_set(c, "icosphere20k", &mesh, 4.0);
}

fn bench_torus(c: &mut Criterion) {
    let mesh = torus(1.0, 0.35, 128, 64);
    bench_set(c, "torus16k", &mesh, 4.0);
}

fn bench_small(c: &mut Criterion) {
    let mesh = icosphere(1.0, 2);
    bench_set(c, "icosphere320", &mesh, 4.0);
}

criterion_group!(ray_benches, bench_icosphere, bench_torus, bench_small);
criterion_main!(ray_benches);
