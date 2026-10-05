#![cfg(feature = "bvh")]

use glam::Vec3;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tesh::prelude::*;
use tesh::{primitives, triangle};

fn brute(mesh: &TriMesh, origin: Vec3, dir: Vec3) -> Option<(u32, f32)> {
    mesh.triangles()
        .enumerate()
        .filter_map(|(i, [a, b, c])| {
            triangle::ray_intersect(origin, dir, a, b, c).map(|t| (i as u32, t))
        })
        .min_by(|x, y| x.1.total_cmp(&y.1))
}

fn rays(rng: &mut SmallRng, radius: f32) -> Vec<(Vec3, Vec3)> {
    let mut v = |lo: f32, hi: f32| {
        Vec3::new(
            rng.random_range(lo..hi),
            rng.random_range(lo..hi),
            rng.random_range(lo..hi),
        )
    };
    let mut out = Vec::new();
    for _ in 0..400 {
        let origin = v(-radius, radius);
        out.push((origin, v(-1.0, 1.0).normalize_or(Vec3::X)));
    }
    for _ in 0..200 {
        let dir = v(-1.0, 1.0).normalize_or(Vec3::Y);
        out.push((-dir * radius * 2.0, dir));
    }
    let axes = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    for i in 0..300 {
        let dir = axes[i % 6];
        let origin = v(-radius, radius) - dir * radius * 2.0;
        out.push((origin, dir));
    }
    for i in 0..16 {
        for j in 0..16 {
            let eye = Vec3::new(0.0, 0.0, -radius * 2.0);
            let target = Vec3::new(i as f32 / 7.5 - 1.0, j as f32 / 7.5 - 1.0, 0.0);
            out.push((eye, (target - eye).normalize()));
        }
    }
    out
}

fn check(tag: &str, mesh: &TriMesh, radius: f32) {
    let mut rng = SmallRng::seed_from_u64(0xB4B4);
    let mut hits = 0;
    for (origin, dir) in rays(&mut rng, radius) {
        let fast = mesh.ray_intersect(origin, dir);
        let slow = brute(mesh, origin, dir);
        match (fast, slow) {
            (Some(h), Some((face, t))) => {
                assert!(
                    (h.t - t).abs() <= 1e-5 * t.max(1.0),
                    "{tag}: t {} vs brute {t} at {origin} {dir}",
                    h.t
                );
                if h.face != face {
                    let [a, b, c] = mesh.triangle(h.face as usize);
                    let t_own = triangle::ray_intersect(origin, dir, a, b, c).unwrap();
                    assert!(
                        (t_own - t).abs() <= 1e-5 * t.max(1.0),
                        "{tag}: face {} is no tie",
                        h.face
                    );
                }
                assert!(h.point.abs_diff_eq(origin + dir * h.t, 1e-5));
                hits += 1;
            }
            (None, None) => {}
            (f, s) => panic!("{tag}: hit/miss mismatch at {origin} {dir}: bvh {f:?}, brute {s:?}"),
        }
    }
    assert!(hits > 100, "{tag}: only {hits} hits");
}

#[test]
fn icosphere_matches_brute_force() {
    check("icosphere", &primitives::icosphere(1.0, 4), 1.5);
}

#[test]
fn torus_matches_brute_force() {
    check("torus", &primitives::torus(1.0, 0.35, 48, 24), 1.5);
}

#[test]
fn box_matches_brute_force() {
    check("box", &primitives::box3(Vec3::new(2.0, 1.0, 0.5)), 1.5);
}

#[test]
fn single_leaf_mesh_matches_brute_force() {
    let mesh = primitives::tetrahedron(1.0);
    assert!(mesh.element_count() <= 16);
    check("tetrahedron", &mesh, 1.5);
}

#[test]
fn tet_surface_matches_brute_force() {
    check(
        "tet_grid",
        &primitives::tet_grid([6, 5, 4], Vec3::new(2.0, 1.6, 1.2)).surface(),
        1.5,
    );
}
