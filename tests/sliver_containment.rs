#![cfg(feature = "bvh")]

use glam::Vec3;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tesh::prelude::*;

/// A closed box whose four long faces are each two fans of `n` needle triangles, one from
/// each end of the face's diagonal, the way bisecting a long thin triangle along its length
/// leaves it. Every needle has the same small area.
fn needle_box(len: f32, side: f32, n: usize) -> TriMesh {
    let corners = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, side, 0.0),
        Vec3::new(0.0, side, side),
        Vec3::new(0.0, 0.0, side),
    ];
    let mut verts = Vec::new();
    for corner in corners {
        verts.extend((0..=n).map(|i| corner + Vec3::X * (len * i as f32 / n as f32)));
    }
    let at = |k: usize, i: usize| ((k % 4) * (n + 1) + i) as u32;
    let mut faces = Vec::new();
    for k in 0..4 {
        for i in 0..n {
            faces.push([at(k + 1, n), at(k, i), at(k, i + 1)]);
            faces.push([at(k, 0), at(k + 1, i), at(k + 1, i + 1)]);
        }
    }
    faces.push([at(0, 0), at(1, 0), at(2, 0)]);
    faces.push([at(0, 0), at(2, 0), at(3, 0)]);
    faces.push([at(0, n), at(2, n), at(1, n)]);
    faces.push([at(0, n), at(3, n), at(2, n)]);
    TriMesh::new(verts, faces)
}

#[test]
fn points_inside_a_box_of_needle_triangles_read_inside() {
    let (len, side) = (0.02, 0.002);
    let mesh = needle_box(len, side, 400);
    assert!(mesh.is_watertight());
    let mut rng = SmallRng::seed_from_u64(3);
    let misses = (0..20_000)
        .filter(|_| {
            let p = Vec3::new(
                rng.random_range(0.05..0.95) * len,
                rng.random_range(0.05..0.95) * side,
                rng.random_range(0.05..0.95) * side,
            );
            !mesh.contains_point(p)
        })
        .count();
    assert_eq!(misses, 0, "{misses} interior points read outside");
}
