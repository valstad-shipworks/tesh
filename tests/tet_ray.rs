use std::collections::BTreeMap;

use glam::{Vec2, Vec3};
use tesh::prelude::*;
use tesh::{primitives, triangle};

const EPS: f32 = 1e-4;

/// The four faces of a tetrahedron, each as (corners, opposite apex).
const TET_FACES: [([usize; 3], usize); 4] = [
    ([1, 2, 3], 0),
    ([0, 2, 3], 1),
    ([0, 1, 3], 2),
    ([0, 1, 2], 3),
];

fn fan(n: usize, radius: f32) -> Vec<(Vec3, Vec3)> {
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).sqrt();
            let a = i as f32 * 2.399_963;
            let d = Vec3::new(r * a.cos(), r * a.sin(), z);
            let jitter = d.any_orthogonal_vector().normalize() * (0.37 * (i % 5) as f32 / 5.0);
            (-d * radius + jitter, d)
        })
        .collect()
}

fn brute(mesh: &TetMesh, origin: Vec3, dir: Vec3) -> Option<f32> {
    mesh.boundary_faces()
        .into_iter()
        .filter_map(|f| {
            let [a, b, c] = f.map(|i| mesh.vertices()[i as usize]);
            triangle::ray_intersect(origin, dir, a, b, c)
        })
        .min_by(f32::total_cmp)
}

fn assert_hit_consistent(mesh: &TetMesh, origin: Vec3, dir: Vec3, hit: &TetRayHit) {
    assert!(hit.point.abs_diff_eq(origin + dir * hit.t, EPS));
    assert!((hit.normal.length() - 1.0).abs() < EPS);

    let tet = mesh.tets()[hit.tet as usize];
    let (idx, apex) = TET_FACES[hit.face as usize];
    let [a, b, c] = idx.map(|k| mesh.vertices()[tet[k] as usize]);
    let plane_n = triangle::normal(a, b, c);
    assert!(
        (hit.point - a).dot(plane_n).abs() < EPS,
        "hit point off its owner face"
    );
    assert!(
        (hit.point - triangle::closest_point(hit.point, a, b, c)).length() < EPS,
        "hit point outside its owner face"
    );
    let apex = mesh.vertices()[tet[apex] as usize];
    assert!(
        hit.normal.dot(apex - a) < 0.0,
        "normal points into the owner tet"
    );
}

#[test]
fn matches_brute_force_on_grid() {
    let mesh = primitives::tet_grid([4, 3, 5], Vec3::new(2.0, 1.5, 2.5));
    let mut hits = 0;
    for (origin, dir) in fan(300, 4.0) {
        let fast = mesh.ray_intersect(origin, dir);
        let slow = brute(&mesh, origin, dir);
        assert_eq!(
            fast.is_some(),
            slow.is_some(),
            "hit/miss mismatch at {origin} {dir}"
        );
        if let (Some(hit), Some(t)) = (fast, slow) {
            assert!((hit.t - t).abs() < EPS);
            assert!(hit.normal.dot(dir) < 0.0, "outside rays enter the volume");
            assert_hit_consistent(&mesh, origin, dir, &hit);
            hits += 1;
        }
    }
    assert!(hits > 100, "fan should mostly hit, got {hits}");
}

#[test]
fn matches_brute_force_on_extrusion() {
    let section = [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];
    let mesh = primitives::extrude_section(&section, &[[0, 1, 2], [0, 2, 3]], &[-1.0, 0.0, 1.0]);
    for (origin, dir) in fan(200, 3.0) {
        let fast = mesh.ray_intersect(origin, dir).map(|h| h.t);
        let slow = brute(&mesh, origin, dir);
        match (fast, slow) {
            (Some(a), Some(b)) => assert!((a - b).abs() < EPS),
            (None, None) => {}
            (f, s) => panic!("hit/miss mismatch at {origin} {dir}: fast {f:?}, brute {s:?}"),
        }
    }
}

#[test]
fn exits_from_inside() {
    let mesh = primitives::tet_grid([3, 3, 3], Vec3::splat(2.0));
    let origin = Vec3::new(0.1, -0.2, 0.3);
    assert!(mesh.contains_point(origin));
    let hit = mesh.ray_intersect(origin, Vec3::X).unwrap();
    assert!((hit.t - 0.9).abs() < EPS);
    assert!(hit.normal.abs_diff_eq(Vec3::X, EPS));
    assert!(hit.normal.dot(Vec3::X) > 0.0, "ray leaves the volume");
    assert_hit_consistent(&mesh, origin, Vec3::X, &hit);
}

#[test]
fn unnormalized_direction_keeps_point() {
    let mesh = primitives::tet_grid([2, 2, 2], Vec3::splat(2.0));
    let origin = Vec3::new(-5.0, 0.25, 0.1);
    let hit = mesh.ray_intersect(origin, Vec3::X * 2.0).unwrap();
    assert!((hit.t - 2.0).abs() < EPS);
    assert!(hit.point.abs_diff_eq(Vec3::new(-1.0, 0.25, 0.1), EPS));
    assert!(hit.normal.abs_diff_eq(Vec3::NEG_X, EPS));
}

#[test]
fn misses_and_empty() {
    let mesh = primitives::tet_grid([2, 2, 2], Vec3::ONE);
    assert!(
        mesh.ray_intersect(Vec3::new(0.0, 3.0, 0.0), Vec3::X)
            .is_none()
    );
    assert!(
        mesh.ray_intersect(Vec3::new(3.0, 0.0, 0.0), Vec3::X)
            .is_none()
    );
    assert!(
        TetMesh::default()
            .ray_intersect(Vec3::ZERO, Vec3::X)
            .is_none()
    );
}

#[test]
fn deformation_refreshes_the_boundary() {
    let mut mesh = primitives::tet_grid([2, 2, 2], Vec3::splat(2.0));
    let before = mesh
        .ray_intersect(Vec3::new(-5.0, 0.1, 0.2), Vec3::X)
        .unwrap();
    assert!((before.point.x + 1.0).abs() < EPS);
    mesh.apply_transform(glam::Affine3A::from_translation(Vec3::new(-0.5, 0.0, 0.0)));
    let after = mesh
        .ray_intersect(Vec3::new(-5.0, 0.1, 0.2), Vec3::X)
        .unwrap();
    assert!((after.point.x + 1.5).abs() < EPS);
}

#[test]
fn batch_matches_scalar() {
    let mesh = primitives::tet_grid([3, 2, 4], Vec3::new(1.0, 2.0, 1.5));
    let rays = fan(64, 3.0);
    let batch = mesh.ray_intersects(&rays);
    for (&(o, d), b) in rays.iter().zip(batch) {
        assert_eq!(mesh.ray_intersect(o, d), b);
    }
}

#[test]
fn boundary_faces_match_reference() {
    let meshes = [
        primitives::tet_grid([3, 4, 2], Vec3::ONE),
        primitives::tetrahedralize_grid(&primitives::icosphere(1.0, 2), 0.3),
    ];
    for mesh in meshes {
        let mut seen: BTreeMap<[u32; 3], (u32, [u32; 3])> = BTreeMap::new();
        for tet in mesh.tets() {
            for (idx, _) in TET_FACES {
                let tri = idx.map(|k| tet[k]);
                let mut key = tri;
                key.sort_unstable();
                seen.entry(key).or_insert((0, tri)).0 += 1;
            }
        }
        let reference: Vec<[u32; 3]> = seen
            .into_iter()
            .filter(|(_, (count, _))| *count == 1)
            .map(|(key, _)| key)
            .collect();
        let got: Vec<[u32; 3]> = mesh
            .boundary_faces()
            .into_iter()
            .map(|mut f| {
                f.sort_unstable();
                f
            })
            .collect();
        assert_eq!(got, reference);
    }
}
