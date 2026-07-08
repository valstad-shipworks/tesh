#![cfg(all(feature = "kdtree", feature = "bvh"))]

use glam::Vec3;
use tesh::prelude::*;

const EPS: f32 = 1e-4;

fn octahedron() -> TriMesh {
    let v = vec![
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    let f = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    TriMesh::new(v, f)
}

#[test]
fn bvh_closest_point_matches_brute_force() {
    let mesh = octahedron();
    let bvh = mesh.bvh();
    let queries = [
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.3, 0.3, 0.3),
        Vec3::new(-1.5, 0.7, 0.2),
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::splat(0.1),
    ];
    for q in queries {
        let brute = mesh.closest_point(q);
        let fast = bvh.closest_point(q);
        assert!(
            fast.abs_diff_eq(brute, EPS),
            "closest_point mismatch at {q}: brute {brute}, bvh {fast}"
        );
        assert!((bvh.distance(q) - mesh.distance(q)).abs() < EPS);
    }
}

#[test]
fn bvh_ray_matches_brute_force() {
    let mesh = octahedron();
    let bvh = mesh.bvh();

    let hit_fast = bvh.ray_intersect(Vec3::new(0.2, 0.1, 3.0), Vec3::NEG_Z);
    let hit_brute = mesh.ray_intersect(Vec3::new(0.2, 0.1, 3.0), Vec3::NEG_Z);
    let (a, b) = (hit_fast.unwrap(), hit_brute.unwrap());
    assert!((a.t - b.t).abs() < EPS);
    assert!(a.point.abs_diff_eq(b.point, EPS));

    // A ray that misses the mesh entirely.
    assert!(bvh.ray_intersect(Vec3::new(5.0, 5.0, 5.0), Vec3::Z).is_none());
}

#[test]
fn vertices_mut_refreshes_cache_after_mutation() {
    let mut mesh = octahedron();
    // First query builds and caches the BVH.
    let d0 = mesh.distance(Vec3::new(2.0, 0.0, 0.0));

    for v in mesh.vertices_mut() {
        v.x += 10.0;
    }

    let d1 = mesh.distance(Vec3::new(12.0, 0.0, 0.0));
    assert!((d0 - d1).abs() < EPS, "query reflects moved geometry after vertices_mut");
}

#[test]
fn tet_point_location_uses_bvh() {
    use tesh::TetMesh;
    let tet = TetMesh::new(
        vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
        vec![[0, 1, 2, 3]],
    );
    assert_eq!(tet.find_containing_tet(Vec3::splat(0.1)), Some(0));
    assert_eq!(tet.find_containing_tet(Vec3::splat(0.9)), None);
    assert!(tet.closest_surface_point(Vec3::new(2.0, 0.0, 0.0)).is_finite());
}

#[test]
fn kdtree_nearest_vertex() {
    let mesh = octahedron();
    let kd = mesh.kdtree();

    let (idx, dist2) = kd.nearest_vertex(Vec3::new(0.9, 0.05, 0.0)).unwrap();
    assert_eq!(idx, 0, "closest to +X vertex");
    assert!(dist2 > 0.0);

    let within = kd.vertices_within(Vec3::ZERO, 1.0 + EPS);
    assert_eq!(within.len(), 6, "all six unit vertices lie within radius 1");

    let three = kd.k_nearest_vertices(Vec3::X, 3);
    assert_eq!(three.len(), 3);
    assert_eq!(three[0].0, 0);
}
