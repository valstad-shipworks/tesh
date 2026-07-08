//! Regression tests pinning the corrected behaviour of the AUDIT.md findings.

use glam::{Affine3A, Vec3};
use tesh::prelude::*;
use tesh::primitives;

const EPS: f32 = 1e-4;

#[test]
fn transform_invalidates_query_cache() {
    let mut mesh = primitives::cube(1.0);
    let d0 = mesh.distance(Vec3::ZERO);
    assert!((d0 - 0.5).abs() < EPS);

    mesh.apply_transform(Affine3A::from_translation(Vec3::splat(100.0)));
    let d1 = mesh.distance(Vec3::ZERO);
    let expected = (Vec3::splat(100.0).length()) - (0.5_f32 * 3.0_f32.sqrt());
    assert!(
        (d1 - expected).abs() < 1e-2,
        "distance after transform: got {d1}, expected ~{expected}"
    );
}

#[test]
fn vertices_mut_invalidates_query_cache() {
    let mut mesh = primitives::cube(1.0);
    let _ = mesh.distance(Vec3::ZERO);
    for v in mesh.vertices_mut() {
        v.x += 10.0;
    }
    let d = mesh.distance(Vec3::new(10.0, 0.0, 0.0));
    assert!((d - 0.5).abs() < EPS, "query answers against moved geometry, got {d}");
}

#[test]
fn contains_point_far_from_origin_terminates() {
    let mesh = primitives::cube(1.0).transformed(Affine3A::from_translation(Vec3::splat(2000.0)));
    assert!(mesh.contains_point(Vec3::splat(2000.0)));
    assert!(!mesh.contains_point(Vec3::splat(2001.0)));
}

#[test]
fn degenerate_tet_contains_nothing() {
    let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    // Tet 0 is degenerate (repeated vertex); tet 1 is the real unit tet.
    let mesh = TetMesh::new(verts, vec![[0, 0, 1, 2], [0, 1, 2, 3]]);

    assert_eq!(
        mesh.find_containing_tet(Vec3::new(0.9, 0.9, 0.0)),
        None,
        "point outside the volume is claimed by the degenerate tet"
    );
    assert_eq!(mesh.find_containing_tet(Vec3::splat(0.1)), Some(1));
    assert!(!tesh::tetrahedron::contains_point(
        Vec3::new(0.9, 0.9, 0.0),
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::X,
        Vec3::Y
    ));
}

#[test]
fn fill_holes_survives_pinch_vertex() {
    // Two holes sharing one vertex: remove two faces of a cube that share exactly one corner.
    let cube = primitives::cube(1.0);
    let mut faces = cube.faces().to_vec();
    let mut pair = None;
    'outer: for i in 0..faces.len() {
        for j in i + 1..faces.len() {
            let shared = faces[i]
                .iter()
                .filter(|v| faces[j].contains(v))
                .count();
            if shared == 1 {
                pair = Some((i, j));
                break 'outer;
            }
        }
    }
    let (i, j) = pair.expect("a cube has face pairs sharing exactly one vertex");
    faces.remove(j);
    faces.remove(i);
    let mut mesh = TriMesh::new(cube.vertices().to_vec(), faces);
    assert!(!mesh.is_watertight());

    let filled = mesh.fill_holes();
    assert!(filled >= 1, "no loops were filled");
    assert!(
        mesh.is_watertight(),
        "{} boundary edges remain after fill_holes",
        mesh.boundary_edges().len()
    );
}

#[test]
fn empty_mesh_queries_are_consistent() {
    let mesh = TriMesh::default();
    let p = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(mesh.closest_point(p), p);
    assert!(mesh.distance(p).is_infinite());
    assert!(mesh.signed_distance(p).is_infinite());
    assert!(!mesh.contains_point(p));
    assert!(mesh.surface_hit(p).is_none());
    assert!(mesh.nearest_vertex(p).is_none());
    assert!(mesh.ray_intersect(p, Vec3::X).is_none());
}

#[cfg(feature = "kdtree")]
#[test]
fn empty_kdtree_returns_none() {
    let mesh = TriMesh::default();
    let kd = mesh.kdtree();
    assert_eq!(kd.nearest_vertex(Vec3::ZERO), None);
    assert!(kd.k_nearest_vertices(Vec3::ZERO, 3).is_empty());
    assert!(kd.vertices_within(Vec3::ZERO, 1.0).is_empty());
}

#[test]
fn merge_vertices_normalizes_negative_zero() {
    let mut mesh = TriMesh::new(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(-0.0, 0.0, 0.0),
            Vec3::X,
            Vec3::Y,
        ],
        vec![[0, 2, 3], [1, 3, 2]],
    );
    mesh.merge_vertices(0.0);
    assert_eq!(mesh.vertices().len(), 3, "-0.0 and +0.0 vertices should weld");
}

#[test]
fn merge_vertices_handles_non_finite_coordinates() {
    let mut mesh = TriMesh::new(
        vec![
            Vec3::new(f32::NAN, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
        ],
        vec![[1, 2, 3]],
    );
    // Must neither panic nor weld the NaN vertex into cell 0.
    mesh.merge_vertices(0.5);
    assert_eq!(mesh.vertices().len(), 4);
}

#[test]
fn face_adjacency_emits_all_pairs_on_nonmanifold_edges() {
    // Three faces fanned around one shared edge (0, 1).
    let mesh = TriMesh::new(
        vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.0, -1.0, 0.0)],
        vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]],
    );
    assert_eq!(mesh.face_adjacency().len(), 3, "k faces on an edge give k(k-1)/2 pairs");
}

#[test]
fn interpolate_matches_barycentric_field() {
    let mesh = TetMesh::new(
        vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
        vec![[0, 1, 2, 3]],
    );
    // A linear field is reproduced exactly.
    let values = [5.0, 7.0, 8.0, 9.0]; // 5 + 2x + 3y + 4z
    let p = Vec3::new(0.2, 0.3, 0.1);
    let got = mesh.interpolate(p, &values).unwrap();
    assert!((got - (5.0 + 2.0 * p.x + 3.0 * p.y + 4.0 * p.z)).abs() < EPS);
}
