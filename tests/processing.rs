use core::f32::consts::PI;

use glam::Vec3;
use tesh::prelude::*;
use tesh::primitives;

fn rel(a: f32, b: f32) -> f32 {
    (a - b).abs() / b.abs()
}

#[test]
fn closed_primitives_are_watertight_and_outward() {
    // (mesh, euler characteristic): sphere-topology shapes are 2, the torus is 0 (genus 1).
    let shapes = [
        (primitives::box3(Vec3::new(2.0, 3.0, 4.0)), 2),
        (primitives::cube(2.0), 2),
        (primitives::tetrahedron(1.0), 2),
        (primitives::uv_sphere(1.0, 48, 24), 2),
        (primitives::icosphere(1.0, 3), 2),
        (primitives::cylinder(1.0, 2.0, 48), 2),
        (primitives::cone(1.0, 2.0, 48), 2),
        (primitives::torus(2.0, 0.5, 48, 24), 0),
        (primitives::capsule(1.0, 2.0, 48, 12), 2),
    ];
    for (i, (m, euler)) in shapes.iter().enumerate() {
        assert!(m.is_valid(), "shape {i} valid indices");
        assert!(m.is_watertight(), "shape {i} watertight");
        assert!(m.signed_volume() > 0.0, "shape {i} wound outward");
        assert_eq!(m.euler_number(), *euler, "shape {i} topology");
    }
}

#[test]
fn primitive_volumes_match_formulas() {
    assert!(rel(primitives::box3(Vec3::new(2.0, 3.0, 4.0)).volume(), 24.0) < 1e-3);
    assert!(rel(primitives::uv_sphere(1.0, 64, 32).volume(), 4.0 / 3.0 * PI) < 0.02);
    assert!(rel(primitives::icosphere(1.0, 4).volume(), 4.0 / 3.0 * PI) < 0.02);
    assert!(rel(primitives::cylinder(1.0, 2.0, 128).volume(), PI * 2.0) < 0.02);
    assert!(rel(primitives::cone(1.0, 2.0, 128).volume(), PI * 2.0 / 3.0) < 0.02);
    // Capsule = cylinder(r,h) + sphere(r).
    let capsule = PI * 2.0 + 4.0 / 3.0 * PI;
    assert!(rel(primitives::capsule(1.0, 2.0, 64, 24).volume(), capsule) < 0.03);
}

#[test]
fn grid_is_open() {
    let g = primitives::grid(4.0, 2.0, 4, 2);
    assert_eq!(g.face_count(), 2 * 4 * 2);
    assert!(!g.is_watertight());
    assert!(!g.boundary_edges().is_empty());
}

#[test]
fn merge_vertices_welds_coincident() {
    let p = [
        Vec3::ZERO,
        Vec3::X,
        Vec3::Y,
        Vec3::X, // duplicate of vertex 1
        Vec3::Y, // duplicate of vertex 2
        Vec3::new(1.0, 1.0, 0.0),
    ];
    let mut mesh = TriMesh::new(p.to_vec(), vec![[0, 1, 2], [3, 4, 5]]);
    mesh.merge_vertices(1e-6);
    assert_eq!(mesh.vertex_count(), 4);
    assert_eq!(mesh.faces(), [[0, 1, 2], [1, 2, 3]]);
}

#[test]
fn cleanup_removes_junk() {
    let mut mesh = TriMesh::new(
        vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(9.0, 9.0, 9.0)],
        vec![
            [0, 1, 2],
            [0, 1, 2], // duplicate
            [0, 0, 1], // degenerate (repeated index)
        ],
    );
    mesh.remove_degenerate_faces(1e-12);
    mesh.remove_duplicate_faces();
    assert_eq!(mesh.face_count(), 1);
    mesh.remove_unreferenced_vertices();
    assert_eq!(mesh.vertex_count(), 3);
}

#[test]
fn orient_outward_fixes_scrambled_winding() {
    let base = primitives::cube(2.0);
    // Scramble: flip every other face's winding so the mesh is inconsistently wound.
    let mut faces = base.faces().to_vec();
    for (i, f) in faces.iter_mut().enumerate() {
        if i % 2 == 0 {
            f.swap(1, 2);
        }
    }
    let mut cube = TriMesh::new(base.vertices().to_vec(), faces);
    cube.orient_outward();
    assert!(cube.is_watertight());
    assert!(cube.signed_volume() > 0.0);
    assert!(rel(cube.volume(), 8.0) < 1e-4);
}

#[test]
fn fill_holes_closes_open_surface() {
    // A cube with its +z face (two triangles) removed.
    let full = primitives::cube(2.0);
    let verts = full.vertices().to_vec();
    let faces = full
        .faces()
        .iter()
        .copied()
        .filter(|f| {
            let [a, b, c] = *f;
            let n = tesh::triangle::normal(verts[a as usize], verts[b as usize], verts[c as usize]);
            n.dot(Vec3::Z) < 0.9
        })
        .collect();
    let mut open = TriMesh::new(verts, faces);
    assert!(!open.is_watertight());

    let filled = open.fill_holes();
    assert_eq!(filled, 1);
    assert!(open.is_watertight(), "filled surface is closed");
    open.orient_outward();
    assert!(rel(open.volume(), 8.0) < 1e-4);
}
