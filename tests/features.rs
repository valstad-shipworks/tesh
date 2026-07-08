//! Tests for the consumer-driven additions (surface hits, batch queries, interop, tet
//! primitives, boundary attributes, sampling).

use glam::Vec3;
use tesh::prelude::*;
use tesh::primitives;

const EPS: f32 = 1e-4;

#[test]
fn surface_hit_face_interior() {
    let mesh = primitives::cube(1.0);
    let hit = mesh.surface_hit(Vec3::new(1.5, 0.0, 0.0)).unwrap();
    assert!((hit.signed_distance - 1.0).abs() < EPS);
    assert!(hit.point.abs_diff_eq(Vec3::new(0.5, 0.0, 0.0), EPS));
    assert!(hit.normal.abs_diff_eq(Vec3::X, EPS));
    assert!((hit.distance() - 1.0).abs() < EPS);
    let [a, b, c] = mesh.triangle(hit.face as usize);
    let reconstructed = tesh::triangle::from_barycentric(hit.barycentric, a, b, c);
    assert!(reconstructed.abs_diff_eq(hit.point, EPS));
}

#[test]
fn surface_hit_inside_is_negative_with_outward_normal() {
    let mesh = primitives::cube(1.0);
    let hit = mesh.surface_hit(Vec3::new(0.4, 0.0, 0.0)).unwrap();
    assert!((hit.signed_distance + 0.1).abs() < EPS);
    assert!(hit.normal.abs_diff_eq(Vec3::X, EPS), "normal should point outward");
}

#[test]
fn surface_hit_vertex_feature() {
    // Closest feature is a cube corner: the sign needs the containment fallback and the
    // normal is the offset direction.
    let mesh = primitives::cube(1.0);
    let hit = mesh.surface_hit(Vec3::splat(1.0)).unwrap();
    assert!(hit.point.abs_diff_eq(Vec3::splat(0.5), EPS));
    assert!((hit.signed_distance - 0.75_f32.sqrt()).abs() < EPS);
    assert!(hit.normal.abs_diff_eq(Vec3::splat(1.0).normalize(), EPS));
}

#[test]
fn surface_hit_on_surface_uses_face_normal() {
    let mesh = primitives::cube(1.0);
    let hit = mesh.surface_hit(Vec3::new(0.5, 0.1, 0.1)).unwrap();
    assert!(hit.signed_distance.abs() < EPS);
    assert!(hit.normal.abs_diff_eq(Vec3::X, EPS));
}

#[test]
fn batch_queries_match_scalar() {
    let mesh = primitives::icosphere(1.0, 1);
    let points = [
        Vec3::new(2.0, 0.1, -0.3),
        Vec3::ZERO,
        Vec3::new(-0.4, 0.9, 1.4),
        Vec3::splat(0.2),
    ];
    let sd = mesh.signed_distances(&points);
    let cp = mesh.closest_points(&points);
    let inside = mesh.contains_points(&points);
    for (i, &p) in points.iter().enumerate() {
        assert_eq!(sd[i], mesh.signed_distance(p));
        assert_eq!(cp[i], mesh.closest_point(p));
        assert_eq!(inside[i], mesh.contains_point(p));
    }

    let rays = [(Vec3::new(0.0, 0.0, 3.0), Vec3::NEG_Z), (Vec3::splat(5.0), Vec3::Z)];
    let hits = mesh.ray_intersects(&rays);
    assert!(hits[0].is_some());
    assert!(hits[1].is_none());

    let sh = mesh.surface_hits(&points);
    assert_eq!(sh.len(), points.len());
    assert!(sh.iter().all(Option::is_some));
}

#[test]
fn flat_buffer_roundtrip() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = [0u32, 1, 2];
    let mesh = TriMesh::from_flat(&positions, &indices);
    assert_eq!(mesh.vertices().len(), 3);
    assert_eq!(mesh.faces(), [[0, 1, 2]]);
    assert_eq!(mesh.flat_indices(), indices);
}

#[test]
fn from_triangles_and_append() {
    let mut mesh = TriMesh::from_triangles([[Vec3::ZERO, Vec3::X, Vec3::Y]]);
    assert_eq!(mesh.face_count(), 1);
    assert_eq!(mesh.vertices().len(), 3);

    let other = primitives::cube(1.0);
    mesh.append(&other);
    assert!(mesh.is_valid());
    assert_eq!(mesh.face_count(), 1 + other.face_count());
    assert_eq!(mesh.vertices().len(), 3 + other.vertices().len());
    // Appended faces reference the offset copies of the other mesh's vertices.
    let last = *mesh.faces().last().unwrap();
    assert!(last.iter().all(|&i| i >= 3));
}

#[test]
fn tet_grid_volume_and_orientation() {
    let mesh = primitives::tet_grid([3, 2, 4], Vec3::new(0.3, 0.2, 0.4));
    assert!(mesh.is_valid());
    assert_eq!(mesh.tet_count(), 3 * 2 * 4 * 6);
    assert!((mesh.volume() - 0.3 * 0.2 * 0.4).abs() < 1e-5);
    assert!(
        mesh.signed_volumes().iter().all(|&v| v > 0.0),
        "every Kuhn tet should be positively oriented"
    );
    let surface = mesh.surface();
    assert!(surface.is_watertight());
    // Centred on the origin.
    assert!(mesh.centroid().abs_diff_eq(Vec3::ZERO, 1e-5));
}

#[test]
fn tetrahedralize_grid_fills_a_sphere() {
    let surface = primitives::icosphere(1.0, 2);
    let mesh = primitives::tetrahedralize_grid(&surface, 0.25);
    assert!(mesh.is_valid());
    assert!(mesh.tet_count() > 0);
    assert!(mesh.signed_volumes().iter().all(|&v| v > 0.0));

    // The staircase volume approximates the sphere's at this resolution.
    let sphere_volume = 4.0 / 3.0 * core::f32::consts::PI;
    let v = mesh.volume();
    assert!(
        (v - sphere_volume).abs() / sphere_volume < 0.35,
        "voxelized volume {v} too far from sphere volume {sphere_volume}"
    );
    assert!(mesh.surface().is_watertight());
    assert!(mesh.contains_point(Vec3::ZERO));
}

#[test]
fn tetrahedralize_grid_of_empty_mesh_is_empty() {
    let mesh = primitives::tetrahedralize_grid(&TriMesh::default(), 0.1);
    assert_eq!(mesh.tet_count(), 0);
}

#[test]
fn boundary_carries_areas_and_outward_normals() {
    let mesh = primitives::tet_grid([2, 2, 2], Vec3::splat(1.0));
    let boundary = mesh.boundary();
    let total_area: f32 = boundary.iter().map(|f| f.area).sum();
    assert!((total_area - 6.0).abs() < 1e-4, "cube surface area, got {total_area}");
    for face in &boundary {
        let centroid = face
            .nodes
            .iter()
            .map(|&i| mesh.vertices()[i as usize])
            .sum::<Vec3>()
            / 3.0;
        assert!(
            face.normal.dot(centroid) > 0.0,
            "normal at {centroid} points inward"
        );
    }
}

#[test]
fn shape_gradients_reproduce_linear_fields() {
    let mesh = primitives::tet_grid([2, 1, 1], Vec3::new(0.4, 0.3, 0.2));
    let grads = mesh.shape_gradients();
    // The gradient of f(p) = 2x + 3y + 4z + 5 recovered per element from nodal values.
    let field = |p: Vec3| 2.0 * p.x + 3.0 * p.y + 4.0 * p.z + 5.0;
    for (i, tet) in mesh.tets().iter().enumerate() {
        let mut g = Vec3::ZERO;
        for (k, &vi) in tet.iter().enumerate() {
            g += grads[i][k] * field(mesh.vertices()[vi as usize]);
        }
        assert!(
            g.abs_diff_eq(Vec3::new(2.0, 3.0, 4.0), 1e-2),
            "element {i} gradient {g}"
        );
        assert!(
            (grads[i][0] + grads[i][1] + grads[i][2] + grads[i][3]).abs_diff_eq(Vec3::ZERO, 1e-3)
        );
    }
}

#[test]
fn degenerate_tet_shape_gradients_are_zero() {
    let mesh = TetMesh::new(
        vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(0.5, 0.5, 0.0)],
        vec![[0, 1, 2, 3]],
    );
    assert_eq!(mesh.shape_gradients()[0], [Vec3::ZERO; 4]);
}

#[test]
fn surface_samples_cover_and_dedupe() {
    let mesh = primitives::cube(1.0);
    let samples = mesh.surface_samples(0.3);
    assert!(samples.len() > mesh.vertices().len(), "long edges should add midpoints");
    // Deterministic across calls.
    assert_eq!(samples, mesh.surface_samples(0.3));
    // Every sample lies on the surface.
    for &s in &samples {
        assert!(mesh.distance(s) < 1e-5);
    }
}

#[test]
fn surface_with_map_drops_interior_vertices() {
    // A 3x3x3-cell grid has exactly 8 interior lattice vertices.
    let mesh = primitives::tet_grid([3, 3, 3], Vec3::splat(1.0));
    let (surface, map) = mesh.surface_with_map();
    assert_eq!(mesh.vertices().len(), 64);
    assert_eq!(surface.vertices().len(), 64 - 8);
    assert_eq!(map.len(), surface.vertices().len());
    for (si, &vi) in map.iter().enumerate() {
        assert_eq!(surface.vertices()[si], mesh.vertices()[vi as usize]);
    }
    assert!(surface.is_watertight());
    assert!((surface.volume() - 1.0).abs() < 1e-4);
}

#[test]
fn tet_split_separates_bodies() {
    let mut verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    verts.extend([Vec3::splat(10.0), Vec3::splat(10.0) + Vec3::X, Vec3::splat(10.0) + Vec3::Y, Vec3::splat(10.0) + Vec3::Z]);
    let mesh = TetMesh::new(verts, vec![[0, 1, 2, 3], [4, 5, 6, 7]]);
    let parts = mesh.split();
    assert_eq!(parts.len(), 2);
    for part in &parts {
        assert_eq!(part.tet_count(), 1);
        assert_eq!(part.vertices().len(), 4);
        assert!(part.is_valid());
    }
}

#[test]
fn mesh_level_nearest_vertex() {
    let mesh = primitives::cube(1.0);
    let (idx, d2) = mesh.nearest_vertex(Vec3::splat(0.45)).unwrap();
    assert!(mesh.vertices()[idx as usize].abs_diff_eq(Vec3::splat(0.5), EPS));
    assert!((d2 - 3.0 * 0.05_f32 * 0.05).abs() < 1e-5);
}

#[test]
fn aabb_merged() {
    let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
    let b = Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0));
    let m = a.merged(b);
    assert_eq!(m.min, Vec3::ZERO);
    assert_eq!(m.max, Vec3::splat(3.0));
}

#[test]
fn face_area_naming() {
    let mesh = primitives::cube(1.0);
    let per_face = mesh.face_areas();
    assert_eq!(per_face.len(), mesh.face_count());
    assert!((mesh.face_area(0) - per_face[0]).abs() < EPS);
    assert!((mesh.area() - 6.0).abs() < EPS);
}
