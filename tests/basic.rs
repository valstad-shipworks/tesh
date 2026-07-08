use glam::{Affine3A, Vec3};
use tesh::prelude::*;
use tesh::{tetrahedron, triangle};

const EPS: f32 = 1e-4;

/// A unit cube `[0,1]^3` as a watertight, outward-wound triangle mesh (12 triangles).
fn unit_cube() -> TriMesh {
    let v = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
    ];
    let f = vec![
        [0, 2, 1],
        [0, 3, 2], // bottom (z=0), normal -z
        [4, 5, 6],
        [4, 6, 7], // top (z=1), normal +z
        [0, 1, 5],
        [0, 5, 4], // y=0
        [2, 3, 7],
        [2, 7, 6], // y=1
        [1, 2, 6],
        [1, 6, 5], // x=1
        [3, 0, 4],
        [3, 4, 7], // x=0
    ];
    TriMesh::new(v, f)
}

/// A single positively-oriented tetrahedron.
fn unit_tet() -> TetMesh {
    let v = vec![
        Vec3::ZERO,
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
    ];
    TetMesh::new(v, vec![[0, 1, 2, 3]])
}

#[test]
fn triangle_primitives() {
    let (a, b, c) = (Vec3::ZERO, Vec3::X, Vec3::Y);
    assert!((triangle::area(a, b, c) - 0.5).abs() < EPS);
    assert!(triangle::normal(a, b, c).abs_diff_eq(Vec3::Z, EPS));

    let p = Vec3::new(0.25, 0.25, 5.0);
    let bary = triangle::barycentric(p, a, b, c);
    assert!((bary.element_sum() - 1.0).abs() < EPS);
    assert!(triangle::closest_point(p, a, b, c).abs_diff_eq(Vec3::new(0.25, 0.25, 0.0), EPS));
}

#[test]
fn tetrahedron_primitives() {
    let (a, b, c, d) = (Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z);
    assert!((tetrahedron::signed_volume(a, b, c, d) - 1.0 / 6.0).abs() < EPS);
    assert!(tetrahedron::contains_point(Vec3::splat(0.1), a, b, c, d));
    assert!(!tetrahedron::contains_point(Vec3::splat(0.5), a, b, c, d));
    let bary = tetrahedron::barycentric(Vec3::splat(0.25), a, b, c, d);
    assert!((bary.iter().sum::<f32>() - 1.0).abs() < EPS);
}

#[test]
fn cube_geometry_and_topology() {
    let cube = unit_cube();
    assert!(cube.is_valid());
    assert!((cube.area() - 6.0).abs() < EPS);
    assert!((cube.volume() - 1.0).abs() < EPS);
    assert!(cube.signed_volume() > 0.0, "outward winding => positive volume");
    assert!(cube.centroid().abs_diff_eq(Vec3::splat(0.5), EPS));

    assert!(cube.is_watertight());
    assert_eq!(cube.edges_unique().len(), 18);
    assert_eq!(cube.euler_number(), 2, "V-E+F for a sphere-topology surface");
    assert!(cube.boundary_edges().is_empty());
    assert_eq!(cube.body_count(), 1);
}

#[test]
fn cube_queries() {
    let cube = unit_cube();
    assert!(cube.contains_point(Vec3::splat(0.5)));
    assert!(!cube.contains_point(Vec3::splat(2.0)));

    let inside = Vec3::new(0.5, 0.5, 0.5);
    assert!(cube.signed_distance(inside) < 0.0);
    assert!(cube.signed_distance(Vec3::new(0.5, 0.5, 2.0)) > 0.0);
    assert!((cube.distance(Vec3::new(0.5, 0.5, 2.0)) - 1.0).abs() < EPS);

    let hit = cube
        .ray_intersect(Vec3::new(0.5, 0.5, -1.0), Vec3::Z)
        .expect("ray through the cube hits it");
    assert!((hit.t - 1.0).abs() < EPS);
    assert!(hit.point.abs_diff_eq(Vec3::new(0.5, 0.5, 0.0), EPS));
}

#[test]
fn vertex_normals_point_outward() {
    let cube = unit_cube();
    let normals = cube.vertex_normals();
    assert_eq!(normals.len(), 8);
    // corner (0,0,0): outward diagonal normal has all-negative components.
    assert!(normals[0].x < 0.0 && normals[0].y < 0.0 && normals[0].z < 0.0);
}

#[test]
fn transform_moves_every_vertex() {
    let moved = unit_cube().transformed(Affine3A::from_translation(Vec3::splat(10.0)));
    assert!(moved.centroid().abs_diff_eq(Vec3::splat(10.5), EPS));
    assert!((moved.volume() - 1.0).abs() < EPS);
}

#[test]
fn tet_volume_surface_and_interpolation() {
    let tet = unit_tet();
    assert!(tet.is_valid());
    assert!((tet.volume() - 1.0 / 6.0).abs() < EPS);
    assert!(tet.signed_volumes()[0] > 0.0);
    assert!(tet.quality(0) > 0.0 && tet.quality(0) <= 1.0);

    let surface = tet.surface();
    assert_eq!(surface.face_count(), 4, "a lone tet has 4 boundary faces");
    assert!(surface.is_watertight());

    // Boundary faces must all wind outward: normals point away from the centroid.
    let centroid = tet.tet_centroid(0);
    for i in 0..surface.face_count() {
        let n = surface.face_normal(i);
        let outward = surface.face_centroid(i) - centroid;
        assert!(n.dot(outward) > 0.0, "face {i} winds outward");
    }

    assert_eq!(tet.find_containing_tet(Vec3::splat(0.1)), Some(0));
    assert_eq!(tet.find_containing_tet(Vec3::splat(0.5)), None);

    // Linear field f(x,y,z)=x reproduced exactly by barycentric interpolation.
    let field = vec![0.0, 1.0, 0.0, 0.0];
    let p = Vec3::new(0.3, 0.2, 0.1);
    assert!((tet.interpolate(p, &field).unwrap() - p.x).abs() < EPS);
}

#[test]
fn cube_mass_properties() {
    let mp = unit_cube().mass_properties();
    assert!((mp.volume - 1.0).abs() < EPS);
    assert!(mp.center_of_mass.abs_diff_eq(Vec3::splat(0.5), EPS));

    // Unit cube, unit density: I = m(h² + d²)/12 = 1/6 on the diagonal, ~0 off-diagonal.
    let i = mp.inertia;
    assert!((i.x_axis.x - 1.0 / 6.0).abs() < EPS);
    assert!((i.y_axis.y - 1.0 / 6.0).abs() < EPS);
    assert!((i.z_axis.z - 1.0 / 6.0).abs() < EPS);
    assert!(i.x_axis.y.abs() < EPS && i.x_axis.z.abs() < EPS && i.y_axis.z.abs() < EPS);

    let (moments, _axes) = mp.principal();
    for m in [moments.x, moments.y, moments.z] {
        assert!((m - 1.0 / 6.0).abs() < EPS, "principal moment {m}");
    }
    assert!((mp.mass(2.0) - 2.0).abs() < EPS);
}

#[test]
fn tet_mass_properties() {
    // Single unit tet: volume 1/6, centroid at (1/4,1/4,1/4).
    let mp = unit_tet().mass_properties();
    assert!((mp.volume - 1.0 / 6.0).abs() < EPS);
    assert!(mp.center_of_mass.abs_diff_eq(Vec3::splat(0.25), EPS));
}

#[test]
fn tet_bounds_via_trait() {
    let tet = unit_tet();
    let b = tet.bounds();
    assert!(b.min.abs_diff_eq(Vec3::ZERO, EPS));
    assert!(b.max.abs_diff_eq(Vec3::ONE, EPS));
}
