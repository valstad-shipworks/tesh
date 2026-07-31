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

/// Independent Möller–Trumbore, matching `triangle::ray_intersect`'s epsilon and two-sided
/// acceptance, so the BVH walk can be checked against an exhaustive scan.
fn ray_tri_ref(origin: Vec3, dir: Vec3, [a, b, c]: [Vec3; 3]) -> Option<f32> {
    const E: f32 = 1e-7;
    let (e1, e2) = (b - a, c - a);
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < E {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = origin - a;
    let u = tvec.dot(pvec) * inv;
    let qvec = tvec.cross(e1);
    let w = dir.dot(qvec) * inv;
    let t = e2.dot(qvec) * inv;
    ((0.0..=1.0).contains(&u) && w >= 0.0 && u + w <= 1.0 && t >= E).then_some(t)
}

fn nearest_ref(mesh: &TriMesh, origin: Vec3, dir: Vec3) -> Option<f32> {
    mesh.triangles()
        .filter_map(|tri| ray_tri_ref(origin, dir, tri))
        .fold(None, |best: Option<f32>, t| {
            Some(best.map_or(t, |b| b.min(t)))
        })
}

fn dist2_ref(mesh: &TriMesh, p: Vec3) -> f32 {
    mesh.triangles()
        .map(|[a, b, c]| {
            // Sample the triangle densely enough to bound the true closest distance.
            let mut best = f32::INFINITY;
            const N: usize = 12;
            for i in 0..=N {
                for j in 0..=(N - i) {
                    let (bu, bv) = (i as f32 / N as f32, j as f32 / N as f32);
                    let q = a + (b - a) * bu + (c - a) * bv;
                    best = best.min((p - q).length_squared());
                }
            }
            best
        })
        .fold(f32::INFINITY, f32::min)
}

/// The wide leaf walk reads whole registers, so its last chunk runs past the leaf into the next
/// leaf's triangles (or, at the end, into the repeated padding). Those are real faces of the
/// mesh, so the answer must be unchanged — checked here against an exhaustive scan, over meshes
/// whose leaves are deliberately not a whole number of lanes wide.
#[test]
fn bvh_ray_matches_exhaustive_scan_across_leaf_widths() {
    let mut seed = 0x1234_5678u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };

    // Face counts spanning every remainder against any lane width up to sixteen.
    for drop in 0..17 {
        let full = tesh::primitives::icosphere(1.0, 2);
        let faces: Vec<[u32; 3]> = full
            .faces()
            .iter()
            .take(full.faces().len() - drop)
            .copied()
            .collect();
        let mesh = TriMesh::new(full.vertices().to_vec(), faces);

        for _ in 0..400 {
            let origin = Vec3::new(next(), next(), next()) * 3.0;
            let target = Vec3::new(next(), next(), next());
            let dir = (target - origin).normalize();

            let got = mesh.ray_intersect(origin, dir);
            let want = nearest_ref(&mesh, origin, dir);
            match (got, want) {
                (None, None) => {}
                (Some(h), Some(t)) => {
                    assert!(
                        (h.t - t).abs() <= 1e-4 * t.abs().max(1.0),
                        "drop {drop}: t {} vs {t}",
                        h.t
                    );
                    // The reported face must be the one that actually produced that `t`.
                    let tri = mesh.triangle(h.face as usize);
                    let own = ray_tri_ref(origin, dir, tri).expect("reported face is hit");
                    assert!((own - h.t).abs() <= 1e-4 * own.abs().max(1.0));
                    assert!(h.point.abs_diff_eq(origin + dir * h.t, 1e-4));
                }
                _ => panic!("drop {drop}: {got:?} vs {want:?}"),
            }
        }
    }
}

/// The same over-read applies to the closest-point walk, where a stray lane would show up as a
/// distance that no triangle actually realises.
#[test]
fn bvh_closest_point_matches_exhaustive_scan_across_leaf_widths() {
    let mut seed = 0x9E37_79B9u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };

    for drop in 0..17 {
        let full = tesh::primitives::icosphere(1.0, 2);
        let faces: Vec<[u32; 3]> = full
            .faces()
            .iter()
            .take(full.faces().len() - drop)
            .copied()
            .collect();
        let mesh = TriMesh::new(full.vertices().to_vec(), faces);

        for _ in 0..60 {
            let p = Vec3::new(next(), next(), next()) * 2.0;
            let got = mesh.distance(p);
            let bound = dist2_ref(&mesh, p).sqrt();
            // The sampled reference is an upper bound on the true distance and cannot be beaten
            // by more than the sampling step.
            assert!(
                got <= bound + 1e-3,
                "drop {drop}: bvh distance {got} exceeds sampled bound {bound} at {p}"
            );
            assert!(
                (p - mesh.closest_point(p)).length() - got < 1e-4,
                "drop {drop}: closest_point disagrees with distance"
            );
        }
    }
}

/// Containment counts ray crossings rather than ranking them, so it is the one walk whose lanes
/// past the leaf must be masked back out. An unmasked lane would double-count a crossing and
/// flip the parity.
#[test]
fn bvh_containment_matches_winding_number() {
    use core::f32::consts::PI;

    fn winding_inside(mesh: &TriMesh, p: Vec3) -> bool {
        let mut omega = 0.0;
        for [a, b, c] in mesh.triangles() {
            let (a, b, c) = (a - p, b - p, c - p);
            let (la, lb, lc) = (a.length(), b.length(), c.length());
            let numer = a.dot(b.cross(c));
            let denom = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
            omega += 2.0 * numer.atan2(denom);
        }
        (omega / (4.0 * PI)).abs() > 0.5
    }

    let mut seed = 0xDEAD_BEEFu64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };

    let meshes = [
        tesh::primitives::icosphere(1.0, 1),
        tesh::primitives::icosphere(1.0, 2),
        tesh::primitives::cube(1.4),
        tesh::primitives::torus(1.0, 0.35, 24, 12),
        tesh::primitives::cylinder(0.7, 1.6, 21),
    ];
    for mesh in &meshes {
        for _ in 0..300 {
            let p = Vec3::new(next(), next(), next()) * 1.4;
            // Skip points within sampling noise of the surface, where the two tests are
            // legitimately allowed to disagree.
            if mesh.distance(p) < 1e-3 {
                continue;
            }
            assert_eq!(
                mesh.contains_point(p),
                winding_inside(mesh, p),
                "containment disagrees at {p}"
            );
        }
    }
}
