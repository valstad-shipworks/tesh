use alloc::vec::Vec;
use glam::Vec3;

use crate::tet::TetMesh;
use crate::{tetrahedron, triangle};

/// A ray hit against the boundary of a [`TetMesh`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetRayHit {
    /// The tetrahedron owning the boundary face that was hit.
    pub tet: u32,
    /// Local face of [`tet`](TetRayHit::tet) that was hit: the face opposite corner `face`.
    pub face: u8,
    /// Ray parameter of the hit.
    pub t: f32,
    /// World-space hit point, `origin + t * dir`.
    pub point: Vec3,
    /// Unit outward normal of the hit face: `dir · normal < 0` where the ray enters the
    /// volume, `> 0` where it leaves.
    pub normal: Vec3,
}

impl TetRayHit {
    #[inline]
    fn new(owner: u32, t: f32, point: Vec3, [a, b, c]: [Vec3; 3]) -> Self {
        Self {
            tet: owner / 4,
            face: (owner % 4) as u8,
            t,
            point,
            normal: triangle::normal(a, b, c),
        }
    }
}

impl TetMesh {
    /// Index of a tetrahedron containing `p`, or `None` if `p` is outside the volume.
    #[cfg(feature = "bvh")]
    #[must_use]
    pub fn find_containing_tet(&self, p: Vec3) -> Option<usize> {
        self.tet_bvh().find_containing_tet(p)
    }

    /// Index of a tetrahedron containing `p`, or `None` if `p` is outside the volume
    /// (brute force over elements).
    #[cfg(not(feature = "bvh"))]
    #[must_use]
    pub fn find_containing_tet(&self, p: Vec3) -> Option<usize> {
        (0..self.tets.len()).find(|&i| {
            let [a, b, c, d] = self.tetrahedron(i);
            tetrahedron::contains_point(p, a, b, c, d)
        })
    }

    /// Whether `p` lies inside the volume.
    #[inline]
    #[must_use]
    pub fn contains_point(&self, p: Vec3) -> bool {
        self.find_containing_tet(p).is_some()
    }

    /// Barycentric coordinates of `p` within tetrahedron `i`.
    ///
    /// # Panics
    /// Panics if `i` is out of range.
    #[inline]
    #[must_use]
    pub fn barycentric(&self, i: usize, p: Vec3) -> [f32; 4] {
        let [a, b, c, d] = self.tetrahedron(i);
        tetrahedron::barycentric(p, a, b, c, d)
    }

    /// Linearly interpolates a per-vertex scalar field at `p`, or `None` if `p` is outside
    /// the volume. `values` is indexed in parallel with the vertex buffer.
    ///
    /// # Panics
    /// Panics if `values` is shorter than the vertex buffer.
    #[must_use]
    pub fn interpolate(&self, p: Vec3, values: &[f32]) -> Option<f32> {
        let i = self.find_containing_tet(p)?;
        let bary = self.barycentric(i, p);
        let tet = self.tets[i];
        Some(
            bary[0] * values[tet[0] as usize]
                + bary[1] * values[tet[1] as usize]
                + bary[2] * values[tet[2] as usize]
                + bary[3] * values[tet[3] as usize],
        )
    }

    /// Nearest crossing of the ray `origin + t * dir` (`t >= 0`) with the volume's boundary.
    /// From an origin inside the volume this is where the ray leaves it (check
    /// [`TetRayHit::normal`] against `dir`, or [`contains_point`](TetMesh::contains_point)).
    ///
    /// Traverses the BVH over the cached boundary surface; the first call extracts the
    /// boundary and builds the tree.
    #[cfg(feature = "bvh")]
    #[must_use]
    pub fn ray_intersect(&self, origin: Vec3, dir: Vec3) -> Option<TetRayHit> {
        let boundary = self.boundary_cached();
        let hit = boundary.mesh.bvh().ray_intersect(origin, dir)?;
        let face = hit.face as usize;
        Some(TetRayHit::new(
            boundary.owners[face],
            hit.t,
            hit.point,
            boundary.mesh.triangle(face),
        ))
    }

    /// Nearest crossing of the ray `origin + t * dir` (`t >= 0`) with the volume's boundary.
    /// From an origin inside the volume this is where the ray leaves it. This extracts the
    /// boundary each call; enable the `bvh` feature to cache it.
    #[cfg(not(feature = "bvh"))]
    #[must_use]
    pub fn ray_intersect(&self, origin: Vec3, dir: Vec3) -> Option<TetRayHit> {
        let mut best: Option<(f32, u32, [Vec3; 3])> = None;
        for (tri, owner) in self.boundary_with_owners() {
            let [a, b, c] = tri.map(|i| self.vertices[i as usize]);
            if let Some(t) = triangle::ray_intersect(origin, dir, a, b, c)
                && best.is_none_or(|(bt, ..)| t < bt)
            {
                best = Some((t, owner, [a, b, c]));
            }
        }
        best.map(|(t, owner, tri)| TetRayHit::new(owner, t, origin + dir * t, tri))
    }

    /// Nearest boundary crossing for each `(origin, dir)` ray (see
    /// [`ray_intersect`](TetMesh::ray_intersect)).
    #[cfg(feature = "bvh")]
    #[must_use]
    pub fn ray_intersects(&self, rays: &[(Vec3, Vec3)]) -> Vec<Option<TetRayHit>> {
        rays.iter()
            .map(|&(origin, dir)| self.ray_intersect(origin, dir))
            .collect()
    }

    /// Nearest boundary crossing for each `(origin, dir)` ray, extracting the boundary once
    /// for the whole batch.
    #[cfg(not(feature = "bvh"))]
    #[must_use]
    pub fn ray_intersects(&self, rays: &[(Vec3, Vec3)]) -> Vec<Option<TetRayHit>> {
        let (faces, owners): (Vec<[u32; 3]>, Vec<u32>) =
            self.boundary_with_owners().into_iter().unzip();
        let surface = crate::TriMesh::new(self.vertices.clone(), faces);
        rays.iter()
            .map(|&(origin, dir)| {
                let hit = surface.ray_intersect(origin, dir)?;
                let face = hit.face as usize;
                Some(TetRayHit::new(
                    owners[face],
                    hit.t,
                    hit.point,
                    surface.triangle(face),
                ))
            })
            .collect()
    }

    /// Closest point on the boundary surface to `p`, using the cached boundary mesh.
    #[cfg(feature = "bvh")]
    #[must_use]
    pub fn closest_surface_point(&self, p: Vec3) -> Vec3 {
        self.surface_cached().closest_point(p)
    }

    /// Closest point on the boundary surface to `p`. This extracts the boundary each call;
    /// enable the `bvh` feature to cache it.
    #[cfg(not(feature = "bvh"))]
    #[must_use]
    pub fn closest_surface_point(&self, p: Vec3) -> Vec3 {
        self.surface().closest_point(p)
    }
}
