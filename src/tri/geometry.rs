use alloc::vec::Vec;
use glam::Vec3;

use crate::compute;
use crate::traits::Mesh;
use crate::triangle;
use crate::tri::TriMesh;

impl TriMesh {
    /// Total surface area.
    #[must_use]
    pub fn area(&self) -> f32 {
        // A scalar accumulation: summing to one float does not pay for the kernel's
        // per-face output buffers.
        self.triangles()
            .map(|[a, b, c]| triangle::area(a, b, c))
            .sum()
    }

    /// Area of face `i`.
    #[inline]
    #[must_use]
    pub fn face_area(&self, i: usize) -> f32 {
        let [a, b, c] = self.triangle(i);
        triangle::area(a, b, c)
    }

    /// Per-face areas (SPMD kernel).
    #[must_use]
    pub fn face_areas(&self) -> Vec<f32> {
        compute::face_areas(&self.vertices, &self.faces)
    }

    /// Unit normal of face `i`.
    #[inline]
    #[must_use]
    pub fn face_normal(&self, i: usize) -> Vec3 {
        let [a, b, c] = self.triangle(i);
        triangle::normal(a, b, c)
    }

    /// Per-face unit normals (SPMD kernel).
    #[must_use]
    pub fn face_normals(&self) -> Vec<Vec3> {
        compute::face_normals(&self.vertices, &self.faces)
    }

    /// Area-weighted per-vertex normals (SPMD kernel + scalar scatter).
    #[must_use]
    pub fn vertex_normals(&self) -> Vec<Vec3> {
        compute::vertex_normals(&self.vertices, &self.faces)
    }

    /// Centroid of face `i`.
    #[inline]
    #[must_use]
    pub fn face_centroid(&self, i: usize) -> Vec3 {
        let [a, b, c] = self.triangle(i);
        triangle::centroid(a, b, c)
    }

    /// Area-weighted centroid of the surface (the center of mass of a uniform shell).
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        let areas = self.face_areas();
        let mut sum = Vec3::ZERO;
        let mut total = 0.0;
        for (i, &a) in areas.iter().enumerate() {
            sum += self.face_centroid(i) * a;
            total += a;
        }
        if total > 0.0 { sum / total } else { self.vertex_centroid() }
    }

    /// Signed volume enclosed by the surface, via the divergence-theorem sum of signed
    /// tetrahedra from the origin. Meaningful only for a closed (watertight), consistently
    /// wound mesh; the sign is positive when the normals point outward.
    #[must_use]
    pub fn signed_volume(&self) -> f32 {
        let mut v = 0.0;
        for [a, b, c] in self.triangles() {
            v += a.dot(b.cross(c));
        }
        v / 6.0
    }

    /// Absolute enclosed volume (see [`signed_volume`](TriMesh::signed_volume)).
    #[inline]
    #[must_use]
    pub fn volume(&self) -> f32 {
        self.signed_volume().abs()
    }

    /// Volume, center of mass, and inertia tensor of the enclosed solid at unit density.
    /// Meaningful only for a closed (watertight), consistently wound surface.
    #[inline]
    #[must_use]
    pub fn mass_properties(&self) -> crate::MassProperties {
        crate::mass::from_boundary(self.triangles())
    }

    /// A deterministic surface sampling at roughly `spacing` density: vertices deduplicated
    /// on a grid of that spacing, plus midpoints of longer edges, in stable index order.
    #[must_use]
    pub fn surface_samples(&self, spacing: f32) -> Vec<Vec3> {
        use alloc::collections::BTreeSet;

        #[cfg(not(feature = "std"))]
        #[allow(unused_imports)]
        use crate::math::F32Ext;

        let inv = 1.0 / spacing.max(1e-6);
        let mut seen: BTreeSet<[i32; 3]> = BTreeSet::new();
        let mut out = Vec::new();
        let mut push = |p: Vec3, out: &mut Vec<Vec3>| {
            let cell = [
                (p.x * inv).round() as i32,
                (p.y * inv).round() as i32,
                (p.z * inv).round() as i32,
            ];
            if seen.insert(cell) {
                out.push(p);
            }
        };
        for &v in &self.vertices {
            push(v, &mut out);
        }
        for [a, b] in self.edges_unique() {
            let pa = self.vertices[a as usize];
            let pb = self.vertices[b as usize];
            if pa.distance_squared(pb) > spacing * spacing {
                push((pa + pb) * 0.5, &mut out);
            }
        }
        out
    }
}
