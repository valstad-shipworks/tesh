use glam::Vec3;

use crate::tet::TetMesh;
use crate::tetrahedron;

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
