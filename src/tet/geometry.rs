use alloc::vec::Vec;
use glam::Vec3;

use crate::compute;
use crate::tet::TetMesh;
use crate::tetrahedron;
use crate::traits::Mesh;

impl TetMesh {
    /// Total volume (sum of absolute per-element volumes).
    #[must_use]
    pub fn volume(&self) -> f32 {
        self.signed_volumes().iter().map(|v| v.abs()).sum()
    }

    /// Per-tetrahedron signed volumes (SPMD kernel). A negative entry flags an inverted
    /// element.
    #[must_use]
    pub fn signed_volumes(&self) -> Vec<f32> {
        compute::tet_signed_volumes(&self.vertices, &self.tets)
    }

    /// Signed volume of tetrahedron `i`.
    #[inline]
    #[must_use]
    pub fn tet_signed_volume(&self, i: usize) -> f32 {
        let [a, b, c, d] = self.tetrahedron(i);
        tetrahedron::signed_volume(a, b, c, d)
    }

    /// Centroid of tetrahedron `i`.
    #[inline]
    #[must_use]
    pub fn tet_centroid(&self, i: usize) -> Vec3 {
        let [a, b, c, d] = self.tetrahedron(i);
        tetrahedron::centroid(a, b, c, d)
    }

    /// Volume-weighted centroid (the solid's center of mass at uniform density).
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        let vols = self.signed_volumes();
        let mut sum = Vec3::ZERO;
        let mut total = 0.0;
        for (i, sv) in vols.iter().enumerate() {
            let v = sv.abs();
            sum += self.tet_centroid(i) * v;
            total += v;
        }
        if total > 0.0 { sum / total } else { self.vertex_centroid() }
    }

    /// Radius-ratio quality of tetrahedron `i`, in `[0, 1]` (`1` = regular, `0` = degenerate).
    #[inline]
    #[must_use]
    pub fn quality(&self, i: usize) -> f32 {
        let [a, b, c, d] = self.tetrahedron(i);
        tetrahedron::quality(a, b, c, d)
    }

    /// Per-tetrahedron radius-ratio qualities.
    #[must_use]
    pub fn qualities(&self) -> Vec<f32> {
        (0..self.tets.len()).map(|i| self.quality(i)).collect()
    }

    /// Volume, center of mass, and inertia tensor of the solid at unit density, computed
    /// from the boundary surface.
    #[must_use]
    pub fn mass_properties(&self) -> crate::MassProperties {
        #[cfg(feature = "bvh")]
        {
            crate::mass::from_boundary(self.surface_cached().triangles())
        }
        #[cfg(not(feature = "bvh"))]
        {
            crate::mass::from_boundary(
                self.boundary_faces()
                    .into_iter()
                    .map(|f| f.map(|i| self.vertices[i as usize])),
            )
        }
    }

    /// Constant shape-function gradients `∇Nᵢ`, four per element: for corners `x₀..x₃`
    /// with `D = [x₀−x₃ | x₁−x₃ | x₂−x₃]`, the rows of `D⁻¹` and `∇N₃ = −(∇N₀+∇N₁+∇N₂)`.
    /// A degenerate element yields zeros; screen with
    /// [`signed_volumes`](TetMesh::signed_volumes).
    #[must_use]
    pub fn shape_gradients(&self) -> Vec<[Vec3; 4]> {
        self.tetrahedra()
            .map(|[a, b, c, d]| {
                let m = glam::Mat3::from_cols(a - d, b - d, c - d);
                if m.determinant().abs() <= f32::MIN_POSITIVE {
                    return [Vec3::ZERO; 4];
                }
                let inv = m.inverse();
                let g0 = Vec3::new(inv.x_axis.x, inv.y_axis.x, inv.z_axis.x);
                let g1 = Vec3::new(inv.x_axis.y, inv.y_axis.y, inv.z_axis.y);
                let g2 = Vec3::new(inv.x_axis.z, inv.y_axis.z, inv.z_axis.z);
                [g0, g1, g2, -(g0 + g1 + g2)]
            })
            .collect()
    }
}
