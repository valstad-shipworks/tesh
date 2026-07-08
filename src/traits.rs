use glam::{Affine3A, Vec3};

use crate::aabb::Aabb;
use crate::sealed::Sealed;

/// Behaviour shared by every mesh: indexed vertices connected into fixed-arity elements
/// (triangles for [`TriMesh`](crate::TriMesh), tetrahedra for [`TetMesh`](crate::TetMesh)).
///
/// The trait is sealed; the two concrete meshes are the only implementors.
pub trait Mesh: Sealed {
    /// Vertices per element: `3` for a triangle mesh, `4` for a tetrahedral mesh.
    const VERTS_PER_ELEM: usize;

    /// The mesh vertices, indexed by the element connectivity.
    fn vertices(&self) -> &[Vec3];

    /// Mutable access to the vertices; the connectivity is left untouched. Taking this
    /// discards any cached query acceleration structures, since the geometry may change.
    #[must_use]
    fn vertices_mut(&mut self) -> &mut [Vec3];

    /// Number of elements (triangles or tetrahedra).
    fn element_count(&self) -> usize;

    /// Vertex indices of element `i`, a slice of length [`VERTS_PER_ELEM`](Mesh::VERTS_PER_ELEM).
    ///
    /// # Panics
    /// Panics if `i` is out of range.
    fn element(&self, i: usize) -> &[u32];

    /// Number of vertices.
    #[inline]
    #[must_use]
    fn vertex_count(&self) -> usize {
        self.vertices().len()
    }

    /// Whether the mesh has no elements.
    #[inline]
    #[must_use]
    fn is_empty(&self) -> bool {
        self.element_count() == 0
    }

    /// Unweighted average of the vertex positions.
    ///
    /// This is the centroid of the *vertices*, not of the surface or volume; the
    /// area- and volume-weighted centroids live on the concrete types.
    #[inline]
    #[must_use]
    fn vertex_centroid(&self) -> Vec3 {
        let verts = self.vertices();
        if verts.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = verts.iter().copied().sum();
        sum / verts.len() as f32
    }
}

/// The axis-aligned bounding box of a mesh's vertices.
pub trait Bounds {
    #[must_use]
    fn bounds(&self) -> Aabb;
}

impl<M: Mesh> Bounds for M {
    #[inline]
    fn bounds(&self) -> Aabb {
        Aabb::from_points(self.vertices())
    }
}

/// Rigid or affine transformation of a mesh, applied to every vertex in place.
pub trait Transform: Sized {
    /// Transforms every vertex by `m`.
    fn apply_transform(&mut self, m: Affine3A);

    /// Consumes the mesh and returns it transformed by `m`.
    #[inline]
    #[must_use]
    fn transformed(mut self, m: Affine3A) -> Self {
        self.apply_transform(m);
        self
    }
}

impl<M: Mesh> Transform for M {
    #[inline]
    fn apply_transform(&mut self, m: Affine3A) {
        for v in self.vertices_mut() {
            *v = m.transform_point3(*v);
        }
    }
}
