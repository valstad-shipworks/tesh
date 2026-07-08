//! The [`TriMesh`]: an indexed triangle surface mesh.

use alloc::vec::Vec;
use glam::Vec3;

use crate::sealed::Sealed;
use crate::traits::Mesh;

mod geometry;
mod query;
mod repair;
mod topology;

pub use query::{RayHit, SurfaceHit};
pub use topology::Components;

/// An indexed triangle mesh: a shared vertex buffer plus triangles referencing it by index.
///
/// The winding of each face fixes its outward normal (counter-clockwise seen from outside).
/// Read the buffers through [`vertices`](TriMesh::vertices) and [`faces`](TriMesh::faces);
/// build one with [`TriMesh::new`]. Spatial queries lazily cache an acceleration structure
/// keyed to the geometry, so the connectivity cannot be edited in place — edit vertex
/// positions through [`vertices_mut`](Mesh::vertices_mut), which drops the stale cache, and
/// rebuild with [`TriMesh::new`] to change the topology.
///
/// # Panics
/// Methods index through the connectivity unchecked: an out-of-range face index panics.
/// Constructors `debug_assert` [`is_valid`](TriMesh::is_valid); deserialized meshes are used
/// as-is.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TriMesh {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) faces: Vec<[u32; 3]>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) cache: crate::cache::TriCache,
}

impl TriMesh {
    /// Builds a mesh from a vertex buffer and triangle connectivity.
    #[inline]
    #[must_use]
    pub fn new(vertices: Vec<Vec3>, faces: Vec<[u32; 3]>) -> Self {
        let mesh = Self {
            vertices,
            faces,
            cache: crate::cache::TriCache::default(),
        };
        debug_assert!(mesh.is_valid(), "face index out of range");
        mesh
    }

    /// Builds a mesh from loader/renderer-layout buffers: one `[x, y, z]` per vertex,
    /// three indices per face.
    ///
    /// # Panics
    /// Panics if `indices.len()` is not a multiple of 3.
    #[must_use]
    pub fn from_flat(positions: &[[f32; 3]], indices: &[u32]) -> Self {
        assert!(
            indices.len().is_multiple_of(3),
            "index buffer length {} is not a multiple of 3",
            indices.len()
        );
        Self::new(
            positions.iter().map(|&p| Vec3::from_array(p)).collect(),
            indices.chunks_exact(3).map(|f| [f[0], f[1], f[2]]).collect(),
        )
    }

    /// Builds a mesh from triangle soup: each triangle gets its own three vertices. Follow
    /// with [`merge_vertices`](TriMesh::merge_vertices) to weld shared corners.
    #[must_use]
    pub fn from_triangles(triangles: impl IntoIterator<Item = [Vec3; 3]>) -> Self {
        let mut mesh = Self::default();
        for tri in triangles {
            let base = mesh.vertices.len() as u32;
            mesh.vertices.extend(tri);
            mesh.faces.push([base, base + 1, base + 2]);
        }
        mesh
    }

    /// Appends another mesh's geometry to this one, offsetting its face indices past this
    /// mesh's vertex buffer.
    pub fn append(&mut self, other: &TriMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.faces
            .extend(other.faces.iter().map(|f| f.map(|i| i + base)));
        self.reset_cache();
    }

    /// The mesh vertices, indexed by the face connectivity.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// The triangle connectivity: three vertex indices per face.
    #[inline]
    #[must_use]
    pub fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }

    /// The face indices flattened to three `u32` per triangle.
    #[must_use]
    pub fn flat_indices(&self) -> Vec<u32> {
        self.faces.iter().flatten().copied().collect()
    }

    #[inline]
    pub(crate) fn reset_cache(&mut self) {
        self.cache = crate::cache::TriCache::default();
    }

    /// Number of triangles.
    #[inline]
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// The three corner positions of face `i`.
    ///
    /// # Panics
    /// Panics if `i` or a face index is out of range.
    #[inline]
    #[must_use]
    pub fn triangle(&self, i: usize) -> [Vec3; 3] {
        let [a, b, c] = self.faces[i];
        [
            self.vertices[a as usize],
            self.vertices[b as usize],
            self.vertices[c as usize],
        ]
    }

    /// Iterator over every triangle's corner positions.
    #[inline]
    pub fn triangles(&self) -> impl Iterator<Item = [Vec3; 3]> + '_ {
        (0..self.faces.len()).map(move |i| self.triangle(i))
    }

    /// Whether every face index is within the vertex buffer.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let n = self.vertices.len() as u32;
        self.faces.iter().flatten().all(|&i| i < n)
    }

    /// The vertex nearest to `p`, as `(index, squared_distance)`, or `None` for a mesh with
    /// no vertices. Uses the cached k-d tree under the `kdtree` feature, else a linear scan.
    #[must_use]
    pub fn nearest_vertex(&self, p: Vec3) -> Option<(u32, f32)> {
        #[cfg(feature = "kdtree")]
        {
            self.kdtree().nearest_vertex(p)
        }
        #[cfg(not(feature = "kdtree"))]
        {
            crate::internal::nearest_vertex_brute(&self.vertices, p)
        }
    }

    /// The cached [`VertexKdTree`](crate::accel::VertexKdTree) over the vertices, built on
    /// first use. Reused by the nearest-vertex queries.
    #[cfg(feature = "kdtree")]
    #[must_use]
    pub fn kdtree(&self) -> &crate::accel::VertexKdTree {
        self.cache
            .kdtree
            .get_or_init(|| crate::accel::VertexKdTree::new(&self.vertices))
    }

    /// The cached [`TriBvh`](crate::accel::TriBvh) over the faces, built on first use.
    /// Reused by the ray and closest-point queries.
    #[cfg(feature = "bvh")]
    #[must_use]
    pub fn bvh(&self) -> &crate::accel::TriBvh {
        self.cache
            .bvh
            .get_or_init(|| crate::accel::TriBvh::new(self))
    }
}

impl Sealed for TriMesh {}

impl Mesh for TriMesh {
    const VERTS_PER_ELEM: usize = 3;

    #[inline]
    fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    #[inline]
    fn vertices_mut(&mut self) -> &mut [Vec3] {
        self.reset_cache();
        &mut self.vertices
    }

    #[inline]
    fn element_count(&self) -> usize {
        self.faces.len()
    }

    #[inline]
    fn element(&self, i: usize) -> &[u32] {
        &self.faces[i]
    }
}
