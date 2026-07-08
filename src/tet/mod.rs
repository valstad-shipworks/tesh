//! The [`TetMesh`]: an indexed tetrahedron volume mesh.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use glam::Vec3;

use crate::sealed::Sealed;
use crate::traits::Mesh;
use crate::tri::TriMesh;

mod geometry;
mod query;
mod topology;

/// The four faces of a tetrahedron `[a, b, c, d]`, each as (triangle, opposite apex).
const TET_FACES: [([usize; 3], usize); 4] = [
    ([1, 2, 3], 0),
    ([0, 2, 3], 1),
    ([0, 1, 3], 2),
    ([0, 1, 2], 3),
];

/// One boundary face of a [`TetMesh`], with its geometry: outward-wound vertex indices into
/// the volume's vertex buffer, area, and unit outward normal.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundaryFace {
    pub nodes: [u32; 3],
    pub area: f32,
    pub normal: Vec3,
}

/// An indexed tetrahedral mesh: a shared vertex buffer plus tetrahedra referencing it by
/// index. Its boundary is a [`TriMesh`] recovered by [`TetMesh::surface`].
///
/// Read the buffers through [`vertices`](TetMesh::vertices) and [`tets`](TetMesh::tets); build
/// one with [`TetMesh::new`]. Spatial queries lazily cache acceleration structures keyed to
/// the geometry, so the connectivity cannot be edited in place — edit vertex positions through
/// [`vertices_mut`](Mesh::vertices_mut), which drops the stale cache, and rebuild with
/// [`TetMesh::new`] to change the topology.
///
/// # Panics
/// Methods index through the connectivity unchecked: an out-of-range tet index panics.
/// Constructors `debug_assert` [`is_valid`](TetMesh::is_valid); deserialized meshes are used
/// as-is.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TetMesh {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) tets: Vec<[u32; 4]>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) cache: crate::cache::TetCache,
}

impl TetMesh {
    /// Builds a mesh from a vertex buffer and tetrahedron connectivity.
    #[inline]
    #[must_use]
    pub fn new(vertices: Vec<Vec3>, tets: Vec<[u32; 4]>) -> Self {
        let mesh = Self {
            vertices,
            tets,
            cache: crate::cache::TetCache::default(),
        };
        debug_assert!(mesh.is_valid(), "tet index out of range");
        mesh
    }

    /// The mesh vertices, indexed by the tetrahedron connectivity.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// The tetrahedron connectivity: four vertex indices per element.
    #[inline]
    #[must_use]
    pub fn tets(&self) -> &[[u32; 4]] {
        &self.tets
    }

    #[inline]
    pub(crate) fn reset_cache(&mut self) {
        self.cache = crate::cache::TetCache::default();
    }

    /// Number of tetrahedra.
    #[inline]
    #[must_use]
    pub fn tet_count(&self) -> usize {
        self.tets.len()
    }

    /// The four corner positions of tetrahedron `i`.
    ///
    /// # Panics
    /// Panics if `i` or a tet index is out of range.
    #[inline]
    #[must_use]
    pub fn tetrahedron(&self, i: usize) -> [Vec3; 4] {
        let [a, b, c, d] = self.tets[i];
        [
            self.vertices[a as usize],
            self.vertices[b as usize],
            self.vertices[c as usize],
            self.vertices[d as usize],
        ]
    }

    /// Iterator over every tetrahedron's corner positions.
    #[inline]
    pub fn tetrahedra(&self) -> impl Iterator<Item = [Vec3; 4]> + '_ {
        (0..self.tets.len()).map(move |i| self.tetrahedron(i))
    }

    /// Whether every tet index is within the vertex buffer.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let n = self.vertices.len() as u32;
        self.tets.iter().flatten().all(|&i| i < n)
    }

    /// Winds triangle `tri` so its normal points away from `apex`.
    fn orient_outward(&self, tri: [u32; 3], apex: u32) -> [u32; 3] {
        let x = self.vertices[tri[0] as usize];
        let y = self.vertices[tri[1] as usize];
        let z = self.vertices[tri[2] as usize];
        let normal = (y - x).cross(z - x);
        let outward = (x + y + z) * (1.0 / 3.0) - self.vertices[apex as usize];
        if normal.dot(outward) < 0.0 {
            [tri[0], tri[2], tri[1]]
        } else {
            tri
        }
    }

    /// The boundary faces of the volume — every triangular face belonging to exactly one
    /// tetrahedron — each wound so its normal points outward. Indices refer to this mesh's
    /// vertex buffer, so they line up with the tetrahedra.
    #[must_use]
    pub fn boundary_faces(&self) -> Vec<[u32; 3]> {
        let mut seen: BTreeMap<[u32; 3], (u32, [u32; 3], u32)> = BTreeMap::new();
        for tet in &self.tets {
            for (idx, apex) in TET_FACES {
                let tri = [tet[idx[0]], tet[idx[1]], tet[idx[2]]];
                let mut key = tri;
                key.sort_unstable();
                let entry = seen.entry(key).or_insert((0, tri, tet[apex]));
                entry.0 += 1;
            }
        }
        seen.into_values()
            .filter(|(count, ..)| *count == 1)
            .map(|(_, tri, apex)| self.orient_outward(tri, apex))
            .collect()
    }

    /// The boundary faces with their geometry: outward-wound vertex indices, area, and
    /// unit normal per face.
    #[must_use]
    pub fn boundary(&self) -> Vec<BoundaryFace> {
        self.boundary_faces()
            .into_iter()
            .map(|nodes| {
                let [a, b, c] = nodes.map(|i| self.vertices[i as usize]);
                let av = crate::triangle::area_vector(a, b, c);
                BoundaryFace {
                    nodes,
                    area: av.length() * 0.5,
                    normal: av.normalize_or_zero(),
                }
            })
            .collect()
    }

    /// The boundary of the volume as a [`TriMesh`]. The vertex buffer is a copy of this
    /// mesh's, so face indices correspond one-to-one with the volume's vertex indices.
    #[must_use]
    pub fn surface(&self) -> TriMesh {
        TriMesh::new(self.vertices.clone(), self.boundary_faces())
    }

    /// The boundary as a compact [`TriMesh`] holding only the vertices it uses, plus the
    /// map from each surface vertex back to its volume vertex index.
    #[must_use]
    pub fn surface_with_map(&self) -> (TriMesh, Vec<u32>) {
        let faces = self.boundary_faces();
        let mut volume_index = Vec::new();
        let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
        let compact_faces: Vec<[u32; 3]> = faces
            .iter()
            .map(|f| {
                f.map(|vi| {
                    *remap.entry(vi).or_insert_with(|| {
                        volume_index.push(vi);
                        volume_index.len() as u32 - 1
                    })
                })
            })
            .collect();
        let vertices = volume_index
            .iter()
            .map(|&vi| self.vertices[vi as usize])
            .collect();
        (TriMesh::new(vertices, compact_faces), volume_index)
    }

    /// Splits the mesh into one [`TetMesh`] per connected component, each with its own
    /// compacted vertex buffer.
    #[must_use]
    pub fn split(&self) -> Vec<TetMesh> {
        let comp = self.connected_components();
        let mut meshes = alloc::vec![TetMesh::default(); comp.count];
        let mut remap: Vec<BTreeMap<u32, u32>> = alloc::vec![BTreeMap::new(); comp.count];

        for (ti, t) in self.tets.iter().enumerate() {
            let c = comp.labels[ti] as usize;
            let mesh = &mut meshes[c];
            let map = &mut remap[c];
            let mut nt = [0u32; 4];
            for (j, &vi) in t.iter().enumerate() {
                nt[j] = *map.entry(vi).or_insert_with(|| {
                    let idx = mesh.vertices.len() as u32;
                    mesh.vertices.push(self.vertices[vi as usize]);
                    idx
                });
            }
            mesh.tets.push(nt);
        }
        meshes
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
    /// first use.
    #[cfg(feature = "kdtree")]
    #[must_use]
    pub fn kdtree(&self) -> &crate::accel::VertexKdTree {
        self.cache
            .kdtree
            .get_or_init(|| crate::accel::VertexKdTree::new(&self.vertices))
    }

    /// The cached boundary [`TriMesh`], extracted on first use.
    #[cfg(feature = "bvh")]
    #[must_use]
    pub(crate) fn surface_cached(&self) -> &TriMesh {
        self.cache.surface.get_or_init(|| self.surface())
    }

    /// The cached [`TetBvh`](crate::accel::TetBvh) over the elements, built on first use.
    /// Reused by point-location queries.
    #[cfg(feature = "bvh")]
    #[must_use]
    pub fn tet_bvh(&self) -> &crate::accel::TetBvh {
        self.cache
            .tet_bvh
            .get_or_init(|| crate::accel::TetBvh::new(self))
    }
}

impl Sealed for TetMesh {}

impl Mesh for TetMesh {
    const VERTS_PER_ELEM: usize = 4;

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
        self.tets.len()
    }

    #[inline]
    fn element(&self, i: usize) -> &[u32] {
        &self.tets[i]
    }
}
