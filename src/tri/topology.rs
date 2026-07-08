use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use crate::internal::{UnionFind, edge_key};
use crate::tri::TriMesh;

/// Connected-component (body) labelling of a mesh: one label per element (per face for a
/// [`TriMesh`], per tetrahedron for a [`TetMesh`](crate::TetMesh)).
#[derive(Debug, Clone)]
pub struct Components {
    /// Component id of each element, in `0..count`.
    pub labels: Vec<u32>,
    /// Number of distinct components.
    pub count: usize,
}


impl TriMesh {
    /// Every undirected edge mapped to the faces that use it.
    pub(crate) fn edge_faces(&self) -> BTreeMap<(u32, u32), Vec<u32>> {
        let mut map: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();
        for (fi, f) in self.faces.iter().enumerate() {
            for k in 0..3 {
                let key = edge_key(f[k], f[(k + 1) % 3]);
                map.entry(key).or_default().push(fi as u32);
            }
        }
        map
    }

    /// The unique undirected edges as a set, without the per-edge face lists.
    fn edge_set(&self) -> BTreeSet<(u32, u32)> {
        let mut set = BTreeSet::new();
        for f in &self.faces {
            for k in 0..3 {
                set.insert(edge_key(f[k], f[(k + 1) % 3]));
            }
        }
        set
    }

    /// The unique undirected edges, each `[min, max]`, in sorted order.
    #[must_use]
    pub fn edges_unique(&self) -> Vec<[u32; 2]> {
        self.edge_set().into_iter().map(|(a, b)| [a, b]).collect()
    }

    /// Edges used by exactly one face — the open boundary of the surface.
    #[must_use]
    pub fn boundary_edges(&self) -> Vec<[u32; 2]> {
        self.edge_faces()
            .into_iter()
            .filter(|(_, fs)| fs.len() == 1)
            .map(|((a, b), _)| [a, b])
            .collect()
    }

    /// Whether the surface is a closed 2-manifold: every edge shared by exactly two faces.
    #[must_use]
    pub fn is_watertight(&self) -> bool {
        !self.faces.is_empty() && self.edge_faces().values().all(|fs| fs.len() == 2)
    }

    /// Every pair of faces sharing an edge; a non-manifold edge with `k` faces contributes
    /// all `k·(k−1)/2` pairs.
    #[must_use]
    pub fn face_adjacency(&self) -> Vec<[u32; 2]> {
        let mut pairs = Vec::new();
        for fs in self.edge_faces().values() {
            for (i, &a) in fs.iter().enumerate() {
                for &b in &fs[i + 1..] {
                    pairs.push([a, b]);
                }
            }
        }
        pairs
    }

    /// For each vertex, the sorted list of vertices it shares an edge with.
    #[must_use]
    pub fn vertex_adjacency(&self) -> Vec<Vec<u32>> {
        let mut adj = vec![Vec::new(); self.vertices.len()];
        for (a, b) in self.edge_set() {
            adj[a as usize].push(b);
            adj[b as usize].push(a);
        }
        adj
    }

    /// Euler characteristic `V - E + F`.
    #[must_use]
    pub fn euler_number(&self) -> i64 {
        self.vertices.len() as i64 - self.edge_set().len() as i64 + self.faces.len() as i64
    }

    /// Labels each face by its connected component (bodies joined through shared edges).
    #[must_use]
    pub fn connected_components(&self) -> Components {
        let mut uf = UnionFind::new(self.faces.len());
        for fs in self.edge_faces().values() {
            for w in fs.windows(2) {
                uf.union(w[0], w[1]);
            }
        }
        crate::internal::compact_labels(&mut uf, self.faces.len())
    }

    /// Number of connected components (bodies).
    #[inline]
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.connected_components().count
    }

    /// Splits the mesh into one [`TriMesh`] per connected component, each with its own
    /// compacted vertex buffer.
    #[must_use]
    pub fn split(&self) -> Vec<TriMesh> {
        let comp = self.connected_components();
        let mut meshes = vec![TriMesh::default(); comp.count];
        let mut remap: Vec<BTreeMap<u32, u32>> = vec![BTreeMap::new(); comp.count];

        for (fi, f) in self.faces.iter().enumerate() {
            let c = comp.labels[fi] as usize;
            let mesh = &mut meshes[c];
            let map = &mut remap[c];
            let mut nf = [0u32; 3];
            for (j, &vi) in f.iter().enumerate() {
                nf[j] = *map.entry(vi).or_insert_with(|| {
                    let idx = mesh.vertices.len() as u32;
                    mesh.vertices.push(self.vertices[vi as usize]);
                    idx
                });
            }
            mesh.faces.push(nf);
        }
        meshes
    }
}
