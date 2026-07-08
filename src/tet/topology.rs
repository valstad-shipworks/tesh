use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::{TET_FACES, TetMesh};
use crate::internal::{UnionFind, edge_key};
use crate::tri::Components;

const TET_EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

impl TetMesh {
    /// The unique undirected edges of the tetrahedralization, each `[min, max]`.
    #[must_use]
    pub fn edges_unique(&self) -> Vec<[u32; 2]> {
        let mut set: BTreeSet<(u32, u32)> = BTreeSet::new();
        for tet in &self.tets {
            for (i, j) in TET_EDGES {
                set.insert(edge_key(tet[i], tet[j]));
            }
        }
        set.into_iter().map(|(a, b)| [a, b]).collect()
    }

    /// Every triangular face mapped to the tetrahedra that use it.
    fn face_tets(&self) -> BTreeMap<[u32; 3], Vec<u32>> {
        let mut map: BTreeMap<[u32; 3], Vec<u32>> = BTreeMap::new();
        for (ti, tet) in self.tets.iter().enumerate() {
            for (idx, _apex) in TET_FACES {
                let mut key = [tet[idx[0]], tet[idx[1]], tet[idx[2]]];
                key.sort_unstable();
                map.entry(key).or_default().push(ti as u32);
            }
        }
        map
    }

    /// Every pair of tetrahedra sharing a triangular face; a non-manifold face with `k`
    /// tetrahedra contributes all `k·(k−1)/2` pairs.
    #[must_use]
    pub fn tet_adjacency(&self) -> Vec<[u32; 2]> {
        let mut pairs = Vec::new();
        for ts in self.face_tets().values() {
            for (i, &a) in ts.iter().enumerate() {
                for &b in &ts[i + 1..] {
                    pairs.push([a, b]);
                }
            }
        }
        pairs
    }

    /// Labels each tetrahedron by its connected component (joined through shared faces).
    #[must_use]
    pub fn connected_components(&self) -> Components {
        let mut uf = UnionFind::new(self.tets.len());
        for ts in self.face_tets().values() {
            for w in ts.windows(2) {
                uf.union(w[0], w[1]);
            }
        }
        crate::internal::compact_labels(&mut uf, self.tets.len())
    }

    /// Number of connected components (bodies).
    #[inline]
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.connected_components().count
    }
}
