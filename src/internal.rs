use alloc::vec;
use alloc::vec::Vec;

/// Undirected edge key, endpoints in ascending order.
#[inline]
pub(crate) fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Linear-scan nearest vertex, as `(index, squared_distance)`; the fallback behind the
/// mesh-level query when the `kdtree` feature is off.
#[cfg(not(feature = "kdtree"))]
pub(crate) fn nearest_vertex_brute(
    vertices: &[glam::Vec3],
    p: glam::Vec3,
) -> Option<(u32, f32)> {
    vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v.distance_squared(p)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

/// Dense root-to-label compaction of union-find results: labels number the components
/// `0..count` in first-appearance order.
pub(crate) fn compact_labels(uf: &mut UnionFind, n: usize) -> crate::tri::Components {
    let mut label_of = vec![u32::MAX; n];
    let mut labels = vec![0u32; n];
    let mut count = 0u32;
    for (i, label) in labels.iter_mut().enumerate() {
        let root = uf.find(i as u32) as usize;
        if label_of[root] == u32::MAX {
            label_of[root] = count;
            count += 1;
        }
        *label = label_of[root];
    }
    crate::tri::Components {
        labels,
        count: count as usize,
    }
}

/// Union-find with path halving and union by rank, over `0..n`.
pub(crate) struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    pub(crate) fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let g = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = g;
            x = g;
        }
        x
    }

    pub(crate) fn union(&mut self, a: u32, b: u32) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.rank[ra as usize] < self.rank[rb as usize] {
            core::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        if self.rank[ra as usize] == self.rank[rb as usize] {
            self.rank[ra as usize] += 1;
        }
    }
}
