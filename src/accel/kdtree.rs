use alloc::vec::Vec;
use core::num::NonZero;

use glam::Vec3;
use kiddo::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;

/// An immutable k-d tree over a mesh's vertices, for nearest-neighbour and radius queries
/// (backed by [`kiddo`]).
///
/// Build it once from a vertex buffer, then query repeatedly. Results carry the vertex index
/// into the buffer the tree was built from; reported distances are **squared**.
pub struct VertexKdTree {
    tree: ImmutableKdTree<f32, u32, 3, 32>,
    len: usize,
}

impl VertexKdTree {
    /// Builds the tree from a vertex buffer.
    #[must_use]
    pub fn new(vertices: &[Vec3]) -> Self {
        let points: Vec<[f32; 3]> = vertices.iter().map(Vec3::to_array).collect();
        Self {
            tree: ImmutableKdTree::new_from_slice(&points),
            len: vertices.len(),
        }
    }

    /// The vertex nearest to `p`, as `(index, squared_distance)`, or `None` for a tree built
    /// over no vertices.
    #[must_use]
    pub fn nearest_vertex(&self, p: Vec3) -> Option<(u32, f32)> {
        if self.len == 0 {
            return None;
        }
        let nn = self.tree.nearest_one::<SquaredEuclidean>(&p.to_array());
        Some((nn.item, nn.distance))
    }

    /// The `k` vertices nearest to `p`, ascending by distance, as `(index, squared_distance)`.
    /// Empty if `k == 0`.
    #[must_use]
    pub fn k_nearest_vertices(&self, p: Vec3, k: usize) -> Vec<(u32, f32)> {
        let Some(k) = NonZero::new(k) else {
            return Vec::new();
        };
        if self.len == 0 {
            return Vec::new();
        }
        self.tree
            .nearest_n::<SquaredEuclidean>(&p.to_array(), k)
            .into_iter()
            .map(|n| (n.item, n.distance))
            .collect()
    }

    /// Indices of every vertex within `radius` of `p`.
    #[must_use]
    pub fn vertices_within(&self, p: Vec3, radius: f32) -> Vec<u32> {
        self.tree
            .within::<SquaredEuclidean>(&p.to_array(), radius * radius)
            .into_iter()
            .map(|n| n.item)
            .collect()
    }
}
