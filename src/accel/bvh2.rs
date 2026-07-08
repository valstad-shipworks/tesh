//! Shared BVH core: a binary tree over primitive bounding boxes, built top-down with
//! binned surface-area-heuristic splits (Wald 2007).

use alloc::vec::Vec;
use glam::{Vec3, Vec3A};

use crate::aabb::Aabb;

const BINS: usize = 16;

/// Bounds are stored as [`Vec3A`] so the per-node traversal tests stay on the SIMD path.
#[derive(Clone, Copy)]
pub(crate) struct Node {
    pub(crate) min: Vec3A,
    pub(crate) max: Vec3A,
    /// Leaf: offset into `prim_indices`. Internal: index of the left child (right is `+ 1`).
    pub(crate) first: u32,
    /// Primitives in the leaf; `0` marks an internal node.
    pub(crate) count: u32,
}

impl Node {
    #[inline]
    pub(crate) fn is_leaf(&self) -> bool {
        self.count > 0
    }

    /// Squared distance from `p` to the node bounds (zero inside).
    #[inline]
    pub(crate) fn distance_squared(&self, p: Vec3A) -> f32 {
        (p - p.clamp(self.min, self.max)).length_squared()
    }

    #[inline]
    pub(crate) fn contains(&self, p: Vec3A) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

pub(crate) struct Bvh2 {
    pub(crate) nodes: Vec<Node>,
    /// Maps a leaf's primitive slots back to input primitive indices.
    pub(crate) prim_indices: Vec<u32>,
}

/// Half the surface area; relative SAH costs only need proportionality.
#[inline]
fn half_area(aabb: &Aabb) -> f32 {
    let e = aabb.extents().max(Vec3::ZERO);
    e.x * e.y + e.y * e.z + e.z * e.x
}

impl Bvh2 {
    /// Builds the tree over per-primitive bounding boxes, splitting until leaves hold at
    /// most `max_leaf` primitives. Empty input yields an empty tree.
    pub(crate) fn build(prim_aabbs: &[Aabb], max_leaf: u32) -> Self {
        let n = prim_aabbs.len();
        let mut prim_indices: Vec<u32> = (0..n as u32).collect();
        if n == 0 {
            return Self {
                nodes: Vec::new(),
                prim_indices,
            };
        }
        let centroids: Vec<Vec3> = prim_aabbs.iter().map(Aabb::center).collect();

        let empty = Node {
            min: Vec3A::INFINITY,
            max: Vec3A::NEG_INFINITY,
            first: 0,
            count: n as u32,
        };
        let mut nodes = Vec::with_capacity(2 * n);
        nodes.push(empty);

        // Nodes awaiting processing still hold their primitive range in (first, count).
        let mut stack = alloc::vec![0usize];
        while let Some(ni) = stack.pop() {
            let start = nodes[ni].first as usize;
            let count = nodes[ni].count as usize;
            let range = &prim_indices[start..start + count];

            let mut aabb = Aabb::EMPTY;
            let mut cb = Aabb::EMPTY;
            for &pi in range {
                aabb.union(&prim_aabbs[pi as usize]);
                cb.expand(centroids[pi as usize]);
            }
            nodes[ni].min = aabb.min.into();
            nodes[ni].max = aabb.max.into();
            if count as u32 <= max_leaf {
                continue;
            }

            let mid =
                Self::split(&mut prim_indices[start..start + count], prim_aabbs, &centroids, &cb)
                    .map_or(start + count / 2, |m| start + m);

            let left = nodes.len();
            nodes.push(Node {
                first: start as u32,
                count: (mid - start) as u32,
                ..empty
            });
            nodes.push(Node {
                first: mid as u32,
                count: (start + count - mid) as u32,
                ..empty
            });
            nodes[ni].first = left as u32;
            nodes[ni].count = 0;
            stack.push(left);
            stack.push(left + 1);
        }

        Self {
            nodes,
            prim_indices,
        }
    }

    /// Partitions `range` around the best binned-SAH split of the centroid bounds `cb`,
    /// returning the split offset within the range. `None` when no split separates the
    /// primitives (degenerate or coincident centroids); the caller falls back to a median.
    fn split(
        range: &mut [u32],
        prim_aabbs: &[Aabb],
        centroids: &[Vec3],
        cb: &Aabb,
    ) -> Option<usize> {
        let ext = cb.extents();
        let axis = if ext.x >= ext.y && ext.x >= ext.z {
            0
        } else if ext.y >= ext.z {
            1
        } else {
            2
        };
        let extent = ext[axis];
        // `<=` alone would let a NaN extent (from NaN vertices) through to the binning math.
        if extent.is_nan() || extent <= 0.0 {
            return None;
        }
        let scale = BINS as f32 / extent;
        let origin = cb.min[axis];
        let bin_of = |pi: u32| -> usize {
            (((centroids[pi as usize][axis] - origin) * scale) as usize).min(BINS - 1)
        };

        let mut bin_aabbs = [Aabb::EMPTY; BINS];
        let mut bin_counts = [0u32; BINS];
        for &pi in range.iter() {
            let b = bin_of(pi);
            bin_counts[b] += 1;
            bin_aabbs[b].union(&prim_aabbs[pi as usize]);
        }

        // Suffix pass: cost of the right side for every split position.
        let mut right_cost = [f32::INFINITY; BINS];
        let mut acc = Aabb::EMPTY;
        let mut cnt = 0u32;
        for k in (1..BINS).rev() {
            acc.union(&bin_aabbs[k]);
            cnt += bin_counts[k];
            right_cost[k] = if cnt > 0 {
                half_area(&acc) * cnt as f32
            } else {
                f32::INFINITY
            };
        }

        let mut best_k = 0;
        let mut best_cost = f32::INFINITY;
        let mut acc = Aabb::EMPTY;
        let mut cnt = 0u32;
        for k in 1..BINS {
            acc.union(&bin_aabbs[k - 1]);
            cnt += bin_counts[k - 1];
            if cnt == 0 {
                continue;
            }
            let cost = half_area(&acc) * cnt as f32 + right_cost[k];
            if cost < best_cost {
                best_cost = cost;
                best_k = k;
            }
        }
        if best_k == 0 {
            return None;
        }

        let mut i = 0;
        let mut j = range.len();
        while i < j {
            if bin_of(range[i]) < best_k {
                i += 1;
            } else {
                j -= 1;
                range.swap(i, j);
            }
        }
        (i > 0 && i < range.len()).then_some(i)
    }
}
