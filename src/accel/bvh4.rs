//! A 4-wide BVH collapsed from a [`Bvh2`], for single-ray traversal: one node visit tests
//! all four child boxes in one SoA pass, with no horizontal reductions, and the tree is half
//! as deep.

use alloc::vec::Vec;
use glam::{Vec3, Vec3A, Vec4};

use crate::aabb::Aabb;
use crate::accel::bvh2::{Bvh2, Node, half_area};

/// Widening of the slab exit that absorbs the rounding of `(bound - origin) * inv_dir`
/// (Ize, *Robust BVH Ray Traversal*: `1 + 2γ₃`), so a ray grazing a box corner or face is
/// never culled from a triangle it touches.
pub(crate) const SLAB_EXIT_SCALE: f32 =
    1.0 + 2.0 * (3.0 * f32::EPSILON * 0.5) / (1.0 - 3.0 * f32::EPSILON * 0.5);

/// Four child boxes in SoA form (`min[axis]` holds that axis for every child). Unused lanes
/// have both bounds at `+∞`, which every finite ray misses.
#[derive(Clone, Copy)]
pub(crate) struct Node4 {
    pub(crate) min: [Vec4; 3],
    pub(crate) max: [Vec4; 3],
    /// Leaf lane: first primitive slot. Internal lane: index of the child [`Node4`].
    pub(crate) child: [u32; 4],
    /// Primitives in a leaf lane; `0` marks an internal (or unused) lane.
    pub(crate) count: [u32; 4],
}

impl Node4 {
    const EMPTY: Self = Self {
        min: [Vec4::INFINITY; 3],
        max: [Vec4::INFINITY; 3],
        child: [0; 4],
        count: [0; 4],
    };
}

/// Collapses a binary tree into 4-wide nodes, repeatedly opening the internal child with the
/// largest surface area (the one a ray is most likely to enter) until four children remain.
/// Leaves keep their primitive ranges, so slot-ordered primitive data carries over unchanged.
pub(crate) fn collapse(bvh: &Bvh2) -> Vec<Node4> {
    let mut out = Vec::new();
    if !bvh.nodes.is_empty() {
        collapse_node(bvh, 0, &mut out);
    }
    out
}

fn collapse_node(bvh: &Bvh2, ni: u32, out: &mut Vec<Node4>) -> u32 {
    let idx = out.len();
    out.push(Node4::EMPTY);

    let node = &bvh.nodes[ni as usize];
    let mut kids = [0u32; 4];
    let mut len = if node.is_leaf() {
        kids[0] = ni;
        1
    } else {
        kids[0] = node.first;
        kids[1] = node.first + 1;
        2
    };
    while len < 4 {
        let open = (0..len)
            .filter(|&j| !bvh.nodes[kids[j] as usize].is_leaf())
            .max_by(|&a, &b| {
                area(&bvh.nodes[kids[a] as usize]).total_cmp(&area(&bvh.nodes[kids[b] as usize]))
            });
        let Some(j) = open else { break };
        let first = bvh.nodes[kids[j] as usize].first;
        kids[j] = first;
        kids[len] = first + 1;
        len += 1;
    }

    for (lane, &k) in kids[..len].iter().enumerate() {
        let c = bvh.nodes[k as usize];
        let (lo, hi) = (Vec3::from(c.min), Vec3::from(c.max));
        for axis in 0..3 {
            out[idx].min[axis][lane] = lo[axis];
            out[idx].max[axis][lane] = hi[axis];
        }
        if c.is_leaf() {
            out[idx].child[lane] = c.first;
            out[idx].count[lane] = c.count;
        } else {
            out[idx].child[lane] = collapse_node(bvh, k, out);
        }
    }
    idx as u32
}

#[inline]
fn area(node: &Node) -> f32 {
    half_area(&Aabb::new(node.min.into(), node.max.into()))
}

/// A ray prepared for slab tests against [`Node4`]s: origin and inverse direction splatted
/// per axis. Zero direction components invert to a huge finite value rather than `∞`, so a
/// ray lying in a box's face plane yields `0 · big = 0` instead of a NaN that the vector
/// `min`/`max` would propagate into a spurious miss.
#[derive(Clone, Copy)]
pub(crate) struct Ray4 {
    origin: [Vec4; 3],
    inv_dir: [Vec4; 3],
}

impl Ray4 {
    /// `inv_dir` is [`safe_inv_dir`] of the ray direction.
    #[inline]
    pub(crate) fn new(origin: Vec3, inv: Vec3) -> Self {
        Self {
            origin: [origin.x, origin.y, origin.z].map(Vec4::splat),
            inv_dir: [inv.x, inv.y, inv.z].map(Vec4::splat),
        }
    }

    /// Per-lane entry `t` into each child box, or `INFINITY` where the ray misses it or
    /// enters past `tmax`.
    #[inline(always)]
    pub(crate) fn hit_children(&self, n: &Node4, tmax: f32) -> [f32; 4] {
        let [ox, oy, oz] = self.origin;
        let [ix, iy, iz] = self.inv_dir;
        let tx0 = (n.min[0] - ox) * ix;
        let tx1 = (n.max[0] - ox) * ix;
        let ty0 = (n.min[1] - oy) * iy;
        let ty1 = (n.max[1] - oy) * iy;
        let tz0 = (n.min[2] - oz) * iz;
        let tz1 = (n.max[2] - oz) * iz;
        let near = tx0
            .min(tx1)
            .max(ty0.min(ty1))
            .max(tz0.min(tz1).max(Vec4::ZERO));
        let far = tx0.max(tx1).min(ty0.max(ty1)).min(tz0.max(tz1)) * SLAB_EXIT_SCALE;
        let far = far.min(Vec4::splat(tmax));
        Vec4::select(near.cmple(far), near, Vec4::INFINITY).to_array()
    }
}

/// `1 / dir` with zero components mapped to a signed huge finite value (see [`Ray4`]).
#[inline]
pub(crate) fn safe_inv_dir(dir: Vec3) -> Vec3 {
    let tiny = dir.abs().cmplt(Vec3::splat(1e-30));
    Vec3::select(tiny, Vec3::splat(1e30).copysign(dir), dir.recip())
}

/// Scalar slab test of the ray against one binary node: the cheap whole-tree reject.
#[inline]
pub(crate) fn hits_node(node: &Node, origin: Vec3, inv_dir: Vec3) -> bool {
    let (o, inv) = (Vec3A::from(origin), Vec3A::from(inv_dir));
    let t1 = (node.min - o) * inv;
    let t2 = (node.max - o) * inv;
    let near = t1.min(t2).max_element().max(0.0);
    let far = t1.max(t2).min_element() * SLAB_EXIT_SCALE;
    near <= far
}
