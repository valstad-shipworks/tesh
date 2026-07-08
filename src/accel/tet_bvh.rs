use alloc::vec::Vec;
use core::array;

use glam::{Vec3, Vec3A};
use hydroplane::{Backend, BackendAll, Gang, GangGlamExt, Kernel, Varying, Vec3Wide, dispatch};

use crate::aabb::Aabb;
use crate::accel::TraversalStack;
use crate::accel::bvh2::Bvh2;
use crate::tet::TetMesh;

/// Wide leaves: every tetrahedron in a leaf is containment-tested in one SPMD pass, so the
/// tree trades depth for SIMD-friendly leaf width.
const MAX_LEAF: u32 = 16;

/// A BVH over a tetrahedral mesh's element bounding boxes, for accelerated point location.
pub struct TetBvh {
    bvh: Bvh2,
    /// Corner components in leaf-slot order, one column per `(corner, axis)`:
    /// `ax ay az bx by bz cx cy cz dx dy dz`. Slot order keeps each leaf's lanes contiguous.
    corners: [Vec<f32>; 12],
}

impl TetBvh {
    /// Builds a BVH over the mesh's elements.
    #[must_use]
    pub fn new(mesh: &TetMesh) -> Self {
        let tets: Vec<[Vec3; 4]> = mesh.tetrahedra().collect();
        let aabbs: Vec<Aabb> = tets
            .iter()
            .map(|&[a, b, c, d]| {
                let mut aabb = Aabb::from_point(a);
                aabb.expand(b);
                aabb.expand(c);
                aabb.expand(d);
                aabb
            })
            .collect();
        let bvh = Bvh2::build(&aabbs, MAX_LEAF);

        let mut corners: [Vec<f32>; 12] = array::from_fn(|_| Vec::with_capacity(tets.len()));
        for &pi in &bvh.prim_indices {
            for (k, v) in tets[pi as usize].iter().enumerate() {
                corners[3 * k].push(v.x);
                corners[3 * k + 1].push(v.y);
                corners[3 * k + 2].push(v.z);
            }
        }
        Self { bvh, corners }
    }

    /// Index of a tetrahedron containing `p`, or `None` if `p` is outside the volume.
    #[must_use]
    pub fn find_containing_tet(&self, p: Vec3) -> Option<usize> {
        if self.bvh.nodes.is_empty() {
            return None;
        }
        dispatch(Locate { tree: self, p }).map(|slot| self.bvh.prim_indices[slot] as usize)
    }
}

struct Locate<'a> {
    tree: &'a TetBvh,
    p: Vec3,
}

impl Kernel<f32> for Locate<'_> {
    /// Slot index of a containing tetrahedron.
    type Output = Option<usize>;

    fn run<S: BackendAll + Backend<f32>>(self, g: Gang<S>) -> Option<usize> {
        let nodes = &self.tree.bvh.nodes;
        let pa = Vec3A::from(self.p);
        let pw = g.splat_vec3(self.p);

        if !nodes[0].contains(pa) {
            return None;
        }
        let mut stack: TraversalStack<u32> = TraversalStack::new();
        let mut ni = 0u32;
        loop {
            let node = &nodes[ni as usize];
            if node.is_leaf() {
                let start = node.first as usize;
                let end = start + node.count as usize;
                let cols: [&[f32]; 12] = array::from_fn(|k| &self.tree.corners[k][start..end]);
                let mut found = None;
                g.for_each_hit_n(
                    cols,
                    |v| tet_contains_wide(g, pw, v),
                    |i| found = found.or(Some(start + i)),
                );
                if found.is_some() {
                    return found;
                }
            } else {
                let l = node.first;
                let in_l = nodes[l as usize].contains(pa);
                let in_r = nodes[l as usize + 1].contains(pa);
                if in_l {
                    if in_r {
                        stack.push(l + 1);
                    }
                    ni = l;
                    continue;
                }
                if in_r {
                    ni = l + 1;
                    continue;
                }
            }
            match stack.pop() {
                Some(n) => ni = n,
                None => return None,
            }
        }
    }
}

#[inline(always)]
fn cross<S: Backend<f32>>(u: Vec3Wide<S>, v: Vec3Wide<S>) -> Vec3Wide<S> {
    let [ux, uy, uz] = u.0;
    let [vx, vy, vz] = v.0;
    Vec3Wide([uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx])
}

/// Lane-parallel [`tetrahedron::contains_point`](crate::tetrahedron::contains_point): the
/// barycentric test over one register of tetrahedra, matching the scalar semantics exactly
/// (a degenerate tetrahedron contains nothing).
#[inline(always)]
fn tet_contains_wide<S: Backend<f32>>(
    g: Gang<S>,
    p: Vec3Wide<S>,
    v: [Varying<f32, S>; 12],
) -> hydroplane::Mask<f32, S> {
    let a = Vec3Wide([v[0], v[1], v[2]]);
    let b = Vec3Wide([v[3], v[4], v[5]]);
    let c = Vec3Wide([v[6], v[7], v[8]]);
    let d = Vec3Wide([v[9], v[10], v[11]]);

    let vab = b - a;
    let vac = c - a;
    let vad = d - a;
    let vap = p - a;

    let n = cross(vab, vac);
    let denom = n.dot(vad);
    let one = g.splat(1.0);
    let inv = one / denom;
    let l1 = cross(vap, vac).dot(vad) * inv;
    let l2 = cross(vab, vap).dot(vad) * inv;
    let l3 = n.dot(vap) * inv;
    let l0 = one - l1 - l2 - l3;

    let min_l = l0.min(l1).min(l2).min(l3);
    denom.abs().gt(g.splat(f32::MIN_POSITIVE)) & min_l.ge(g.splat(0.0))
}
