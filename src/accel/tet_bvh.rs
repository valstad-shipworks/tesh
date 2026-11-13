use alloc::vec::Vec;
use core::array;
use core::ops::{ControlFlow, Range};

use glam::Vec3;
use hydroplane::{Backend, BackendAll, Gang, GangGlamExt, Varying, Vec3Wide};
use hydroplane_solvers::bvh::{Bvh, Options, OverlapLeaves};

use crate::accel::prim_bounds;
use crate::tet::TetMesh;

/// Wide leaves: every tetrahedron in a leaf is containment-tested in one SPMD pass, so the
/// tree trades depth for SIMD-friendly leaf width.
const MAX_LEAF: usize = hydroplane_solvers::bvh::DEFAULT_MAX_LEAF;

/// A BVH over a tetrahedral mesh's element bounding boxes, for accelerated point location.
pub struct TetBvh {
    bvh: Bvh,
    /// Corner components in leaf-slot order, one column per `(corner, axis)`:
    /// `ax ay az bx by bz cx cy cz dx dy dz`. Slot order keeps each leaf's lanes contiguous.
    corners: [Vec<f32>; 12],
}

impl TetBvh {
    /// Builds a BVH over the mesh's elements.
    #[must_use]
    pub fn new(mesh: &TetMesh) -> Self {
        let tets: Vec<[Vec3; 4]> = mesh.tetrahedra().collect();
        let boxes: Vec<_> = tets.iter().map(prim_bounds).collect();
        let options = Options {
            max_leaf: Some(MAX_LEAF),
            ..Options::default()
        };
        let bvh = Bvh::with_options(&boxes, options);

        let mut corners: [Vec<f32>; 12] = array::from_fn(|_| Vec::with_capacity(tets.len()));
        for &pi in bvh.order() {
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
        let mut leaves = Locate {
            tree: self,
            p,
            found: None,
        };
        let _ = self.bvh.containing(p.to_array(), &mut leaves);
        leaves.found.map(|slot| self.bvh.order()[slot] as usize)
    }
}

/// Stops the descent at the first leaf holding a tetrahedron that contains `p`.
struct Locate<'a> {
    tree: &'a TetBvh,
    p: Vec3,
    /// Slot of the containing tetrahedron.
    found: Option<usize>,
}

impl OverlapLeaves for Locate<'_> {
    #[inline(always)]
    fn visit<S: BackendAll + Backend<f32>>(
        &mut self,
        g: Gang<S>,
        slots: Range<usize>,
    ) -> ControlFlow<()> {
        let p = g.splat_vec3(self.p);
        let start = slots.start;
        let cols: [&[f32]; 12] = array::from_fn(|k| &self.tree.corners[k][slots.clone()]);
        let mut found = None;
        g.for_each_hit_n(
            cols,
            |v| tet_contains_wide(g, p, v),
            |i| found = found.or(Some(start + i)),
        );
        self.found = found;
        if found.is_some() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
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
