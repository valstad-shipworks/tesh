use alloc::vec::Vec;
use core::array;
use core::ops::Range;

use glam::Vec3;
use hydroplane::{Backend, BackendAll, Gang, GangGlamExt, Mask, Varying, Vec3Wide};
use hydroplane_solvers::bvh::{Bvh, NearestLeaves, Options, RayLeaves};

use crate::accel::prim_bounds;
use crate::tri::{RayHit, TriMesh};
use crate::triangle;

/// Wide leaves: every triangle in a leaf is tested in one SPMD pass, so the tree trades
/// depth for SIMD-friendly leaf width.
const MAX_LEAF: usize = hydroplane_solvers::bvh::DEFAULT_MAX_LEAF;

/// Slots of padding past the last triangle, so a leaf whose length is not a whole number of
/// registers can still finish on a full-width load instead of a staged partial one.
///
/// Every lane a leaf's walk loads therefore still holds a real face of this mesh: the lanes past
/// a leaf read the next leaf's triangles, and past the last leaf the padding repeats the final
/// triangle. Ranking one of those early is harmless — the traversal only ever prunes on hits it
/// was entitled to find — so only [`ParityLeaf`], which counts crossings rather than ranking
/// them, has to mask the lanes past its leaf back out.
const COLUMN_PAD: usize = hydroplane::MAX_LANES;

/// A BVH over a triangle mesh's faces, for accelerated ray and closest-point queries.
///
/// Build it once from a [`TriMesh`]; queries reuse the tree. Face indices in results refer
/// to the mesh the tree was built from.
pub struct TriBvh {
    /// Nodes as wide as the SIMD registers of the machine it was built on.
    bvh: Bvh,
    /// Corner components in leaf-slot order, one column per `(corner, axis)`:
    /// `ax ay az bx by bz cx cy cz`. Slot order keeps each leaf's lanes contiguous.
    corners: [Vec<f32>; 9],
}

impl TriBvh {
    /// Builds a BVH over the mesh's faces.
    #[must_use]
    pub fn new(mesh: &TriMesh) -> Self {
        let tris: Vec<[Vec3; 3]> = mesh.triangles().collect();
        let boxes: Vec<_> = tris.iter().map(prim_bounds).collect();
        let options = Options {
            max_leaf: Some(MAX_LEAF),
            ..Options::default()
        };
        let bvh = Bvh::with_options(&boxes, options);

        let mut corners: [Vec<f32>; 9] =
            array::from_fn(|_| Vec::with_capacity(tris.len() + COLUMN_PAD));
        for &pi in bvh.order() {
            for (k, v) in tris[pi as usize].iter().enumerate() {
                corners[3 * k].push(v.x);
                corners[3 * k + 1].push(v.y);
                corners[3 * k + 2].push(v.z);
            }
        }
        for col in &mut corners {
            let last = col.last().copied().unwrap_or(0.0);
            col.resize(col.len() + COLUMN_PAD, last);
        }
        Self { bvh, corners }
    }

    /// The triangle behind BVH primitive slot `slot`.
    #[inline]
    fn tri_at(&self, slot: usize) -> [Vec3; 3] {
        let c = &self.corners;
        [
            Vec3::new(c[0][slot], c[1][slot], c[2][slot]),
            Vec3::new(c[3][slot], c[4][slot], c[5][slot]),
            Vec3::new(c[6][slot], c[7][slot], c[8][slot]),
        ]
    }

    /// Nearest intersection of the ray `origin + t * dir` (`t >= 0`) with the surface.
    #[must_use]
    pub fn ray_intersect(&self, origin: Vec3, dir: Vec3) -> Option<RayHit> {
        let mut leaves = RayLeaf {
            tree: self,
            origin,
            dir,
            slot: usize::MAX,
        };
        let t = self
            .bvh
            .ray(origin.to_array(), dir.to_array(), f32::INFINITY, &mut leaves)?;
        Some(RayHit {
            face: self.bvh.order()[leaves.slot],
            t,
            point: origin + dir * t,
        })
    }

    /// The point on the surface closest to `p`, via a distance-pruned BVH descent.
    #[must_use]
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        self.closest(p).1
    }

    /// The face holding the point closest to `p`, or `None` for an empty mesh.
    #[must_use]
    pub fn closest_face(&self, p: Vec3) -> Option<u32> {
        self.closest_slot(p).map(|slot| self.bvh.order()[slot])
    }

    /// Unsigned distance from `p` to the surface.
    #[inline]
    #[must_use]
    pub fn distance(&self, p: Vec3) -> f32 {
        self.closest(p).0.sqrt()
    }

    /// Whether `p` is enclosed by the surface, by counting the crossings of one ray in a
    /// single traversal (parity). Robust for watertight, consistently wound meshes.
    #[must_use]
    pub fn contains_point(&self, p: Vec3) -> bool {
        if self.bvh.is_empty() {
            return false;
        }
        let mut inside = 0;
        for (i, &dir) in PARITY_DIRS.iter().enumerate() {
            let mut leaves = ParityLeaf {
                tree: self,
                origin: p,
                dir: dir.normalize(),
                crossings: 0,
                ambiguous: false,
            };
            self.bvh
                .ray(p.to_array(), leaves.dir.to_array(), f32::INFINITY, &mut leaves);
            if !leaves.ambiguous {
                return leaves.crossings % 2 == 1;
            }
            inside += usize::from(leaves.crossings % 2 == 1);
            if inside * 2 > PARITY_DIRS.len() || (i + 1 - inside) * 2 > PARITY_DIRS.len() {
                break;
            }
        }
        inside * 2 > PARITY_DIRS.len()
    }

    /// Signed distance to the surface: negative inside, positive outside.
    #[must_use]
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        let d = self.distance(p);
        if self.contains_point(p) { -d } else { d }
    }

    /// `(squared_distance, closest_point)` — the shared core of the distance queries.
    fn closest(&self, p: Vec3) -> (f32, Vec3) {
        match self.closest_slot(p) {
            // The wide pass only ranks lanes; the winning triangle is re-solved exactly.
            Some(slot) => {
                let [a, b, c] = self.tri_at(slot);
                let q = triangle::closest_point(p, a, b, c);
                ((p - q).length_squared(), q)
            }
            None => (f32::INFINITY, p),
        }
    }

    /// Slot of the triangle closest to `p`.
    fn closest_slot(&self, p: Vec3) -> Option<usize> {
        let mut leaves = NearestLeaf {
            tree: self,
            p,
            slot: usize::MAX,
        };
        self.bvh.nearest(p.to_array(), f32::INFINITY, &mut leaves)?;
        Some(leaves.slot)
    }

    /// A leaf's corner columns, extended to a whole number of `lanes`-wide registers so every
    /// chunk of the walk is a full load. Safe to over-read by [`COLUMN_PAD`]'s contract.
    #[inline(always)]
    fn leaf_cols(&self, slots: &Range<usize>, lanes: usize) -> [&[f32]; 9] {
        let end = slots.start + slots.len().next_multiple_of(lanes);
        array::from_fn(|k| &self.corners[k][slots.start..end])
    }
}

#[inline(always)]
fn cross<S: Backend<f32>>(u: Vec3Wide<S>, v: Vec3Wide<S>) -> Vec3Wide<S> {
    let [ux, uy, uz] = u.0;
    let [vx, vy, vz] = v.0;
    Vec3Wide([uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx])
}

/// Lane-parallel Möller–Trumbore with [`triangle::ray_intersect`]'s exact semantics:
/// per-lane `t`, or `INFINITY` for a miss / parallel ray / `t` below the epsilon.
#[inline(always)]
fn ray_tri_wide<S: Backend<f32>>(
    g: Gang<S>,
    origin: Vec3Wide<S>,
    dir: Vec3Wide<S>,
    v: [Varying<f32, S>; 9],
) -> (Varying<f32, S>, Mask<f32, S>) {
    const EPS: f32 = 1e-7;
    use triangle::PARALLEL_COS;
    let a = Vec3Wide([v[0], v[1], v[2]]);
    let b = Vec3Wide([v[3], v[4], v[5]]);
    let c = Vec3Wide([v[6], v[7], v[8]]);

    let e1 = b - a;
    let e2 = c - a;
    let pvec = cross(dir, e2);
    let det = e1.dot(pvec);
    let inv = g.splat(1.0) / det;
    let tvec = origin - a;
    let u = tvec.dot(pvec) * inv;
    let qvec = cross(tvec, e1);
    let w = dir.dot(qvec) * inv;
    let t = e2.dot(qvec) * inv;

    let n = cross(e1, e2);
    let zero = g.splat(0.0);
    let one = g.splat(1.0);
    let eps = g.splat(EPS);
    let valid = (det * det).gt(g.splat(PARALLEL_COS * PARALLEL_COS) * n.dot(n) * dir.dot(dir))
        & u.ge(zero)
        & u.le(one)
        & w.ge(zero)
        & (u + w).le(one)
        & t.ge(eps);
    (t.select(valid, g.splat(f32::INFINITY)), valid)
}

/// [`ray_tri_wide`]'s crossing test for a parity count: whether the ray crosses each lane's
/// triangle, and whether it passes within [`EDGE_BAND`] of an edge of it, where rounding
/// decides which of the triangles sharing that edge the crossing lands on.
#[inline(always)]
fn parity_tri_wide<S: Backend<f32>>(
    g: Gang<S>,
    origin: Vec3Wide<S>,
    dir: Vec3Wide<S>,
    v: [Varying<f32, S>; 9],
) -> (Mask<f32, S>, Mask<f32, S>) {
    const EPS: f32 = 1e-7;
    use triangle::PARALLEL_COS;
    let a = Vec3Wide([v[0], v[1], v[2]]);
    let b = Vec3Wide([v[3], v[4], v[5]]);
    let c = Vec3Wide([v[6], v[7], v[8]]);

    let e1 = b - a;
    let e2 = c - a;
    let pvec = cross(dir, e2);
    let det = e1.dot(pvec);
    let inv = g.splat(1.0) / det;
    let tvec = origin - a;
    let u = tvec.dot(pvec) * inv;
    let qvec = cross(tvec, e1);
    let w = dir.dot(qvec) * inv;
    let t = e2.dot(qvec) * inv;
    let n = cross(e1, e2);

    let zero = g.splat(0.0);
    let one = g.splat(1.0);
    let band = g.splat(EDGE_BAND);
    let facing = (det * det).gt(g.splat(PARALLEL_COS * PARALLEL_COS) * n.dot(n));
    let ahead = t.ge(g.splat(EPS));
    let rest = one - u - w;
    let valid = facing & ahead & u.ge(zero) & u.le(one) & w.ge(zero) & rest.ge(zero);
    let near = u.ge(-band) & w.ge(-band) & rest.ge(-band);
    let edge = u.lt(band) | w.lt(band) | rest.lt(band);
    (valid, facing & ahead & near & edge)
}

/// Folds a chunk's per-lane `t` into the running best, resolving the winning lane in the
/// vector unit: the minimum is one reduction, and the lane holding it is the lowest set bit of
/// `t <= min` — no round trip through memory, and no per-lane branch.
#[inline(always)]
fn take_nearest<S: Backend<f32>>(
    g: Gang<S>,
    t: Varying<f32, S>,
    base: usize,
    best_t: &mut f32,
    best_slot: &mut usize,
) {
    let chunk_t = t.reduce_min();
    if chunk_t < *best_t {
        *best_t = chunk_t;
        // `t >= chunk_t` in every lane, so `<=` selects exactly the lanes holding the minimum;
        // taking the lowest keeps the earliest slot on a tie, as a forward scan would.
        *best_slot = base + t.le(g.splat(chunk_t)).to_bitmask().trailing_zeros() as usize;
    }
}

/// Nearest-hit leaf test for [`TriBvh::ray_intersect`]; `slot` ends on the winning triangle.
struct RayLeaf<'a> {
    tree: &'a TriBvh,
    origin: Vec3,
    dir: Vec3,
    slot: usize,
}

impl RayLeaves for RayLeaf<'_> {
    /// The chunk minima are combined lane-wise and reduced once per leaf, keeping the
    /// horizontal reduction off the per-chunk critical path; the winning lane is recovered only
    /// when the leaf improves on `t_max`.
    #[inline(always)]
    fn hit<S: BackendAll + Backend<f32>>(
        &mut self,
        g: Gang<S>,
        slots: Range<usize>,
        t_max: f32,
    ) -> Option<f32> {
        let lanes = g.lanes::<f32>();
        let cols = self.tree.leaf_cols(&slots, lanes);
        let chunks = cols[0].len() / lanes;
        let origin = g.splat_vec3(self.origin);
        let dir = g.splat_vec3(self.dir);
        let inf = g.splat(f32::INFINITY);
        let mut chunk_t = [inf; MAX_LEAF];
        let mut leaf_min = inf;
        for (ci, slot) in chunk_t.iter_mut().enumerate().take(chunks) {
            let off = ci * lanes;
            let v = array::from_fn(|k| g.load(&cols[k][off..off + lanes]));
            let (t, _) = ray_tri_wide(g, origin, dir, v);
            *slot = t;
            leaf_min = leaf_min.min(t);
        }
        let leaf_t = leaf_min.reduce_min();
        if leaf_t >= t_max {
            return None;
        }
        let m = g.splat(leaf_t);
        // The first chunk holding the minimum, lowest lane on a tie, as a forward scan would.
        for (ci, t) in chunk_t[..chunks].iter().enumerate() {
            let bits = t.le(m).to_bitmask();
            if bits != 0 {
                self.slot = slots.start + ci * lanes + bits.trailing_zeros() as usize;
                break;
            }
        }
        Some(leaf_t)
    }
}

/// Ray directions for the parity test, tried in order until one gives an unambiguous count.
/// None is axis-aligned or near another, so a ray that grazes an edge along one is unlikely
/// to graze one along the next.
const PARITY_DIRS: [Vec3; 3] = [
    Vec3::new(0.573_1, 0.602_9, 0.555_2),
    Vec3::new(-0.651_8, 0.270_4, 0.708_4),
    Vec3::new(0.319_2, -0.811_5, 0.489_6),
];

/// A crossing this close to a triangle's edge, in barycentric terms, may be counted by both
/// triangles that share the edge or by neither, so the count it is part of is not trusted.
const EDGE_BAND: f32 = 1e-3;

/// Crossing counter for [`TriBvh::contains_point`]: it never reports a hit, so the traversal
/// visits every leaf the ray enters.
struct ParityLeaf<'a> {
    tree: &'a TriBvh,
    origin: Vec3,
    dir: Vec3,
    crossings: u32,
    /// Whether any crossing grazed an edge.
    ambiguous: bool,
}

impl RayLeaves for ParityLeaf<'_> {
    #[inline(always)]
    fn hit<S: BackendAll + Backend<f32>>(
        &mut self,
        g: Gang<S>,
        slots: Range<usize>,
        _: f32,
    ) -> Option<f32> {
        let lanes = g.lanes::<f32>();
        let cols = self.tree.leaf_cols(&slots, lanes);
        let origin = g.splat_vec3(self.origin);
        let dir = g.splat_vec3(self.dir);
        for off in (0..cols[0].len()).step_by(lanes) {
            let v = array::from_fn(|k| g.load(&cols[k][off..off + lanes]));
            let (valid, grazed) = parity_tri_wide(g, origin, dir, v);
            let active = g.active_mask::<f32>((slots.len() - off).min(lanes));
            self.crossings += (valid & active).to_bitmask().count_ones();
            self.ambiguous |= (grazed & active).to_bitmask() != 0;
        }
        None
    }
}

/// Lane-parallel squared distance from `p` to each lane's triangle — the branchless form of
/// [`triangle::closest_point`] (Ericson's region tests turned into reverse-priority selects,
/// so the first matching region wins exactly as the scalar branch chain does).
#[inline(always)]
fn dist2_tri_wide<S: Backend<f32>>(
    g: Gang<S>,
    p: Vec3Wide<S>,
    v: [Varying<f32, S>; 9],
) -> Varying<f32, S> {
    let a = Vec3Wide([v[0], v[1], v[2]]);
    let b = Vec3Wide([v[3], v[4], v[5]]);
    let c = Vec3Wide([v[6], v[7], v[8]]);
    let zero = g.splat(0.0);

    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);

    let va = d3 * d6 - d5 * d4;
    let vb = d5 * d2 - d1 * d6;
    let vc = d1 * d4 - d3 * d2;

    // Region masks, in the scalar chain's priority order.
    let m_a = d1.le(zero) & d2.le(zero);
    let m_b = d3.ge(zero) & d4.le(d3);
    let m_ab = vc.le(zero) & d1.ge(zero) & d3.le(zero);
    let m_c = d6.ge(zero) & d5.le(d6);
    let m_ac = vb.le(zero) & d2.ge(zero) & d6.le(zero);
    let m_bc = va.le(zero) & (d4 - d3).ge(zero) & (d5 - d6).ge(zero);

    // Candidate points; untaken lanes may divide by zero, but their results are discarded
    // by the selects below (the scalar chain has the same blind spots).
    let q_ab = a.add_scaled(ab, d1 / (d1 - d3));
    let q_ac = a.add_scaled(ac, d2 / (d2 - d6));
    let w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    let q_bc = b.add_scaled(c - b, w_bc);
    let denom = g.splat(1.0) / (va + vb + vc);
    let q_face = a.add_scaled(ab, vb * denom).add_scaled(ac, vc * denom);

    // Reverse-priority selects: the last applied (highest-priority) mask wins.
    let mut q = q_face;
    q = q_bc.select(m_bc, q);
    q = q_ac.select(m_ac, q);
    q = c.select(m_c, q);
    q = q_ab.select(m_ab, q);
    q = b.select(m_b, q);
    q = a.select(m_a, q);

    (p - q).length_squared()
}

/// Closest-triangle leaf test for [`TriBvh::closest_point`]; `slot` ends on the winner.
struct NearestLeaf<'a> {
    tree: &'a TriBvh,
    p: Vec3,
    slot: usize,
}

impl NearestLeaves for NearestLeaf<'_> {
    #[inline(always)]
    fn nearest<S: BackendAll + Backend<f32>>(
        &mut self,
        g: Gang<S>,
        slots: Range<usize>,
        best_d2: f32,
    ) -> Option<f32> {
        let lanes = g.lanes::<f32>();
        let cols = self.tree.leaf_cols(&slots, lanes);
        let p = g.splat_vec3(self.p);
        let mut best = best_d2;
        for off in (0..cols[0].len()).step_by(lanes) {
            let v = array::from_fn(|k| g.load(&cols[k][off..off + lanes]));
            let d2 = dist2_tri_wide(g, p, v);
            take_nearest(g, d2, slots.start + off, &mut best, &mut self.slot);
        }
        (best < best_d2).then_some(best)
    }
}
