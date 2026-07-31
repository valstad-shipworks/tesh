use alloc::vec::Vec;
use core::array;

use glam::{Vec3, Vec3A};
use hydroplane::{
    Backend, BackendAll, Gang, GangGlamExt, Kernel, Mask, Varying, Vec3Wide, dispatch,
};

use crate::aabb::Aabb;
use crate::accel::TraversalStack;
use crate::accel::bvh2::{Bvh2, Node};
use crate::tri::{RayHit, TriMesh};
use crate::triangle;

/// Wide leaves: every triangle in a leaf is tested in one SPMD pass, so the tree trades
/// depth for SIMD-friendly leaf width.
const MAX_LEAF: u32 = 16;

/// Slots of padding past the last triangle, so a leaf whose length is not a whole number of
/// registers can still finish on a full-width load instead of a staged partial one.
///
/// Every lane a leaf's walk loads therefore still holds a real face of this mesh: the lanes past
/// a leaf read the next leaf's triangles, and past the last leaf the padding repeats the final
/// triangle. Ranking one of those early is harmless — the traversal only ever prunes on hits it
/// was entitled to find — so only [`ParityQuery`], which counts crossings rather than ranking
/// them, has to mask the lanes past its leaf back out.
const COLUMN_PAD: usize = hydroplane::MAX_LANES;

/// A BVH over a triangle mesh's faces, for accelerated ray and closest-point queries.
///
/// Build it once from a [`TriMesh`]; queries reuse the tree. Face indices in results refer
/// to the mesh the tree was built from.
pub struct TriBvh {
    bvh: Bvh2,
    /// Corner components in leaf-slot order, one column per `(corner, axis)`:
    /// `ax ay az bx by bz cx cy cz`. Slot order keeps each leaf's lanes contiguous.
    corners: [Vec<f32>; 9],
}

/// `t` at which the ray enters the node's bounds, or `INFINITY` when it misses or enters
/// past `tmax`.
#[inline]
fn ray_aabb(node: &Node, origin: Vec3A, inv_dir: Vec3A, tmax: f32) -> f32 {
    let t1 = (node.min - origin) * inv_dir;
    let t2 = (node.max - origin) * inv_dir;
    let t_near = t1.min(t2).max_element().max(0.0);
    let t_far = t1.max(t2).min_element().min(tmax);
    if t_near <= t_far { t_near } else { f32::INFINITY }
}

impl TriBvh {
    /// Builds a BVH over the mesh's faces.
    #[must_use]
    pub fn new(mesh: &TriMesh) -> Self {
        let tris: Vec<[Vec3; 3]> = mesh.triangles().collect();
        let aabbs: Vec<Aabb> = tris
            .iter()
            .map(|&[a, b, c]| {
                let mut aabb = Aabb::from_point(a);
                aabb.expand(b);
                aabb.expand(c);
                aabb
            })
            .collect();
        let bvh = Bvh2::build(&aabbs, MAX_LEAF);

        let mut corners: [Vec<f32>; 9] =
            array::from_fn(|_| Vec::with_capacity(tris.len() + COLUMN_PAD));
        for &pi in &bvh.prim_indices {
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
        if self.bvh.nodes.is_empty() {
            return None;
        }
        dispatch(RayQuery {
            tree: self,
            origin,
            dir,
        })
        .map(|(slot, t)| RayHit {
            face: self.bvh.prim_indices[slot],
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
        if self.bvh.nodes.is_empty() {
            return None;
        }
        dispatch(ClosestQuery { tree: self, p }).map(|slot| self.bvh.prim_indices[slot])
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
        if self.bvh.nodes.is_empty() {
            return false;
        }
        let crossings = dispatch(ParityQuery { tree: self, p });
        crossings % 2 == 1
    }

    /// Signed distance to the surface: negative inside, positive outside.
    #[must_use]
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        let d = self.distance(p);
        if self.contains_point(p) { -d } else { d }
    }

    /// `(squared_distance, closest_point)` — the shared core of the distance queries.
    fn closest(&self, p: Vec3) -> (f32, Vec3) {
        if self.bvh.nodes.is_empty() {
            return (f32::INFINITY, p);
        }
        match dispatch(ClosestQuery { tree: self, p }) {
            // The wide pass only ranks lanes; the winning triangle is re-solved exactly.
            Some(slot) => {
                let [a, b, c] = self.tri_at(slot);
                let q = triangle::closest_point(p, a, b, c);
                ((p - q).length_squared(), q)
            }
            None => (f32::INFINITY, p),
        }
    }

    /// A leaf's corner columns, extended to a whole number of `lanes`-wide registers so every
    /// chunk of the walk is a full load. Safe to over-read by [`COLUMN_PAD`]'s contract.
    #[inline]
    fn leaf_cols(&self, start: usize, len: usize, lanes: usize) -> [&[f32]; 9] {
        let end = start + len.next_multiple_of(lanes);
        array::from_fn(|k| &self.corners[k][start..end])
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

    let zero = g.splat(0.0);
    let one = g.splat(1.0);
    let eps = g.splat(EPS);
    let valid = det.abs().ge(eps)
        & u.ge(zero)
        & u.le(one)
        & w.ge(zero)
        & (u + w).le(one)
        & t.ge(eps);
    (t.select(valid, g.splat(f32::INFINITY)), valid)
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

struct RayQuery<'a> {
    tree: &'a TriBvh,
    origin: Vec3,
    dir: Vec3,
}

impl Kernel<f32> for RayQuery<'_> {
    /// `(slot, t)` of the nearest hit.
    type Output = Option<(usize, f32)>;

    fn run<S: BackendAll + Backend<f32>>(self, g: Gang<S>) -> Self::Output {
        let nodes = &self.tree.bvh.nodes;
        let origin_a = Vec3A::from(self.origin);
        let inv_dir = Vec3A::from(self.dir).recip();
        let ow = g.splat_vec3(self.origin);
        let dw = g.splat_vec3(self.dir);

        let mut best_t = f32::INFINITY;
        let mut best_slot = usize::MAX;
        if ray_aabb(&nodes[0], origin_a, inv_dir, best_t).is_infinite() {
            return None;
        }
        let lanes = g.lanes::<f32>();
        let mut stack: TraversalStack<(u32, f32)> = TraversalStack::new();
        let mut ni = 0u32;
        'traverse: loop {
            let node = &nodes[ni as usize];
            if node.is_leaf() {
                let start = node.first as usize;
                let len = node.count as usize;
                let cols = self.tree.leaf_cols(start, len, lanes);
                for off in (0..cols[0].len()).step_by(lanes) {
                    let v = array::from_fn(|k| g.load(&cols[k][off..off + lanes]));
                    let (t, _) = ray_tri_wide(g, ow, dw, v);
                    take_nearest(g, t, start + off, &mut best_t, &mut best_slot);
                }
            } else {
                let l = node.first;
                let tl = ray_aabb(&nodes[l as usize], origin_a, inv_dir, best_t);
                let tr = ray_aabb(&nodes[l as usize + 1], origin_a, inv_dir, best_t);
                // Descend into the nearer hit child, deferring the farther one.
                let (near, far) = if tl <= tr { (l, l + 1) } else { (l + 1, l) };
                let (t_near, t_far) = if tl <= tr { (tl, tr) } else { (tr, tl) };
                if t_near.is_finite() {
                    if t_far.is_finite() {
                        stack.push((far, t_far));
                    }
                    ni = near;
                    continue;
                }
            }
            while let Some((n, t)) = stack.pop() {
                if t < best_t {
                    ni = n;
                    continue 'traverse;
                }
            }
            break;
        }
        (best_slot != usize::MAX).then_some((best_slot, best_t))
    }
}

struct ParityQuery<'a> {
    tree: &'a TriBvh,
    p: Vec3,
}

impl Kernel<f32> for ParityQuery<'_> {
    /// Number of ray crossings.
    type Output = u32;

    fn run<S: BackendAll + Backend<f32>>(self, g: Gang<S>) -> u32 {
        let nodes = &self.tree.bvh.nodes;
        let dir = Vec3::new(0.573_1, 0.602_9, 0.555_2).normalize();
        let origin_a = Vec3A::from(self.p);
        let inv_dir = Vec3A::from(dir).recip();
        let ow = g.splat_vec3(self.p);
        let dw = g.splat_vec3(dir);
        let lanes = g.lanes::<f32>();

        if ray_aabb(&nodes[0], origin_a, inv_dir, f32::INFINITY).is_infinite() {
            return 0;
        }
        let mut crossings = 0u32;
        let mut stack: TraversalStack<u32> = TraversalStack::new();
        let mut ni = 0u32;
        loop {
            let node = &nodes[ni as usize];
            if node.is_leaf() {
                let start = node.first as usize;
                let len = node.count as usize;
                let cols = self.tree.leaf_cols(start, len, lanes);
                for off in (0..cols[0].len()).step_by(lanes) {
                    let v = array::from_fn(|k| g.load(&cols[k][off..off + lanes]));
                    let (_, valid) = ray_tri_wide(g, ow, dw, v);
                    let active = g.active_mask::<f32>((len - off).min(lanes));
                    crossings += (valid & active).to_bitmask().count_ones();
                }
            } else {
                let l = node.first;
                let hit_l =
                    ray_aabb(&nodes[l as usize], origin_a, inv_dir, f32::INFINITY).is_finite();
                let hit_r =
                    ray_aabb(&nodes[l as usize + 1], origin_a, inv_dir, f32::INFINITY).is_finite();
                if hit_l {
                    if hit_r {
                        stack.push(l + 1);
                    }
                    ni = l;
                    continue;
                }
                if hit_r {
                    ni = l + 1;
                    continue;
                }
            }
            match stack.pop() {
                Some(n) => ni = n,
                None => return crossings,
            }
        }
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

struct ClosestQuery<'a> {
    tree: &'a TriBvh,
    p: Vec3,
}

impl Kernel<f32> for ClosestQuery<'_> {
    /// Slot of the closest triangle.
    type Output = Option<usize>;

    fn run<S: BackendAll + Backend<f32>>(self, g: Gang<S>) -> Self::Output {
        let nodes = &self.tree.bvh.nodes;
        let pa = Vec3A::from(self.p);
        let pw = g.splat_vec3(self.p);
        let lanes = g.lanes::<f32>();

        let mut best_d2 = f32::INFINITY;
        let mut best_slot = usize::MAX;
        let mut stack: TraversalStack<(u32, f32)> = TraversalStack::new();
        let mut ni = 0u32;
        'traverse: loop {
            let node = &nodes[ni as usize];
            if node.is_leaf() {
                let start = node.first as usize;
                let len = node.count as usize;
                let cols = self.tree.leaf_cols(start, len, lanes);
                for off in (0..cols[0].len()).step_by(lanes) {
                    let v = array::from_fn(|k| g.load(&cols[k][off..off + lanes]));
                    let d2 = dist2_tri_wide(g, pw, v);
                    take_nearest(g, d2, start + off, &mut best_d2, &mut best_slot);
                }
            } else {
                let l = node.first;
                let dl = nodes[l as usize].distance_squared(pa);
                let dr = nodes[l as usize + 1].distance_squared(pa);
                // Descend into the nearer child, deferring the farther one.
                let (near, far) = if dl <= dr { (l, l + 1) } else { (l + 1, l) };
                let (d_near, d_far) = if dl <= dr { (dl, dr) } else { (dr, dl) };
                if d_near < best_d2 {
                    if d_far < best_d2 {
                        stack.push((far, d_far));
                    }
                    ni = near;
                    continue;
                }
            }
            while let Some((n, d)) = stack.pop() {
                if d < best_d2 {
                    ni = n;
                    continue 'traverse;
                }
            }
            break;
        }
        (best_slot != usize::MAX).then_some(best_slot)
    }
}
