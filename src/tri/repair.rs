//! Mesh conditioning: welding, cleanup, orientation, hole filling. Each method rebuilds
//! in place and drops the query cache.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::math::{F32Ext, F64Ext};
use crate::internal::edge_key;
use crate::tri::TriMesh;
use crate::triangle;

#[inline]
fn face_has_directed(f: [u32; 3], u: u32, v: u32) -> bool {
    (f[0] == u && f[1] == v) || (f[1] == u && f[2] == v) || (f[2] == u && f[0] == v)
}

impl TriMesh {
    /// Merges vertices sharing a grid cell of spacing `tolerance` and rewrites the faces.
    /// The grid is the guarantee: cell-mates up to `√3 · tolerance` apart merge, closer
    /// pairs straddling a cell boundary do not. `tolerance == 0` welds exact coincidents
    /// only. Degenerate faces created by the merge are left in place; follow with
    /// [`remove_degenerate_faces`](TriMesh::remove_degenerate_faces).
    pub fn merge_vertices(&mut self, tolerance: f32) {
        let inv = if tolerance > 0.0 {
            f64::from(tolerance).recip()
        } else {
            0.0
        };
        let quantize = |x: f32| -> i64 {
            let cell = f64::from(x) * inv;
            if inv > 0.0 && cell.is_finite() && cell.abs() < 2e18 {
                cell.round() as i64
            } else {
                // Exact-bits fallback: `tolerance == 0`, a spacing the coordinates cannot
                // resolve, or a non-finite coordinate. `-0.0` normalized so it welds with
                // `+0.0`; NaNs weld only with bit-identical NaNs. Offset past the grid-cell
                // range so the two key kinds cannot collide.
                let x = if x == 0.0 { 0.0 } else { x };
                (1i64 << 62) + i64::from(x.to_bits())
            }
        };

        let mut map: BTreeMap<[i64; 3], u32> = BTreeMap::new();
        let mut new_verts: Vec<glam::Vec3> = Vec::with_capacity(self.vertices.len());
        let mut remap = vec![0u32; self.vertices.len()];
        for (i, v) in self.vertices.iter().enumerate() {
            let key = [quantize(v.x), quantize(v.y), quantize(v.z)];
            remap[i] = *map.entry(key).or_insert_with(|| {
                let id = new_verts.len() as u32;
                new_verts.push(*v);
                id
            });
        }
        for f in &mut self.faces {
            for k in f {
                *k = remap[*k as usize];
            }
        }
        self.vertices = new_verts;
        self.reset_cache();
    }

    /// Removes faces that are topologically or geometrically degenerate: any with a repeated
    /// vertex index, or with area at or below `area_eps`.
    pub fn remove_degenerate_faces(&mut self, area_eps: f32) {
        // area > eps ⇔ |cross|² > (2·eps)², sparing the per-face sqrt.
        let threshold = (2.0 * area_eps) * (2.0 * area_eps);
        let verts = &self.vertices;
        self.faces.retain(|&[a, b, c]| {
            if a == b || b == c || a == c {
                return false;
            }
            let av = triangle::area_vector(
                verts[a as usize],
                verts[b as usize],
                verts[c as usize],
            );
            av.length_squared() > threshold
        });
        self.reset_cache();
    }

    /// Removes duplicate faces (those with the same set of vertices), keeping the first.
    pub fn remove_duplicate_faces(&mut self) {
        let mut seen: BTreeSet<[u32; 3]> = BTreeSet::new();
        self.faces.retain(|f| {
            let mut key = *f;
            key.sort_unstable();
            seen.insert(key)
        });
        self.reset_cache();
    }

    /// Drops vertices not referenced by any face and compacts the vertex buffer.
    pub fn remove_unreferenced_vertices(&mut self) {
        let mut remap = vec![u32::MAX; self.vertices.len()];
        let mut new_verts = Vec::with_capacity(self.vertices.len());
        for f in &self.faces {
            for &vi in f {
                if remap[vi as usize] == u32::MAX {
                    remap[vi as usize] = new_verts.len() as u32;
                    new_verts.push(self.vertices[vi as usize]);
                }
            }
        }
        for f in &mut self.faces {
            for k in f {
                *k = remap[*k as usize];
            }
        }
        self.vertices = new_verts;
        self.reset_cache();
    }

    /// Makes every face wind consistently with its neighbours, propagating an orientation
    /// across each connected component (does not by itself decide inward vs outward).
    pub fn orient_consistent(&mut self) {
        let ef = self.edge_faces();
        let n = self.faces.len();
        let mut visited = vec![false; n];
        let mut stack: Vec<u32> = Vec::new();

        for seed in 0..n {
            if visited[seed] {
                continue;
            }
            visited[seed] = true;
            stack.push(seed as u32);
            while let Some(f) = stack.pop() {
                let face = self.faces[f as usize];
                let edges = [
                    (face[0], face[1]),
                    (face[1], face[2]),
                    (face[2], face[0]),
                ];
                for (u, v) in edges {
                    let Some(neighbours) = ef.get(&edge_key(u, v)) else {
                        continue;
                    };
                    for &g in neighbours {
                        if g == f || visited[g as usize] {
                            continue;
                        }
                        // Consistent neighbours traverse the shared edge the opposite way; if
                        // `g` traverses it the same way (u -> v), flip it.
                        if face_has_directed(self.faces[g as usize], u, v) {
                            self.faces[g as usize].swap(1, 2);
                        }
                        visited[g as usize] = true;
                        stack.push(g);
                    }
                }
            }
        }
        self.reset_cache();
    }

    /// Reverses every face's winding.
    pub fn flip_faces(&mut self) {
        for f in &mut self.faces {
            f.swap(1, 2);
        }
        self.reset_cache();
    }

    /// Flips the whole mesh if it is wound inward, so the faces face outward (positive
    /// enclosed volume). Assumes a consistently wound, closed surface — call
    /// [`orient_consistent`](TriMesh::orient_consistent) first if unsure.
    pub fn make_outward(&mut self) {
        if self.signed_volume() < 0.0 {
            self.flip_faces();
        }
    }

    /// Orients the mesh into a clean outward-facing state: consistent winding across every
    /// component, then flipped outward.
    #[inline]
    pub fn orient_outward(&mut self) {
        self.orient_consistent();
        self.make_outward();
    }

    /// Fills boundary holes by fanning each closed loop of single-face edges, returning the
    /// number of loops filled (holes pinched at a shared vertex may walk as one). Winding
    /// matches the surrounding faces; run [`orient_outward`](TriMesh::orient_outward) after
    /// if the input winding was inconsistent.
    pub fn fill_holes(&mut self) -> usize {
        // Directed boundary half-edges: the single-face edges, in their face's direction.
        // A pinch vertex has several outgoing half-edges, so each maps to a list.
        let mut next: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (key, faces) in self.edge_faces() {
            if faces.len() != 1 {
                continue;
            }
            let f = self.faces[faces[0] as usize];
            let (u, v) = if face_has_directed(f, key.0, key.1) {
                (key.0, key.1)
            } else {
                (key.1, key.0)
            };
            next.entry(u).or_default().push(v);
        }

        let mut filled = 0;
        while let Some((&start, _)) = next.iter().next() {
            // Walk boundary half-edges from `start`, consuming each as it is traversed.
            let mut loop_verts = Vec::new();
            let mut cur = start;
            let closed = loop {
                let Some(outs) = next.get_mut(&cur) else {
                    break false;
                };
                let nxt = outs.pop().expect("emptied entries are removed");
                if outs.is_empty() {
                    next.remove(&cur);
                }
                loop_verts.push(cur);
                cur = nxt;
                if cur == start {
                    break true;
                }
            };
            // Only a walk that returned to its start bounds a hole; an unclosed chain
            // (non-manifold boundary) is dropped rather than fanned open.
            if !closed || loop_verts.len() < 3 {
                continue;
            }
            // Fan the loop; reverse each boundary edge so the patch closes the surface.
            let apex = loop_verts[0];
            for w in loop_verts[1..].windows(2) {
                if w[0] != apex && w[1] != apex {
                    self.faces.push([apex, w[1], w[0]]);
                }
            }
            filled += 1;
        }
        self.reset_cache();
        filled
    }
}
