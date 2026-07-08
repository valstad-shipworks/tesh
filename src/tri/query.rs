use alloc::vec::Vec;
use glam::Vec3;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::math::F32Ext;
use crate::tri::TriMesh;
use crate::triangle;

/// A ray hit against a mesh: the face index, ray parameter `t`, and world-space point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub face: u32,
    pub t: f32,
    pub point: Vec3,
}

/// The full result of a closest-point query, resolved in a single traversal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceHit {
    /// Face holding the closest point.
    pub face: u32,
    /// The closest point on the surface.
    pub point: Vec3,
    /// Barycentric coordinates of [`point`](SurfaceHit::point) within its face.
    pub barycentric: Vec3,
    /// Negative inside, positive outside.
    pub signed_distance: f32,
    /// Unit outward normal at the closest point: the offset direction to the query point
    /// when it is off the surface, else the face normal.
    pub normal: Vec3,
}

impl SurfaceHit {
    /// Unsigned distance to the surface.
    #[inline]
    #[must_use]
    pub fn distance(&self) -> f32 {
        self.signed_distance.abs()
    }
}

/// Offset length below which f32 rounding dominates the offset's direction, so the face
/// normal stands in for it (and the sign, at that magnitude irrelevant, defaults outside).
const ON_SURFACE_EPS: f32 = 1e-6;
/// Barycentric margin inside which the closest point counts as interior to its face.
const INTERIOR_EPS: f32 = 1e-5;

impl TriMesh {
    /// Builds the [`SurfaceHit`] for a query point whose closest face is already known.
    /// The face normal signs interior hits; on an edge or vertex a single face's normal
    /// cannot decide, so containment (one extra traversal under `bvh`) settles those.
    fn assemble_surface_hit(&self, p: Vec3, face: u32) -> SurfaceHit {
        let [a, b, c] = self.triangle(face as usize);
        let q = triangle::closest_point(p, a, b, c);
        let barycentric = triangle::barycentric(q, a, b, c);
        let face_normal = triangle::normal(a, b, c);
        let delta = p - q;
        let dist = delta.length();

        let on_surface = dist <= ON_SURFACE_EPS;
        let interior = barycentric.min_element() > INTERIOR_EPS;
        let inside = if on_surface {
            false
        } else if interior {
            delta.dot(face_normal) < 0.0
        } else {
            self.contains_point(p)
        };
        let sign = if inside { -1.0 } else { 1.0 };
        SurfaceHit {
            face,
            point: q,
            barycentric,
            signed_distance: sign * dist,
            normal: if on_surface {
                face_normal
            } else {
                delta * (sign / dist)
            },
        }
    }

    /// [`SurfaceHit`] for each query point (see [`surface_hit`](TriMesh::surface_hit)).
    #[must_use]
    pub fn surface_hits(&self, points: &[Vec3]) -> Vec<Option<SurfaceHit>> {
        points.iter().map(|&p| self.surface_hit(p)).collect()
    }

    /// The closest surface point for each query point (see
    /// [`closest_point`](TriMesh::closest_point)).
    #[must_use]
    pub fn closest_points(&self, points: &[Vec3]) -> Vec<Vec3> {
        points.iter().map(|&p| self.closest_point(p)).collect()
    }

    /// Unsigned distance to the surface for each query point (see
    /// [`distance`](TriMesh::distance)).
    #[must_use]
    pub fn distances(&self, points: &[Vec3]) -> Vec<f32> {
        points.iter().map(|&p| self.distance(p)).collect()
    }

    /// Signed distance to the surface for each query point (see
    /// [`signed_distance`](TriMesh::signed_distance)).
    #[must_use]
    pub fn signed_distances(&self, points: &[Vec3]) -> Vec<f32> {
        points.iter().map(|&p| self.signed_distance(p)).collect()
    }

    /// Containment of each query point (see [`contains_point`](TriMesh::contains_point)).
    #[must_use]
    pub fn contains_points(&self, points: &[Vec3]) -> Vec<bool> {
        points.iter().map(|&p| self.contains_point(p)).collect()
    }

    /// Nearest surface intersection for each `(origin, dir)` ray (see
    /// [`ray_intersect`](TriMesh::ray_intersect)).
    #[must_use]
    pub fn ray_intersects(&self, rays: &[(Vec3, Vec3)]) -> Vec<Option<RayHit>> {
        rays.iter()
            .map(|&(origin, dir)| self.ray_intersect(origin, dir))
            .collect()
    }
}

#[cfg(feature = "bvh")]
impl TriMesh {
    /// The point on the surface closest to `p`, or `p` itself for an empty mesh.
    #[inline]
    #[must_use]
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        self.bvh().closest_point(p)
    }

    /// Unsigned distance from `p` to the surface; infinite for an empty mesh.
    #[inline]
    #[must_use]
    pub fn distance(&self, p: Vec3) -> f32 {
        self.bvh().distance(p)
    }

    /// Signed distance to the surface: negative inside, positive outside. Infinite for an
    /// empty mesh.
    #[inline]
    #[must_use]
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        self.bvh().signed_distance(p)
    }

    /// Whether `p` is enclosed by the surface (watertight, consistently wound meshes).
    #[inline]
    #[must_use]
    pub fn contains_point(&self, p: Vec3) -> bool {
        self.bvh().contains_point(p)
    }

    /// Nearest intersection of the ray `origin + t * dir` (`t >= 0`) with the surface.
    #[inline]
    #[must_use]
    pub fn ray_intersect(&self, origin: Vec3, dir: Vec3) -> Option<RayHit> {
        self.bvh().ray_intersect(origin, dir)
    }

    /// The [`SurfaceHit`] closest to `p`, or `None` for an empty mesh. One closest-point
    /// traversal, plus a containment traversal only for edge/vertex features.
    #[must_use]
    pub fn surface_hit(&self, p: Vec3) -> Option<SurfaceHit> {
        let face = self.bvh().closest_face(p)?;
        Some(self.assemble_surface_hit(p, face))
    }
}

#[cfg(not(feature = "bvh"))]
mod brute {
    use core::f32::consts::PI;

    use glam::Vec3;

    #[cfg(not(feature = "std"))]
    #[allow(unused_imports)]
    use crate::math::F32Ext;
    use crate::tri::TriMesh;
    use crate::tri::query::{RayHit, SurfaceHit};
    use crate::triangle;

    impl TriMesh {
        /// Slot of the face closest to `p`, by brute force.
        fn closest_face_brute(&self, p: Vec3) -> Option<u32> {
            let mut best = None;
            let mut best_d = f32::INFINITY;
            for (i, [a, b, c]) in self.triangles().enumerate() {
                let d = triangle::distance_squared(p, a, b, c);
                if d < best_d {
                    best_d = d;
                    best = Some(i as u32);
                }
            }
            best
        }

        /// The point on the surface closest to `p` (brute force over faces), or `p` itself
        /// for an empty mesh.
        #[must_use]
        pub fn closest_point(&self, p: Vec3) -> Vec3 {
            match self.closest_face_brute(p) {
                Some(face) => {
                    let [a, b, c] = self.triangle(face as usize);
                    triangle::closest_point(p, a, b, c)
                }
                None => p,
            }
        }

        /// Unsigned distance from `p` to the surface; infinite for an empty mesh.
        #[must_use]
        pub fn distance(&self, p: Vec3) -> f32 {
            if self.faces.is_empty() {
                return f32::INFINITY;
            }
            (p - self.closest_point(p)).length()
        }

        /// Signed distance to the surface: negative inside, positive outside. Infinite for
        /// an empty mesh.
        #[must_use]
        pub fn signed_distance(&self, p: Vec3) -> f32 {
            let d = self.distance(p);
            if self.contains_point(p) { -d } else { d }
        }

        /// Whether `p` is enclosed by the surface, via the generalized winding number.
        #[must_use]
        pub fn contains_point(&self, p: Vec3) -> bool {
            let mut omega = 0.0;
            for [a, b, c] in self.triangles() {
                let a = a - p;
                let b = b - p;
                let c = c - p;
                let la = a.length();
                let lb = b.length();
                let lc = c.length();
                let numer = a.dot(b.cross(c));
                let denom = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
                omega += 2.0 * numer.atan2(denom);
            }
            (omega / (4.0 * PI)).abs() > 0.5
        }

        /// Nearest intersection of the ray `origin + t * dir` (`t >= 0`) with the surface,
        /// via Möller–Trumbore against every face (brute force).
        #[must_use]
        pub fn ray_intersect(&self, origin: Vec3, dir: Vec3) -> Option<RayHit> {
            let mut hit: Option<RayHit> = None;
            for (i, [a, b, c]) in self.triangles().enumerate() {
                let Some(t) = triangle::ray_intersect(origin, dir, a, b, c) else {
                    continue;
                };
                if hit.is_none_or(|h| t < h.t) {
                    hit = Some(RayHit {
                        face: i as u32,
                        t,
                        point: origin + dir * t,
                    });
                }
            }
            hit
        }

        /// The [`SurfaceHit`] closest to `p` (brute force), or `None` for an empty mesh.
        #[must_use]
        pub fn surface_hit(&self, p: Vec3) -> Option<SurfaceHit> {
            let face = self.closest_face_brute(p)?;
            Some(self.assemble_surface_hit(p, face))
        }
    }
}
