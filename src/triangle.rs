//! Free-standing math on a single triangle's corner points.

use glam::Vec3;

/// Twice the area vector: `(b - a) × (c - a)`. Its length is twice the triangle area
/// and its direction is the face normal (unnormalized), following the winding `a → b → c`.
#[inline]
#[must_use]
pub fn area_vector(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a)
}

/// Unit face normal for winding `a → b → c`, or zero for a degenerate triangle.
#[inline]
#[must_use]
pub fn normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    area_vector(a, b, c).normalize_or_zero()
}

/// Triangle area.
#[inline]
#[must_use]
pub fn area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    area_vector(a, b, c).length() * 0.5
}

/// Centroid (average of the three corners).
#[inline]
#[must_use]
pub fn centroid(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (a + b + c) * (1.0 / 3.0)
}

/// Barycentric coordinates `(u, v, w)` of `p` projected onto the triangle's plane, with
/// `u + v + w = 1` weighting `a`, `b`, `c` respectively. Returns zeros for a degenerate
/// triangle.
#[must_use]
pub fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= f32::MIN_POSITIVE {
        return Vec3::ZERO;
    }
    let inv = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv;
    let w = (d00 * d21 - d01 * d20) * inv;
    Vec3::new(1.0 - v - w, v, w)
}

/// Cartesian point for barycentric weights `bary = (u, v, w)` over corners `a`, `b`, `c`.
#[inline]
#[must_use]
pub fn from_barycentric(bary: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    a * bary.x + b * bary.y + c * bary.z
}

/// The point on triangle `abc` closest to `p` (Ericson, *Real-Time Collision Detection*).
#[must_use]
pub fn closest_point(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Squared distance from `p` to the closest point on triangle `abc`.
#[inline]
#[must_use]
pub fn distance_squared(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (p - closest_point(p, a, b, c)).length_squared()
}

/// Ray parameter `t` of the intersection of `origin + t * dir` with triangle `abc`
/// (Möller–Trumbore), or `None` for a miss, a ray parallel to the plane, or `t` below a
/// small epsilon (so an origin on the surface does not hit its own triangle).
#[must_use]
pub fn ray_intersect(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    const EPS: f32 = 1e-7;
    let e1 = b - a;
    let e2 = c - a;
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < EPS {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = origin - a;
    let u = tvec.dot(pvec) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(qvec) * inv;
    (t >= EPS).then_some(t)
}
