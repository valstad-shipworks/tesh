//! Free-standing math on a single tetrahedron's corner points. Corner order `a, b, c, d`
//! is positively oriented (`d` on the positive side of plane `abc`), making
//! [`signed_volume`] non-negative.

use glam::Vec3;

/// Signed volume: `((b - a) × (c - a)) · (d - a) / 6`. Positive when `a, b, c, d` is
/// positively oriented, negative when inverted — the sign detects element inversion.
#[inline]
#[must_use]
pub fn signed_volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f32 {
    (b - a).cross(c - a).dot(d - a) * (1.0 / 6.0)
}

/// Absolute volume.
#[inline]
#[must_use]
pub fn volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f32 {
    signed_volume(a, b, c, d).abs()
}

/// Centroid (average of the four corners).
#[inline]
#[must_use]
pub fn centroid(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Vec3 {
    (a + b + c + d) * 0.25
}

/// Barycentric coordinates `(λ0, λ1, λ2, λ3)` of `p` with respect to corners
/// `a, b, c, d`, summing to 1. `p` is inside a non-degenerate tetrahedron iff all four
/// are `>= 0`. Returns `[0; 4]` for a degenerate (zero-volume) tetrahedron — a sentinel
/// (real coordinates sum to 1) that [`contains_point`] treats as containing nothing.
#[must_use]
pub fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> [f32; 4] {
    let vap = p - a;
    let vab = b - a;
    let vac = c - a;
    let vad = d - a;
    let denom = vab.cross(vac).dot(vad);
    if denom.abs() <= f32::MIN_POSITIVE {
        return [0.0; 4];
    }
    let inv = 1.0 / denom;
    let l1 = vap.cross(vac).dot(vad) * inv;
    let l2 = vab.cross(vap).dot(vad) * inv;
    let l3 = vab.cross(vac).dot(vap) * inv;
    [1.0 - l1 - l2 - l3, l1, l2, l3]
}

/// Cartesian point for barycentric weights `bary` over corners `a, b, c, d`.
#[inline]
#[must_use]
pub fn from_barycentric(bary: [f32; 4], a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Vec3 {
    a * bary[0] + b * bary[1] + c * bary[2] + d * bary[3]
}

/// Whether `p` lies inside (or on the boundary of) tetrahedron `abcd`. A degenerate
/// (zero-volume) tetrahedron contains nothing.
#[inline]
#[must_use]
pub fn contains_point(p: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> bool {
    let bary = barycentric(p, a, b, c, d);
    // A degenerate tetrahedron yields `[0; 4]`; real coordinates sum to 1.
    bary.iter().all(|&l| l >= 0.0) && bary.iter().sum::<f32>() > 0.5
}

/// Radius-ratio quality in `[0, 1]`: `3 * r_in / r_circ`, normalized so a regular
/// tetrahedron scores `1` and a degenerate (flat) one scores `0`.
#[must_use]
pub fn quality(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f32 {
    let vol = volume(a, b, c, d);
    if vol <= f32::MIN_POSITIVE {
        return 0.0;
    }
    let faces = crate::triangle::area(b, c, d)
        + crate::triangle::area(a, c, d)
        + crate::triangle::area(a, b, d)
        + crate::triangle::area(a, b, c);
    let r_in = 3.0 * vol / faces;

    let (p, q, r) = (b - a, c - a, d - a);
    let r_circ = (p.length_squared() * q.cross(r)
        + q.length_squared() * r.cross(p)
        + r.length_squared() * p.cross(q))
        .length()
        / (12.0 * vol);
    (3.0 * r_in / r_circ).clamp(0.0, 1.0)
}
