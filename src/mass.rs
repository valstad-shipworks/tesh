//! Mass properties from a closed, outward-wound triangle boundary, via the signed
//! origin-tetrahedra sum (Blow & Binstock's covariance method).

use glam::{Mat3, Vec3};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::math::F32Ext;

/// Volume, center of mass, and inertia tensor of a solid at unit density.
///
/// The [`inertia`](MassProperties::inertia) tensor is taken about the
/// [`center_of_mass`](MassProperties::center_of_mass) and scales linearly with density.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MassProperties {
    pub volume: f32,
    pub center_of_mass: Vec3,
    /// Inertia tensor about the center of mass, at unit density.
    pub inertia: Mat3,
}

impl MassProperties {
    /// Mass at the given uniform density.
    #[inline]
    #[must_use]
    pub fn mass(&self, density: f32) -> f32 {
        self.volume * density
    }

    /// Inertia tensor about the center of mass at the given uniform density.
    #[inline]
    #[must_use]
    pub fn inertia_at(&self, density: f32) -> Mat3 {
        self.inertia * density
    }

    /// Principal inertia: the three moments (eigenvalues) and their axes (unit eigenvectors
    /// as the columns of the returned matrix), at unit density.
    #[must_use]
    pub fn principal(&self) -> (Vec3, Mat3) {
        jacobi_eigen_symmetric(self.inertia)
    }
}

#[inline]
fn outer(u: Vec3, v: Vec3) -> Mat3 {
    Mat3::from_cols(u * v.x, u * v.y, u * v.z)
}

#[inline]
fn trace(m: Mat3) -> f32 {
    m.x_axis.x + m.y_axis.y + m.z_axis.z
}

/// Accumulates mass properties from a closed, outward-wound triangle boundary.
pub(crate) fn from_boundary(tris: impl Iterator<Item = [Vec3; 3]>) -> MassProperties {
    // Covariance of the canonical tetrahedron (origin, e_x, e_y, e_z), scaled by 1/120.
    let canon =
        Mat3::from_cols_array(&[2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]) * (1.0 / 120.0);

    let mut covariance = Mat3::ZERO;
    let mut vol6 = 0.0;
    let mut moment = Vec3::ZERO;
    for [a, b, c] in tris {
        let m = Mat3::from_cols(a, b, c);
        let det = m.determinant();
        vol6 += det;
        moment += (a + b + c) * det;
        covariance += (m * canon * m.transpose()) * det;
    }

    if vol6.abs() <= f32::MIN_POSITIVE {
        return MassProperties {
            volume: 0.0,
            center_of_mass: Vec3::ZERO,
            inertia: Mat3::ZERO,
        };
    }

    let sign = vol6.signum();
    let volume = vol6 * sign / 6.0;
    let center_of_mass = moment / (4.0 * vol6);
    let covariance = covariance * sign;

    // Shift the covariance to the center of mass, then convert to an inertia tensor.
    let central = covariance - outer(center_of_mass, center_of_mass) * volume;
    let inertia = Mat3::IDENTITY * trace(central) - central;

    MassProperties {
        volume,
        center_of_mass,
        inertia,
    }
}

/// Cyclic Jacobi eigen-decomposition of a symmetric 3×3 matrix. Returns the eigenvalues and
/// a rotation whose columns are the corresponding unit eigenvectors.
fn jacobi_eigen_symmetric(m: Mat3) -> (Vec3, Mat3) {
    let mut a = m;
    let mut v = Mat3::IDENTITY;

    for _ in 0..24 {
        // Largest off-diagonal magnitude and its position.
        let (mut p, mut q) = (0, 1);
        let mut max = a.col(1).x.abs();
        let off02 = a.col(2).x.abs();
        let off12 = a.col(2).y.abs();
        if off02 > max {
            max = off02;
            p = 0;
            q = 2;
        }
        if off12 > max {
            max = off12;
            p = 1;
            q = 2;
        }
        if max <= 1e-9 {
            break;
        }

        let apq = a.col(q)[p];
        let app = a.col(p)[p];
        let aqq = a.col(q)[q];
        let theta = (aqq - app) / (2.0 * apq);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let cos = 1.0 / (t * t + 1.0).sqrt();
        let sin = t * cos;

        let mut rot = Mat3::IDENTITY;
        rot.col_mut(p)[p] = cos;
        rot.col_mut(q)[q] = cos;
        rot.col_mut(q)[p] = sin;
        rot.col_mut(p)[q] = -sin;

        a = rot.transpose() * a * rot;
        v *= rot;
    }

    (Vec3::new(a.col(0).x, a.col(1).y, a.col(2).z), v)
}
