//! Float methods the `libm`-only build lacks, supplied from [`libm`]. Import
//! unconditionally where needed; under `std` the inherent methods win.

#[cfg(not(feature = "std"))]
pub(crate) trait F32Ext {
    fn sqrt(self) -> f32;
    fn atan2(self, x: f32) -> f32;
    fn sin(self) -> f32;
    fn cos(self) -> f32;
    fn round(self) -> f32;
    fn ceil(self) -> f32;
}

#[cfg(not(feature = "std"))]
impl F32Ext for f32 {
    #[inline]
    fn sqrt(self) -> f32 {
        libm::sqrtf(self)
    }
    #[inline]
    fn atan2(self, x: f32) -> f32 {
        libm::atan2f(self, x)
    }
    #[inline]
    fn sin(self) -> f32 {
        libm::sinf(self)
    }
    #[inline]
    fn cos(self) -> f32 {
        libm::cosf(self)
    }
    #[inline]
    fn round(self) -> f32 {
        libm::roundf(self)
    }
    #[inline]
    fn ceil(self) -> f32 {
        libm::ceilf(self)
    }
}

#[cfg(not(feature = "std"))]
pub(crate) trait F64Ext {
    fn round(self) -> f64;
}

#[cfg(not(feature = "std"))]
impl F64Ext for f64 {
    #[inline]
    fn round(self) -> f64 {
        libm::round(self)
    }
}
