#![cfg_attr(not(feature = "std"), no_std)]
//! Triangle surface meshes ([`TriMesh`]) and tetrahedron volume meshes ([`TetMesh`]),
//! built on [`glam`] with per-element compute through [`hydroplane`] SPMD kernels.
//! `no_std` with mandatory `alloc`: enable `std` (default) or `libm` for float math.
//!
//! Features: `bvh` and `kdtree` (default) accelerate the spatial queries — without them
//! the same methods run brute force; `serde`, `bytemuck`, `approx` add the usual derives.

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("enable either the `std` or `libm` feature for floating-point math");

extern crate alloc;

mod aabb;
mod cache;
mod internal;
mod mass;
mod math;
mod traits;

#[cfg(any(feature = "kdtree", feature = "bvh"))]
pub mod accel;
pub mod compute;
pub mod primitives;
pub mod tet;
pub mod tetrahedron;
pub mod tri;
pub mod triangle;

pub use aabb::Aabb;
pub use mass::MassProperties;
pub use tet::{BoundaryFace, TetMesh};
pub use traits::{Bounds, Mesh, Transform};
pub use tri::{Components, RayHit, SurfaceHit, TriMesh};

pub(crate) mod sealed {
    /// Private supertrait: keeps `tesh`'s core traits from being implemented downstream,
    /// so their invariants stay under our control.
    pub trait Sealed {}
}

/// Common imports: the two mesh types, their query result types, and the shared traits.
pub mod prelude {
    pub use crate::aabb::Aabb;
    pub use crate::mass::MassProperties;
    pub use crate::tet::{BoundaryFace, TetMesh};
    pub use crate::traits::{Bounds, Mesh, Transform};
    pub use crate::tri::{Components, RayHit, SurfaceHit, TriMesh};
}
