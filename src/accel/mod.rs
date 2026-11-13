//! Feature-gated acceleration structures, built and cached lazily by the mesh query
//! methods: [`TriBvh`]/[`TetBvh`] ([`hydroplane_solvers::bvh`], nodes as wide as the SIMD
//! registers) behind `bvh`, [`VertexKdTree`] ([`kiddo`]) behind `kdtree`. Brute-force fallbacks can differ slightly — inside tests use ray
//! parity here but the winding number there.

#[cfg(feature = "bvh")]
mod bvh;
#[cfg(feature = "kdtree")]
mod kdtree;
#[cfg(feature = "bvh")]
mod tet_bvh;

#[cfg(feature = "bvh")]
pub use bvh::TriBvh;
#[cfg(feature = "kdtree")]
pub use kdtree::VertexKdTree;
#[cfg(feature = "bvh")]
pub use tet_bvh::TetBvh;

#[cfg(feature = "bvh")]
fn prim_bounds<const N: usize>(corners: &[glam::Vec3; N]) -> hydroplane_solvers::bvh::Aabb {
    hydroplane_solvers::bvh::Aabb::from_points(&corners.map(|v| v.to_array()))
}
