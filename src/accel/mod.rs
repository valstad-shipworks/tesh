//! Feature-gated acceleration structures, built and cached lazily by the mesh query
//! methods: [`TriBvh`]/[`TetBvh`] (binned SAH) behind `bvh`, [`VertexKdTree`] ([`kiddo`])
//! behind `kdtree`. Brute-force fallbacks can differ slightly — inside tests use ray
//! parity here but the winding number there.

#[cfg(feature = "bvh")]
mod bvh;
#[cfg(feature = "bvh")]
mod bvh2;
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

/// LIFO traversal stack for the BVH queries: a fixed inline array covers any realistic
/// tree depth, spilling to the heap only for pathological builds, so the hot query paths
/// stay allocation-free.
#[cfg(feature = "bvh")]
const STACK_INLINE: usize = 64;

#[cfg(feature = "bvh")]
pub(crate) struct TraversalStack<T> {
    inline: [T; STACK_INLINE],
    len: usize,
    spill: alloc::vec::Vec<T>,
}

#[cfg(feature = "bvh")]
impl<T: Copy + Default> TraversalStack<T> {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            inline: [T::default(); STACK_INLINE],
            len: 0,
            spill: alloc::vec::Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, v: T) {
        if self.len < STACK_INLINE {
            self.inline[self.len] = v;
            self.len += 1;
        } else {
            self.spill.push(v);
        }
    }

    #[inline]
    pub(crate) fn pop(&mut self) -> Option<T> {
        if let Some(v) = self.spill.pop() {
            return Some(v);
        }
        self.len.checked_sub(1).map(|l| {
            self.len = l;
            self.inline[l]
        })
    }
}
