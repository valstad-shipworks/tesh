//! Lazily-built query caches embedded in the meshes: built behind `&self` on first query,
//! excluded from equality/serde, dropped on clone.

#[cfg(any(feature = "bvh", feature = "kdtree"))]
mod lazy {
    use core::fmt;
    use std::sync::OnceLock;

    /// A write-once cache cell: [`get_or_init`](Lazy::get_or_init) builds the value behind a
    /// shared reference, and the standard derives treat it as absent (clone resets it,
    /// equality ignores it).
    pub struct Lazy<T>(OnceLock<T>);

    impl<T> Lazy<T> {
        #[inline]
        pub fn get_or_init(&self, f: impl FnOnce() -> T) -> &T {
            self.0.get_or_init(f)
        }
    }

    impl<T> Default for Lazy<T> {
        #[inline]
        fn default() -> Self {
            Self(OnceLock::new())
        }
    }

    impl<T> Clone for Lazy<T> {
        #[inline]
        fn clone(&self) -> Self {
            Self::default()
        }
    }

    impl<T> PartialEq for Lazy<T> {
        #[inline]
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }

    impl<T> fmt::Debug for Lazy<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("Lazy(..)")
        }
    }
}

#[cfg(any(feature = "bvh", feature = "kdtree"))]
pub(crate) use lazy::Lazy;

/// Query caches for a [`TriMesh`](crate::TriMesh).
#[derive(Default, Clone, PartialEq, Debug)]
pub(crate) struct TriCache {
    #[cfg(feature = "bvh")]
    pub(crate) bvh: Lazy<crate::accel::TriBvh>,
    #[cfg(feature = "kdtree")]
    pub(crate) kdtree: Lazy<crate::accel::VertexKdTree>,
}

/// Query caches for a [`TetMesh`](crate::TetMesh).
#[derive(Default, Clone, PartialEq, Debug)]
pub(crate) struct TetCache {
    #[cfg(feature = "kdtree")]
    pub(crate) kdtree: Lazy<crate::accel::VertexKdTree>,
    #[cfg(feature = "bvh")]
    pub(crate) surface: Lazy<crate::tri::TriMesh>,
    #[cfg(feature = "bvh")]
    pub(crate) tet_bvh: Lazy<crate::accel::TetBvh>,
}
