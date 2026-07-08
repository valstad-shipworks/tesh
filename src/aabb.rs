use glam::Vec3;

/// An axis-aligned bounding box, defined by its minimum and maximum corners.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// An empty box, inverted so the first [`expand`](Aabb::expand) snaps it onto a real point.
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    #[inline]
    #[must_use]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Bounding box of a single point.
    #[inline]
    #[must_use]
    pub const fn from_point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    /// Bounding box enclosing every point in `points`, or [`EMPTY`](Aabb::EMPTY) if none.
    #[must_use]
    pub fn from_points(points: &[Vec3]) -> Self {
        let mut aabb = Self::EMPTY;
        for &p in points {
            aabb.expand(p);
        }
        aabb
    }

    /// Grows the box to include `p`.
    #[inline]
    pub fn expand(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    /// Grows the box to include `other`.
    #[inline]
    pub fn union(&mut self, other: &Aabb) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// The smallest box enclosing both `self` and `other`.
    #[inline]
    #[must_use]
    pub fn merged(self, other: Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Whether the box is inverted — no point has been added to [`EMPTY`](Aabb::EMPTY).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min.cmpgt(self.max).any()
    }

    #[inline]
    #[must_use]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Full side lengths along each axis.
    #[inline]
    #[must_use]
    pub fn extents(&self) -> Vec3 {
        self.max - self.min
    }

    #[inline]
    #[must_use]
    pub fn contains(&self, p: Vec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }

    /// Center and radius of the smallest sphere enclosing this box.
    #[inline]
    #[must_use]
    pub fn bounding_sphere(&self) -> (Vec3, f32) {
        let center = self.center();
        (center, (self.max - center).length())
    }

    /// Closest point to `p` lying inside (or on) the box.
    #[inline]
    #[must_use]
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        p.clamp(self.min, self.max)
    }

    /// Squared distance from `p` to the box (zero when `p` is inside).
    #[inline]
    #[must_use]
    pub fn distance_squared(&self, p: Vec3) -> f32 {
        (p - self.closest_point(p)).length_squared()
    }
}
