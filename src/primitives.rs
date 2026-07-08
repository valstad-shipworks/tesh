//! Procedural primitives. Closed surfaces are watertight and outward-wound, origin-centred,
//! principal axis along `+Z`; [`plane`] and [`grid`] are open. Volume meshes ([`tet_grid`],
//! [`tetrahedralize_grid`]) have positively oriented elements.

use alloc::vec::Vec;
use core::f32::consts::{PI, TAU};

use glam::Vec3;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::math::F32Ext;
use crate::tet::TetMesh;
use crate::tri::TriMesh;

struct Builder {
    verts: Vec<Vec3>,
    faces: Vec<[u32; 3]>,
}

impl Builder {
    fn new() -> Self {
        Self {
            verts: Vec::new(),
            faces: Vec::new(),
        }
    }

    #[inline]
    fn vertex(&mut self, p: Vec3) -> u32 {
        let i = self.verts.len() as u32;
        self.verts.push(p);
        i
    }

    #[inline]
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.faces.push([a, b, c]);
    }

    #[inline]
    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.tri(a, b, c);
        self.tri(a, c, d);
    }

    fn open(self) -> TriMesh {
        TriMesh::new(self.verts, self.faces)
    }

    /// Finish a closed solid, normalizing it to consistent outward winding.
    fn closed(self) -> TriMesh {
        let mut mesh = TriMesh::new(self.verts, self.faces);
        mesh.orient_outward();
        mesh
    }
}

/// An axis-aligned box of the given full side lengths, centred on the origin.
#[must_use]
pub fn box3(extents: Vec3) -> TriMesh {
    let h = extents * 0.5;
    let mut b = Builder::new();
    let signs = [-1.0f32, 1.0];
    let mut c = [0u32; 8];
    for (i, slot) in c.iter_mut().enumerate() {
        let x = signs[i & 1];
        let y = signs[(i >> 1) & 1];
        let z = signs[(i >> 2) & 1];
        *slot = b.vertex(Vec3::new(x * h.x, y * h.y, z * h.z));
    }
    // Faces per axis pair (indices are corner bit patterns x + 2y + 4z).
    b.quad(c[0], c[2], c[3], c[1]); // -z
    b.quad(c[4], c[5], c[7], c[6]); // +z
    b.quad(c[0], c[1], c[5], c[4]); // -y
    b.quad(c[2], c[6], c[7], c[3]); // +y
    b.quad(c[0], c[4], c[6], c[2]); // -x
    b.quad(c[1], c[3], c[7], c[5]); // +x
    b.closed()
}

/// A cube of the given side length.
#[inline]
#[must_use]
pub fn cube(side: f32) -> TriMesh {
    box3(Vec3::splat(side))
}

/// A regular tetrahedron whose vertices lie on a sphere of the given radius.
#[must_use]
pub fn tetrahedron(radius: f32) -> TriMesh {
    let s = radius / 3.0f32.sqrt();
    let mut b = Builder::new();
    let v0 = b.vertex(Vec3::new(s, s, s));
    let v1 = b.vertex(Vec3::new(s, -s, -s));
    let v2 = b.vertex(Vec3::new(-s, s, -s));
    let v3 = b.vertex(Vec3::new(-s, -s, s));
    b.tri(v0, v1, v2);
    b.tri(v0, v3, v1);
    b.tri(v0, v2, v3);
    b.tri(v1, v3, v2);
    b.closed()
}

/// A latitude–longitude sphere with `sectors` meridians and `stacks` latitude bands
/// (`stacks - 1` interior parallels between the poles).
#[must_use]
pub fn uv_sphere(radius: f32, sectors: usize, stacks: usize) -> TriMesh {
    let sectors = sectors.max(3);
    let stacks = stacks.max(2);
    let mut b = Builder::new();

    let top = b.vertex(Vec3::new(0.0, 0.0, radius));
    let mut rings = Vec::new();
    for i in 1..stacks {
        let phi = PI * i as f32 / stacks as f32;
        let (sp, cp) = (phi.sin(), phi.cos());
        let start = b.verts.len() as u32;
        rings.push(start);
        for j in 0..sectors {
            let theta = TAU * j as f32 / sectors as f32;
            b.vertex(Vec3::new(radius * sp * theta.cos(), radius * sp * theta.sin(), radius * cp));
        }
    }
    let bottom = b.vertex(Vec3::new(0.0, 0.0, -radius));

    let sec = sectors as u32;
    let first = rings[0];
    for j in 0..sec {
        b.tri(top, first + j, first + (j + 1) % sec);
    }
    for k in 0..rings.len() - 1 {
        let (ra, rb) = (rings[k], rings[k + 1]);
        for j in 0..sec {
            let j1 = (j + 1) % sec;
            b.quad(ra + j, rb + j, rb + j1, ra + j1);
        }
    }
    let last = *rings.last().unwrap();
    for j in 0..sec {
        b.tri(bottom, last + (j + 1) % sec, last + j);
    }
    b.closed()
}

/// An icosphere: a subdivided icosahedron projected onto a sphere. Each subdivision
/// quadruples the triangle count (0 gives the 20-face icosahedron).
#[must_use]
pub fn icosphere(radius: f32, subdivisions: usize) -> TriMesh {
    let t = (1.0 + 5.0f32.sqrt()) * 0.5;
    let mut verts: Vec<Vec3> = [
        Vec3::new(-1.0, t, 0.0),
        Vec3::new(1.0, t, 0.0),
        Vec3::new(-1.0, -t, 0.0),
        Vec3::new(1.0, -t, 0.0),
        Vec3::new(0.0, -1.0, t),
        Vec3::new(0.0, 1.0, t),
        Vec3::new(0.0, -1.0, -t),
        Vec3::new(0.0, 1.0, -t),
        Vec3::new(t, 0.0, -1.0),
        Vec3::new(t, 0.0, 1.0),
        Vec3::new(-t, 0.0, -1.0),
        Vec3::new(-t, 0.0, 1.0),
    ]
    .into_iter()
    .map(|v| v.normalize())
    .collect();

    let mut faces: Vec<[u32; 3]> = alloc::vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    for _ in 0..subdivisions {
        let mut midpoints: alloc::collections::BTreeMap<(u32, u32), u32> =
            alloc::collections::BTreeMap::new();
        let mut mid = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            *midpoints.entry(key).or_insert_with(|| {
                let m = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
                let id = verts.len() as u32;
                verts.push(m);
                id
            })
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in faces {
            let ab = mid(a, b, &mut verts);
            let bc = mid(b, c, &mut verts);
            let ca = mid(c, a, &mut verts);
            next.push([a, ab, ca]);
            next.push([b, bc, ab]);
            next.push([c, ca, bc]);
            next.push([ab, bc, ca]);
        }
        faces = next;
    }

    for v in &mut verts {
        *v *= radius;
    }
    let mut mesh = TriMesh::new(verts, faces);
    mesh.orient_outward();
    mesh
}

/// A cylinder of the given radius and height, axis along `+Z`, centred on the origin.
#[must_use]
pub fn cylinder(radius: f32, height: f32, sectors: usize) -> TriMesh {
    let sectors = sectors.max(3);
    let h = height * 0.5;
    let mut b = Builder::new();
    let top_c = b.vertex(Vec3::new(0.0, 0.0, h));
    let bot_c = b.vertex(Vec3::new(0.0, 0.0, -h));

    let ring_top = b.verts.len() as u32;
    for j in 0..sectors {
        let a = TAU * j as f32 / sectors as f32;
        b.vertex(Vec3::new(radius * a.cos(), radius * a.sin(), h));
    }
    let ring_bot = b.verts.len() as u32;
    for j in 0..sectors {
        let a = TAU * j as f32 / sectors as f32;
        b.vertex(Vec3::new(radius * a.cos(), radius * a.sin(), -h));
    }

    let sec = sectors as u32;
    for j in 0..sec {
        let j1 = (j + 1) % sec;
        b.tri(top_c, ring_top + j, ring_top + j1);
        b.tri(bot_c, ring_bot + j1, ring_bot + j);
        b.quad(ring_top + j, ring_bot + j, ring_bot + j1, ring_top + j1);
    }
    b.closed()
}

/// A cone of the given base radius and height, centred on the origin: apex at `+height/2`,
/// base disc at `-height/2`.
#[must_use]
pub fn cone(radius: f32, height: f32, sectors: usize) -> TriMesh {
    let sectors = sectors.max(3);
    let h = height * 0.5;
    let mut b = Builder::new();
    let apex = b.vertex(Vec3::new(0.0, 0.0, h));
    let base_c = b.vertex(Vec3::new(0.0, 0.0, -h));
    let ring = b.verts.len() as u32;
    for j in 0..sectors {
        let a = TAU * j as f32 / sectors as f32;
        b.vertex(Vec3::new(radius * a.cos(), radius * a.sin(), -h));
    }
    let sec = sectors as u32;
    for j in 0..sec {
        let j1 = (j + 1) % sec;
        b.tri(apex, ring + j, ring + j1);
        b.tri(base_c, ring + j1, ring + j);
    }
    b.closed()
}

/// A torus of the given ring radius and tube radius in the `XY` plane, centred on the origin.
#[must_use]
pub fn torus(radius: f32, tube: f32, sectors: usize, sides: usize) -> TriMesh {
    let sectors = sectors.max(3);
    let sides = sides.max(3);
    let mut b = Builder::new();
    for i in 0..sectors {
        let u = TAU * i as f32 / sectors as f32;
        let (cu, su) = (u.cos(), u.sin());
        for j in 0..sides {
            let v = TAU * j as f32 / sides as f32;
            let (cv, sv) = (v.cos(), v.sin());
            let r = radius + tube * cv;
            b.vertex(Vec3::new(r * cu, r * su, tube * sv));
        }
    }
    let (s, d) = (sectors as u32, sides as u32);
    for i in 0..s {
        for j in 0..d {
            let i1 = (i + 1) % s;
            let j1 = (j + 1) % d;
            b.quad(i * d + j, i1 * d + j, i1 * d + j1, i * d + j1);
        }
    }
    b.closed()
}

/// A capsule: a cylinder of the given radius and cylindrical height, capped by hemispheres,
/// axis along `+Z`, centred on the origin.
#[must_use]
pub fn capsule(radius: f32, height: f32, sectors: usize, stacks: usize) -> TriMesh {
    let sectors = sectors.max(3);
    let stacks = stacks.max(1);
    let h = height * 0.5;
    let mut b = Builder::new();
    let sec = sectors as u32;

    let top = b.vertex(Vec3::new(0.0, 0.0, h + radius));
    let mut rings = Vec::new();

    // Upper hemisphere.
    for i in 1..=stacks {
        let phi = 0.5 * PI * i as f32 / stacks as f32;
        let (sp, cp) = (phi.sin(), phi.cos());
        rings.push(b.verts.len() as u32);
        for j in 0..sectors {
            let a = TAU * j as f32 / sectors as f32;
            b.vertex(Vec3::new(radius * sp * a.cos(), radius * sp * a.sin(), h + radius * cp));
        }
    }
    // Lower hemisphere.
    for i in 0..stacks {
        let phi = 0.5 * PI + 0.5 * PI * i as f32 / stacks as f32;
        let (sp, cp) = (phi.sin(), phi.cos());
        rings.push(b.verts.len() as u32);
        for j in 0..sectors {
            let a = TAU * j as f32 / sectors as f32;
            b.vertex(Vec3::new(radius * sp * a.cos(), radius * sp * a.sin(), -h + radius * cp));
        }
    }
    let bottom = b.vertex(Vec3::new(0.0, 0.0, -h - radius));

    let first = rings[0];
    for j in 0..sec {
        b.tri(top, first + j, first + (j + 1) % sec);
    }
    for k in 0..rings.len() - 1 {
        let (ra, rb) = (rings[k], rings[k + 1]);
        for j in 0..sec {
            let j1 = (j + 1) % sec;
            b.quad(ra + j, rb + j, rb + j1, ra + j1);
        }
    }
    let last = *rings.last().unwrap();
    for j in 0..sec {
        b.tri(bottom, last + (j + 1) % sec, last + j);
    }
    b.closed()
}

/// A flat rectangular grid in the `XY` plane of the given full width and depth, tessellated
/// into `nx` by `ny` cells. Open and single-sided, with normals along `+Z`.
#[must_use]
pub fn grid(width: f32, depth: f32, nx: usize, ny: usize) -> TriMesh {
    let nx = nx.max(1);
    let ny = ny.max(1);
    let mut b = Builder::new();
    for iy in 0..=ny {
        for ix in 0..=nx {
            let x = (ix as f32 / nx as f32 - 0.5) * width;
            let y = (iy as f32 / ny as f32 - 0.5) * depth;
            b.vertex(Vec3::new(x, y, 0.0));
        }
    }
    let stride = (nx + 1) as u32;
    for iy in 0..ny as u32 {
        for ix in 0..nx as u32 {
            let a = iy * stride + ix;
            b.quad(a, a + 1, a + 1 + stride, a + stride);
        }
    }
    b.open()
}

/// A square [`grid`] of the given side length.
#[inline]
#[must_use]
pub fn plane(size: f32, subdivisions: usize) -> TriMesh {
    grid(size, size, subdivisions.max(1), subdivisions.max(1))
}

/// The six diagonal walks of the Kuhn subdivision (conforming across cells), with each
/// order's permutation parity: on an axis-aligned grid the walk's signed volume carries the
/// parity's sign, so odd orders need a corner swap.
const KUHN_AXIS_ORDERS: [([usize; 3], bool); 6] = [
    ([0, 1, 2], false),
    ([0, 2, 1], true),
    ([1, 0, 2], true),
    ([1, 2, 0], false),
    ([2, 0, 1], false),
    ([2, 1, 0], true),
];

/// Emits the six Kuhn tetrahedra of one grid cell at `(i, j, k)`, positively oriented,
/// resolving lattice corners to vertex indices through `id`.
fn kuhn_cell(
    (i, j, k): (usize, usize, usize),
    mut id: impl FnMut(usize, usize, usize) -> u32,
    tets: &mut Vec<[u32; 4]>,
) {
    for (order, odd) in KUHN_AXIS_ORDERS {
        let mut corner = [i, j, k];
        let mut tet = [id(corner[0], corner[1], corner[2]), 0, 0, 0];
        for (n, &axis) in order.iter().enumerate() {
            corner[axis] += 1;
            tet[n + 1] = id(corner[0], corner[1], corner[2]);
        }
        if odd {
            tet.swap(2, 3);
        }
        tets.push(tet);
    }
}

/// A `cells[0] × cells[1] × cells[2]` axis-aligned box of `size` full extents, centred on
/// the origin, with each cell split into six tetrahedra (the Kuhn subdivision, conforming
/// across neighbouring cells).
#[must_use]
pub fn tet_grid(cells: [usize; 3], size: Vec3) -> TetMesh {
    let [nx, ny, nz] = cells.map(|c| c.max(1));
    let step = Vec3::new(
        size.x / nx as f32,
        size.y / ny as f32,
        size.z / nz as f32,
    );
    let origin = size * -0.5;
    let id = |i: usize, j: usize, k: usize| ((k * (ny + 1) + j) * (nx + 1) + i) as u32;

    let mut vertices = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                vertices.push(origin + Vec3::new(i as f32, j as f32, k as f32) * step);
            }
        }
    }

    let mut tets = Vec::with_capacity(nx * ny * nz * 6);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                kuhn_cell((i, j, k), id, &mut tets);
            }
        }
    }
    TetMesh::new(vertices, tets)
}

/// Tetrahedralizes the interior of a closed, consistently wound surface on a uniform grid:
/// every cell of spacing `cell` whose centre the surface encloses is Kuhn-split. The
/// boundary is the voxelized (staircase) approximation, not body-fitted; features thinner
/// than a cell vanish, and a `cell` coarser than the solid yields an empty mesh.
#[must_use]
pub fn tetrahedralize_grid(surface: &TriMesh, cell: f32) -> TetMesh {
    use alloc::collections::BTreeMap;

    use crate::traits::Bounds;

    let cell = cell.max(1e-6);
    let bounds = surface.bounds();
    if bounds.is_empty() {
        return TetMesh::default();
    }
    let extents = bounds.extents();
    let nx = (extents.x / cell).ceil().max(1.0) as usize;
    let ny = (extents.y / cell).ceil().max(1.0) as usize;
    let nz = (extents.z / cell).ceil().max(1.0) as usize;
    // Centre the lattice on the bounds so the staircase error splits evenly.
    let origin = bounds.center()
        - Vec3::new(nx as f32, ny as f32, nz as f32) * (cell * 0.5);
    let lattice =
        |i: usize, j: usize, k: usize| origin + Vec3::new(i as f32, j as f32, k as f32) * cell;

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut ids: BTreeMap<[usize; 3], u32> = BTreeMap::new();
    let mut tets = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let centre = lattice(i, j, k) + Vec3::splat(cell * 0.5);
                if !surface.contains_point(centre) {
                    continue;
                }
                // Interned lattice corners keep the mesh conforming across cells.
                let id = |i: usize, j: usize, k: usize| -> u32 {
                    *ids.entry([i, j, k]).or_insert_with(|| {
                        vertices.push(lattice(i, j, k));
                        vertices.len() as u32 - 1
                    })
                };
                kuhn_cell((i, j, k), id, &mut tets);
            }
        }
    }
    TetMesh::new(vertices, tets)
}
