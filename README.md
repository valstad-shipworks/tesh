# tesh

Triangle surface meshes and tetrahedron volume meshes for Rust. Built on `glam` for math and [`hydroplane`](../hydroplane) for runtime-dispatched SIMD acceleration (SSE4 / AVX2 / AVX-512 / NEON, selected per host CPU).

`no_std` with mandatory `alloc` — enable `std` (default) or `libm` for the float backend. Spatial queries are fast by default: a BVH and kd-tree build lazily on first use and cache on the mesh; with those features off, the same methods fall back to brute force.

## Mesh types

- **`TriMesh`** — a triangle surface mesh: a vertex buffer plus `[u32; 3]` faces.
- **`TetMesh`** — a tetrahedron volume mesh: a vertex buffer plus `[u32; 4]` tets. Its boundary surface is available as a `TriMesh`.

The buffers are read-only accessors — `vertices()` and `faces()`/`tets()` — so the cached accelerators can't fall out of sync with the geometry. Build a mesh with `new(..)`, move vertex positions in place through `vertices_mut()` (which drops the stale cache), and rebuild with `new(..)` to change the topology.

## Traits

- `Mesh` — shared accessors: `vertices()`, `element_count()`, `vertex_count()`, `is_empty()`, `vertex_centroid()`.
- `Bounds` — `bounds()` returns the axis-aligned bounding box of the vertices.
- `Transform` — `apply_transform(Affine3A)` in place, or `transformed(Affine3A)` for an owned copy.

These traits are sealed; only `TriMesh` and `TetMesh` implement them.

## Primitives

`tesh::primitives` builds common shapes. Closed solids come out watertight and outward-oriented; `grid`/`plane` are open surfaces.

```rust
use glam::Vec3;
use tesh::primitives::*;

let c = cube(1.0);
let s = uv_sphere(1.0, 32, 16);
let s2 = icosphere(1.0, 3);
let cyl = cylinder(0.5, 2.0, 32);
let cn = cone(0.5, 1.0, 32);
let t = torus(1.0, 0.25, 48, 24);
let cap = capsule(0.3, 2.0, 24, 8);
let b = box3(Vec3::new(1.0, 2.0, 3.0));

// volume meshes
let vol = tet_grid([8, 8, 8], Vec3::splat(1.0));
let filled = tetrahedralize_grid(&cube(1.0), 0.1);
```

## Building meshes

```rust
use glam::Vec3;
use tesh::TriMesh;

// from vertices + indexed faces
let mesh = TriMesh::new(
    vec![Vec3::ZERO, Vec3::X, Vec3::Y],
    vec![[0, 1, 2]],
);

// from flat buffers (e.g. GPU / glTF interop)
let mesh = TriMesh::from_flat(
    &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    &[0, 1, 2],
);

// from raw triangles (welds nothing)
let mesh = TriMesh::from_triangles([[Vec3::ZERO, Vec3::X, Vec3::Y]]);

// compose scenes
let mut scene = cube_a;
scene.append(&cube_b);
```

## Spatial queries

Every query takes `&self` and lazily builds the accelerator behind a cache. Single-point and batched (`_s` plural) variants are both available; batched calls dispatch the leaf tests SIMD-wide.

```rust
use glam::Vec3;
use tesh::{TriMesh, primitives::cube};

let mesh = cube(1.0);
let p = Vec3::new(2.0, 0.0, 0.0);

let cp   = mesh.closest_point(p);        // nearest point on the surface
let d    = mesh.distance(p);             // unsigned distance
let sd    = mesh.signed_distance(p);      // negative inside (ray-parity, watertight)
let inside = mesh.contains_point(p);      // point-in-solid
let hit    = mesh.ray_intersect(Vec3::ZERO, Vec3::X);   // Option<RayHit>
let surf   = mesh.surface_hit(p);         // Option<SurfaceHit>: point, normal, face
let (vi, dist) = mesh.nearest_vertex(p).unwrap();       // kd-tree vertex lookup

// batched forms
let ds     = mesh.distances(&[p, -p]);
let cps    = mesh.closest_points(&[p, -p]);
let ins    = mesh.contains_points(&[p, -p]);
```

## Geometry & mass properties

```rust
use tesh::primitives::cube;

let mesh = cube(2.0);

let area   = mesh.area();
let vol    = mesh.volume();
let normals = mesh.face_normals();       // or vertex_normals()
let vn      = mesh.vertex_normals();
let c       = mesh.centroid();
let samples = mesh.surface_samples(0.1); // Poisson-ish surface points

// mass properties: volume, center of mass, inertia tensor
let mp = mesh.mass_properties();
let mass = mp.mass(1000.0);              // density → scalar mass
let inertia = mp.inertia_at(1000.0);
let (axes, moments) = mp.principal();    // Jacobi eigensolve
```

## Topology

```rust
let watertight = mesh.is_watertight();
let boundary   = mesh.boundary_edges();
let edges      = mesh.edges_unique();
let euler      = mesh.euler_number();
let bodies     = mesh.body_count();
let parts      = mesh.split();           // one TriMesh per connected component
```

## Repair

All repair ops take `&mut self` and reset the query cache.

```rust
let mut mesh = /* noisy imported mesh */;

mesh.merge_vertices(1e-5);            // weld coincident vertices (grid snap)
mesh.remove_degenerate_faces(1e-9);
mesh.remove_duplicate_faces();
mesh.remove_unreferenced_vertices();
mesh.orient_consistent();             // BFS-consistent winding
mesh.orient_outward();                // outward-facing normals
let filled = mesh.fill_holes();       // returns the number of holes closed
```

## Volume meshes

```rust
use tesh::primitives::tet_grid;
use glam::Vec3;

let vol = tet_grid([8, 8, 8], Vec3::splat(1.0));

let v      = vol.volume();
let q      = vol.qualities();               // per-tet quality
let mp     = vol.mass_properties();
let tet    = vol.find_containing_tet(Vec3::splat(0.5));   // Option<usize>
let val    = vol.interpolate(Vec3::splat(0.5), &field);  // barycentric interpolation
let grads  = vol.shape_gradients();         // per-tet ∇ of the linear basis

// boundary surface, sharing the vertex indices
let surface: tesh::TriMesh = vol.surface();
let (surf, vertex_map) = vol.surface_with_map();
```

## Features

- `std` *(default)* — `std` float backend.
- `libm` — `no_std` float backend (mutually sufficient with `std`; one is required).
- `bvh` *(default)* — BVH-accelerated distance / ray / containment queries.
- `kdtree` *(default)* — kd-tree nearest-vertex queries.
- `serde` — `Serialize`/`Deserialize` for the mesh types.
- `bytemuck` — `Pod`/`Zeroable` derives for interop.
- `approx` — approximate-equality derives.

## License

Apache-2.0.
