# tesh audit — 2026-07-02

Multi-agent audit (7 finders / dedup / adversarial verify) + manual verification.
Baseline: clippy clean, 17/17 tests pass.

> **Addendum (2026-07-06): every open item below is resolved.** #2 (`vertices_mut` and the
> blanket `Transform` now reset the query caches), #4 (degenerate tets contain nothing —
> scalar and SPMD paths), #5 (`fill_holes` keys half-edges per directed edge and only fans
> walks that closed), #6/#10 (`nearest_vertex` returns `Option`, constructors
> `debug_assert!(is_valid())`, panic contracts documented), #8 (`serde/alloc`), #9 (brute
> and BVH agree on empty meshes: `closest_point` → `p`, `distance` → `∞`), #11
> (grid-snap semantics documented; `-0.0` normalized; non-finite/overflowing coordinates
> fall back to collision-free bit keys). Low items: `interpolate` documents its panic;
> `face_adjacency`/`tet_adjacency` emit all pairs on non-manifold edges. All easy
> optimizations, ergonomics, and doc corrections applied (`area_faces` → `face_areas` +
> `face_area(i)`, `signed_volume(i)` → `tet_signed_volume(i)`, `TetMesh::split`,
> mesh-level `nearest_vertex`, `append`/`from_triangles`, root re-exports, `Aabb::merged`,
> `#[must_use]`, BVH traversal stacks inline, kernels pack only what's returned).
> Regression tests live in `tests/audit_fixes.rs`; consumer-driven features
> (`SurfaceHit`, batch queries, flat-buffer interop, `tet_grid`/`tetrahedralize_grid`,
> `boundary()` with areas/normals, `shape_gradients`, `surface_samples`,
> `surface_with_map`) in `tests/features.rs`. 52/52 tests pass, clippy clean, full
> feature matrix builds, girder + rigmarole check and test green.

> **Addendum (same day):** obvhs has been replaced with a native binned-SAH BVH
> (`src/accel/bvh2.rs` + rewritten `bvh.rs`/`tet_bvh.rs`; `triangle::ray_intersect` added,
> shared with the brute path). Both BVHs run their leaf tests SPMD-wide via hydroplane
> (`Kernel` + `dispatch`, 16-prim leaves, leaf-slot-ordered SoA corner columns): lane-parallel
> Möller–Trumbore for rays/parity, a branchless lane-parallel Ericson distance² for
> closest-point (winning lane re-solved exactly), and lane-parallel barycentric containment
> for tet location. Benchmarked against obvhs with zero correctness mismatches: rays
> 0.82–0.88x, closest ~0.41–0.55x, contains ~0.35–0.41x, builds ~parity (0.78–1.05x),
> tet locate 50 ns vs 45 ns (1.12x, the one remaining gap). This **resolves #1** (contains_point hang — parity is now one
> traversal, no ray-marching; repro passes), **#3** (glam duplication — obvhs gone),
> **#7** (NaN vertex now degrades to a query miss instead of a debug panic), and moots the
> `BvhBuildParams` re-export item (**#23**, `with_params` removed) and the `accel` module
> doc rewrite (**#31**, rewritten). The BVH/brute semantic split for `contains_point`
> (parity vs winding number) still exists. Everything else below is still open.

**Verification legend:**
- ✅ = reproduced by running code (repro crate: scratchpad `audit-verify`)
- 📖 = confirmed by direct code reading
- ⚖ = confirmed by an independent judge agent
- The judge-agent verify pass for panic/error/perf findings died on a transient login error; those were verified manually instead.

## High severity

### 1. ✅ `TriBvh::contains_point` hangs forever on meshes far from the origin — `src/accel/bvh.rs:89`
The ray-march advances by absolute `STEP_EPS = 1e-5`. For hit points with coordinates ≳128, f32 ULP exceeds the step, so `origin = hit.point + dir * STEP_EPS` rounds back to the same origin and the same hit repeats (obvhs `Ray::new_inf` has `tmin = 0.0`). Infinite loop; `crossings: u32` would also overflow-panic in debug.
**Repro:** `cube(1.0)` translated to (2000,2000,2000), `contains_point(Vec3::splat(2000.0))` → spins forever. Reachable via `contains_point` / `signed_distance` under default features.
**Fix:** advance `tmin` past the previous `hit.t` instead of re-basing the origin, or scale the step by `hit.point.abs().max_element()`. Also note the BVH path (ray parity) and brute path (winding number) disagree on non-watertight input.

### 2. ✅ `apply_transform` / `vertices_mut()` leave the query cache stale — `src/traits.rs:81`
Blanket `impl<M: Mesh> Transform for M` mutates through `vertices_mut()`, which never resets `TriCache`/`TetCache`. Any query after a transform that followed an earlier query silently answers against the old geometry.
**Repro:** `cube(1.0)`; `distance(ZERO)` = 0.5; translate by (100,100,100); `distance(ZERO)` still = 0.5 (should be ~172.9).
**Fix:** have `TriMesh::vertices_mut`/`TetMesh::vertices_mut` call `reset_cache()`, and/or make `Transform` impls invalidate.

### 3. ✅ tesh does not compile in a fresh downstream project — duplicate glam (`Cargo.toml`)
A fresh resolve gives obvhs (`glam >=0.30.10,<0.34`) glam **0.33.2** while tesh pins **0.32** → two glam versions, and tesh's `.into()` conversions at the obvhs boundary fail (`Vec3A: From<Vec3>` not satisfied, 12 errors). tesh's own lockfile happens to unify at 0.32.1, so this is invisible in-repo. *(Found during verification, not by the workflow.)*
**Repro:** new crate depending on `tesh = {path}` + `glam = "0.32"` → E0277; `cargo update glam@0.33.2 --precise 0.32.1` fixes it.
**Fix options:** convert at the obvhs boundary via arrays (`Vec3A::from_array(v.to_array())` etc.) so no shared glam is needed; or constrain versions so unification is forced. Same issue latent for kiddo.

## Errors (medium)

### 4. ✅ Degenerate tet contains every point — `src/tetrahedron.rs:64`
`barycentric` returns `[0.0; 4]` for zero-volume tets; `contains_point` checks `all(l >= 0.0)` → true for any p. Under bvh the damage is bounded by the tet's AABB; under brute force one degenerate tet makes `contains_point` true everywhere and `interpolate` returns garbage.
**Repro:** `TetMesh` with tets `[[0,0,1,2],[0,1,2,3]]`, `find_containing_tet((0.9,0.9,0.0))` → `Some(0)` (the degenerate tet).
**Fix:** return false / signal degeneracy instead of `[0;4]`.

### 5. ✅ `fill_holes` breaks when two holes share a vertex — `src/tri/repair.rs:188`
Half-edges stored as `BTreeMap<u32, u32>` keyed by source vertex; a pinch vertex with two outgoing boundary half-edges silently overwrites one. Chains truncate, unclosed chains still get fanned and counted.
**Repro:** cube with two faces sharing one vertex removed → `fill_holes()` returns 1, `is_watertight()` false, 3 boundary edges remain. Violates the documented "surface becomes closed" postcondition that mass properties rely on.
**Fix:** key per directed edge / multimap; only fan chains that actually closed (`cur == start`).

### 6. ✅ `VertexKdTree::nearest_vertex` on an empty mesh fabricates index 0 — `src/accel/kdtree.rs:29`
kiddo's empty-tree sentinel `(0, f32::MAX)` is forwarded as-is; indistinguishable from a real vertex 0, and indexing with it panics.
**Repro:** empty `TriMesh` → `nearest_vertex(ZERO)` = `(0, 3.4e38)`.
**Fix:** return `Option<(u32, f32)>` or check `vertices.is_empty()`.

### 7. ✅ One NaN vertex panics the first BVH query in debug builds — `src/accel/bvh.rs:43`
obvhs PLOC builder `debug_assert!(!aabb.min.is_nan())`. Release builds instead build a garbage tree silently. `ray_intersect` with non-finite origin/dir hits similar asserts. TetBvh same exposure.
**Repro:** mesh with one NaN vertex, `.distance(ZERO)` → `assertion failed: !aabb.min.is_nan()`.
**Fix:** document finite-geometry requirement or filter/assert at the tesh boundary.

### 8. ✅ `--no-default-features --features libm,serde` does not build — `Cargo.toml:29`
serde gets neither `std` nor `alloc`, so `Vec<T>` has no `Serialize`/`Deserialize`.
**Fix:** `serde = ["dep:serde", "serde/alloc", "glam/serde"]`.

### 9. 📖 Brute-force empty-mesh `distance` = 0; BVH path returns inf — `src/tri/query.rs:85`
Brute `closest_point` initializes `best = p`, so empty mesh → distance 0 ("every point is on the surface"). BVH path returns infinity. Feature flags change the answer.
**Fix:** initialize consistently; derive distance from `best_d`.

### 10. 📖 Out-of-range face indices panic deep inside nearly every method — `src/tri/mod.rs:69`
Unchecked `vertices[f[k] as usize]` throughout (accessors, kernels, topology, repair, BVH builds). serde Deserialize constructs unvalidated meshes, so untrusted input panics on first use. `is_valid()` exists but nothing calls or references it; no `# Panics` docs anywhere.
**Fix:** document the panic contract on accessors + `debug_assert!(is_valid())` in constructors; consider validating Deserialize.

### 11. 📖 `merge_vertices` contract mismatches — `src/tri/repair.rs:28`
Grid snap ≠ "within tolerance": pairs closer than tol straddling a cell boundary don't merge; pairs up to √3·tol in one cell do. `tolerance == 0` path uses `to_bits`, so `-0.0` ≠ `+0.0`. Also (low): `1/tolerance` overflow for tiny tolerances saturates the cast and welds whole octants; NaN coords weld with cell 0.
**Fix:** document grid semantics as the real guarantee; normalize -0.0; guard non-finite `x * inv`.

## Errors / panics (low)

- 📖 `TetMesh::interpolate` panics if `values` shorter than the vertex buffer — `src/tet/query.rs:48`. Return None or document.
- 📖 `face_adjacency` emits only consecutive pairs for non-manifold edges (k>2 faces → k−1 pairs, not all pairs) — `src/tri/topology.rs:59`. Document or emit all pairs.

## Easy optimizations (all 📖)

| Where | Issue | Fix |
|---|---|---|
| `src/compute/mod.rs:136` | `face_areas`/`face_normals` compute + discard the other kernel output incl. a packed `Vec<Vec3>`; `TriMesh::area` allocates ~13n floats to sum one scalar | pack only what's returned |
| `src/tri/topology.rs:32` | `edges_unique`/`euler_number` build a `Vec<u32>` per edge then drop all values | use `BTreeSet` like the TetMesh twin already does |
| `src/tet/geometry.rs:72` | `mass_properties` clones the whole vertex buffer + re-extracts the boundary every call | use `surface_cached()` under bvh; iterate `boundary_faces()` without cloning otherwise |
| `src/accel/bvh.rs:112` | `closest` heap-allocates the traversal stack per query (hottest path) | fixed array + depth counter, or SmallVec |
| `src/tri/repair.rs:66` | `remove_degenerate_faces` takes a sqrt per face | compare `length_squared()` vs `(2·eps)²` |
| `src/tri/topology.rs:94` (+ tet twin :59) | root→label via `BTreeMap` | dense `vec![u32::MAX; n]` |
| `src/tri/repair.rs:39,85` | missing `with_capacity` on vertex-sized Vecs | `Vec::with_capacity(vertices.len())` |

## Ergonomics (all ⚖ confirmed)

- `invalidate()` disappears when both accel features are off, though docs unconditionally require calling it — `src/tri/mod.rs:47`, `src/tet/mod.rs:52`. Compile it unconditionally (reset is a no-op then).
- `with_params` takes `obvhs::BvhBuildParams` but it isn't re-exported → users need a version-locked direct obvhs dep — `src/accel/bvh.rs:34`. Add `pub use obvhs::BvhBuildParams;`.
- `area_faces()` breaks the `face_*` naming scheme (`face_normals`, `face_centroid`); no scalar `face_area(i)` — `src/tri/geometry.rs:18`. Rename + add.
- `signed_volume` = whole mesh on TriMesh but per-element on TetMesh; TetMesh lacks `split()` — `src/tet/geometry.rs:26`. Rename to `tet_signed_volume(i)`; add `TetMesh::split()`.
- No mesh-level nearest-vertex query (kdtree-only, no brute fallback), though `kdtree()` docs claim reuse — `src/tri/mod.rs:91`. Add `nearest_vertex` with fallback, mirroring `closest_point`.
- No `append`/`from_triangles` — every scene-composition user hand-writes the offset loop — `src/tri/mod.rs:36`.
- `Components`/`RayHit` not re-exported at root/prelude; TetMesh users must import from `tri` — `src/tri/mod.rs:15`.
- `Aabb::union` takes `&Aabb` (24-byte Copy type); no owned `merged`/`including` — `src/aabb.rs:52`.
- Missing `#[must_use]` on `Mesh::vertex_count`/`is_empty`/`vertex_centroid`, `Bounds::bounds` — `src/traits.rs:28`.

## Docs (all ⚖ confirmed; incorrect > verbose)

- `src/accel/mod.rs:4` — claims brute-force methods "stay available in every build" (they're compiled out when accel is on) and tells users to build accelerators manually (they're lazy + cached). Rewrite.
- `src/primitives.rs:240` — cone: "base centred on the origin plane" is wrong; solid is origin-centred, base at −h/2.
- `src/primitives.rs:147` — icosphere: each subdivision **quadruples** (not doubles) the face count.
- `src/primitives.rs:107` — uv_sphere: `stacks` is bands; mesh has stacks−1 parallels.
- `src/tet/mod.rs:126` — `surface()` says "sharing this mesh's vertex buffer"; it clones it (indices correspond — say that instead).
- `src/tri/topology.rs:8` — `Components` says "per face" but is also TetMesh's per-tet labelling. Say "per element".
- `src/tetrahedron.rs:5` — claims the orientation convention "matches the faces extracted in crate::tet"; boundary extraction reorients geometrically and doesn't rely on it. Drop the sentence.
- `src/lib.rs:9` — crate docs cover std/libm but omit `kdtree`/`bvh`/`serde`/`bytemuck`/`approx`. Add a terse feature list.

## Not yet done

- Fixes: nothing has been changed; the repo is untouched.
- Repro crate for re-running verifications: `/private/tmp/claude-501/-Users-bowan-Crates-tesh/568c6770-6801-4724-8228-5a8e767e9ffa/scratchpad/audit-verify` (session-scoped; may be gone next session — cases were: stale, kdtree-empty, nan, hang, degenerate-tet, fill-holes).
- Suggested fix order: #3 (packaging, blocks any downstream use) → #2 → #1 → #5/#8 → the rest.
