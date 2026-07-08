//! Per-element math through [`hydroplane`] SPMD kernels: corners gathered into SoA
//! columns, one lockstep kernel pass, results repacked. Scatter steps (per-vertex normal
//! accumulation) stay scalar — the CPU backends have no atomic accumulate.

use alloc::vec;
use alloc::vec::Vec;

use glam::Vec3;
use hydroplane::{Gang, GangGlamExt, Vec3Wide, kernel};

#[kernel]
fn tri_normal_area_kernel<'a>(
    ctx: Gang,
    a: [&'a [f32]; 3],
    b: [&'a [f32]; 3],
    c: [&'a [f32]; 3],
    out_n: [&'a mut [f32]; 3],
    out_area: &'a mut [f32],
) {
    let [onx, ony, onz] = out_n;
    ctx.for_each_chunk::<f32>(out_area.len(), |off, cnt| {
        let end = off + cnt;
        let pa = ctx.load_partial_vec3([&a[0][off..end], &a[1][off..end], &a[2][off..end]], 0.0);
        let pb = ctx.load_partial_vec3([&b[0][off..end], &b[1][off..end], &b[2][off..end]], 0.0);
        let pc = ctx.load_partial_vec3([&c[0][off..end], &c[1][off..end], &c[2][off..end]], 0.0);

        let [ex, ey, ez] = (pb - pa).0;
        let [fx, fy, fz] = (pc - pa).0;
        let cross = Vec3Wide([
            ey * fz - ez * fy,
            ez * fx - ex * fz,
            ex * fy - ey * fx,
        ]);

        let len = cross.length();
        let area = len * ctx.splat(0.5);
        let positive = len.gt(ctx.splat(0.0));
        let inv = len.recip().select(positive, ctx.splat(0.0));
        let normal = cross * inv;

        normal.store_partial([&mut onx[off..end], &mut ony[off..end], &mut onz[off..end]]);
        area.store_partial(&mut out_area[off..end]);
    });
}

#[kernel]
fn tet_volume_kernel<'a>(
    ctx: Gang,
    a: [&'a [f32]; 3],
    b: [&'a [f32]; 3],
    c: [&'a [f32]; 3],
    d: [&'a [f32]; 3],
    out: &'a mut [f32],
) {
    ctx.for_each_chunk::<f32>(out.len(), |off, cnt| {
        let end = off + cnt;
        let pa = ctx.load_partial_vec3([&a[0][off..end], &a[1][off..end], &a[2][off..end]], 0.0);
        let pb = ctx.load_partial_vec3([&b[0][off..end], &b[1][off..end], &b[2][off..end]], 0.0);
        let pc = ctx.load_partial_vec3([&c[0][off..end], &c[1][off..end], &c[2][off..end]], 0.0);
        let pd = ctx.load_partial_vec3([&d[0][off..end], &d[1][off..end], &d[2][off..end]], 0.0);

        let [ex, ey, ez] = (pb - pa).0;
        let [fx, fy, fz] = (pc - pa).0;
        let [gx, gy, gz] = (pd - pa).0;
        let cx = ey * fz - ez * fy;
        let cy = ez * fx - ex * fz;
        let cz = ex * fy - ey * fx;
        let vol = (cx * gx + cy * gy + cz * gz) * ctx.splat(1.0 / 6.0);

        vol.store_partial(&mut out[off..end]);
    });
}

/// Splits vertex `k` of each element into three contiguous component columns, indexing
/// through the element connectivity.
fn corner_columns<const N: usize>(
    vertices: &[Vec3],
    elements: &[[u32; N]],
    k: usize,
) -> [Vec<f32>; 3] {
    let n = elements.len();
    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    let mut z = vec![0.0f32; n];
    for (i, e) in elements.iter().enumerate() {
        let v = vertices[e[k] as usize];
        x[i] = v.x;
        y[i] = v.y;
        z[i] = v.z;
    }
    [x, y, z]
}

fn pack_normals(nx: &[f32], ny: &[f32], nz: &[f32]) -> Vec<Vec3> {
    (0..nx.len())
        .map(|i| Vec3::new(nx[i], ny[i], nz[i]))
        .collect()
}

/// The raw kernel outputs: unpacked normal component columns and areas.
fn tri_normals_areas_raw(
    vertices: &[Vec3],
    faces: &[[u32; 3]],
) -> ([Vec<f32>; 3], Vec<f32>) {
    let n = faces.len();
    let [ax, ay, az] = corner_columns(vertices, faces, 0);
    let [bx, by, bz] = corner_columns(vertices, faces, 1);
    let [cx, cy, cz] = corner_columns(vertices, faces, 2);

    let mut nx = vec![0.0f32; n];
    let mut ny = vec![0.0f32; n];
    let mut nz = vec![0.0f32; n];
    let mut area = vec![0.0f32; n];

    tri_normal_area_kernel(
        [&ax, &ay, &az],
        [&bx, &by, &bz],
        [&cx, &cy, &cz],
        [&mut nx, &mut ny, &mut nz],
        &mut area,
    );

    ([nx, ny, nz], area)
}

/// Per-face unit normal and area, computed together in one SPMD pass.
#[must_use]
pub fn face_normals_areas(vertices: &[Vec3], faces: &[[u32; 3]]) -> (Vec<Vec3>, Vec<f32>) {
    let ([nx, ny, nz], area) = tri_normals_areas_raw(vertices, faces);
    (pack_normals(&nx, &ny, &nz), area)
}

/// Per-face unit normals.
#[must_use]
pub fn face_normals(vertices: &[Vec3], faces: &[[u32; 3]]) -> Vec<Vec3> {
    let ([nx, ny, nz], _area) = tri_normals_areas_raw(vertices, faces);
    pack_normals(&nx, &ny, &nz)
}

/// Per-face areas.
#[must_use]
pub fn face_areas(vertices: &[Vec3], faces: &[[u32; 3]]) -> Vec<f32> {
    tri_normals_areas_raw(vertices, faces).1
}

/// Area-weighted vertex normals: each face adds its (area-scaled) normal to its three
/// vertices, then every vertex normal is renormalized. The scatter runs on the scalar
/// side; the per-face normals and areas come from the kernel.
#[must_use]
pub fn vertex_normals(vertices: &[Vec3], faces: &[[u32; 3]]) -> Vec<Vec3> {
    let (normals, areas) = face_normals_areas(vertices, faces);
    let mut acc = vec![Vec3::ZERO; vertices.len()];
    for (f, tri) in faces.iter().enumerate() {
        let contribution = normals[f] * areas[f];
        for &vi in tri {
            acc[vi as usize] += contribution;
        }
    }
    for n in &mut acc {
        *n = n.normalize_or_zero();
    }
    acc
}

/// Per-tetrahedron signed volumes (positive for positively-oriented elements).
#[must_use]
pub fn tet_signed_volumes(vertices: &[Vec3], tets: &[[u32; 4]]) -> Vec<f32> {
    let n = tets.len();
    let [ax, ay, az] = corner_columns(vertices, tets, 0);
    let [bx, by, bz] = corner_columns(vertices, tets, 1);
    let [cx, cy, cz] = corner_columns(vertices, tets, 2);
    let [dx, dy, dz] = corner_columns(vertices, tets, 3);

    let mut vol = vec![0.0f32; n];
    tet_volume_kernel(
        [&ax, &ay, &az],
        [&bx, &by, &bz],
        [&cx, &cy, &cz],
        [&dx, &dy, &dz],
        &mut vol,
    );
    vol
}
