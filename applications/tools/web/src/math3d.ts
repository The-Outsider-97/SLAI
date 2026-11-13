/** Small 3D maths shared by the SLAI applications that draw a model: column-major 4x4 matrices (as WebGL wants them), a Z-up orbit
 *  camera, ray picking and edge finding. SPADRA's building view and StringLab's string viewport both use this module; neither carries
 *  its own copy. It has no dependency on the DOM and is tested in tests_ts/math3d.test.ts. */

export type Pt3 = [number, number, number];

export type Mat4 = Float32Array;

export const sub = (a: Pt3, b: Pt3): Pt3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: Pt3, b: Pt3): Pt3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scaleV = (a: Pt3, k: number): Pt3 => [a[0] * k, a[1] * k, a[2] * k];
export const dot = (a: Pt3, b: Pt3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Pt3, b: Pt3): Pt3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const length = (a: Pt3): number => Math.hypot(a[0], a[1], a[2]);
export function normalise(a: Pt3): Pt3 { const l = length(a); return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 1]; }

export function identity(): Mat4 { const m = new Float32Array(16); m[0] = m[5] = m[10] = m[15] = 1; return m; }

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let sum = 0;
    for (let k = 0; k < 4; k++) sum += (a[k * 4 + r] ?? 0) * (b[c * 4 + k] ?? 0);
    out[c * 4 + r] = sum;
  }
  return out;
}

export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2), m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f; m[10] = (far + near) / (near - far); m[11] = -1; m[14] = (2 * far * near) / (near - far);
  return m;
}

export function ortho(halfW: number, halfH: number, near: number, far: number): Mat4 {
  const m = new Float32Array(16);
  m[0] = 1 / halfW; m[5] = 1 / halfH; m[10] = -2 / (far - near); m[14] = -(far + near) / (far - near); m[15] = 1;
  return m;
}

export function lookAt(eye: Pt3, target: Pt3, up: Pt3): Mat4 {
  const f = normalise(sub(target, eye));
  const s = normalise(cross(f, up));
  const u = cross(s, f);
  const m = identity();
  m[0] = s[0]; m[4] = s[1]; m[8] = s[2];
  m[1] = u[0]; m[5] = u[1]; m[9] = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[12] = -dot(s, eye); m[13] = -dot(u, eye); m[14] = dot(f, eye);
  return m;
}

export function invert(m: Mat4): Mat4 | null {
  const a = Array.from(m);
  const inv = Array.from(identity());
  for (let col = 0; col < 4; col++) {
    let pivot = col;
    for (let r = col + 1; r < 4; r++) if (Math.abs(a[col * 4 + r] ?? 0) > Math.abs(a[col * 4 + pivot] ?? 0)) pivot = r;
    if (Math.abs(a[col * 4 + pivot] ?? 0) < 1e-14) return null;
    if (pivot !== col) for (let c = 0; c < 4; c++) {
      [a[c * 4 + col], a[c * 4 + pivot]] = [a[c * 4 + pivot] as number, a[c * 4 + col] as number];
      [inv[c * 4 + col], inv[c * 4 + pivot]] = [inv[c * 4 + pivot] as number, inv[c * 4 + col] as number];
    }
    const d = a[col * 4 + col] as number;
    for (let c = 0; c < 4; c++) { a[c * 4 + col] = (a[c * 4 + col] as number) / d; inv[c * 4 + col] = (inv[c * 4 + col] as number) / d; }
    for (let r = 0; r < 4; r++) {
      if (r === col) continue;
      const f = a[col * 4 + r] as number;
      if (f === 0) continue;
      for (let c = 0; c < 4; c++) { a[c * 4 + r] = (a[c * 4 + r] as number) - f * (a[c * 4 + col] as number); inv[c * 4 + r] = (inv[c * 4 + r] as number) - f * (inv[c * 4 + col] as number); }
    }
  }
  return Float32Array.from(inv);
}

/** Apply a matrix to a point; returns clip-space x, y, z and w. */
export function transform(m: Mat4, p: Pt3): [number, number, number, number] {
  const g = (i: number): number => m[i] ?? 0;
  return [
    g(0) * p[0] + g(4) * p[1] + g(8) * p[2] + g(12), g(1) * p[0] + g(5) * p[1] + g(9) * p[2] + g(13),
    g(2) * p[0] + g(6) * p[1] + g(10) * p[2] + g(14), g(3) * p[0] + g(7) * p[1] + g(11) * p[2] + g(15),
  ];
}

export interface Ray { origin: Pt3; dir: Pt3 }

/** The world ray through a pixel (ndcX, ndcY in -1..1), valid for perspective and orthographic cameras alike. */
export function rayThrough(viewProjection: Mat4, ndcX: number, ndcY: number): Ray | null {
  const inv = invert(viewProjection);
  if (!inv) return null;
  const unproject = (z: number): Pt3 => { const [x, y, zz, w] = transform(inv, [ndcX, ndcY, z]); const d = w === 0 ? 1 : w; return [x / d, y / d, zz / d]; };
  const near = unproject(-1), far = unproject(1);
  return { origin: near, dir: normalise(sub(far, near)) };
}

/** Moeller-Trumbore. Returns the distance along the ray, or null. Hits from either side count (walls are seen from both). */
export function rayTriangle(ray: Ray, a: Pt3, b: Pt3, c: Pt3): number | null {
  const e1 = sub(b, a), e2 = sub(c, a);
  const p = cross(ray.dir, e2);
  const det = dot(e1, p);
  if (Math.abs(det) < 1e-12) return null;
  const inv = 1 / det;
  const t0 = sub(ray.origin, a);
  const u = dot(t0, p) * inv;
  if (u < 0 || u > 1) return null;
  const q = cross(t0, e1);
  const v = dot(ray.dir, q) * inv;
  if (v < 0 || u + v > 1) return null;
  const t = dot(e2, q) * inv;
  return t >= 0 ? t : null;
}

/** Where a ray meets the horizontal plane z = height. */
export function rayPlaneZ(ray: Ray, height: number): Pt3 | null {
  if (Math.abs(ray.dir[2]) < 1e-9) return null;
  const t = (height - ray.origin[2]) / ray.dir[2];
  return t < 0 ? null : add(ray.origin, scaleV(ray.dir, t));
}

export interface Camera { yaw: number; pitch: number; distance: number; target: Pt3; ortho: boolean }

/** Z-up orbit: yaw turns about Z (0 looks along +Y from -Y), pitch lifts above the horizon. */
export function cameraEye(cam: Camera): Pt3 {
  const cp = Math.cos(cam.pitch);
  return [cam.target[0] + cam.distance * cp * Math.sin(cam.yaw), cam.target[1] - cam.distance * cp * Math.cos(cam.yaw), cam.target[2] + cam.distance * Math.sin(cam.pitch)];
}

/** Where the camera stands, as a sheet's 3D view states it: degrees from +X counter-clockwise in plan (-180 to 180) and degrees above the horizon (inside ±89). */
export function viewerDirection(cam: Pick<Camera, "yaw" | "pitch">): { azimuth: number; elevation: number } {
  const wrap = (a: number): number => ((((a + 180) % 360) + 360) % 360) - 180;
  return { azimuth: Math.round(wrap((cam.yaw * 180) / Math.PI - 90) * 100) / 100, elevation: Math.round(Math.max(-89, Math.min(89, (cam.pitch * 180) / Math.PI)) * 100) / 100 };
}

export function cameraMatrices(cam: Camera, width: number, height: number, extent: number): { view: Mat4; projection: Mat4; viewProjection: Mat4 } {
  const view = lookAt(cameraEye(cam), cam.target, [0, 0, 1]);
  const aspect = width / Math.max(1, height);
  const far = Math.max(cam.distance + extent * 4, 10);
  const projection = cam.ortho
    ? ortho((cam.distance * Math.tan(Math.PI / 8)) * aspect, cam.distance * Math.tan(Math.PI / 8), -far, far)
    : perspective(Math.PI / 4, aspect, Math.max(cam.distance * 0.01, 0.01), far);
  return { view, projection, viewProjection: multiply(projection, view) };
}

export const HOME_VIEW = { yaw: Math.PI / 4, pitch: Math.atan(1 / Math.SQRT2) };   // true isometric: equal foreshortening on all three axes

/** Edges worth drawing: boundary edges and edges whose neighbouring faces differ by more than `angleDeg`. */
export function featureEdges(vertices: readonly Pt3[], faces: readonly number[][], angleDeg = 30): Array<[number, number]> {
  const key = (i: number): string => { const v = vertices[i] as Pt3; return `${v[0].toFixed(6)},${v[1].toFixed(6)},${v[2].toFixed(6)}`; };
  const edges = new Map<string, { a: number; b: number; normals: Pt3[] }>();
  for (const f of faces) {
    const [i, j, k] = f;
    if (i === undefined || j === undefined || k === undefined) continue;
    const vi = vertices[i], vj = vertices[j], vk = vertices[k];
    if (!vi || !vj || !vk) continue;
    const n = normalise(cross(sub(vj, vi), sub(vk, vi)));
    for (const [p, q] of [[i, j], [j, k], [k, i]] as const) {
      const kp = key(p), kq = key(q);
      const id = kp < kq ? `${kp}|${kq}` : `${kq}|${kp}`;
      const entry = edges.get(id);
      if (entry) entry.normals.push(n); else edges.set(id, { a: p, b: q, normals: [n] });
    }
  }
  const limit = Math.cos((angleDeg * Math.PI) / 180);
  const out: Array<[number, number]> = [];
  for (const e of edges.values()) {
    const [n1, n2] = e.normals;
    if (e.normals.length === 1 || (n1 && n2 && e.normals.length === 2 && dot(n1, n2) < limit)) out.push([e.a, e.b]);
  }
  return out;
}

// ------------------------------------------------------------------------------------------------------------ moving the camera
/** Where a ray meets the plane through ``point`` with the given normal (in front of or behind the ray's origin: the camera's near plane can be on either side). */
export function rayPlane(ray: Ray, point: Pt3, normal: Pt3): Pt3 | null {
  const denom = dot(ray.dir, normal);
  if (Math.abs(denom) < 1e-9) return null;
  return add(ray.origin, scaleV(ray.dir, dot(sub(point, ray.origin), normal) / denom));
}

/** Turn ``v`` about a unit axis by an angle (Rodrigues). */
function turn(v: Pt3, axis: Pt3, angle: number): Pt3 {
  const c = Math.cos(angle), s = Math.sin(angle), k = cross(axis, v), d = dot(axis, v) * (1 - c);
  return [v[0] * c + k[0] * s + axis[0] * d, v[1] * c + k[1] * s + axis[1] * d, v[2] * c + k[2] * s + axis[2] * d];
}

/** Orbit the camera about a point that need not be its target: the eye and the target turn together about ``pivot`` (about the vertical axis by
 *  ``dyaw``, about the horizontal axis across the view by ``dpitch``), so the pivot stays where it is on the screen. Pitch stops at ``limit``. */
export function orbitAbout(cam: Camera, pivot: Pt3, dyaw: number, dpitch: number, limit: number): void {
  const pitch = Math.min(limit, Math.max(-limit, cam.pitch + dpitch));
  const yaw = cam.yaw + dyaw;
  const across: Pt3 = [-Math.cos(yaw), -Math.sin(yaw), 0];              // the axis that lifts the eye when turned positively
  const move = (p: Pt3): Pt3 => add(pivot, turn(turn(sub(p, pivot), [0, 0, 1], dyaw), across, pitch - cam.pitch));
  const eye = move(cameraEye(cam)), target = move(cam.target);
  cam.target = target;
  cam.yaw = yaw;
  cam.pitch = pitch;
  cam.distance = length(sub(eye, target)) || cam.distance;
}

/** Look around from where the eye is: the eye stays still and the target swings. */
export function lookAround(cam: Camera, dyaw: number, dpitch: number, limit: number): void {
  const eye = cameraEye(cam);
  cam.yaw += dyaw;
  cam.pitch = Math.min(limit, Math.max(-limit, cam.pitch + dpitch));
  const cp = Math.cos(cam.pitch);
  cam.target = [eye[0] - cam.distance * cp * Math.sin(cam.yaw), eye[1] + cam.distance * cp * Math.cos(cam.yaw), eye[2] - cam.distance * Math.sin(cam.pitch)];
}

/** Walk: move the eye and the target together, ``forward`` along the horizontal direction of view, ``right`` across it and ``up`` vertically (metres). */
export function walk(cam: Camera, forward: number, right: number, up: number): void {
  const fx = -Math.sin(cam.yaw), fy = Math.cos(cam.yaw);                // horizontal direction of view
  cam.target = [cam.target[0] + fx * forward + fy * right, cam.target[1] + fy * forward - fx * right, cam.target[2] + up];
}

/** Zoom by ``factor`` (below 1 closer) keeping the point of the model under the pointer where it is on the screen. */
export function zoomAt(cam: Camera, factor: number, ndcX: number, ndcY: number, width: number, height: number, extent: number, min = 0.05, max = 1e6): void {
  const focus = (): Pt3 | null => {
    const ray = rayThrough(cameraMatrices(cam, width, height, extent).viewProjection, ndcX, ndcY);
    return ray ? rayPlane(ray, cam.target, normalise(sub(cam.target, cameraEye(cam)))) : null;
  };
  const before = focus();
  cam.distance = Math.min(max, Math.max(min, cam.distance * factor));
  const after = focus();
  if (before && after) cam.target = add(cam.target, sub(before, after));
}

/** The middle of the box that holds some points, or null for none. */
export function centreOf(points: readonly Pt3[]): Pt3 | null {
  if (!points.length) return null;
  let lo: Pt3 = [Infinity, Infinity, Infinity], hi: Pt3 = [-Infinity, -Infinity, -Infinity];
  for (const p of points) { lo = [Math.min(lo[0], p[0]), Math.min(lo[1], p[1]), Math.min(lo[2], p[2])]; hi = [Math.max(hi[0], p[0]), Math.max(hi[1], p[1]), Math.max(hi[2], p[2])]; }
  return [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
}
