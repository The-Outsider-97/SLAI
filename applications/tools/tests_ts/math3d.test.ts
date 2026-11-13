import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { centreOf, lookAround, orbitAbout, rayPlane, walk, zoomAt, type Camera, cameraEye, cameraMatrices, featureEdges, HOME_VIEW, identity, invert, lookAt, multiply, rayPlaneZ, rayThrough, rayTriangle, transform, viewerDirection } from "../web/src/math3d.js";
import type { Pt3 } from "../web/src/math3d.js";

const near = (a: number, b: number, eps = 1e-5): void => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

// the same unit box the server tessellates for a wall (8 shared vertices, 12 triangles)
const box: Pt3[] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const tris = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]];

describe("matrices", () => {
  it("multiplies and inverts", () => {
    const m = lookAt([5, -4, 3], [0, 0, 1], [0, 0, 1]);
    const prod = multiply(m, invert(m) ?? identity());
    for (let i = 0; i < 16; i++) near(prod[i] ?? 0, identity()[i] ?? 0);
    assert.equal(invert(new Float32Array(16)), null);
  });
  it("puts the target at the centre of the view", () => {
    const cam = { yaw: HOME_VIEW.yaw, pitch: HOME_VIEW.pitch, distance: 20, target: [2, 3, 1] as Pt3, ortho: true };
    const { viewProjection } = cameraMatrices(cam, 800, 600, 10);
    const [x, y] = transform(viewProjection, cam.target);
    near(x, 0); near(y, 0);
  });
  it("home view looks from the south-west-above with equal foreshortening on the axes", () => {
    const eye = cameraEye({ ...HOME_VIEW, distance: 10, target: [0, 0, 0], ortho: true });
    assert.ok(eye[0] > 0 && eye[1] < 0 && eye[2] > 0);
    near(Math.abs(eye[0]), Math.abs(eye[1]));
    near(Math.hypot(...eye), 10);
  });
});

describe("picking", () => {
  const cam = { ...HOME_VIEW, distance: 12, target: [0.5, 0.5, 0.5] as Pt3, ortho: false };
  it("a ray through the centre of the screen hits the box and one through a corner misses", () => {
    const { viewProjection } = cameraMatrices(cam, 800, 600, 4);
    const hit = rayThrough(viewProjection, 0, 0);
    assert.ok(hit);
    const ts = tris.map(([a, b, c]) => rayTriangle(hit, box[a as number] as Pt3, box[b as number] as Pt3, box[c as number] as Pt3)).filter((t): t is number => t !== null);
    assert.ok(ts.length >= 2, "enters and leaves the box");
    const miss = rayThrough(viewProjection, 0.98, 0.98);
    assert.ok(miss);
    assert.equal(tris.map(([a, b, c]) => rayTriangle(miss, box[a as number] as Pt3, box[b as number] as Pt3, box[c as number] as Pt3)).filter((t) => t !== null).length, 0);
  });
  it("works for an orthographic camera too, and finds the ground under the pointer", () => {
    const { viewProjection } = cameraMatrices({ ...cam, ortho: true }, 800, 600, 4);
    const ray = rayThrough(viewProjection, 0, 0);
    assert.ok(ray);
    const ground = rayPlaneZ(ray, 0);
    assert.ok(ground);
    near(ground[2], 0);
    // the point it found must project back to the centre of the screen along the ray
    const [x, y] = transform(viewProjection, [ground[0], ground[1], 0]);
    near(x / 1, 0, 1e-4); near(y, 0, 1e-4);
    assert.equal(rayPlaneZ({ origin: [0, 0, 1], dir: [1, 0, 0] }, 0), null);
  });
});

describe("edges", () => {
  it("a box has twelve feature edges, not the eighteen triangle edges", () => {
    assert.equal(featureEdges(box, tris).length, 12);
  });
  it("a flat sheet of two triangles has four boundary edges and no diagonal", () => {
    assert.equal(featureEdges([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 3]]).length, 4);
  });
});

describe("moving the camera", () => {
  const cam = (over: Partial<Camera> = {}): Camera => ({ yaw: 0.7, pitch: 0.5, distance: 20, target: [3, 4, 1], ortho: false, ...over });
  const inView = (c: Camera, p: Pt3): number[] => transform(cameraMatrices(c, 800, 600, 10).view, p);

  it("finds where a ray meets any plane, whichever side of the ray's origin it lies", () => {
    const ray = { origin: [0, 0, 5] as Pt3, dir: [0, 0, -1] as Pt3 };
    assert.deepEqual(rayPlane(ray, [0, 0, 2], [0, 0, 1]), [0, 0, 2]);
    assert.deepEqual(rayPlane(ray, [0, 0, 9], [0, 0, 1]), [0, 0, 9]);
    assert.equal(rayPlane(ray, [0, 0, 2], [1, 0, 0]), null);
  });

  it("orbits about a point that is not the target and leaves that point exactly where it was on the screen", () => {
    for (const ortho of [false, true]) {
      const c = cam({ ortho }), pivot: Pt3 = [8, -2, 2.5];
      const before = inView(c, pivot);
      orbitAbout(c, pivot, 0.9, -0.3, 1.5);
      const after = inView(c, pivot);
      for (let i = 0; i < 3; i++) near(before[i] ?? 0, after[i] ?? 0, 1e-4);
    }
  });

  it("orbits about the target exactly as turning yaw and pitch would", () => {
    const c = cam(), d = cam();
    orbitAbout(c, c.target, 0.4, 0.2, 1.5);
    d.yaw += 0.4; d.pitch += 0.2;
    near(c.yaw, d.yaw); near(c.pitch, d.pitch); near(c.distance, d.distance);
    for (let i = 0; i < 3; i++) near(c.target[i] ?? 0, d.target[i] ?? 0);
  });

  it("stops the pitch at the limit and keeps the distance", () => {
    const c = cam();
    orbitAbout(c, [0, 0, 0], 0, 5, 1.2);
    near(c.pitch, 1.2); near(c.distance, 20, 1e-6);
  });

  it("looks around from the eye without moving it", () => {
    const c = cam(), eye = cameraEye(c);
    lookAround(c, 0.6, -0.2, 1.5);
    const now = cameraEye(c);
    for (let i = 0; i < 3; i++) near(now[i] ?? 0, eye[i] ?? 0, 1e-9);
    near(c.distance, 20, 1e-9);
  });

  it("walks along the horizontal direction of view, across it, and up", () => {
    const c = cam({ yaw: 0, pitch: 0.4 });                    // yaw 0 looks along +Y
    walk(c, 2, 0, 0);
    assert.deepEqual(c.target.map((v) => Math.round(v * 1e9) / 1e9), [3, 6, 1]);
    walk(c, 0, 1, 0);
    assert.deepEqual(c.target.map((v) => Math.round(v * 1e9) / 1e9), [4, 6, 1]);        // to the right of a view along +Y is +X
    walk(c, 0, 0, -0.5);
    near(c.target[2] ?? 0, 0.5);
  });

  it("zooms towards the pointer: the point under it stays under it", () => {
    for (const ortho of [false, true]) {
      const c = cam({ ortho });
      const ndc: [number, number] = [0.5, -0.3];
      const ray = rayThrough(cameraMatrices(c, 800, 600, 10).viewProjection, ndc[0], ndc[1]);
      assert.ok(ray);
      // the point on the plane through the target is the one that is kept in place, so use that plane's point
      const focus = rayPlane(ray, c.target, (() => { const e = cameraEye(c); const f: Pt3 = [c.target[0] - e[0], c.target[1] - e[1], c.target[2] - e[2]]; const l = Math.hypot(...f); return [f[0] / l, f[1] / l, f[2] / l] as Pt3; })());
      assert.ok(focus);
      zoomAt(c, 0.5, ndc[0], ndc[1], 800, 600, 10);
      near(c.distance, 10, 1e-9);
      const q = transform(cameraMatrices(c, 800, 600, 10).viewProjection, focus);
      near(q[0] / q[3], ndc[0], 1e-4); near(q[1] / q[3], ndc[1], 1e-4);
    }
  });

  it("zooms about the target when the pointer is at the centre, and respects the limits", () => {
    const c = cam();
    zoomAt(c, 0.5, 0, 0, 800, 600, 10);
    for (let i = 0; i < 3; i++) near(c.target[i] ?? 0, [3, 4, 1][i] ?? 0, 1e-3);
    zoomAt(c, 1e-9, 0, 0, 800, 600, 10);
    near(c.distance, 0.05, 1e-9);
  });

  it("finds the middle of some points", () => {
    assert.equal(centreOf([]), null);
    assert.deepEqual(centreOf([[0, 0, 0], [4, 2, 6], [1, 1, 1]]), [2, 1, 3]);
  });
});

describe("the direction a sheet's 3D view is drawn from", () => {
  it("is the direction from the target to the eye, in the server's terms (azimuth from +X, elevation above the horizon)", () => {
    for (const [yaw, pitch] of [[HOME_VIEW.yaw, HOME_VIEW.pitch], [0, 0.3], [Math.PI / 2, 0.8], [-2.5, -0.4], [4, 1.2]] as const) {
      const cam: Camera = { yaw, pitch, distance: 7, target: [1, 2, 3], ortho: false };
      const eye = cameraEye(cam);
      const d = viewerDirection(cam);
      const az = (d.azimuth * Math.PI) / 180, el = (d.elevation * Math.PI) / 180;
      near((eye[0] - 1) / 7, Math.cos(el) * Math.cos(az), 1e-3);
      near((eye[1] - 2) / 7, Math.cos(el) * Math.sin(az), 1e-3);
      near((eye[2] - 3) / 7, Math.sin(el), 1e-3);
      assert.ok(d.azimuth >= -180 && d.azimuth <= 180);
    }
  });
  it("keeps the elevation inside what the server accepts", () => {
    assert.equal(viewerDirection({ yaw: 0, pitch: Math.PI / 2 }).elevation, 89);
    assert.equal(viewerDirection({ yaw: 0, pitch: -Math.PI / 2 }).elevation, -89);
  });
});
