// Small 2D geometry helpers. Points are [x, y] arrays.
export const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
export const mul = (a, s) => [a[0] * s, a[1] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
export const len = (a) => Math.hypot(a[0], a[1]);
export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
export const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
export const mid = (a, b) => lerp(a, b, 0.5);
export const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l]; };
export const perp = (a) => [-a[1], a[0]]; // rotates +90° in image coords (y down): right -> down
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const smoothstep = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };

export function bilinear(q, u, v) {
  const a = (1 - u) * (1 - v), b = u * (1 - v), c = u * v, d = (1 - u) * v;
  return [a * q[0][0] + b * q[1][0] + c * q[2][0] + d * q[3][0],
          a * q[0][1] + b * q[1][1] + c * q[2][1] + d * q[3][1]];
}

/** Inverse bilinear by Newton iterations (also valid outside the quad). */
export function invBilinear(q, p) {
  let u = 0.5, v = 0.5;
  for (let it = 0; it < 12; it++) {
    const P = bilinear(q, u, v);
    const ex = P[0] - p[0], ey = P[1] - p[1];
    if (ex * ex + ey * ey < 1e-6) break;
    // partial derivatives
    const dPu = [(1 - v) * (q[1][0] - q[0][0]) + v * (q[2][0] - q[3][0]),
                 (1 - v) * (q[1][1] - q[0][1]) + v * (q[2][1] - q[3][1])];
    const dPv = [(1 - u) * (q[3][0] - q[0][0]) + u * (q[2][0] - q[1][0]),
                 (1 - u) * (q[3][1] - q[0][1]) + u * (q[2][1] - q[1][1])];
    const det = dPu[0] * dPv[1] - dPu[1] * dPv[0];
    if (Math.abs(det) < 1e-9) break;
    u -= (ex * dPv[1] - ey * dPv[0]) / det;
    v -= (dPu[0] * ey - dPu[1] * ex) / det;
  }
  return [u, v];
}

export function distToSegment(p, a, b) {
  const ab = sub(b, a), ap = sub(p, a);
  const t = clamp(dot(ap, ab) / (dot(ab, ab) || 1), 0, 1);
  return dist(p, add(a, mul(ab, t)));
}

export function distToPolyline(p, pts) {
  let d = Infinity;
  for (let i = 0; i + 1 < pts.length; i++) d = Math.min(d, distToSegment(p, pts[i], pts[i + 1]));
  if (pts.length === 1) d = dist(p, pts[0]);
  return d;
}

export function pointInPolygon(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if (((yi > p[1]) !== (yj > p[1])) && (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}

/** Signed distance to a polygon (negative inside). */
export function signedDistPolygon(p, poly) {
  let d = Infinity;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) d = Math.min(d, distToSegment(p, poly[j], poly[i]));
  return pointInPolygon(p, poly) ? -d : d;
}

/** Principal axis of a point set: returns {mean, axis, major, minor}. */
export function pca(xs, ys) {
  const n = xs.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  sxx /= n; syy /= n; sxy /= n;
  const tr = sxx + syy, det = sxx * syy - sxy * sxy;
  const l1 = tr / 2 + Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l2 = tr / 2 - Math.sqrt(Math.max(0, tr * tr / 4 - det));
  let axis = Math.abs(sxy) > 1e-9 ? [l1 - syy, sxy] : (sxx >= syy ? [1, 0] : [0, 1]);
  axis = norm(axis);
  return { mean: [mx, my], axis, major: Math.sqrt(l1), minor: Math.sqrt(Math.max(0, l2)) };
}

export function percentile(arr, p) {
  if (!arr.length) return NaN;
  const s = Float64Array.from(arr).sort();
  const i = clamp((s.length - 1) * p, 0, s.length - 1);
  const lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}
