// 2D skinning rig: a mesh over the garment texture, deformed every frame by
// a torso bilinear map plus per-limb bone transforms (linear blend skinning).
import { add, sub, mul, dist, mid, norm, perp, clamp, smoothstep, bilinear, invBilinear,
         distToSegment, distToPolyline, signedDistPolygon } from './geom.js';
import { LM } from './garment.js';

const DEFAULT_CALIB = { R: 1.35, H: 0.62, U: 0.85, F: 0.75, T: 2.1, S: 2.0 };
const RANGE = { R: [1.0, 1.9], H: [0.45, 0.9], U: [0.6, 1.2], F: [0.5, 1.1], T: [1.5, 3.0], S: [1.4, 3.0] };

/** Per-user body proportions, learnt while the person faces the camera. */
export class BodyCalib {
  constructor() { Object.assign(this, DEFAULT_CALIB); this.version = 0; this._snap = { ...DEFAULT_CALIB }; this.samples = 0; }
  values() { const o = {}; for (const k in DEFAULT_CALIB) o[k] = this[k]; return o; }
  _ema(key, val, rate) {
    const [lo, hi] = RANGE[key];
    if (!Number.isFinite(val)) return;
    this[key] += (clamp(val, lo, hi) - this[key]) * rate;
  }
  update(lm, world) {
    if (!lm) return false;
    const v = (i) => lm[i].v;
    const P = (i) => [lm[i].x, lm[i].y];
    const frontal = world ? Math.abs(world[LM.LS].z - world[LM.RS].z) < 0.12 : true;
    if (!(v(LM.LS) > 0.8 && v(LM.RS) > 0.8 && frontal)) return false;
    const sw = dist(P(LM.LS), P(LM.RS));
    const warm = this.samples < 60 ? 0.15 : 0.03;
    if (v(LM.LH) > 0.8 && v(LM.RH) > 0.8) {
      const hipW = dist(P(LM.LH), P(LM.RH));
      this._ema('R', dist(mid(P(LM.LS), P(LM.RS)), mid(P(LM.LH), P(LM.RH))) / sw, warm);
      this._ema('H', hipW / sw, warm);
      if (v(LM.LK) > 0.8 && v(LM.RK) > 0.8) {
        this._ema('T', (dist(P(LM.LH), P(LM.LK)) + dist(P(LM.RH), P(LM.RK))) / 2 / hipW, warm);
        if (v(LM.LA) > 0.8 && v(LM.RA) > 0.8)
          this._ema('S', (dist(P(LM.LK), P(LM.LA)) + dist(P(LM.RK), P(LM.RA))) / 2 / hipW, warm);
      }
      this.samples++;
    }
    // limb lengths: projected lengths only shrink (foreshortening), so track the upper envelope
    for (const [S, E, W] of [[LM.LS, LM.LE, LM.LW], [LM.RS, LM.RE, LM.RW]]) {
      if (v(E) > 0.8) {
        const u = dist(P(S), P(E)) / sw;
        this._ema('U', u, u > this.U ? 0.08 : 0.004);
        if (v(W) > 0.8) { const f = dist(P(E), P(W)) / sw; this._ema('F', f, f > this.F ? 0.08 : 0.004); }
      }
    }
    let changed = false;
    for (const k in DEFAULT_CALIB) if (Math.abs(this[k] - this._snap[k]) / this._snap[k] > 0.04) changed = true;
    if (changed) { this._snap = this.values(); this.version++; }
    return changed;
  }
}

// --------------------------------------------------------------------------- //
function affineFromBone(g0, g1, root, target, sPerp, sLo, sHi) {
  const dg = sub(g1, g0); const lg = Math.hypot(dg[0], dg[1]) || 1;
  const dirG = [dg[0] / lg, dg[1] / lg], perpG = perp(dirG);
  const db = sub(target, root); const lb = Math.hypot(db[0], db[1]);
  const dirB = lb > 1e-3 ? [db[0] / lb, db[1] / lb] : dirG, perpB = perp(dirB);
  const along = clamp(lb / lg, sLo, sHi);
  // M = along * dirB (x) dirG + sPerp * perpB (x) perpG
  const a = along * dirB[0] * dirG[0] + sPerp * perpB[0] * perpG[0];
  const b = along * dirB[0] * dirG[1] + sPerp * perpB[0] * perpG[1];
  const c = along * dirB[1] * dirG[0] + sPerp * perpB[1] * perpG[0];
  const d = along * dirB[1] * dirG[1] + sPerp * perpB[1] * perpG[1];
  const tx = root[0] - (a * g0[0] + b * g0[1]);
  const ty = root[1] - (c * g0[0] + d * g0[1]);
  return [a, b, c, d, tx, ty];
}
const applyAff = (m, p) => [m[0] * p[0] + m[1] * p[1] + m[4], m[2] * p[0] + m[3] * p[1] + m[5]];

export class Rig {
  constructor(model, calib) {
    this.model = model;
    this.buildMesh();
    this.prepare(calib);
  }

  // ---- mesh -----------------------------------------------------------------
  buildMesh() {
    const { w, h, tex } = this.model;
    const cols = 48, rows = clamp(Math.round(cols * h / w), 24, 96);
    this.cols = cols; this.rows = rows;
    const n = (cols + 1) * (rows + 1);
    this.gpos = new Float32Array(n * 2);
    this.uvs = new Float32Array(n * 2);
    for (let j = 0; j <= rows; j++) for (let i = 0; i <= cols; i++) {
      const k = j * (cols + 1) + i;
      this.gpos[k * 2] = i / cols * w; this.gpos[k * 2 + 1] = j / rows * h;
      this.uvs[k * 2] = i / cols; this.uvs[k * 2 + 1] = j / rows;
    }
    // keep cells that touch the garment (max alpha over a 2x supersampled grid, dilated by one cell)
    const sc = document.createElement('canvas'); sc.width = cols * 2; sc.height = rows * 2;
    const sctx = sc.getContext('2d', { willReadFrequently: true });
    sctx.drawImage(tex, 0, 0, sc.width, sc.height);
    const a = sctx.getImageData(0, 0, sc.width, sc.height).data;
    const occ = new Uint8Array(cols * rows);
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      let m = 0;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) m = Math.max(m, a[((j * 2 + dy) * sc.width + i * 2 + dx) * 4 + 3]);
      occ[j * cols + i] = m > 4 ? 1 : 0;
    }
    this.cells = [];
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      let keep = false;
      for (let dj = -1; dj <= 1 && !keep; dj++) for (let di = -1; di <= 1; di++) {
        const jj = j + dj, ii = i + di;
        if (jj >= 0 && jj < rows && ii >= 0 && ii < cols && occ[jj * cols + ii]) { keep = true; break; }
      }
      if (keep) this.cells.push(j * (cols + 1) + i);
    }
    // part mask: 1 where a texel lies outside the torso panel (sleeve/leg side). Used to
    // clip the duplicated triangles along a hard split at texel precision.
    const pw = Math.max(16, Math.round(Math.min(256, w) )), ph = Math.max(16, Math.round(pw * h / w));
    this.partW = pw; this.partH = ph;
    this.part = new Uint8Array(pw * ph);
    const poly = this.model.torsoPoly;
    for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
      const gp = [(x + 0.5) / pw * w, (y + 0.5) / ph * h];
      this.part[y * pw + x] = signedDistPolygon(gp, poly) > 0 ? 255 : 0;
    }
    this.n0 = n;              // grid vertices; prepare() may append duplicates
    this.baseGpos = this.gpos; this.baseUvs = this.uvs;
    this.n = n;
    this.out = new Float32Array(n * 2);
  }

  // ---- garment-space skeleton -------------------------------------------------
  garmentSkeleton(calib) {
    const m = this.model, g = m.g;
    if (m.category === 'lower') {
      const HR = g.HR, HL = g.HL, hipW = dist(HR, HL) || 1;
      const across = norm(sub(HL, HR)), down = perp(across);
      const up = mul(down, -0.55 * hipW);
      const quad = [add(HR, up), add(HL, up), HL, HR];
      const chains = [];
      for (const side of ['R', 'L']) {
        const H = side === 'R' ? HR : HL;
        let K, A, halfW;
        if (m.mode === 'worn') { K = g['K' + side]; A = g['A' + side]; halfW = m.legs[side].halfW; }
        else {
          const leg = m.legs?.[side];
          if (!leg) continue;
          K = add(H, mul(leg.axis, calib.T * hipW)); A = add(K, mul(leg.axis, calib.S * hipW)); halfW = leg.halfW;
        }
        chains.push({ side, pts: [H, K, A], halfW, line: [H, K, A], long: true, rootUV: side === 'R' ? [0, 1] : [1, 1] });
      }
      return { quad, chains, width: hipW };
    }
    const SR = g.SR, SL = g.SL, sw = dist(SR, SL) || 1;
    const across = norm(sub(SL, SR)), down = perp(across);
    let HR, HL;
    if (m.mode === 'worn') { HR = g.HR; HL = g.HL; }
    else {
      const hm = add(mid(SR, SL), mul(down, calib.R * sw));
      HR = add(hm, mul(across, -calib.H * sw / 2)); HL = add(hm, mul(across, calib.H * sw / 2));
    }
    const quad = [SR, SL, HL, HR];
    const chains = [];
    for (const side of ['R', 'L']) {
      const S = side === 'R' ? SR : SL;
      const sl = m.sleeves?.[side];
      if (!sl) continue;
      let R, E, W, line;
      if (m.mode === 'worn') { R = S; E = g['E' + side]; W = g['W' + side]; line = [S, E, W]; }
      else {
        // the sleeve hangs from the armhole centre; that point is carried by the torso map
        // at run time, so sleeve and body stay stitched together
        R = sl.line[0];
        E = add(R, mul(sl.axis, calib.U * sw)); W = add(E, mul(sl.axis, calib.F * sw)); line = sl.line;
      }
      chains.push({ side, pts: [R, E, W], halfW: sl.halfW, line, long: sl.long, armhole: sl.armhole || null,
                    rootUV: invBilinear(quad, R) });
    }
    return { quad, chains, width: sw };
  }

  // ---- uv + weights (recomputed when the calibration drifts) -------------------
  prepare(calib) {
    const m = this.model;
    const skel = this.garmentSkeleton(calib);
    this.skel = skel;
    const n = this.n0, B = Math.max(2, m.band || 10);
    this.gpos = this.baseGpos; this.uvs = this.baseUvs;
    this.U = new Float32Array(n); this.V = new Float32Array(n);
    this.W = new Float32Array(n * 5); // torso, R upper, R lower, L upper, L lower
    this.limb = new Float32Array(n);
    const poly = m.torsoPoly;
    for (let k = 0; k < n; k++) {
      const p = [this.gpos[k * 2], this.gpos[k * 2 + 1]];
      const [u, v] = invBilinear(skel.quad, p);
      this.U[k] = u; this.V[k] = v;
      const sd = signedDistPolygon(p, poly);
      // Sleeve and body fabric are joined only along the armhole seam: blend softly
      // there, but split sharply elsewhere (a sleeve hanging beside the body must not
      // drag the body panel with it when the arm lifts).
      let nearSeam = 0;
      for (const c of skel.chains) if (c.armhole) nearSeam = Math.max(nearSeam, 1 - smoothstep(0, 1.5 * B, distToSegment(p, c.armhole[0], c.armhole[1])));
      if (!skel.chains.some(c => c.armhole)) nearSeam = 1;
      const soft = smoothstep(-B / 2, B / 2, sd);
      const hard = smoothstep(-0.08 * B, 0.08 * B, sd);
      const outside = hard + (soft - hard) * nearSeam;
      const w = [1 - outside + 1e-3, 0, 0, 0, 0];
      for (const c of skel.chains) {
        const d = distToPolyline(p, c.line);
        // generous reach so cuffs and sleeve corners never fall back to the body panel
        const wa = (1 - smoothstep(1.2 * c.halfW, 1.2 * c.halfW + 2.5 * B, d)) * outside;
        if (wa <= 0) continue;
        let f = 0;
        if (c.long) {
          const dU = distToSegment(p, c.pts[0], c.pts[1]), dF = distToSegment(p, c.pts[1], c.pts[2]);
          f = smoothstep(-0.5 * c.halfW, 0.5 * c.halfW, dU - dF);
        }
        const base = c.side === 'R' ? 1 : 3;
        w[base] += wa * (1 - f); w[base + 1] += wa * f;
      }
      const s = w[0] + w[1] + w[2] + w[3] + w[4];
      for (let i = 0; i < 5; i++) this.W[k * 5 + i] = w[i] / s;
      this.limb[k] = 1 - w[0] / s;
    }
    // Triangles: torso first, limbs drawn on top. A triangle that straddles a hard
    // body/sleeve split would stretch into a streak when the arm moves, so it is given
    // its own copies of the vertices, all weighted to one side: the mesh is cut along the
    // split (no streaks), yet nothing is missing in the rest pose.
    const torso = [], limb = [];
    const C = this.cols + 1;
    const L = this.limb, Wt = this.W;
    const extra = { gpos: [], uvs: [], U: [], V: [], W: [], limb: [], cut: [] };
    const dupCache = new Map();
    let next = n;
    const dup = (k, group, donor) => {
      const key = k * 2 + group;
      if (dupCache.has(key)) return dupCache.get(key);
      let w = [Wt[k * 5], Wt[k * 5 + 1], Wt[k * 5 + 2], Wt[k * 5 + 3], Wt[k * 5 + 4]];
      if (group === 0) w = [1, 0, 0, 0, 0];
      else {
        let sl = w[1] + w[2] + w[3] + w[4];
        if (sl < 0.05) { w = [Wt[donor * 5], Wt[donor * 5 + 1], Wt[donor * 5 + 2], Wt[donor * 5 + 3], Wt[donor * 5 + 4]]; sl = w[1] + w[2] + w[3] + w[4]; }
        w = [0, w[1] / sl, w[2] / sl, w[3] / sl, w[4] / sl];
      }
      extra.gpos.push(this.baseGpos[k * 2], this.baseGpos[k * 2 + 1]);
      extra.uvs.push(this.baseUvs[k * 2], this.baseUvs[k * 2 + 1]);
      extra.U.push(this.U[k]); extra.V.push(this.V[k]); extra.W.push(...w); extra.limb.push(group); extra.cut.push(1);
      dupCache.set(key, next);
      return next++;
    };
    for (const k of this.cells) {
      for (const tri of [[k, k + 1, k + C + 1], [k, k + C + 1, k + C]]) {
        const a = L[tri[0]], b = L[tri[1]], c = L[tri[2]];
        const isLimb = (a + b + c) / 3 > 0.5;
        if (Math.max(a, b, c) - Math.min(a, b, c) <= 0.55) { (isLimb ? limb : torso).push(...tri); continue; }
        // render it twice: once riding with the body, once with the limb; the fragment
        // shader keeps only the texels of the matching side (part mask)
        const donor = tri.reduce((best, v) => (L[v] > L[best] ? v : best), tri[0]);
        torso.push(...tri.map(v => dup(v, 0, donor)));
        limb.push(...tri.map(v => dup(v, 1, donor)));
      }
    }
    if (next > n) {
      const grow = (base, add) => { const out = new Float32Array(base.length + add.length); out.set(base); out.set(add, base.length); return out; };
      this.gpos = grow(this.baseGpos, extra.gpos); this.uvs = grow(this.baseUvs, extra.uvs);
      this.U = grow(this.U, extra.U); this.V = grow(this.V, extra.V); this.W = grow(this.W, extra.W);
      this.limb = grow(this.limb, extra.limb);
    }
    this.cut = new Float32Array(next);
    this.cut.fill(1, n);
    this.n = next;
    if (!this.out || this.out.length !== next * 2) this.out = new Float32Array(next * 2);
    const IA = next > 65535 ? Uint32Array : Uint16Array;
    this.idxTorso = IA.from(torso); this.idxLimb = IA.from(limb);
    this.calibVersion = calib.version;
    this.indicesVersion = (this.indicesVersion || 0) + 1;
  }

  // ---- per frame -----------------------------------------------------------------
  /**
   * sk: body skeleton (frame px) from skeletonFromLandmarks
   * fit: {size, offsetY, length}
   */
  update(sk, fit) {
    const m = this.model, skel = this.skel;
    const size = fit.size ?? 1, length = fit.length ?? 1, off = fit.offsetY ?? 0;
    let bq;
    if (m.category === 'lower') {
      const across = norm(sub(sk.HL, sk.HR)), down = perp(across);
      const up = mul(down, -0.55 * sk.hipW);
      bq = [add(sk.HR, up), add(sk.HL, up), sk.HL, sk.HR];
    } else {
      bq = [sk.SR, sk.SL, sk.HL, sk.HR];
    }
    // fit: scale about the quad centre, shift along the body's down axis
    const c = [(bq[0][0] + bq[1][0] + bq[2][0] + bq[3][0]) / 4, (bq[0][1] + bq[1][1] + bq[2][1] + bq[3][1]) / 4];
    const torsoLen = dist(mid(bq[0], bq[1]), mid(bq[2], bq[3]));
    const shift = mul(sk.down, off * torsoLen);
    bq = bq.map(p => add(add(c, mul(sub(p, c), size)), shift));
    const bodyWidth = dist(bq[0], bq[1]);
    const sG = bodyWidth / (skel.width || 1);
    // limb transforms
    const aff = [null, null, null, null];
    const target = m.category === 'lower'
      ? { R: [sk.KR, sk.AR], L: [sk.KL, sk.AL] }
      : { R: [sk.ER, sk.WR], L: [sk.EL, sk.WL] };
    for (const ch of skel.chains) {
      const [g0, g1, g2] = ch.pts;
      const root = bilinear(bq, ch.rootUV[0], ch.rootUV[1] * length);
      const [t1, t2] = target[ch.side].map(p => add(p, shift));
      const A1 = affineFromBone(g0, g1, root, t1, sG, 0.3 * sG, 1.6 * sG);
      const e = applyAff(A1, g1);
      const A2 = affineFromBone(g1, g2, e, t2, sG, 0.3 * sG, 1.6 * sG);
      const base = ch.side === 'R' ? 0 : 2;
      aff[base] = A1; aff[base + 1] = A2;
    }
    const out = this.out, W = this.W, gp = this.gpos, U = this.U, V = this.V;
    const q0 = bq[0], q1 = bq[1], q2 = bq[2], q3 = bq[3];
    for (let k = 0; k < this.n; k++) {
      const u = U[k], v = V[k] * length;
      const a = (1 - u) * (1 - v), b = u * (1 - v), cc = u * v, d = (1 - u) * v;
      const wt = W[k * 5];
      let x = wt * (a * q0[0] + b * q1[0] + cc * q2[0] + d * q3[0]);
      let y = wt * (a * q0[1] + b * q1[1] + cc * q2[1] + d * q3[1]);
      const gx = gp[k * 2], gy = gp[k * 2 + 1];
      for (let i = 0; i < 4; i++) {
        const wi = W[k * 5 + 1 + i];
        if (wi > 1e-4 && aff[i]) {
          const M = aff[i];
          x += wi * (M[0] * gx + M[1] * gy + M[4]);
          y += wi * (M[2] * gx + M[3] * gy + M[5]);
        } else if (wi > 1e-4) {
          // limb without a bone (should not happen): fall back to the torso map
          x += wi * (a * q0[0] + b * q1[0] + cc * q2[0] + d * q3[0]);
          y += wi * (a * q0[1] + b * q1[1] + cc * q2[1] + d * q3[1]);
        }
      }
      out[k * 2] = x; out[k * 2 + 1] = y;
    }
    this.lastBodyQuad = bq;
    return out;
  }
}
