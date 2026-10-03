// Turns a garment image into a "garment model": a texture plus a 2D skeleton
// expressed in texture pixels, which the rig maps onto the live body.
//
// Two kinds of source:
//  - flat:  product photo (flat-lay / ghost mannequin). Shoulders, hem and sleeves
//           are estimated from the cut-out silhouette.
//  - worn:  a photo of someone wearing it (model photo, or our own HD result).
//           The skeleton comes straight from pose landmarks on that photo.
import { loadImage } from './api.js';
import { add, sub, mul, dist, mid, norm, perp, pca, percentile, clamp } from './geom.js';

export const LM = { NOSE: 0, LS: 11, RS: 12, LE: 13, RE: 14, LW: 15, RW: 16, LH: 23, RH: 24, LK: 25, RK: 26, LA: 27, RA: 28 };
const MAX_TEX = 1024;

function toCanvas(img, maxSide = MAX_TEX, crop = null) {
  const sx = crop ? crop[0] : 0, sy = crop ? crop[1] : 0;
  const sw = crop ? crop[2] - crop[0] : (img.naturalWidth || img.width);
  const sh = crop ? crop[3] - crop[1] : (img.naturalHeight || img.height);
  const s = Math.min(1, maxSide / Math.max(sw, sh));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(sw * s)); c.height = Math.max(1, Math.round(sh * s));
  c.getContext('2d').drawImage(img, sx, sy, sw, sh, 0, 0, c.width, c.height);
  return { canvas: c, scale: s, ox: sx, oy: sy };
}

function alphaMask(canvas, targetH = 400) {
  const s = Math.min(1, targetH / canvas.height);
  const w = Math.max(8, Math.round(canvas.width * s)), h = Math.max(8, Math.round(canvas.height * s));
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(canvas, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  const m = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) m[i] = d[i * 4 + 3] > 127 ? 1 : 0;
  return { m, w, h, s };
}

/**
 * Runs of garment pixels in a row. Gaps up to `gap` px are merged, but only near
 * the centre (an open jacket front) when `cx` is given — never the gap between a
 * sleeve and the body.
 */
function rowRuns(m, w, y, gap, cx = null, zone = Infinity) {
  const runs = [];
  let start = -1;
  for (let x = 0; x <= w; x++) {
    const on = x < w && m[y * w + x];
    if (on && start < 0) start = x;
    if (!on && start >= 0) { runs.push([start, x - 1]); start = -1; }
  }
  const merged = [];
  for (const r of runs) {
    const last = merged[merged.length - 1];
    const g = last ? r[0] - last[1] : Infinity;
    const gm = last ? (r[0] + last[1]) / 2 : 0;
    if (last && g <= gap && (cx === null || Math.abs(gm - cx) <= zone)) last[1] = r[1];
    else merged.push([...r]);
  }
  return merged;
}

function centerRun(runs, cx) {
  let best = null, bd = Infinity;
  for (const r of runs) {
    const d = cx < r[0] ? r[0] - cx : cx > r[1] ? cx - r[1] : 0;
    if (d < bd) { bd = d; best = r; }
  }
  return best;
}

function bbox(m, w, h) {
  let x0 = w, x1 = -1, y0 = h, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (m[y * w + x]) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  return x1 < 0 ? null : { x0, x1, y0, y1, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

function topAt(m, w, h, x, y0) {
  x = clamp(Math.round(x), 0, w - 1);
  for (let y = y0; y < h; y++) if (m[y * w + x]) return y;
  return y0;
}

// --------------------------------------------------------------------------- //
// Flat garment analysis
// --------------------------------------------------------------------------- //
function analyzeTop(A, category) {
  const { m, w, h } = A;
  const bb = bbox(m, w, h);
  if (!bb) throw new Error('衣服遮罩是空的');
  const gap = Math.round(bb.w * 0.12);
  let cx = (bb.x0 + bb.x1) / 2;
  const tol = bb.w * 0.03;
  // the run that contains the garment centre line (open fronts are merged by rowRuns)
  const runsAt = (y) => {
    for (const r of rowRuns(m, w, y, gap, cx, bb.w * 0.1)) if (r[0] <= cx + tol && r[1] >= cx - tol) return r;
    return null;
  };
  // torso hem = lowest row crossing the centre line (sleeves may hang lower than the hem)
  let hemY = bb.y1;
  for (let y = bb.y1; y > bb.y0; y--) if (runsAt(y)) { hemY = y; break; }
  const top = bb.y0, th = Math.max(8, hemY - top);
  const rowY = (f) => Math.round(top + th * f);
  let xl, xr;
  if (category === 'dress') {
    // bodice rows (just below the armholes)
    const lefts = [], rights = [];
    for (let y = rowY(0.22); y <= rowY(0.40); y++) { const r = runsAt(y); if (r) { lefts.push(r[0]); rights.push(r[1]); } }
    xl = percentile(lefts, 0.65); xr = percentile(rights, 0.35);
  } else {
    // hem band: torso only, except where cuffs touch it -> keep the narrower rows
    const rows = [];
    for (let y = rowY(0.90); y <= rowY(0.99); y++) { const r = runsAt(y); if (r) rows.push(r); }
    rows.sort((a, b) => (a[1] - a[0]) - (b[1] - b[0]));
    const keep = rows.slice(0, Math.max(1, Math.ceil(rows.length * 0.4)));
    const hemL = percentile(keep.map(r => r[0]), 0.5), hemR = percentile(keep.map(r => r[1]), 0.5);
    const hemW = Math.max(4, hemR - hemL);
    cx = (hemL + hemR) / 2;
    // chest/waist rows that are not widened by sleeves hanging alongside the body
    const lefts = [], rights = [];
    for (let y = rowY(0.45); y <= rowY(0.92); y++) {
      const r = runsAt(y);
      if (r && r[1] - r[0] <= 1.35 * hemW) { lefts.push(r[0]); rights.push(r[1]); }
    }
    if (lefts.length >= 5) { xl = percentile(lefts, 0.5); xr = percentile(rights, 0.5); }
    else { xl = cx - 0.54 * hemW; xr = cx + 0.54 * hemW; }
  }
  if (!Number.isFinite(xl) || !Number.isFinite(xr)) { xl = bb.x0 + bb.w * 0.2; xr = bb.x1 - bb.w * 0.2; }
  const tw = Math.max(4, xr - xl);
  cx = (xl + xr) / 2;
  const seamR = [xl + 0.05 * tw, 0], seamL = [xr - 0.05 * tw, 0];
  seamR[1] = topAt(m, w, h, seamR[0], bb.y0);
  seamL[1] = topAt(m, w, h, seamL[0], bb.y0);

  // sleeves = pixels outside the torso column
  const margin = Math.max(1, 0.03 * tw);
  const area = m.reduce((a, b) => a + b, 0);
  const sleeves = {};
  for (const side of ['R', 'L']) {
    const seam = side === 'R' ? seamR : seamL;
    const edgeX = side === 'R' ? xl : xr;
    const outside = (x) => side === 'R' ? x < xl - margin : x > xr + margin;
    const xs = [], ys = [];
    // for dresses only look above the waist: the flared skirt is not a sleeve
    const yMax = category === 'dress' ? Math.min(hemY, seam[1] + 0.8 * tw) : hemY;
    for (let y = bb.y0; y <= yMax; y++) for (let x = bb.x0; x <= bb.x1; x++) {
      if (m[y * w + x] && outside(x)) { xs.push(x); ys.push(y); }
    }
    if (xs.length < area * 0.015) { sleeves[side] = null; continue; }
    // armpit: where a column just outside the torso stops being covered
    const colX = clamp(Math.round(side === 'R' ? xl - 2 * margin : xr + 2 * margin), 0, w - 1);
    let armpit = null, started = false;
    for (let y = Math.round(seam[1]); y <= hemY; y++) {
      if (m[y * w + colX]) started = true;
      else if (started) { armpit = y; break; }
    }
    const armpitY = Math.min(armpit ?? Infinity, seam[1] + 0.32 * tw);
    const root = [edgeX, (seam[1] + armpitY) / 2];
    // cuff = the sleeve pixels farthest from the armhole
    const d = xs.map((x, i) => Math.hypot(x - root[0], ys[i] - root[1]));
    const dCut = percentile(d, 0.88);
    let cxs = 0, cys = 0, cn = 0;
    for (let i = 0; i < xs.length; i++) if (d[i] >= dCut) { cxs += xs[i]; cys += ys[i]; cn++; }
    const cuff = [cxs / cn, cys / cn];
    const axis = norm(sub(cuff, root));
    const length = dist(cuff, root);
    const pa = perp(axis);
    const perpd = xs.map((x, i) => Math.abs((x - root[0]) * pa[0] + (ys[i] - root[1]) * pa[1]));
    const halfW = Math.max(percentile(perpd, 0.85), tw * 0.08);
    sleeves[side] = { root, axis, length, halfW, long: length > 0.75 * tw, line: [root, cuff],
                      armhole: [[seam[0], seam[1]], [edgeX, armpitY]] };
  }
  return { bb, cx, xl, xr, tw, hemY, seamR, seamL, sleeves };
}

function analyzeBottom(A) {
  const { m, w, h } = A;
  const bb = bbox(m, w, h);
  if (!bb) throw new Error('衣服遮罩是空的');
  const gapSmall = Math.round(bb.w * 0.02);
  const lefts = [], rights = [];
  for (let y = bb.y0 + Math.round(bb.h * 0.01); y <= bb.y0 + bb.h * 0.06; y++) {
    const runs = rowRuns(m, w, y, Math.round(bb.w * 0.1));
    const r = centerRun(runs, (bb.x0 + bb.x1) / 2);
    if (r) { lefts.push(r[0]); rights.push(r[1]); }
  }
  const wl = percentile(lefts, 0.5), wr = percentile(rights, 0.5);
  const waistW = Math.max(4, wr - wl), cx = (wl + wr) / 2;
  // crotch: first row (below 15%) where the garment splits into two legs for a while
  let crotchY = null, streak = 0;
  for (let y = Math.round(bb.y0 + bb.h * 0.15); y < bb.y1; y++) {
    const runs = rowRuns(m, w, y, gapSmall).filter(r => r[1] - r[0] > waistW * 0.12);
    const split = runs.length >= 2 && runs.some(r => r[1] < cx) && runs.some(r => r[0] > cx);
    streak = split ? streak + 1 : 0;
    if (streak >= 6) { crotchY = y - 5; break; }
  }
  const legs = {};
  if (crotchY !== null) {
    for (const side of ['R', 'L']) {
      const xs = [], ys = [];
      for (let y = crotchY; y <= bb.y1; y++) for (let x = bb.x0; x <= bb.x1; x++) {
        if (!m[y * w + x]) continue;
        if (side === 'R' ? x < cx : x > cx) { xs.push(x); ys.push(y); }
      }
      if (xs.length < 30) { legs[side] = null; continue; }
      const P = pca(xs, ys);
      let axis = P.axis[1] < 0 ? mul(P.axis, -1) : P.axis;
      const pa = perp(axis);
      const perpd = xs.map((x, i) => Math.abs((x - P.mean[0]) * pa[0] + (ys[i] - P.mean[1]) * pa[1]));
      legs[side] = { axis, halfW: Math.max(percentile(perpd, 0.9), waistW * 0.12), mean: P.mean };
    }
  }
  return { bb, wl, wr, waistW, cx, crotchY, legs, skirt: crotchY === null };
}

/**
 * Build a flat garment model from the server cut-out.
 * `overrides` (optional, normalised 0..1 texture coords) come from the anchor editor.
 */
export function buildFlatModel(cutoutImg, category, overrides = null) {
  const { canvas } = toCanvas(cutoutImg);
  const A = alphaMask(canvas);
  const k = canvas.width / A.w; // analysis px -> texture px
  const P = (p) => [p[0] * k, p[1] * k];
  const model = { mode: 'flat', category, tex: canvas, w: canvas.width, h: canvas.height };

  if (category === 'lower') {
    const a = analyzeBottom(A);
    const top = a.bb.y0;
    const hipW = 0.62 * a.waistW;
    const gHR = [a.cx - hipW / 2, top + 0.30 * a.waistW], gHL = [a.cx + hipW / 2, top + 0.30 * a.waistW];
    model.kind = a.skirt ? 'skirt' : 'pants';
    model.g = { HR: P(gHR), HL: P(gHL) };
    model.legs = {};
    for (const side of ['R', 'L']) {
      const L = a.legs[side];
      model.legs[side] = L ? { axis: L.axis, halfW: L.halfW * k } : null;
    }
    const yC = (a.crotchY ?? a.bb.y1) * k;
    model.torsoPoly = [[a.bb.x0 * k - 4, top * k - 8], [a.bb.x1 * k + 4, top * k - 8],
                       [a.bb.x1 * k + 4, yC], [a.bb.x0 * k - 4, yC]];
    if (a.skirt) model.torsoPoly = [[-10, -10], [model.w + 10, -10], [model.w + 10, model.h + 10], [-10, model.h + 10]];
    model.band = 0.10 * a.waistW * k;
    model.analysis = a;
  } else {
    const a = analyzeTop(A, category);
    const tw = a.tw;
    const kx = 0.15, ky = 0.07; // seam->joint inset as a fraction of torso width
    let SR = [a.xl + kx * tw, a.seamR[1] + ky * tw];
    let SL = [a.xr - kx * tw, a.seamL[1] + ky * tw];
    model.g = { SR: P(SR), SL: P(SL) };
    model.sleeves = {};
    for (const side of ['R', 'L']) {
      const s = a.sleeves[side];
      model.sleeves[side] = s ? { axis: s.axis, length: s.length * k, halfW: s.halfW * k, long: s.long,
        end: P(s.line[1]), line: s.line.map(P), armhole: s.armhole.map(P) } : null;
    }
    const armpitY = Math.min(a.hemY, Math.max(a.seamR[1], a.seamL[1]) + 0.28 * tw);
    model.torsoPoly = [
      P([a.seamR[0], a.bb.y0 - 12]), P([a.seamL[0], a.bb.y0 - 12]),
      P([a.xr, armpitY]), P([a.xr, a.hemY + 12]), P([a.xl, a.hemY + 12]), P([a.xl, armpitY]),
    ];
    model.band = 0.10 * tw * k;
    model.hemY = a.hemY * k;
    model.analysis = a;
  }
  if (overrides) applyOverrides(model, overrides);
  return model;
}

export function applyOverrides(model, o) {
  const T = (p) => [p[0] * model.w, p[1] * model.h];
  if (model.category === 'lower') {
    if (o.HR) model.g.HR = T(o.HR);
    if (o.HL) model.g.HL = T(o.HL);
  } else {
    if (o.SR) model.g.SR = T(o.SR);
    if (o.SL) model.g.SL = T(o.SL);
    for (const side of ['R', 'L']) {
      const key = side === 'R' ? 'ER' : 'EL';
      const s = model.sleeves[side];
      if (o[key] && s) {
        const root = side === 'R' ? model.g.SR : model.g.SL;
        const end = T(o[key]);
        s.axis = norm(sub(end, root));
        s.end = end;
        s.length = dist(end, root);
        s.line = [root, end];
      }
    }
  }
  model.overridden = true;
}

/** Anchor handles for the editor, normalised to 0..1. */
export function editableAnchors(model) {
  const N = (p) => [p[0] / model.w, p[1] / model.h];
  if (model.mode !== 'flat') return {};
  if (model.category === 'lower') return { HR: N(model.g.HR), HL: N(model.g.HL) };
  const out = { SR: N(model.g.SR), SL: N(model.g.SL) };
  if (model.sleeves.R) out.ER = N(model.sleeves.R.end);
  if (model.sleeves.L) out.EL = N(model.sleeves.L.end);
  return out;
}

// --------------------------------------------------------------------------- //
// Worn garment (photo of a person wearing it, or our HD result)
// --------------------------------------------------------------------------- //
export function isWornPhoto(an) {
  if (!an || !an.lm) return false;
  const lm = an.lm;
  const shoulders = Math.min(lm[LM.LS].v, lm[LM.RS].v);
  const face = lm[LM.NOSE].v;
  let skin = 0;
  if (an.seg) {
    const d = an.seg.data;
    for (let i = 0; i < d.length; i += 4) if (d[i + 1] > 128 || d[i + 2] > 128) skin++;
    skin /= (d.length / 4);
  }
  const sw = dist([lm[LM.LS].x, lm[LM.LS].y], [lm[LM.RS].x, lm[LM.RS].y]);
  return shoulders > 0.8 && face > 0.6 && skin > 0.01 && sw > 0.12 * an.W;
}

const P2 = (p) => [p.x, p.y];

/** Skeleton points from landmarks, with estimates for poorly visible joints. */
export function skeletonFromLandmarks(lm, calib) {
  const SR = P2(lm[LM.RS]), SL = P2(lm[LM.LS]);
  const sw = dist(SR, SL) || 1;
  const across = norm(sub(SL, SR));
  const down = perp(across);
  const smid = mid(SR, SL);
  const vis = (i, lo = 0.35, hi = 0.75) => clamp((lm[i].v - lo) / (hi - lo), 0, 1);
  const mixp = (est, i) => { const t = vis(i); return [est[0] + (lm[i].x - est[0]) * t, est[1] + (lm[i].y - est[1]) * t]; };
  const hipMidEst = add(smid, mul(down, calib.R * sw));
  const HR = mixp(add(hipMidEst, mul(across, -calib.H * sw / 2)), LM.RH);
  const HL = mixp(add(hipMidEst, mul(across, calib.H * sw / 2)), LM.LH);
  const out = { SR, SL, HR, HL, sw, across, down };
  for (const [S, E, W, side, sgn] of [[SR, LM.RE, LM.RW, 'R', -1], [SL, LM.LE, LM.LW, 'L', 1]]) {
    const Eest = add(S, add(mul(down, calib.U * sw), mul(across, sgn * 0.12 * sw)));
    const Ep = mixp(Eest, E);
    const Wdir = norm(sub(Ep, S));
    const Wp = mixp(add(Ep, mul(Wdir, calib.F * sw)), W);
    out['E' + side] = Ep; out['W' + side] = Wp;
  }
  const hipW = dist(HR, HL) || sw * 0.6;
  for (const [H, K, A, side] of [[HR, LM.RK, LM.RA, 'R'], [HL, LM.LK, LM.LA, 'L']]) {
    const Kp = mixp(add(H, mul(down, calib.T * hipW)), K);
    const Ap = mixp(add(Kp, mul(norm(sub(Kp, H)), calib.S * hipW)), A);
    out['K' + side] = Kp; out['A' + side] = Ap;
  }
  out.hipW = hipW;
  return out;
}

/**
 * Build a worn garment model.
 * img: image/canvas; an: {W,H,lm (image px), seg (packed conf)}; restrict: optional mask canvas
 * (same size as img) limiting the garment area (used for HD results).
 */
export function buildWornModel(img, an, category, calib, restrict = null) {
  const W = an.W, H = an.H;
  const sk = skeletonFromLandmarks(an.lm, calib);
  const torsoLen = dist(mid(sk.SR, sk.SL), mid(sk.HR, sk.HL));
  // garment mask from the clothes class
  const mc = document.createElement('canvas'); mc.width = W; mc.height = H;
  const mctx = mc.getContext('2d', { willReadFrequently: true });
  const sc = document.createElement('canvas'); sc.width = an.seg.w; sc.height = an.seg.h;
  const sctx = sc.getContext('2d');
  const sd = sctx.createImageData(an.seg.w, an.seg.h);
  for (let i = 0; i < an.seg.w * an.seg.h; i++) {
    const c = an.seg.data[i * 4];
    sd.data[i * 4] = 255; sd.data[i * 4 + 1] = 255; sd.data[i * 4 + 2] = 255;
    sd.data[i * 4 + 3] = c > 110 ? Math.min(255, (c - 110) * 2.2) : 0;
  }
  sctx.putImageData(sd, 0, 0);
  mctx.imageSmoothingEnabled = true;
  mctx.drawImage(sc, 0, 0, W, H);
  // category region
  mctx.globalCompositeOperation = 'destination-in';
  const hipY = (sk.HR[1] + sk.HL[1]) / 2;
  mctx.fillStyle = '#fff';
  if (category === 'upper') mctx.fillRect(0, 0, W, hipY + 0.45 * torsoLen);
  else if (category === 'lower') mctx.fillRect(0, hipY - 0.45 * torsoLen, W, H);
  else mctx.fillRect(0, 0, W, H);
  if (restrict) mctx.drawImage(restrict, 0, 0, W, H);
  mctx.globalCompositeOperation = 'source-over';
  // the clothes class covers top AND bottoms: split them at the strongest colour change
  if (category !== 'dress') {
    const cut = garmentBoundaryY(img, mc, sk, torsoLen, W, H);
    mctx.clearRect(0, category === 'upper' ? cut : 0, W, category === 'upper' ? H - cut : cut);
  }
  // bbox of the mask
  const md = mctx.getImageData(0, 0, W, H).data;
  let x0 = W, x1 = 0, y0 = H, y1 = 0, cnt = 0;
  for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) if (md[(y * W + x) * 4 + 3] > 40) {
    cnt++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (cnt < 200) throw new Error('在照片上找不到衣服區域');
  const pad = 0.03 * Math.max(W, H);
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad); x1 = Math.min(W, x1 + pad); y1 = Math.min(H, y1 + pad);
  // texture = image * mask, cropped
  const full = document.createElement('canvas'); full.width = W; full.height = H;
  const fctx = full.getContext('2d');
  fctx.drawImage(img, 0, 0, W, H);
  fctx.globalCompositeOperation = 'destination-in';
  fctx.drawImage(mc, 0, 0);
  const { canvas, scale } = toCanvas(full, MAX_TEX, [x0, y0, x1, y1]);
  const T = (p) => [(p[0] - x0) * scale, (p[1] - y0) * scale];
  const g = {};
  for (const key of ['SR', 'SL', 'HR', 'HL', 'ER', 'EL', 'WR', 'WL', 'KR', 'KL', 'AR', 'AL']) g[key] = T(sk[key]);
  const sw = sk.sw * scale;
  const model = { mode: 'worn', category, tex: canvas, w: canvas.width, h: canvas.height, g };
  model.band = 0.12 * sw;
  if (category === 'lower') {
    model.kind = 'pants';
    const up = mul(sub(mid(g.SR, g.SL), mid(g.HR, g.HL)), 0.45);
    model.torsoPoly = [add(g.HR, up), add(g.HL, up), add(g.HL, mul(up, -0.35)), add(g.HR, mul(up, -0.35))];
    model.legs = { R: { halfW: 0.13 * sw }, L: { halfW: 0.13 * sw } };
  } else {
    const across = norm(sub(g.SL, g.SR));
    const up = mul(perp(across), -0.45 * sw);
    const extDown = mul(sub(mid(g.HR, g.HL), mid(g.SR, g.SL)), category === 'dress' ? 2.5 : 0.6);
    const inS = mul(across, 0.12 * sw);
    model.torsoPoly = [add(add(g.SR, inS), up), add(sub(g.SL, inS), up),
                       add(sub(g.HL, mul(across, 0.02 * sw)), extDown), add(add(g.HR, mul(across, 0.02 * sw)), extDown)];
    const downV = perp(across);
    const armhole = (S, sgn) => [add(S, mul(downV, -0.12 * sw)), add(add(S, mul(downV, 0.42 * sw)), mul(across, -sgn * 0.06 * sw))];
    model.sleeves = {
      R: { halfW: 0.11 * sw, long: true, worn: true, armhole: armhole(g.SR, -1) },
      L: { halfW: 0.11 * sw, long: true, worn: true, armhole: armhole(g.SL, 1) },
    };
  }
  return model;
}

/** y (image px) where the top ends and the bottoms begin, found from row colour statistics. */
function garmentBoundaryY(img, maskCanvas, sk, torsoLen, W, H) {
  const hipY = (sk.HR[1] + sk.HL[1]) / 2;
  const fallback = hipY + 0.3 * torsoLen;
  const s = Math.min(1, 200 / W);
  const w = Math.max(8, Math.round(W * s)), h = Math.max(8, Math.round(H * s));
  const a = document.createElement('canvas'); a.width = w; a.height = h;
  const actx = a.getContext('2d', { willReadFrequently: true });
  actx.drawImage(img, 0, 0, w, h);
  const px = actx.getImageData(0, 0, w, h).data;
  actx.clearRect(0, 0, w, h); actx.drawImage(maskCanvas, 0, 0, w, h);
  const mk = actx.getImageData(0, 0, w, h).data;
  const x0 = Math.max(0, Math.round(Math.min(sk.HR[0], sk.SR[0]) * s)), x1 = Math.min(w - 1, Math.round(Math.max(sk.HL[0], sk.SL[0]) * s));
  const y0 = Math.max(0, Math.round((hipY - 0.35 * torsoLen) * s)), y1 = Math.min(h - 1, Math.round((hipY + 0.55 * torsoLen) * s));
  const rows = [];
  for (let y = y0; y <= y1; y++) {
    let r = 0, g = 0, b = 0, n = 0;
    for (let x = Math.min(x0, x1); x <= Math.max(x0, x1); x++) {
      const i = (y * w + x) * 4;
      if (mk[i + 3] > 128) { r += px[i]; g += px[i + 1]; b += px[i + 2]; n++; }
    }
    rows.push(n > 2 ? [r / n, g / n, b / n] : null);
  }
  const k = Math.max(2, Math.round(0.04 * h));
  let best = -1;
  const cands = [];
  const avg = (from, to) => {
    let r = 0, g = 0, b = 0, n = 0;
    for (let i = Math.max(0, from); i < Math.min(rows.length, to); i++) if (rows[i]) { r += rows[i][0]; g += rows[i][1]; b += rows[i][2]; n++; }
    return n ? [r / n, g / n, b / n] : null;
  };
  for (let i = k; i < rows.length - k; i++) {
    const A = avg(i - k, i), B = avg(i, i + k);
    if (!A || !B) continue;
    const d = Math.hypot(A[0] - B[0], A[1] - B[1], A[2] - B[2]);
    cands.push([d, (y0 + i) / s]);
    best = Math.max(best, d);
  }
  if (best <= 30) return fallback;
  // the top/bottoms seam is usually the lowest strong change (prints inside the top come first)
  let y = null;
  for (const [d, cy] of cands) if (d >= Math.max(30, 0.7 * best)) y = y === null ? cy : Math.max(y, cy);
  return y ?? fallback;
}

// --------------------------------------------------------------------------- //
export async function buildGarment(meta, tracker, calib) {
  const cutout = await loadImage(meta.cutout_url);
  let analysis = null;
  try {
    const original = await loadImage(meta.original_url);
    analysis = await tracker.analyzeImage(original);
    if (isWornPhoto(analysis) && analysis.seg) {
      const m = buildWornModel(original, analysis, meta.category, calib);
      m.meta = meta;
      m.fromPhoto = true;
      return m;
    }
  } catch (e) { console.warn('worn analysis failed, using flat', e); }
  const m = buildFlatModel(cutout, meta.category, meta.anchors);
  m.meta = meta;
  return m;
}

/** Mirror a garment model (used so logos read correctly in the mirrored preview). */
export function flipModel(m) {
  const W = m.w;
  const c = document.createElement('canvas'); c.width = m.w; c.height = m.h;
  const ctx = c.getContext('2d'); ctx.translate(W, 0); ctx.scale(-1, 1); ctx.drawImage(m.tex, 0, 0);
  const F = (p) => p && [W - p[0], p[1]];
  const swap = (k) => k.endsWith('R') ? k.slice(0, -1) + 'L' : k.endsWith('L') ? k.slice(0, -1) + 'R' : k;
  const g = {};
  for (const [k, v] of Object.entries(m.g)) g[swap(k)] = F(v);
  const part = (s) => s && ({ ...s, axis: s.axis && [-s.axis[0], s.axis[1]], end: F(s.end), line: s.line && s.line.map(F),
                               armhole: s.armhole && s.armhole.map(F) });
  const sides = (o) => o && ({ R: part(o.L), L: part(o.R) });
  return { ...m, tex: c, g, sleeves: sides(m.sleeves), legs: sides(m.legs), torsoPoly: m.torsoPoly.map(F).reverse(), flipped: !m.flipped };
}
