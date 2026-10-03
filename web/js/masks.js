// Builds the "cloth-agnostic" inpainting mask CatVTON needs, from MediaPipe
// segmentation + pose (replaces the DensePose + SCHP masker of the original repo).
import { dist, mid, add, sub, mul, norm, perp } from './geom.js';
import { LM, skeletonFromLandmarks } from './garment.js';

function segChannelCanvas(seg, ch, thr = 0.5) {
  const c = document.createElement('canvas'); c.width = seg.w; c.height = seg.h;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(seg.w, seg.h);
  const t = thr * 255;
  for (let i = 0; i < seg.w * seg.h; i++) {
    const v = seg.data[i * 4 + ch];
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = 255;
    img.data[i * 4 + 3] = v > t ? 255 : 0;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function thresholdCanvas(c, thr = 128) {
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const img = ctx.getImageData(0, 0, c.width, c.height);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const on = d[i] >= thr ? 255 : 0;
    d[i] = d[i + 1] = d[i + 2] = on; d[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/**
 * Returns {mask: canvas (W x H, white = repaint), crop: [x0,y0,x1,y1]}.
 * lm: landmarks in frame px, seg: packed confidences (R clothes, G body skin, B hair/face, A person).
 */
export function buildAgnosticMask(W, H, lm, seg, category, calib) {
  const sk = skeletonFromLandmarks(lm, calib);
  const sw = sk.sw;
  const torsoLen = dist(mid(sk.SR, sk.SL), mid(sk.HR, sk.HL));
  const hipY = (sk.HR[1] + sk.HL[1]) / 2;
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
  ctx.imageSmoothingEnabled = true;
  const white = '#fff';
  const { across, down } = sk;
  const poly = (pts) => { ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.fill(); };
  const stroke = (pts, width) => {
    ctx.lineWidth = width; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke();
  };
  ctx.fillStyle = white; ctx.strokeStyle = white;

  const doUpper = category === 'upper' || category === 'dress';
  const doLower = category === 'lower' || category === 'dress';

  // 1) current clothes in the relevant body region
  if (seg) {
    ctx.save();
    ctx.beginPath();
    if (category === 'upper') ctx.rect(0, 0, W, hipY + 0.3 * torsoLen);
    else if (category === 'lower') ctx.rect(0, hipY - 0.4 * torsoLen, W, H);
    else ctx.rect(0, 0, W, H);
    ctx.clip();
    ctx.drawImage(segChannelCanvas(seg, 0, 0.45), 0, 0, W, H);
    ctx.restore();
  }
  // 2) generous torso / pelvis hull so a bigger garment has room
  if (doUpper) {
    const out = (p, s) => add(p, mul(across, s * 0.22 * sw));
    const bottom = mul(down, (category === 'dress' ? 1.1 : 0.28) * torsoLen);
    poly([
      add(out(sk.SR, -1), mul(down, -0.12 * sw)), add(out(sk.SL, 1), mul(down, -0.12 * sw)),
      add(add(sk.HL, mul(across, 0.3 * sw)), bottom), add(add(sk.HR, mul(across, -0.3 * sw)), bottom),
    ]);
    // arms (sleeve length may change)
    stroke([sk.SR, sk.ER, sk.WR], 0.36 * sw);
    stroke([sk.SL, sk.EL, sk.WL], 0.36 * sw);
  }
  if (doLower) {
    const hipW = sk.hipW;
    const waist = mul(down, -0.38 * torsoLen);
    poly([add(add(sk.HR, waist), mul(across, -0.55 * hipW)), add(add(sk.HL, waist), mul(across, 0.55 * hipW)),
          add(sk.HL, mul(across, 0.6 * hipW)), add(sk.HR, mul(across, -0.6 * hipW))]);
    stroke([sk.HR, sk.KR, sk.AR], 0.75 * hipW);
    stroke([sk.HL, sk.KL, sk.AL], 0.75 * hipW);
  }
  // 3) dilate + smooth
  const tmp = document.createElement('canvas'); tmp.width = W; tmp.height = H;
  const tctx = tmp.getContext('2d');
  tctx.filter = `blur(${Math.round(Math.max(W, H) * 0.008)}px)`;
  tctx.drawImage(c, 0, 0);
  ctx.drawImage(tmp, 0, 0);
  thresholdCanvas(c, 60);

  // 4) protect: face & hair, hands, feet, and the other half of the body
  ctx.fillStyle = '#000'; ctx.strokeStyle = '#000';
  if (seg) {
    ctx.save();
    ctx.filter = `blur(${Math.round(Math.max(W, H) * 0.002)}px)`;
    ctx.globalCompositeOperation = 'destination-out';
    ctx.drawImage(segChannelCanvas(seg, 2, 0.5), 0, 0, W, H);
    ctx.restore();
  }
  ctx.globalCompositeOperation = 'source-over';
  const P = (i) => [lm[i].x, lm[i].y];
  // wrist + (pinky, index, thumb) landmarks per hand
  for (const [wr, fingers] of [[LM.RW, [18, 20, 22]], [LM.LW, [17, 19, 21]]]) {
    if (lm[wr].v < 0.3) continue;
    const hand = fingers.map(P);
    const center = mul(add(add(hand[0], hand[1]), add(hand[2], P(wr))), 0.25);
    const r = Math.max(dist(P(wr), hand[1]) * 1.25, 0.09 * sw);
    ctx.beginPath(); ctx.arc(center[0], center[1], r, 0, Math.PI * 2); ctx.fill();
  }
  if (category === 'upper') {
    const y = hipY + 0.32 * torsoLen;
    ctx.fillRect(0, y, W, H - y);
  }
  if (category === 'lower') {
    const y = hipY - 0.45 * torsoLen;
    ctx.fillRect(0, 0, W, Math.max(0, y));
  }
  if (doLower) {
    for (const i of [27, 28, 29, 30, 31, 32]) {
      if (lm[i].v < 0.3) continue;
      ctx.beginPath(); ctx.arc(lm[i].x, lm[i].y + 0.02 * sw, 0.12 * sw, 0, Math.PI * 2); ctx.fill();
    }
  }
  thresholdCanvas(c, 128);

  // crop box (3:4) around the person incl. head for context
  const ys = [lm[LM.NOSE].y - 0.6 * sw], xs = [];
  const md = ctx.getImageData(0, 0, W, H).data;
  let x0 = W, x1 = 0, y0 = H, y1 = 0;
  for (let y = 0; y < H; y += 4) for (let x = 0; x < W; x += 4) if (md[(y * W + x) * 4] > 128) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  xs.push(x0, x1, sk.SR[0] - 0.4 * sw, sk.SL[0] + 0.4 * sw); ys.push(y0, y1);
  const bx0 = Math.min(...xs), bx1 = Math.max(...xs), by0 = Math.min(...ys), by1 = Math.max(...ys);
  let bh = Math.max((by1 - by0) * 1.12, (bx1 - bx0) * 4 / 3 * 1.08);
  let bw = bh * 3 / 4;
  const cx = (bx0 + bx1) / 2, cy = (by0 + by1) / 2;
  const crop = [Math.round(cx - bw / 2), Math.round(cy - bh / 2), Math.round(cx + bw / 2), Math.round(cy + bh / 2)];
  return { mask: c, crop };
}

export function canvasToBlob(c, type = 'image/png', q) {
  return new Promise((res) => c.toBlob(res, type, q));
}
