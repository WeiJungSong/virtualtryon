// Developer page: visualise garment analysis and the skinning rig on synthetic poses.
import * as api from './api.js';
import { buildFlatModel, skeletonFromLandmarks } from './garment.js';
import { Rig, BodyCalib } from './rig.js';
import { Renderer } from './renderer.js';

const out = document.getElementById('out');
const calib = new BodyCalib();

function lmFrom(points) {
  const lm = Array.from({ length: 33 }, () => ({ x: 640, y: 360, z: 0, v: 1 }));
  for (const [i, [x, y]] of Object.entries(points)) lm[i] = { x, y, z: 0, v: 1 };
  return lm;
}
const base = { 0: [640, 120], 11: [720, 230], 12: [560, 230], 23: [692, 455], 24: [588, 455], 25: [700, 640], 26: [580, 640], 27: [705, 820], 28: [575, 820],
  19: [800, 520], 20: [480, 520] };
const POSES = {
  'A-pose': { ...base, 13: [770, 360], 15: [800, 480], 14: [510, 360], 16: [480, 480] },
  '舉手': { ...base, 13: [850, 200], 15: [930, 100], 14: [430, 200], 16: [350, 100], 19: [940, 80], 20: [340, 80] },
  '跨步': { ...base, 13: [770, 360], 15: [800, 480], 14: [510, 360], 16: [480, 480], 25: [760, 630], 27: [820, 800], 26: [560, 640], 28: [540, 830] },
  '側身': { ...base, 11: [690, 232], 12: [590, 228], 23: [672, 455], 24: [604, 455], 13: [720, 360], 15: [735, 480], 14: [585, 360], 16: [590, 480] },
};

function drawAnalysis(model, rig) {
  const s = 360 / Math.max(model.w, model.h);
  const c = document.createElement('canvas'); c.width = model.w * s; c.height = model.h * s;
  const ctx = c.getContext('2d');
  ctx.drawImage(model.tex, 0, 0, c.width, c.height);
  const P = (p) => [p[0] * s, p[1] * s];
  const poly = (pts, color, close = true) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    pts.map(P).forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p)); if (close) ctx.closePath(); ctx.stroke();
  };
  // limb weights as dots
  for (let k = 0; k < rig.n; k += 2) {
    const l = rig.limb[k];
    if (l < 0.02) continue;
    ctx.fillStyle = `rgba(80,160,255,${0.15 + 0.6 * l})`;
    ctx.fillRect(rig.gpos[k * 2] * s - 1, rig.gpos[k * 2 + 1] * s - 1, 2, 2);
  }
  poly(model.torsoPoly, '#6f6');
  const sk = rig.skel;
  poly(sk.quad, '#fa0');
  for (const ch of sk.chains) { poly(ch.pts, '#f44', false); poly(ch.line, '#4af', false); }
  for (const p of Object.values(model.g)) { const q = P(p); ctx.fillStyle = '#fa0'; ctx.beginPath(); ctx.arc(q[0], q[1], 4, 0, 7); ctx.fill(); }
  return c;
}

async function main() {
  const { garments } = await api.listGarments();
  const bg = document.createElement('canvas'); bg.width = 1280; bg.height = 720;
  const bctx = bg.getContext('2d'); bctx.fillStyle = '#9a9a9a'; bctx.fillRect(0, 0, 1280, 720);
  const cv = document.createElement('canvas');
  const r = new Renderer(cv);
  for (const g of garments) {
    const row = document.createElement('div'); row.className = 'row';
    out.appendChild(row);
    const lbl = document.createElement('div'); lbl.className = 'lbl'; row.appendChild(lbl);
    try {
      const img = await api.loadImage(g.cutout_url);
      const model = buildFlatModel(img, g.category, g.anchors);
      const rig = new Rig(model, calib);
      const sl = model.sleeves ? Object.entries(model.sleeves).map(([k, v]) => `${k}:${v ? (v.long ? '長' : '短') + Math.round(v.length) : '無'}`).join(' ') : '';
      lbl.textContent = `${g.id} · ${g.category} · ${model.kind || ''} ${sl}`;
      row.appendChild(drawAnalysis(model, rig));
      for (const [name, pts] of Object.entries(POSES)) {
        r.setGarment(rig);
        const sk = skeletonFromLandmarks(lmFrom(pts), calib);
        const pos = rig.update(sk, { size: 1, offsetY: 0, length: 1 });
        r.render(bg, null, pos, { shade: 0, illum: 0, gain: 1, wb: [1, 1, 1], ao: 0, meanClothes: 0.3, opacity: 1, arms: null, armR: 0, showGarment: true });
        const tall = g.category !== 'upper';
        const c2 = document.createElement('canvas'); c2.width = 420; c2.height = tall ? 630 : 420;
        const x2 = c2.getContext('2d');
        x2.drawImage(cv, 640 - 300, 0, 600, tall ? 900 : 600, 0, 0, 420, tall ? 630 : 420);
        x2.strokeStyle = '#f44'; x2.lineWidth = 2;
        const T = (p) => [(p[0] - 340) * 0.7, p[1] * 0.7];
        for (const [a, b] of [[sk.SR, sk.SL], [sk.SR, sk.ER], [sk.ER, sk.WR], [sk.SL, sk.EL], [sk.EL, sk.WL], [sk.SR, sk.HR], [sk.SL, sk.HL], [sk.HR, sk.HL], [sk.HR, sk.KR], [sk.KR, sk.AR], [sk.HL, sk.KL], [sk.KL, sk.AL]]) {
          x2.beginPath(); x2.moveTo(...T(a)); x2.lineTo(...T(b)); x2.stroke();
        }
        x2.fillStyle = '#fff'; x2.fillText(name, 8, 16);
        row.appendChild(c2);
      }
    } catch (e) { lbl.textContent = `${g.id}: ${e.message}`; console.error(e); }
  }
  document.body.dataset.done = '1';
}
main();
