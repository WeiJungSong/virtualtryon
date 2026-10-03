// Draggable anchor editor for flat garments.
import { editableAnchors } from './garment.js';

const LABEL = { SR: '肩', SL: '肩', ER: '袖', EL: '袖', HR: '髖', HL: '髖' };

export class AnchorEditor {
  constructor(canvas, onCommit) {
    this.c = canvas; this.ctx = canvas.getContext('2d');
    this.onCommit = onCommit;
    this.model = null; this.anchors = {}; this.drag = null;
    canvas.addEventListener('pointerdown', (e) => this._down(e));
    canvas.addEventListener('pointermove', (e) => this._move(e));
    canvas.addEventListener('pointerup', (e) => this._up(e));
    canvas.addEventListener('pointercancel', (e) => this._up(e));
  }
  setModel(model) {
    this.model = model;
    this.anchors = model ? editableAnchors(model) : {};
    this.draw();
  }
  _layout() {
    const m = this.model, W = this.c.width, H = this.c.height;
    const s = Math.min((W - 20) / m.w, (H - 20) / m.h);
    return { s, ox: (W - m.w * s) / 2, oy: (H - m.h * s) / 2 };
  }
  _toCanvas(p) { const L = this._layout(); return [L.ox + p[0] * this.model.w * L.s, L.oy + p[1] * this.model.h * L.s]; }
  _fromEvent(e) {
    const r = this.c.getBoundingClientRect();
    const x = (e.clientX - r.left) * this.c.width / r.width, y = (e.clientY - r.top) * this.c.height / r.height;
    const L = this._layout();
    return { cx: x, cy: y, n: [(x - L.ox) / (this.model.w * L.s), (y - L.oy) / (this.model.h * L.s)] };
  }
  draw() {
    const ctx = this.ctx, W = this.c.width, H = this.c.height;
    ctx.clearRect(0, 0, W, H);
    if (!this.model) return;
    const L = this._layout();
    ctx.drawImage(this.model.tex, L.ox, L.oy, this.model.w * L.s, this.model.h * L.s);
    if (this.model.mode !== 'flat') {
      ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(0, H - 34, W, 34);
      ctx.fillStyle = '#ddd'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('模特兒照／HD 結果：位置由姿勢偵測自動決定', W / 2, H - 13);
      return;
    }
    const a = this.anchors;
    ctx.strokeStyle = 'rgba(232,178,122,.7)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    const line = (k1, k2) => { if (a[k1] && a[k2]) { const p = this._toCanvas(a[k1]), q = this._toCanvas(a[k2]); ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke(); } };
    line('SR', 'SL'); line('SR', 'ER'); line('SL', 'EL'); line('HR', 'HL');
    ctx.setLineDash([]);
    for (const [k, p] of Object.entries(a)) {
      const [x, y] = this._toCanvas(p);
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fillStyle = this.drag === k ? '#fff' : '#e8b27a'; ctx.fill();
      ctx.strokeStyle = '#1a1208'; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = '#1a1208'; ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(LABEL[k] || k, x, y + 0.5);
    }
  }
  _down(e) {
    if (!this.model || this.model.mode !== 'flat') return;
    const { cx, cy } = this._fromEvent(e);
    let best = null, bd = 18;
    for (const [k, p] of Object.entries(this.anchors)) {
      const [x, y] = this._toCanvas(p);
      const d = Math.hypot(x - cx, y - cy);
      if (d < bd) { bd = d; best = k; }
    }
    if (best) { this.drag = best; this.c.setPointerCapture(e.pointerId); this.draw(); }
  }
  _move(e) {
    if (!this.drag) return;
    const { n } = this._fromEvent(e);
    this.anchors[this.drag] = [Math.min(1.1, Math.max(-0.1, n[0])), Math.min(1.1, Math.max(-0.1, n[1]))];
    this.draw();
  }
  _up() {
    if (!this.drag) return;
    this.drag = null; this.draw();
    this.onCommit({ ...this.anchors });
  }
}
