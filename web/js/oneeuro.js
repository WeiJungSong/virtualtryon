// One Euro filter (Casiez et al. 2012): low jitter when still, low lag when moving.
class LowPass {
  constructor() { this.y = null; }
  filter(x, a) { this.y = this.y === null ? x : a * x + (1 - a) * this.y; return this.y; }
}

const alpha = (cutoff, dt) => {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
};

export class OneEuro {
  constructor(minCutoff = 1.2, beta = 8, dCutoff = 1.0) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.x = new LowPass(); this.dx = new LowPass(); this.t = null;
  }
  reset() { this.x = new LowPass(); this.dx = new LowPass(); this.t = null; }
  filter(value, tSec) {
    if (this.t === null) { this.t = tSec; this.dx.filter(0, 1); return this.x.filter(value, 1); }
    const dt = Math.max(1e-3, tSec - this.t);
    this.t = tSec;
    const prev = this.x.y;
    const d = (value - prev) / dt;
    const ed = this.dx.filter(d, alpha(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(ed);
    return this.x.filter(value, alpha(cutoff, dt));
  }
}
