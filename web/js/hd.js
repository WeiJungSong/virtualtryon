// HD capture: freeze a frame, build the inpainting mask, run CatVTON on the local
// server, then (optionally) turn the result into a "worn" garment for live preview.
import * as api from './api.js';
import { buildAgnosticMask, canvasToBlob } from './masks.js';
import { buildWornModel, LM } from './garment.js';

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

export class HDController {
  constructor(app) {
    this.app = app;
    this.busy = false;
  }

  async capture() {
    const app = this.app, s = app.state;
    if (this.busy) return;
    if (!s.current) { app.toast('先在衣櫃選一件衣服', true); return; }
    this.busy = true;
    app.ui.btnHD.disabled = true;
    try {
      if (app.ui.hdCountdown.checked) {
        for (let i = 3; i > 0; i--) { app.ui.countdown.textContent = i; await sleep(900); }
        app.ui.countdown.textContent = '';
      }
      const r = s.last;
      const lm = r && r.found ? r.lm : null;
      if (!lm || Math.min(lm[LM.LS].v, lm[LM.RS].v) < 0.6) throw new Error('沒有偵測到完整的肩膀，請退後一點、面向鏡頭');
      // the frame the landmarks were computed on
      const src = app.frameCanvas, W = src.width, H = src.height;
      const frame = document.createElement('canvas'); frame.width = W; frame.height = H;
      frame.getContext('2d').drawImage(src, 0, 0, W, H);
      app.ui.flash.classList.add('on'); setTimeout(() => app.ui.flash.classList.remove('on'), 60);
      const lmCopy = lm.map(p => ({ ...p }));
      const seg = r.seg ? { ...r.seg, data: new Uint8ClampedArray(r.seg.data) } : null;
      const category = s.current.category;
      const { mask, crop } = buildAgnosticMask(W, H, lmCopy, seg, category, app.calib);
      const fd = new FormData();
      fd.append('frame', await canvasToBlob(frame, 'image/jpeg', 0.95), 'frame.jpg');
      fd.append('mask', await canvasToBlob(mask), 'mask.png');
      fd.append('garment_id', s.current.id);
      fd.append('category', category);
      fd.append('preset', app.ui.hdPreset.value);
      fd.append('seed', String(Math.floor(Math.random() * 1e6)));
      fd.append('crop', JSON.stringify(crop));
      fd.append('garment_source', s.model?.mode === 'worn' && s.model?.fromPhoto ? 'original' : 'cutout');
      this._progress(0, '上傳中…');
      let job = await api.submitHD(fd);
      const t0 = performance.now();
      while (!['done', 'error'].includes(job.status)) {
        await sleep(700);
        job = await api.getJob(job.id);
        const eng = job.engine || {};
        let msg = job.message;
        if (job.status === 'loading' && eng.state === 'downloading') msg = `下載模型 ${eng.download_mb} / ${eng.expected_mb} MB（只需一次）`;
        else if (job.status === 'loading') msg = eng.message || '載入模型…';
        else if (job.status === 'running') msg = `生成中 ${job.progress}/${job.total} 步 · ${((performance.now() - t0) / 1000).toFixed(0)}s`;
        else if (job.status === 'queued') msg = `排隊中（前面 ${job.queue} 個）`;
        const frac = job.status === 'running' ? job.progress / Math.max(1, job.total) : job.status === 'loading' && eng.expected_mb ? 0.02 + 0.2 * Math.min(1, eng.download_mb / eng.expected_mb) : 0.02;
        this._progress(frac, msg);
        app.refreshHDStatus(eng);
      }
      if (job.status === 'error') throw new Error(job.message);
      this._progress(1, `完成，用時 ${job.seconds}s`);
      app.addResult({ result_url: job.result_url, crop_url: job.crop_url, id: job.id }, true);
      app.openResult(job.result_url, frame.toDataURL('image/jpeg', 0.9));
      if (app.ui.hdLive.checked) await this.useAsLiveGarment(job.result_url, lmCopy, mask, category);
    } catch (e) {
      console.error(e);
      this._progress(0, `失敗：${e.message}`);
      app.toast(e.message, true);
    } finally {
      this.busy = false;
      app.ui.btnHD.disabled = false;
      app.ui.countdown.textContent = '';
    }
  }

  /** Cut the generated garment out of the HD result and drive it with the live skeleton. */
  async useAsLiveGarment(url, capturedLm, mask, category) {
    const app = this.app;
    try {
      const img = await api.loadImage(url);
      const an = await app.tracker.analyzeImage(img);
      if (!an.seg) throw new Error('segmenter unavailable');
      const good = an.lm && Math.min(an.lm[LM.LS].v, an.lm[LM.RS].v) > 0.7;
      const restrict = document.createElement('canvas'); restrict.width = mask.width; restrict.height = mask.height;
      const rctx = restrict.getContext('2d');
      rctx.filter = 'blur(4px)';
      rctx.drawImage(mask, 0, 0);
      // mask is white-on-black; convert to alpha
      const d = rctx.getImageData(0, 0, restrict.width, restrict.height);
      for (let i = 0; i < d.data.length; i += 4) { d.data[i + 3] = d.data[i]; d.data[i] = d.data[i + 1] = d.data[i + 2] = 255; }
      rctx.putImageData(d, 0, 0);
      const model = buildWornModel(img, { ...an, lm: good ? an.lm : capturedLm }, category, app.calib, restrict);
      model.fromHD = true;
      app.setHDModel(model);
    } catch (e) {
      console.warn('HD live garment failed', e);
      app.toast('HD 結果無法用於即時預覽（仍可在結果中查看）');
    }
  }

  _progress(frac, msg) {
    const ui = this.app.ui;
    ui.hdProg.hidden = false;
    ui.hdBar.style.width = `${Math.round(frac * 100)}%`;
    ui.hdMsg.textContent = msg;
  }
}
