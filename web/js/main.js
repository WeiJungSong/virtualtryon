import * as api from './api.js';
import { Tracker } from './tracker.js';
import { Renderer } from './renderer.js';
import { Rig, BodyCalib } from './rig.js';
import { buildGarment, buildFlatModel, skeletonFromLandmarks, flipModel, LM } from './garment.js';
import { HDController } from './hd.js';
import { AnchorEditor } from './editor.js';
import { clamp, smoothstep } from './geom.js';

const $ = (id) => document.getElementById(id);
const ui = {
  video: $('video'), view: $('view'), overlay: $('overlay'), hint: $('hint'), countdown: $('countdown'), flash: $('flash'),
  fps: $('fps'), perf: $('perf'), hdState: $('hdState'), pillCam: $('pillCam'), pillTrack: $('pillTrack'), pillHD: $('pillHD'),
  drop: $('drop'), file: $('file'), urlForm: $('urlForm'), urlInput: $('urlInput'), catCtl: $('catCtl'),
  candidates: $('candidates'), candGrid: $('candGrid'), candTitle: $('candTitle'), candClose: $('candClose'),
  addBusy: $('addBusy'), addBusyText: $('addBusyText'), wardrobe: $('wardrobe'), wardCount: $('wardCount'),
  btnHD: $('btnHD'), hdPreset: $('hdPreset'), hdCountdown: $('hdCountdown'), hdLive: $('hdLive'),
  hdProg: $('hdProg'), hdBar: $('hdBar'), hdMsg: $('hdMsg'), btnPreload: $('btnPreload'), hdInfo: $('hdInfo'), results: $('results'),
  modal: $('modal'), modalImg: $('modalImg'), modalDownload: $('modalDownload'), modalClose: $('modalClose'), modalCompare: $('modalCompare'),
  toast: $('toast'), btnMirror: $('btnMirror'), btnCompare: $('btnCompare'), btnDebug: $('btnDebug'), btnSnap: $('btnSnap'), btnReadable: $('btnReadable'),
  anchorCanvas: $('anchorCanvas'), anchorReset: $('anchorReset'), anchorMode: $('anchorMode'),
};

const DEFAULTS = { sSize: 1, sOffset: 0, sLength: 1, sShade: 0.75, sIllum: 0.55, sBright: 1, sWB: 0.6, sAO: 0.35 };
const state = {
  category: 'upper', garments: [], current: null, model: null, rig: null, hdModel: null, hdRig: null,
  mirror: true, readable: true, debug: false, showGarment: true, opacity: 0, last: null,
  sliders: { ...DEFAULTS },
};
const store = {
  get(k, d) { try { const v = localStorage.getItem('vton.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('vton.' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

const app = {
  ui, state, calib: new BodyCalib(), tracker: new Tracker(), renderer: null, video: ui.video,
  toast(msg, err = false) {
    ui.toast.textContent = msg; ui.toast.classList.toggle('err', err); ui.toast.classList.add('on');
    clearTimeout(this._tt); this._tt = setTimeout(() => ui.toast.classList.remove('on'), err ? 5000 : 2600);
  },
  setHDModel(model) {
    state.hdModel = model;
    state.hdRig = makeRig(model);
    activateRig();
    app.toast('即時預覽已改用 HD 生成的衣服質感');
  },
  refreshHDStatus(eng) { renderHDStatus(eng); },
  addResult(r, prepend) { addResultTile(r, prepend); },
  openResult(url, compareUrl) { openModal(url, compareUrl); },
};
window.vtonApp = app; // handy for debugging in the console

// --------------------------------------------------------------------------- //
// Camera + loop
// --------------------------------------------------------------------------- //
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user', frameRate: { ideal: 30 } }, audio: false,
    });
    ui.video.srcObject = stream;
    await ui.video.play();
    setPill(ui.pillCam, 'ok');
    return true;
  } catch (e) {
    setPill(ui.pillCam, 'err');
    ui.hint.textContent = `無法開啟鏡頭：${e.message}（請允許瀏覽器使用相機）`;
    return false;
  }
}

function setPill(el, cls) { el.classList.remove('ok', 'warn', 'err'); if (cls) el.classList.add(cls); }

// In the mirrored preview, flip the garment so prints and logos read the right way round.
function makeRig(model) {
  return new Rig(state.mirror && state.readable ? flipModel(model) : model, app.calib);
}
function rebuildRigs() {
  if (state.model) state.rig = makeRig(state.model);
  if (state.hdModel) state.hdRig = makeRig(state.hdModel);
  activateRig();
}

function activeRig() { return (ui.hdLive.checked && state.hdRig) || state.rig; }
let renderedRig = null;
function activateRig() {
  if (!app.renderer) return;
  const r = activeRig();
  if (r !== renderedRig) { app.renderer.setGarment(r); renderedRig = r; }
}

let lastPrepare = 0, frames = 0, fpsT = performance.now();
// Every frame is copied once so tracking, rendering and HD capture all use the exact same pixels.
const frameCanvas = document.createElement('canvas');
const frameCtx = frameCanvas.getContext('2d', { alpha: false });
app.frameCanvas = frameCanvas;
function onFrame(now) {
  const video = ui.video;
  if (video.readyState >= 2 && app.tracker.ready && video.videoWidth) {
    if (frameCanvas.width !== video.videoWidth || frameCanvas.height !== video.videoHeight) {
      frameCanvas.width = video.videoWidth; frameCanvas.height = video.videoHeight;
    }
    frameCtx.drawImage(video, 0, 0);
    const v = frameCanvas;
    const r = app.tracker.process(v, now);
    if (r) {
      state.last = r;
      if (r.found && app.calib.update(r.lm, r.world)) {
        const t = performance.now();
        if (t - lastPrepare > 1000) {
          lastPrepare = t;
          for (const rig of [state.rig, state.hdRig]) if (rig) rig.prepare(app.calib);
        }
      }
      const rig = activeRig();
      if (rig !== renderedRig) activateRig();
      let positions = null, arms = null, armR = 0, sk = null;
      if (r.found) {
        sk = skeletonFromLandmarks(r.lm, app.calib);
        const vis = Math.min(r.lm[LM.LS].v, r.lm[LM.RS].v);
        const target = smoothstep(0.3, 0.65, vis);
        state.opacity += (target - state.opacity) * 0.25;
        if (rig) positions = rig.update(sk, fitParams());
        const P = (i) => [r.lm[i].x, r.lm[i].y];
        arms = [sk.SR, sk.ER, sk.WR, P(20), sk.SL, sk.EL, sk.WL, P(19)];
        armR = 0.13 * sk.sw;
        updateArmFront(r.world);
      } else state.opacity *= 0.8;
      const s = state.sliders, st = r.stats;
      const gain = clamp(0.55 + 0.45 * st.faceLum / 0.30, 0.6, 1.15) * s.sBright;
      const wb = st.wb.map(x => 1 + (x - 1) * s.sWB);
      app.renderer.render(v, r.seg, positions, {
        shade: s.sShade, illum: s.sIllum, gain, wb, ao: s.sAO, meanClothes: st.clothesLum,
        opacity: state.opacity, arms, armR, armFront: state.armFront, showGarment: state.showGarment,
        shadeLine: sk ? shadeLine(sk) : null,
      });
      drawOverlay(r, sk);
      updateHint(r);
      frames++;
      if (now - fpsT > 1000) {
        ui.fps.textContent = `${Math.round(frames * 1000 / (now - fpsT))} fps`;
        const tk = app.tracker;
        ui.perf.textContent = `姿勢 ${tk.timings.pose.toFixed(1)}ms (${tk.pose?._delegate || ''}) · 分割 ${tk.timings.seg.toFixed(1)}ms (${tk.segDelegate}${tk.segEvery > 1 ? `，每 ${tk.segEvery} 格` : ''})`;
        frames = 0; fpsT = now;
        setPill(ui.pillTrack, r.found ? 'ok' : 'warn');
      }
    }
  }
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(() => onFrame(performance.now()));
  else requestAnimationFrame(onFrame);
}

// Arms only hide the garment when they are in front of the body (crossed arms, hand on
// the chest). Hanging arms are inside the sleeves / beside the body, so the garment wins.
state.armFront = [0, 0, 0, 0, 0, 0];
function updateArmFront(world) {
  if (!world) return;
  const tz = (world[11].z + world[12].z + world[23].z + world[24].z) / 4;
  const segs = [[12, 14], [14, 16], [16, 20], [11, 13], [13, 15], [15, 19]];
  segs.forEach(([a, b], i) => {
    const front = smoothstep(0.07, 0.17, tz - (world[a].z + world[b].z) / 2);
    state.armFront[i] += (front - state.armFront[i]) * 0.3;
  });
}

function shadeLine(sk) {
  const cat = state.current?.category;
  if (!cat || cat === 'dress') return null;
  const hm = [(sk.HR[0] + sk.HL[0]) / 2, (sk.HR[1] + sk.HL[1]) / 2];
  const sm = [(sk.SR[0] + sk.SL[0]) / 2, (sk.SR[1] + sk.SL[1]) / 2];
  const tl = Math.hypot(hm[0] - sm[0], hm[1] - sm[1]);
  const d = sk.down;
  if (cat === 'upper') return { p: [hm[0] + d[0] * 0.02 * tl, hm[1] + d[1] * 0.02 * tl], n: d, w: 0.12 * tl };
  return { p: [hm[0] - d[0] * 0.3 * tl, hm[1] - d[1] * 0.3 * tl], n: [-d[0], -d[1]], w: 0.12 * tl };
}

function fitParams() {
  const s = state.sliders;
  return { size: s.sSize, offsetY: s.sOffset, length: s.sLength };
}

function updateHint(r) {
  let msg = '';
  if (!r.found) msg = '找不到人，請站到鏡頭前（上半身入鏡）';
  else if (!state.current) msg = '加入或選擇一件衣服開始試穿';
  else if (Math.min(r.lm[LM.LS].v, r.lm[LM.RS].v) < 0.5) msg = '請讓雙肩都入鏡';
  else if (state.current.category === 'lower' && Math.min(r.lm[LM.LK].v, r.lm[LM.RK].v) < 0.5) msg = '下身衣物請退後，讓膝蓋入鏡';
  if (ui.hint.textContent !== msg) ui.hint.textContent = msg;
}

function drawOverlay(r, sk) {
  const c = ui.overlay;
  if (c.width !== ui.view.width || c.height !== ui.view.height) { c.width = ui.view.width; c.height = ui.view.height; }
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (!state.debug || !r.found) return;
  ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(120,220,255,.9)'; ctx.fillStyle = '#fff';
  const seg = (a, b) => { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); };
  seg(sk.SR, sk.SL); seg(sk.SR, sk.HR); seg(sk.SL, sk.HL); seg(sk.HR, sk.HL);
  seg(sk.SR, sk.ER); seg(sk.ER, sk.WR); seg(sk.SL, sk.EL); seg(sk.EL, sk.WL);
  seg(sk.HR, sk.KR); seg(sk.KR, sk.AR); seg(sk.HL, sk.KL); seg(sk.KL, sk.AL);
  for (const p of r.lm) { ctx.globalAlpha = Math.max(0.15, p.v); ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 7); ctx.fill(); }
  ctx.globalAlpha = 1;
  const rig = activeRig();
  if (rig?.lastBodyQuad) {
    ctx.strokeStyle = 'rgba(232,178,122,.9)'; ctx.setLineDash([6, 4]);
    const q = rig.lastBodyQuad; ctx.beginPath(); q.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.closePath(); ctx.stroke();
    ctx.setLineDash([]);
  }
}

// --------------------------------------------------------------------------- //
// Garments
// --------------------------------------------------------------------------- //
async function refreshWardrobe(selectId) {
  const { garments } = await api.listGarments();
  state.garments = garments;
  ui.wardCount.textContent = garments.length ? `${garments.length} 件` : '';
  ui.wardrobe.innerHTML = '';
  if (!garments.length) { ui.wardrobe.innerHTML = '<p class="empty">還沒有衣服，先加入一件吧</p>'; return; }
  const catName = { upper: '上衣', dress: '洋裝', lower: '下身' };
  for (const g of garments) {
    const t = document.createElement('div');
    t.className = 'tile' + (state.current?.id === g.id ? ' sel' : '');
    t.dataset.id = g.id;
    t.innerHTML = `<img src="${g.cutout_url}" alt=""><span class="tag">${catName[g.category] || ''}</span><button class="x" title="移到垃圾桶">✕</button>`;
    t.addEventListener('click', (e) => {
      if (e.target.classList.contains('x')) return;
      selectGarment(g);
    });
    t.querySelector('.x').addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm('把這件衣服移到垃圾桶（data/trash）？')) return;
      await api.deleteGarment(g.id);
      if (state.current?.id === g.id) clearGarment();
      refreshWardrobe();
    });
    ui.wardrobe.appendChild(t);
  }
  if (selectId) { const g = garments.find(x => x.id === selectId); if (g) await selectGarment(g); }
}

function clearGarment() {
  state.current = null; state.model = null; state.rig = null; state.hdRig = null; state.hdModel = null;
  activateRig(); editor.setModel(null);
}

async function selectGarment(meta) {
  state.current = meta;
  store.set('lastGarment', meta.id);
  document.querySelectorAll('#wardrobe .tile').forEach(t => t.classList.toggle('sel', t.dataset.id === meta.id));
  setCategory(meta.category, false);
  try {
    busy(true, '分析衣服形狀…');
    const model = await buildGarment(meta, app.tracker, app.calib);
    if (state.current?.id !== meta.id) return; // user clicked another one meanwhile
    state.model = model;
    state.rig = makeRig(model);
    state.hdRig = null; state.hdModel = null;
    activateRig();
    editor.setModel(model);
    ui.anchorMode.textContent = model.mode === 'worn' ? '模特兒照模式' : (meta.anchors ? '已手動校正' : '自動偵測');
  } catch (e) {
    console.error(e);
    app.toast(`衣服分析失敗：${e.message}`, true);
  } finally { busy(false); }
}

function busy(on, text) { ui.addBusy.hidden = !on; if (text) ui.addBusyText.textContent = text; }

async function addFromFile(file) {
  if (!file || !file.type.startsWith('image/')) { app.toast('請提供圖片檔', true); return; }
  busy(true, '去背中…');
  try {
    const g = await api.uploadGarment({ file, category: state.category });
    await refreshWardrobe(g.id);
    app.toast('已加入衣櫃');
  } catch (e) { app.toast(e.message, true); } finally { busy(false); }
}

async function addFromUrl(url) {
  url = url.trim();
  if (!url) return;
  if (url.startsWith('data:image')) {
    const blob = await (await fetch(url)).blob();
    return addFromFile(new File([blob], 'pasted.png', { type: blob.type }));
  }
  busy(true, '讀取網址…');
  try {
    const r = await api.fetchUrl(url);
    if (r.kind === 'image') {
      busy(true, '下載並去背中…');
      const g = await api.uploadGarment({ url: r.url, category: state.category });
      await refreshWardrobe(g.id);
      app.toast('已加入衣櫃');
    } else showCandidates(r);
  } catch (e) { app.toast(e.message, true); } finally { busy(false); }
}

function showCandidates(r) {
  ui.candidates.hidden = false;
  ui.candTitle.textContent = r.title ? `「${r.title.slice(0, 24)}」選一張商品圖` : '選一張商品圖';
  ui.candGrid.innerHTML = '';
  for (const u of r.images) {
    const t = document.createElement('div'); t.className = 'tile';
    const img = document.createElement('img'); img.loading = 'lazy'; img.src = api.proxyUrl(u, r.page);
    img.onerror = () => t.remove();
    t.appendChild(img);
    t.addEventListener('click', async () => {
      ui.candidates.hidden = true;
      busy(true, '下載並去背中…');
      try {
        const g = await api.uploadGarment({ url: u, category: state.category });
        await refreshWardrobe(g.id);
        app.toast('已加入衣櫃');
      } catch (e) { app.toast(e.message, true); } finally { busy(false); }
    });
    ui.candGrid.appendChild(t);
  }
}

function setCategory(cat, patchCurrent) {
  state.category = cat;
  ui.catCtl.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.cat === cat));
  if (patchCurrent && state.current && state.current.category !== cat) {
    api.patchGarment(state.current.id, { category: cat, clear_anchors: true }).then((meta) => {
      const i = state.garments.findIndex(g => g.id === meta.id);
      if (i >= 0) state.garments[i] = meta;
      refreshWardrobe().then(() => selectGarment(meta));
    }).catch(e => app.toast(e.message, true));
  }
}

function extractDroppedUrl(dt) {
  const html = dt.getData('text/html');
  if (html) {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    const img = doc.querySelector('img');
    if (img) {
      const srcset = img.getAttribute('srcset');
      if (srcset) {
        const best = srcset.split(',').map(s => s.trim().split(/\s+/)).sort((a, b) => parseFloat(b[1] || 0) - parseFloat(a[1] || 0))[0];
        if (best?.[0]?.startsWith('http')) return best[0];
      }
      const src = img.getAttribute('src');
      if (src && (src.startsWith('http') || src.startsWith('data:image'))) return src;
    }
  }
  const uri = (dt.getData('text/uri-list') || '').split('\n').find(l => l && !l.startsWith('#'));
  if (uri) return uri.trim();
  const text = dt.getData('text/plain');
  if (text && /^https?:\/\//.test(text.trim())) return text.trim();
  return null;
}

// --------------------------------------------------------------------------- //
// HD status + results
// --------------------------------------------------------------------------- //
function renderHDStatus(eng) {
  if (!eng) return;
  const map = { not_loaded: ['未載入', ''], downloading: ['下載中', 'warn'], loading: ['載入中', 'warn'], ready: ['就緒', 'ok'], error: ['錯誤', 'err'] };
  const [txt, cls] = map[eng.state] || [eng.state, ''];
  ui.hdState.textContent = `${txt} · ${eng.device}`;
  setPill(ui.pillHD, cls);
  ui.hdInfo.textContent = eng.state === 'error' ? eng.error : (eng.state === 'downloading' ? `${eng.download_mb}/${eng.expected_mb} MB` : eng.message || '');
}

async function pollStatus() {
  try { const s = await api.getStatus(); renderHDStatus(s.hd); return s; } catch { setPill(ui.pillHD, 'err'); ui.hdState.textContent = '伺服器離線'; }
}

function addResultTile(r, prepend = false) {
  const t = document.createElement('div'); t.className = 'tile';
  t.innerHTML = `<img src="${r.crop_url || r.result_url}" alt="">`;
  t.addEventListener('click', () => openModal(r.result_url));
  if (prepend) ui.results.prepend(t); else ui.results.appendChild(t);
}

let modalCompare = null;
function openModal(url, compareUrl = null) {
  ui.modalImg.src = url; ui.modalImg.dataset.result = url;
  ui.modalDownload.href = url;
  ui.modalDownload.download = url.split('/').pop();
  modalCompare = compareUrl;
  ui.modalCompare.hidden = !compareUrl;
  ui.modal.hidden = false;
}

// --------------------------------------------------------------------------- //
// UI wiring
// --------------------------------------------------------------------------- //
const editor = new AnchorEditor(ui.anchorCanvas, async (anchors) => {
  if (!state.current || state.model?.mode !== 'flat') return;
  try {
    const meta = await api.patchGarment(state.current.id, { anchors });
    state.current = meta;
    const cut = await api.loadImage(meta.cutout_url);
    const model = buildFlatModel(cut, meta.category, anchors);
    model.meta = meta;
    state.model = model; state.rig = makeRig(model); state.hdRig = null; state.hdModel = null;
    activateRig(); editor.setModel(model);
    ui.anchorMode.textContent = '已手動校正';
  } catch (e) { app.toast(e.message, true); }
});

function wire() {
  ui.catCtl.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setCategory(b.dataset.cat, true); });
  ui.file.addEventListener('change', () => { if (ui.file.files[0]) addFromFile(ui.file.files[0]); ui.file.value = ''; });
  ['dragenter', 'dragover'].forEach(ev => document.addEventListener(ev, (e) => { e.preventDefault(); ui.drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => document.addEventListener(ev, (e) => { if (ev === 'dragleave' && e.relatedTarget) return; ui.drop.classList.remove('over'); }));
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    const dt = e.dataTransfer;
    const url = extractDroppedUrl(dt);
    if (url) addFromUrl(url);
    else if (dt.files?.length) addFromFile(dt.files[0]);
    else app.toast('沒有讀到圖片，試試右鍵「複製圖片」再貼上', true);
  });
  document.addEventListener('paste', (e) => {
    if (e.target === ui.urlInput) return;
    const items = [...(e.clipboardData?.items || [])];
    const img = items.find(i => i.type.startsWith('image/'));
    if (img) { e.preventDefault(); addFromFile(img.getAsFile()); return; }
    const text = e.clipboardData?.getData('text/plain');
    if (text && /^https?:\/\//.test(text.trim())) { e.preventDefault(); addFromUrl(text); }
  });
  ui.urlForm.addEventListener('submit', (e) => { e.preventDefault(); addFromUrl(ui.urlInput.value); ui.urlInput.value = ''; });
  ui.candClose.addEventListener('click', () => { ui.candidates.hidden = true; });

  for (const id of Object.keys(DEFAULTS)) {
    const el = $(id), out = el.nextElementSibling;
    el.value = state.sliders[id];
    const show = () => { out.textContent = Number(el.value).toFixed(2); };
    show();
    el.addEventListener('input', () => { state.sliders[id] = Number(el.value); show(); store.set('sliders', state.sliders); });
  }
  $('resetSliders').addEventListener('click', () => {
    state.sliders = { ...DEFAULTS }; store.set('sliders', state.sliders);
    for (const id of Object.keys(DEFAULTS)) { $(id).value = DEFAULTS[id]; $(id).nextElementSibling.textContent = DEFAULTS[id].toFixed(2); }
  });

  const setMirror = (on) => {
    state.mirror = on; store.set('mirror', on);
    ui.view.classList.toggle('mirror', on); ui.overlay.classList.toggle('mirror', on);
    ui.btnMirror.classList.toggle('on', on);
    ui.btnReadable.disabled = !on;
    rebuildRigs();
  };
  const setReadable = (on) => {
    state.readable = on; store.set('readable', on);
    ui.btnReadable.classList.toggle('on', on);
    rebuildRigs();
  };
  state.readable = store.get('readable', true);
  ui.btnReadable.classList.toggle('on', state.readable);
  setMirror(store.get('mirror', true));
  ui.btnMirror.addEventListener('click', () => setMirror(!state.mirror));
  ui.btnReadable.addEventListener('click', () => setReadable(!state.readable));
  const setDebug = (on) => { state.debug = on; ui.btnDebug.classList.toggle('on', on); };
  ui.btnDebug.addEventListener('click', () => setDebug(!state.debug));
  const hold = (on) => { state.showGarment = !on; ui.btnCompare.classList.toggle('on', on); };
  ui.btnCompare.addEventListener('pointerdown', () => hold(true));
  ['pointerup', 'pointerleave'].forEach(ev => ui.btnCompare.addEventListener(ev, () => hold(false)));
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); hold(true); }
    if (e.key === 'h' || e.key === 'H') hd.capture();
    if (e.key === 'd' || e.key === 'D') setDebug(!state.debug);
    if (e.key === 'm' || e.key === 'M') setMirror(!state.mirror);
    if (e.key === 'Escape') ui.modal.hidden = true;
  });
  document.addEventListener('keyup', (e) => { if (e.code === 'Space') hold(false); });
  ui.btnSnap.addEventListener('click', () => {
    const c = document.createElement('canvas'); c.width = ui.view.width; c.height = ui.view.height;
    const ctx = c.getContext('2d');
    if (state.mirror) { ctx.translate(c.width, 0); ctx.scale(-1, 1); }
    ctx.drawImage(ui.view, 0, 0);
    const a = document.createElement('a'); a.href = c.toDataURL('image/jpeg', 0.92);
    a.download = `tryon-${Date.now()}.jpg`; a.click();
  });

  ui.btnHD.addEventListener('click', () => hd.capture());
  ui.hdPreset.value = store.get('preset', 'fast');
  ui.hdPreset.addEventListener('change', () => store.set('preset', ui.hdPreset.value));
  ui.hdLive.addEventListener('change', () => activateRig());
  ui.btnPreload.addEventListener('click', async () => { try { renderHDStatus(await api.loadHD()); app.toast('開始在背景載入 HD 模型'); } catch (e) { app.toast(e.message, true); } });
  ui.modalClose.addEventListener('click', () => { ui.modal.hidden = true; });
  ui.modal.addEventListener('click', (e) => { if (e.target === ui.modal) ui.modal.hidden = true; });
  ui.modalCompare.addEventListener('pointerdown', () => { if (modalCompare) ui.modalImg.src = modalCompare; });
  ['pointerup', 'pointerleave'].forEach(ev => ui.modalCompare.addEventListener(ev, () => { ui.modalImg.src = ui.modalImg.dataset.result; }));
  ui.anchorReset.addEventListener('click', async () => {
    if (!state.current) return;
    const meta = await api.patchGarment(state.current.id, { clear_anchors: true });
    await selectGarment(meta);
  });
}

const hd = new HDController(app);

// --------------------------------------------------------------------------- //
async function main() {
  state.sliders = { ...DEFAULTS, ...store.get('sliders', {}) };
  wire();
  try { app.renderer = new Renderer(ui.view); } catch (e) { ui.hint.textContent = e.message; return; }
  const status = await pollStatus();
  setInterval(pollStatus, 5000);
  const camOk = await startCamera();
  try {
    const info = await app.tracker.init((m) => { ui.hint.textContent = m; });
    console.log('tracker', info);
    if (!info.multiclass) app.toast('多類別分割模型載入失敗，光影與遮擋效果會降低', true);
  } catch (e) {
    ui.hint.textContent = `MediaPipe 載入失敗：${e.message}`;
    setPill(ui.pillTrack, 'err');
    return;
  }
  ui.hint.textContent = '';
  try {
    await refreshWardrobe();
    const last = store.get('lastGarment', null);
    const g = state.garments.find(x => x.id === last) || state.garments[0];
    if (g) selectGarment(g);
    const { results } = await api.listResults();
    results.forEach(r => addResultTile(r));
  } catch (e) { app.toast(`伺服器連線失敗：${e.message}`, true); }
  if (camOk) onFrame(performance.now());
  if (status && !status.rembg) app.toast('提示：rembg 未安裝，去背品質會降低');
}
main();
