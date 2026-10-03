// MediaPipe pose + multi-class segmentation for the live camera, plus image-mode
// helpers for analysing garment photos and HD results.
import { OneEuro } from './oneeuro.js';

const VERSION = '1.0.1';
const LOCAL = {
  bundle: '/vendor/tasks-vision/vision_bundle.mjs',
  wasm: '/vendor/tasks-vision/wasm',
  pose: '/vendor/models/pose_landmarker_full.task',
  seg: '/vendor/models/selfie_multiclass_256x256.tflite',
};
const CDN = {
  bundle: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/vision_bundle.mjs`,
  wasm: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`,
  pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
  seg: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
};

// setup.sh downloads the runtime + models into web/vendor and writes manifest.json;
// without it everything is loaded from the CDN.
async function vendorManifest() {
  try {
    const r = await fetch('/vendor/manifest.json', { cache: 'no-store' });
    return r.ok ? await r.json() : {};
  } catch { return {}; }
}

let visionPromise = null;
async function loadVision() {
  if (!visionPromise) {
    visionPromise = (async () => {
      const man = await vendorManifest();
      const local = !!man.tasks_vision;
      const vision = await import(local ? LOCAL.bundle : CDN.bundle);
      const fileset = await vision.FilesetResolver.forVisionTasks(local ? LOCAL.wasm : CDN.wasm);
      const poseUrl = man.pose ? LOCAL.pose : CDN.pose;
      const segUrl = man.seg ? LOCAL.seg : CDN.seg;
      return { vision, fileset, poseUrl, segUrl, local };
    })();
  }
  return visionPromise;
}

async function createTask(Cls, fileset, opts, delegates) {
  let lastErr;
  for (const delegate of delegates) {
    try {
      const t = await Cls.createFromOptions(fileset, { ...opts, baseOptions: { ...opts.baseOptions, delegate } });
      t._delegate = delegate;
      return t;
    } catch (e) { lastErr = e; console.warn(`${Cls.name} ${delegate} failed`, e); }
  }
  throw lastErr;
}

// selfie_multiclass categories: 0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others
export function packConfidence(masks, w, h, out, faceOut = null) {
  const n = w * h;
  const bg = masks[0].getAsFloat32Array();
  const hair = masks[1].getAsFloat32Array();
  const body = masks[2].getAsFloat32Array();
  const face = masks[3].getAsFloat32Array();
  const clothes = masks[4].getAsFloat32Array();
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[j] = clothes[i] * 255;
    out[j + 1] = body[i] * 255;
    out[j + 2] = Math.max(hair[i], face[i]) * 255;
    out[j + 3] = (1 - bg[i]) * 255;
    if (faceOut) faceOut[i] = face[i] * 255;
  }
  return out;
}

export class Tracker {
  constructor() {
    this.ready = false;
    this.segW = 320; this.segH = 180;
    this.segCanvas = document.createElement('canvas');
    this.segCtx = this.segCanvas.getContext('2d', { willReadFrequently: true });
    this.filters = Array.from({ length: 33 }, () => [new OneEuro(), new OneEuro()]);
    this.vis = new Float32Array(33);
    this.lastTs = 0;
    this.frame = 0;
    this.seg = null;
    this.segEvery = 1;
    this.stats = { faceLum: 0.35, clothesLum: 0.25, wb: [1, 1, 1], valid: false };
    this.timings = { pose: 0, seg: 0 };
    this.lostFrames = 0;
  }

  async init(onStatus = () => {}) {
    onStatus('載入 MediaPipe…');
    const { vision, fileset, poseUrl, segUrl } = await loadVision();
    this.vision = vision; this.fileset = fileset; this.segUrl = segUrl; this.poseUrl = poseUrl;
    this.multiclass = true;
    onStatus('載入姿勢模型…');
    this.pose = await this._createPose(['GPU', 'CPU']);
    onStatus('載入分割模型…');
    try {
      // GPU is usually ~10x faster than the CPU (wasm) path; its output is validated on
      // the first frames and we fall back to CPU if it comes back empty (seen on some GPUs).
      this.segmenter = await this._createSegmenter(['GPU', 'CPU']);
      this.segValidated = this.segmenter._delegate === 'CPU';
      this.segEmpty = 0;
    } catch (e) {
      console.warn('multiclass segmenter unavailable, falling back to pose mask', e);
      this.multiclass = false;
      this.pose.close();
      this.pose = await this._createPose(['GPU', 'CPU']);
    }
    onStatus('模型暖機…');
    this._warmup();
    // measured on an M-series Mac: pose 18 ms GPU / 20 ms CPU, segmentation 25 ms GPU /
    // 200 ms CPU; software GPUs are 10-100x slower, which is what the benchmark catches
    this.bench = {};
    this._initBench('pose', 40);
    if (this.segmenter) this._initBench('segmenter', 60);
    this.ready = true;
    return { poseDelegate: this.pose._delegate, segDelegate: this.segmenter?._delegate, multiclass: this.multiclass };
  }

  // The first GPU run compiles shaders and uploads weights (0.2-6 s); do it here, behind
  // the loading message, instead of freezing the first camera frame.
  _warmup() {
    try {
      this.setSegSize(this.segW, this.segH);
      this.segCtx.fillStyle = '#808080';
      this.segCtx.fillRect(0, 0, this.segW, this.segH);
      const r = this.pose.detectForVideo(this.segCanvas, 1);
      r.segmentationMasks?.forEach(m => m.close());
      if (this.segmenter) {
        const sr = this.segmenter.segmentForVideo(this.segCanvas, 1);
        sr.confidenceMasks?.forEach(m => m.getAsFloat32Array());
        sr.close?.();
      }
      this.lastTs = 1;
    } catch (e) { console.warn('warm-up failed', e); }
  }

  get poseDelegate() { return this.pose?._delegate || ''; }

  get segDelegate() {
    return this.segmenter?._delegate || (this.multiclass ? '' : 'pose');
  }

  _createPose(delegates) {
    return createTask(this.vision.PoseLandmarker, this.fileset, {
      baseOptions: { modelAssetPath: this.poseUrl }, runningMode: 'VIDEO', numPoses: 1,
      minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
      outputSegmentationMasks: !this.multiclass,
    }, delegates);
  }

  _createSegmenter(delegates) {
    return createTask(this.vision.ImageSegmenter, this.fileset, {
      baseOptions: { modelAssetPath: this.segUrl }, runningMode: 'VIDEO',
      outputCategoryMask: false, outputConfidenceMasks: true,
    }, delegates);
  }

  async _switchSegmenterToCpu() {
    if (this.segSwitching) return;
    this.segSwitching = true;
    if (this.bench.segmenter) this.bench.segmenter.stage = 'done';
    console.warn('GPU segmentation returned empty masks; switching to CPU');
    try {
      const cpu = await this._createSegmenter(['CPU']);
      const old = this.segmenter;
      this.segmenter = cpu;
      this.segValidated = true;
      this.timings.seg = 0;
      old.close?.();
    } finally { this.segSwitching = false; }
  }

  // The GPU delegate wins on most laptops, but on weak or software-emulated GPUs the
  // wasm CPU path is faster. If the GPU is not clearly fast, time the CPU once as well
  // (live: the GPU task keeps working while the CPU one loads) and keep the faster.
  _initBench(key, fastMs) {
    this.bench[key] = { stage: this[key]._delegate === 'GPU' ? 'gpu' : 'done', times: [], fast: fastMs };
  }

  _bench(key, ms) {
    const b = this.bench?.[key];
    if (!b || (b.stage !== 'gpu' && b.stage !== 'cpu')) return;
    if (key === 'segmenter' && !this.segValidated) return;
    const tk = key === 'pose' ? 'pose' : 'seg';
    b.times.push(ms);
    const post = b.times.slice(3);                       // skip warm-up runs
    if (post.length < 3) return;
    const med = post.slice().sort((x, y) => x - y)[post.length >> 1];
    if (!(post.length >= 15 || (b.stage === 'gpu' && med > b.fast * 4))) return;
    b.times = [];
    if (b.stage === 'gpu') {
      b.gpu = med;
      if (med < b.fast) { b.stage = 'done'; return; }
      b.stage = 'cpu-loading';
      (key === 'pose' ? this._createPose(['CPU']) : this._createSegmenter(['CPU'])).then(cpu => {
        b.spare = this[key];
        this[key] = cpu;
        this.timings[tk] = 0;
        b.stage = 'cpu';
      }).catch(e => { console.warn(`${key} CPU benchmark failed`, e); b.stage = 'done'; });
    } else {
      b.stage = 'done';
      if (b.gpu < med) {
        const cpu = this[key];
        this[key] = b.spare;
        cpu.close?.();
      } else {
        b.spare.close?.();
      }
      b.spare = null;
      this.timings[tk] = 0;
      console.info(`${key}: GPU ${b.gpu.toFixed(1)} ms vs CPU ${med.toFixed(1)} ms → ${this[key]._delegate}`);
    }
  }

  setSegSize(w, h) {
    this.segW = w; this.segH = h;
    this.segCanvas.width = w; this.segCanvas.height = h;
    this.segBuf = new Uint8ClampedArray(w * h * 4);
    this.faceBuf = new Uint8ClampedArray(w * h);
  }

  /** source: video or canvas holding the current frame (landmarks are in its pixels). */
  process(video, nowMs) {
    const W = video.videoWidth || video.width, H = video.videoHeight || video.height;
    if (!W || !H) return null;
    const sw = 320, sh = Math.round(320 * H / W);
    if (this.segW !== sw || this.segH !== sh || !this.segBuf) this.setSegSize(sw, sh);
    let ts = Math.max(nowMs, this.lastTs + 1);
    this.lastTs = ts;
    this.frame++;

    // --- pose
    const t0 = performance.now();
    const res = this.pose.detectForVideo(video, ts);
    const t1 = performance.now();
    this.timings.pose = this.timings.pose ? this.timings.pose * 0.9 + (t1 - t0) * 0.1 : t1 - t0;
    this._bench('pose', t1 - t0);
    const tSec = ts / 1000;
    let found = res.landmarks && res.landmarks.length > 0;
    let lm = null, world = null;
    if (found) {
      this.lostFrames = 0;
      const raw = res.landmarks[0];
      world = res.worldLandmarks?.[0] || null;
      lm = raw.map((p, i) => {
        const x = this.filters[i][0].filter(p.x, tSec);
        const y = this.filters[i][1].filter(p.y, tSec);
        const v = p.visibility ?? 1;
        this.vis[i] = this.vis[i] * 0.7 + v * 0.3;
        return { x: x * W, y: y * H, z: p.z, v: this.vis[i] };
      });
      if (!this.multiclass && res.segmentationMasks?.length) {
        this._packPoseMask(res.segmentationMasks[0]);
      }
    } else if (++this.lostFrames > 5) {
      this.filters.forEach(f => { f[0].reset(); f[1].reset(); });
    }
    if (res.segmentationMasks) res.segmentationMasks.forEach(m => m.close());

    // --- segmentation (downscaled frame)
    if (this.multiclass && !this.segSwitching && (this.frame % this.segEvery === 0 || !this.seg)) {
      this.segCtx.drawImage(video, 0, 0, this.segW, this.segH);
      const s0 = performance.now();
      const sres = this.segmenter.segmentForVideo(this.segCanvas, ts);
      if (sres.confidenceMasks && sres.confidenceMasks.length >= 5) {
        packConfidence(sres.confidenceMasks, this.segW, this.segH, this.segBuf, this.faceBuf);
        this.seg = { data: this.segBuf, w: this.segW, h: this.segH, version: this.frame };
      }
      sres.close?.();
      const s1 = performance.now();
      this.timings.seg = this.timings.seg ? this.timings.seg * 0.9 + (s1 - s0) * 0.1 : s1 - s0;
      this._bench('segmenter', s1 - s0);
      // keep segmentation to roughly <= 12 ms per displayed frame on average
      this.segEvery = Math.max(1, Math.min(8, Math.ceil(this.timings.seg / 12)));
      if (!this.segValidated && found && this.seg) {
        let person = 0;
        const d = this.seg.data;
        for (let i = 3; i < d.length; i += 64) person += d[i] > 128 ? 1 : 0;
        if (person > 20) this.segValidated = true;
        else if (++this.segEmpty >= 8) this._switchSegmenterToCpu();
      }
      if (this.frame % 15 === 0) this._updateStats();
    }
    return { W, H, t: tSec, found, lm, world, seg: this.seg, stats: this.stats };
  }

  _packPoseMask(mask) {
    if (!this.segBuf) return;
    // pose mask comes at full frame size; sample it down to seg size
    const f = mask.getAsFloat32Array();
    const mw = mask.width, mh = mask.height;
    const out = this.segBuf;
    for (let y = 0; y < this.segH; y++) {
      const sy = Math.min(mh - 1, Math.round(y * mh / this.segH));
      for (let x = 0; x < this.segW; x++) {
        const sx = Math.min(mw - 1, Math.round(x * mw / this.segW));
        const v = f[sy * mw + sx] * 255, j = (y * this.segW + x) * 4;
        out[j] = v; out[j + 1] = 0; out[j + 2] = 0; out[j + 3] = v;
      }
    }
    this.seg = { data: out, w: this.segW, h: this.segH, version: this.frame };
  }

  _updateStats() {
    if (!this.seg) return;
    const img = this.segCtx.getImageData(0, 0, this.segW, this.segH).data;
    const m = this.seg.data;
    let fs = 0, fn = 0, cs = 0, cn = 0, r = 0, g = 0, b = 0, n = 0;
    const lin = (c) => Math.pow(c / 255, 2.2);
    const face = this.faceBuf;
    for (let i = 0; i < img.length; i += 8) { // every other pixel
      const R = lin(img[i]), G = lin(img[i + 1]), B = lin(img[i + 2]);
      const L = 0.2126 * R + 0.7152 * G + 0.0722 * B;
      if (face && face[i >> 2] > 180) { fs += L; fn++; }
      if (m[i] > 180) { cs += L; cn++; }
      if (m[i + 3] < 60) { r += R; g += G; b += B; n++; }        // background for white balance
    }
    if (fn > 30) this.stats.faceLum = this.stats.faceLum * 0.7 + (fs / fn) * 0.3;
    if (cn > 30) this.stats.clothesLum = this.stats.clothesLum * 0.7 + (cs / cn) * 0.3;
    if (n > 100) {
      const mr = r / n, mg = g / n, mb = b / n;
      const lum = 0.2126 * mr + 0.7152 * mg + 0.0722 * mb + 1e-4;
      const wb = [mr / lum, mg / lum, mb / lum];
      this.stats.wb = this.stats.wb.map((v, i) => v * 0.7 + Math.min(1.6, Math.max(0.6, wb[i])) * 0.3);
    }
    this.stats.valid = true;
  }

  // ---- image-mode helpers (garment photos, HD results) --------------------
  async imageTasks() {
    if (!this._imageTasks) {
      this._imageTasks = (async () => {
        const { vision, fileset, poseUrl, segUrl } = await loadVision();
        const pose = await createTask(vision.PoseLandmarker, fileset, {
          baseOptions: { modelAssetPath: poseUrl }, runningMode: 'IMAGE', numPoses: 1,
          minPoseDetectionConfidence: 0.5,
        }, ['CPU']);
        let seg = null;
        try {
          seg = await createTask(vision.ImageSegmenter, fileset, {
            baseOptions: { modelAssetPath: segUrl }, runningMode: 'IMAGE', outputConfidenceMasks: true, outputCategoryMask: false,
          }, ['CPU']);
        } catch (e) { console.warn('image segmenter unavailable', e); }
        return { pose, seg };
      })();
    }
    return this._imageTasks;
  }

  /** Detect pose + segmentation on a still image. Landmarks in image pixels. */
  async analyzeImage(img) {
    const { pose, seg } = await this.imageTasks();
    const W = img.naturalWidth || img.width, H = img.naturalHeight || img.height;
    const pr = pose.detect(img);
    const lm = pr.landmarks?.[0]?.map(p => ({ x: p.x * W, y: p.y * H, z: p.z, v: p.visibility ?? 1 })) || null;
    let segData = null;
    if (seg) {
      const sr = seg.segment(img);
      if (sr.confidenceMasks?.length >= 5) {
        const m0 = sr.confidenceMasks[0];
        const buf = new Uint8ClampedArray(m0.width * m0.height * 4);
        packConfidence(sr.confidenceMasks, m0.width, m0.height, buf);
        segData = { data: buf, w: m0.width, h: m0.height };
      }
      sr.close?.();
    }
    return { W, H, lm, seg: segData };
  }
}
