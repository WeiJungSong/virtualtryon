// Thin wrapper over the local server API.
const HDR = { 'X-Requested-With': 'vton' };

async function call(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { ...HDR, ...(opts.headers || {}) } });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch { /* ignore */ }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return res.json();
}
const json = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export const getStatus = () => call('/api/status');
export const listGarments = () => call('/api/garments');
export const fetchUrl = (url) => call('/api/fetch', json('POST', { url }));
export const patchGarment = (id, patch) => call(`/api/garments/${id}`, json('PATCH', patch));
export const deleteGarment = (id) => call(`/api/garments/${id}`, { method: 'DELETE' });
export const loadHD = () => call('/api/hd/load', { method: 'POST' });
export const submitHD = (fd) => call('/api/hd/jobs', { method: 'POST', body: fd });
export const getJob = (id) => call(`/api/hd/jobs/${id}`);
export const listResults = () => call('/api/results');
export const proxyUrl = (url, referer) =>
  `/api/proxy?url=${encodeURIComponent(url)}${referer ? `&referer=${encodeURIComponent(referer)}` : ''}`;

export function uploadGarment({ file, url, category, name }) {
  const fd = new FormData();
  if (file) fd.append('file', file);
  if (url) fd.append('url', url);
  fd.append('category', category);
  if (name) fd.append('name', name);
  return call('/api/garments', { method: 'POST', body: fd });
}

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`圖片載入失敗：${src}`));
    img.src = src;
  });
}
