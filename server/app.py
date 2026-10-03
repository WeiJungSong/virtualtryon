"""Local virtual try-on server: static web app + garment library + CatVTON HD jobs."""
from __future__ import annotations

import io
import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

import mimetypes

from . import config

# make sure ES modules and wasm get the right MIME types on every OS
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
from . import garment_proc as gp
from .catvton_engine import PRESETS, CatVTONEngine

app = FastAPI(title="Local Virtual Try-On")
engine = CatVTONEngine()
CATEGORIES = {"upper", "dress", "lower"}
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


# --------------------------------------------------------------------------- #
# Basic protection: only localhost Host headers (DNS rebinding) and a custom
# header on API calls (blocks cross-site form posts from other web pages).
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def guard(request: Request, call_next):
    host = (request.headers.get("host") or "").rsplit(":", 1)[0]
    if host not in ALLOWED_HOSTS:
        return JSONResponse({"detail": "forbidden host"}, status_code=403)
    path = request.url.path
    if path.startswith("/api/") and path != "/api/proxy":
        if request.headers.get("x-requested-with") != "vton":
            return JSONResponse({"detail": "missing header"}, status_code=403)
    response = await call_next(request)
    # app code (html/js/css) is revalidated on every load so an updated copy of the
    # program is picked up without a hard refresh; versioned vendor files stay cached
    if not path.startswith(("/api/", "/data/", "/vendor/")) and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-cache"
    return response


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def _warmup():
    gp.start_rembg_warmup()   # downloads/loads the cut-out model in the background


@app.get("/api/status")
def status():
    return {"hd": engine.info(), "rembg": gp.rembg_available(), "rembg_status": gp.rembg_status(), "version": 1}


# --------------------------------------------------------------------------- #
# Fetching garments from the web
# --------------------------------------------------------------------------- #
class FetchReq(BaseModel):
    url: str


@app.post("/api/fetch")
def fetch_url(req: FetchReq):
    try:
        data, ctype, final = gp.fetch(req.url)
    except gp.FetchError as e:
        raise HTTPException(400, str(e))
    if ctype.startswith("image/") or gp.looks_like_image(data):
        return {"kind": "image", "url": final}
    if "html" in ctype or data[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):
        title, images = gp.extract_page_images(data.decode("utf-8", "ignore"), final)
        if not images:
            raise HTTPException(400, "這個頁面找不到商品圖片（需要登入、或圖片由 JavaScript 載入的頁面抓不到），請改用右鍵「複製圖片」後貼上")
        return {"kind": "page", "title": title, "page": final, "images": images}
    raise HTTPException(400, f"不支援的內容類型：{ctype or '未知'}")


@app.get("/api/proxy")
def proxy(url: str, referer: str | None = None):
    try:
        data, ctype, _ = gp.fetch(url, referer=referer)
    except gp.FetchError as e:
        raise HTTPException(400, str(e))
    if not (ctype.startswith("image/") or gp.looks_like_image(data)):
        raise HTTPException(400, "不是圖片")
    if not ctype.startswith("image/"):
        ctype = "image/jpeg"
    return Response(data, media_type=ctype, headers={"Cache-Control": "max-age=3600"})


# --------------------------------------------------------------------------- #
# Garment library
# --------------------------------------------------------------------------- #
def _meta_path(gid: str):
    return config.GARMENT_DIR / gid / "meta.json"


def _read_meta(gid: str) -> dict:
    p = _meta_path(gid)
    if not p.exists():
        raise HTTPException(404, "找不到這件衣服")
    return json.loads(p.read_text("utf-8"))


def _write_meta(meta: dict):
    _meta_path(meta["id"]).write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")


def _public(meta: dict) -> dict:
    gid = meta["id"]
    v = int(meta.get("updated", 0))
    return {**meta,
            "original_url": f"/data/garments/{gid}/original.png?v={v}",
            "cutout_url": f"/data/garments/{gid}/cutout.png?v={v}"}


@app.post("/api/garments")
def create_garment(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    category: str = Form("upper"),
    name: str | None = Form(None),
):
    if category not in CATEGORIES:
        raise HTTPException(400, "未知的分類")
    if file is not None:
        raw = file.file.read()
        source = file.filename or "upload"
    elif url:
        if url.startswith("data:"):
            import base64
            raw = base64.b64decode(url.split(",", 1)[1])
            source = "clipboard"
        else:
            try:
                raw, _, source = gp.fetch(url)
            except gp.FetchError as e:
                raise HTTPException(400, str(e))
    else:
        raise HTTPException(400, "請提供圖片或網址")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "無法讀取圖片")
    try:
        cutout, method = gp.remove_background(img)
    except ValueError as e:
        raise HTTPException(400, str(e))
    gid = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = config.GARMENT_DIR / gid
    d.mkdir(parents=True)
    img.thumbnail((2048, 2048), Image.LANCZOS)
    img.save(d / "original.png")
    cutout.save(d / "cutout.png")
    meta = {
        "id": gid,
        "name": name or "",
        "category": category,
        "source": source,
        "matting": method,
        "created": time.time(),
        "updated": time.time(),
        "anchors": None,  # manual anchor overrides from the web UI
    }
    _write_meta(meta)
    return _public(meta)


@app.get("/api/garments")
def list_garments():
    metas = []
    for p in sorted(config.GARMENT_DIR.glob("*/meta.json"), reverse=True):
        try:
            metas.append(_public(json.loads(p.read_text("utf-8"))))
        except Exception:
            continue
    return {"garments": metas}


class GarmentPatch(BaseModel):
    name: str | None = None
    category: str | None = None
    anchors: dict | None = None
    clear_anchors: bool = False


@app.patch("/api/garments/{gid}")
def patch_garment(gid: str, patch: GarmentPatch):
    meta = _read_meta(gid)
    if patch.name is not None:
        meta["name"] = patch.name
    if patch.category is not None:
        if patch.category not in CATEGORIES:
            raise HTTPException(400, "未知的分類")
        meta["category"] = patch.category
    if patch.anchors is not None:
        meta["anchors"] = patch.anchors
    if patch.clear_anchors:
        meta["anchors"] = None
    meta["updated"] = time.time()
    _write_meta(meta)
    return _public(meta)


@app.delete("/api/garments/{gid}")
def delete_garment(gid: str):
    """Soft delete: the folder is moved to data/trash so nothing is lost."""
    _read_meta(gid)
    shutil.move(str(config.GARMENT_DIR / gid), str(config.TRASH_DIR / gid))
    return {"ok": True}


# --------------------------------------------------------------------------- #
# HD try-on jobs (single background worker; the model is loaded lazily)
# --------------------------------------------------------------------------- #
jobs: dict[str, dict] = {}
job_queue: "queue.Queue[str]" = queue.Queue()


def _worker():
    while True:
        jid = job_queue.get()
        job = jobs.get(jid)
        if not job:
            continue
        try:
            if engine.state != "ready":
                job.update(status="loading", message="準備模型（第一次需下載約 2.3 GB）")
                engine.load()
            job.update(status="running", message="生成中", started=time.time())

            def prog(i, n):
                job.update(progress=i, total=n)

            meta = _read_meta(job["garment_id"])
            gdir = config.GARMENT_DIR / job["garment_id"]
            if job["garment_source"] == "original":
                garment = Image.open(gdir / "original.png").convert("RGB")
            else:
                garment = gp.flatten_on_white(Image.open(gdir / "cutout.png").convert("RGBA"))
            frame = Image.open(io.BytesIO(job.pop("_frame"))).convert("RGB")
            mask = Image.open(io.BytesIO(job.pop("_mask"))).convert("L")
            composite, crop_result = engine.tryon_frame(
                frame, mask, garment, job["crop"], job["preset"], job["seed"], progress=prog
            )
            rid = jid
            composite.save(config.RESULT_DIR / f"{rid}.jpg", quality=93)
            crop_result.save(config.RESULT_DIR / f"{rid}_crop.jpg", quality=93)
            info = {k: job[k] for k in ("garment_id", "preset", "seed", "crop", "category")}
            info.update(created=time.time(), seconds=round(time.time() - job["started"], 1), garment_name=meta.get("name", ""))
            (config.RESULT_DIR / f"{rid}.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8")
            job.update(
                status="done", message="完成", seconds=info["seconds"],
                result_url=f"/data/results/{rid}.jpg", crop_url=f"/data/results/{rid}_crop.jpg",
            )
        except Exception as e:
            traceback.print_exc()
            job.update(status="error", message=f"{type(e).__name__}: {e}")
            job.pop("_frame", None)
            job.pop("_mask", None)


threading.Thread(target=_worker, daemon=True).start()


@app.post("/api/hd/load")
def hd_load():
    if engine.state in ("not_loaded", "error"):
        def bg():
            try:
                engine.load()
            except Exception:
                traceback.print_exc()
        threading.Thread(target=bg, daemon=True).start()
    return engine.info()


@app.post("/api/hd/jobs")
def hd_submit(
    frame: UploadFile = File(...),
    mask: UploadFile = File(...),
    garment_id: str = Form(...),
    category: str = Form("upper"),
    preset: str = Form("fast"),
    seed: int = Form(42),
    crop: str | None = Form(None),
    garment_source: str = Form("cutout"),
):
    _read_meta(garment_id)
    if preset not in PRESETS:
        raise HTTPException(400, "未知的品質設定")
    box = None
    if crop:
        try:
            box = tuple(int(round(float(v))) for v in json.loads(crop))
            assert len(box) == 4 and box[2] > box[0] and box[3] > box[1]
        except Exception:
            raise HTTPException(400, "crop 格式錯誤")
    jid = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    jobs[jid] = {
        "id": jid, "status": "queued", "message": "排隊中", "progress": 0, "total": PRESETS[preset]["steps"],
        "garment_id": garment_id, "category": category, "preset": preset, "seed": seed, "crop": box,
        "garment_source": garment_source if garment_source in ("cutout", "original") else "cutout",
        "_frame": frame.file.read(), "_mask": mask.file.read(), "created": time.time(),
    }
    job_queue.put(jid)
    return _job_public(jobs[jid])


def _job_public(job: dict) -> dict:
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    out["engine"] = engine.info()
    out["queue"] = job_queue.qsize()
    return out


@app.get("/api/hd/jobs/{jid}")
def hd_job(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "找不到工作")
    return _job_public(job)


@app.get("/api/results")
def list_results():
    items = []
    for p in sorted(config.RESULT_DIR.glob("*.json"), reverse=True)[:60]:
        try:
            info = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        rid = p.stem
        items.append({**info, "id": rid, "result_url": f"/data/results/{rid}.jpg",
                      "crop_url": f"/data/results/{rid}_crop.jpg"})
    return {"results": items}


# static files last so API routes win
app.mount("/data", StaticFiles(directory=config.DATA_DIR), name="data")
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")


def main():
    import socket
    import uvicorn
    url = f"http://{config.HOST}:{config.PORT}"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((config.HOST, config.PORT)) == 0:
            print(f"\n  已經在執行中：{url}（或連接埠 {config.PORT} 被其他程式占用，可用 VTON_PORT 換一個）\n")
            return
    print(f"\n  本機虛擬試穿已啟動：{url}\n  關閉這個視窗或按 Ctrl+C 結束\n")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
