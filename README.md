# 本機虛擬試穿（Local Try-On）

用鏡頭即時試穿網購衣服的本機版本，架構參考 AnyWear。**即時 AR 預覽**和 **HD 逼真生成**都在這台電腦上運算，衣服圖片與鏡頭畫面不會上傳到任何雲端。

## 快速開始

| 系統 | 安裝 | 啟動 |
|---|---|---|
| macOS | `bash setup.sh`（或雙擊 `setup.command`） | `bash run.sh`（或雙擊 `run.command`） |
| Windows | 雙擊 `setup.bat` | 雙擊 `run.bat` |
| Linux | `bash setup.sh` | `bash run.sh` |

- 不需要事先安裝 Python；安裝程式會把所有東西下載到這個資料夾內，最後顯示「自我檢測：全部通過」即完成
- 啟動後以 Chrome 開啟 http://127.0.0.1:8765，允許鏡頭權限
- 換電腦：複製整個資料夾，在新電腦重新執行安裝即可（見第 3 節）

---

## 1. AnyWear 是怎麼做的？和這個專案差在哪？

| | AnyWear（Decart） | 本專案 |
|---|---|---|
| 核心技術 | **Lucy VTON**：即時「影片擴散模型」，每一格畫面都由 AI 重新生成 | 即時：姿勢追蹤＋2D 骨架蒙皮變形＋光影轉移；HD：**CatVTON** 擴散模型 |
| 運算位置 | 雲端 H100 GPU 叢集，透過 WebRTC 串流 | 你的 Mac（Apple GPU / MPS） |
| 即時畫面的光影 | AI 逐格生成，布料垂墜、擺動都是生成出來的 | 從你身上實際衣服抽出皺褶與明暗，套到虛擬衣服上 |
| 照片級真實度 | 即時 | 按「HD 生成」後輸出一張（每張需要等待） |
| 費用 | 依秒計費的雲端 API | 免費（模型下載約 2.5 GB） |

Lucy VTON 是閉源模型，沒有釋出權重；開源的即時影片試穿模型（例如 2026 年 8 月的 LiveVVT）目前也還沒公開程式碼和權重，而且需要 NVIDIA 等級的 GPU。所以本機版採用 **混合架構**：

```
鏡頭 ─┬─> MediaPipe 姿勢 (33 點) ──┐
      └─> MediaPipe 多類別分割 ────┤ (頭髮/臉/皮膚/衣服)
                                   v
衣服圖 ─> 去背 ─> 自動找肩線/袖子/下擺 ─> 2D 網格 + 骨架蒙皮 ─> WebGL 合成
                                              │        (皺褶光影轉移、色溫、
                                              │         接觸陰影、手臂遮擋)
                                              v
                          [HD 生成] 凍結一格 ─> 衣物無關遮罩 ─> CatVTON (MPS)
                                              │
                                              └─> 生成結果再切出衣服，
                                                  回饋給即時預覽當作貼圖
```

最後一步是關鍵：HD 生成的衣服已經「穿」在你身上（領口、垂墜都對），把它切下來當即時預覽的貼圖後，再隨你的動作變形，即時畫面的真實度會明顯提升。

---

## 2. 支援的電腦

| 電腦 | 即時試穿 | HD 生成用的運算 |
|---|---|---|
| Apple Silicon Mac（M1 以上） | ✓ | Apple GPU（MPS） |
| Intel Mac | ✓ | PyTorch 2.2.2；有 AMD 顯卡用 MPS，否則 CPU |
| Windows / Linux + NVIDIA 顯卡 | ✓ | CUDA |
| Windows / Linux 沒有 NVIDIA 顯卡 | ✓ | CPU（HD 每張要等比較久） |

- 不需要事先裝 Python：安裝程式會用 [uv](https://github.com/astral-sh/uv) 下載一份專用的 Python 3.11，連同所有套件和模型都放在這個資料夾裡（`.tools/`、`.venv/`），不動到系統設定
- 瀏覽器用 Google Chrome（Safari／Edge 也能開，MediaPipe 的 GPU 加速以 Chrome 最穩定）
- 第一次安裝需要網路，大約下載：uv 與 Python 約 50 MB、套件（含 PyTorch）數百 MB、MediaPipe 約 60 MB、去背模型約 180 MB；HD 模型約 2.3 GB（第一次按 HD 時才下載，或安裝時加 `--hd`）
- 記憶體建議 16 GB 以上（「高品質」HD 需要更多）

## 3. 安裝與啟動

### macOS
打開「終端機」，執行：
```bash
cd /path/to/virtual-tryon             # 換成你放這個資料夾的位置
bash setup.sh                         # 安裝（最後會跑自我檢測）
bash run.sh                           # 啟動，會用 Chrome 開 http://127.0.0.1:8765
```
也可以在 Finder 雙擊 `setup.command`、`run.command`。從網路下載或 AirDrop 收到的資料夾，macOS 第一次會擋下雙擊執行，這時改在檔案上按右鍵 →「打開」，或直接用上面的 `bash setup.sh`。

### Windows
雙擊 `setup.bat` 安裝，之後雙擊 `run.bat` 啟動。

### Linux
`bash setup.sh`，之後 `bash run.sh`。

第一次開啟網頁時，瀏覽器會詢問鏡頭權限，請允許。

### 安裝程式做了什麼
1. 下載 uv 到 `.tools/uv/`
2. 下載 Python 3.11 到 `.tools/python/`，建立 `.venv/`
3. 依電腦安裝對應的 PyTorch（Apple GPU／CUDA／CPU），再裝其他套件
4. 下載 MediaPipe 執行環境、姿勢與分割模型（`web/vendor/`）、去背模型（`.tools/u2net/`）
5. 自我檢測：套件、PyTorch 裝置、HD 引擎（用小型測試模型實跑一次）、去背、伺服器 API。最後顯示「自我檢測：全部通過」就是可以用了

想同時下載並實測 HD 模型：`bash setup.sh --hd`（Windows：`setup.bat --hd`）；已經裝好的 Mac 也可以雙擊 `tools/hd-selftest.command`。最後一行會顯示這台電腦 HD 生成一張「快速」需要幾秒。

離線或公司管理的電腦：安裝前可設定 `VTON_UV`（已有的 uv 執行檔路徑）與 `VTON_PYTHON`（已有的 Python 3.10–3.13 路徑），安裝程式就不會去下載它們。

### 搬到另一台電腦
- 複製 `virtual-tryon` 資料夾或 zip 到新電腦，再跑一次安裝程式即可
- `.venv/`、`.tools/` 是這台電腦專用的（Python、PyTorch 依 CPU 而不同），不用帶走；帶過去也沒關係，安裝程式偵測到不相容會自動改名保留、重新建立
- 想帶走衣櫃和 HD 結果：一起複製 `data/` 資料夾

### 移除
直接刪掉整個資料夾。所有下載的 Python、套件、模型都在裡面。

## 4. 使用方式

### 加入衣服（4 種方法）
1. **複製貼上（最方便）**：在購物網站的商品圖上按右鍵 →「複製圖片」→ 回到試穿頁按 `⌘V`（Windows：`Ctrl+V`）
2. **拖曳**：把購物網站的圖片直接拖到試穿頁
3. **貼網址**：貼上商品頁網址，會列出頁面上的商品圖讓你挑；也可以直接貼圖片網址
4. **選檔案**：點虛線框選擇本機圖片

加入前先選分類：**上衣／外套**、**洋裝／連身**、**褲子／裙子**。
選錯分類也沒關係，選取那件衣服後再點一次正確的分類即可。

**哪種圖片效果最好**：白底或素色背景的平拍圖、隱形模特兒（ghost mannequin）圖。
模特兒穿著照也可以用，系統會偵測到人體、自動切換成「模特兒照模式」。

### 即時試穿
- 站在鏡頭前，讓 **雙肩到腰部入鏡**；試穿下身衣物要退後讓膝蓋入鏡
- **穿合身、素色、淺色的上衣效果最好**：系統會從你身上衣服的皺褶取光影；條紋或大圖案會被當成皺褶轉移上去
- 光線從正面或側前方打來，背景單純

### 調整
| 滑桿 | 作用 |
|---|---|
| 尺寸 | 衣服整體放大縮小（寬鬆／合身） |
| 上下位置、衣長 | 修正衣服高低與長度 |
| 皺褶光影 | 從你身上衣服轉移過來的皺褶強度 |
| 立體受光 | 身體左右明暗（例如側光）的轉移強度 |
| 亮度、色溫匹配 | 讓商品照的亮度與色溫配合你房間的光線 |
| 接觸陰影 | 衣服邊緣落在皮膚上的陰影 |

衣服位置明顯不對（例如肩膀歪掉）時，打開 **錨點校正**，拖曳圓點：「肩」對準肩膀關節（肩線略內側），「袖」對準袖口。

### HD 逼真生成
1. 選好品質：快速（384×512、20 步）／標準（576×768、30 步）／高品質（768×1024、50 步）
2. 按 **拍照並生成 HD 試穿**（或鍵盤 `H`），3 秒倒數內面向鏡頭、雙手稍微離開身體
3. 第一次會先下載約 2.3 GB 模型；之後只需載入
4. 完成後會跳出結果，可下載；勾選「生成後用 HD 結果強化即時預覽」時，即時畫面會改用 HD 生成的衣服貼圖

### 快捷鍵
`空白鍵`（按住）看原本畫面 · `H` HD 生成 · `M` 鏡像 · `D` 顯示骨架 · `Esc` 關閉結果

「文字正向」按鈕：鏡像畫面中，衣服上的文字和 logo 預設維持正向可讀（關掉則和真正的鏡子一樣左右相反）。

---

## 5. 技術細節

**即時 AR（瀏覽器，WebGL2）**
- 追蹤：MediaPipe Pose Landmarker（full）＋ Selfie Multiclass 分割（頭髮、臉、身體皮膚、衣服），One Euro 濾波去抖動，並學習使用者的身體比例
- 速度：兩個模型預設用 GPU；啟動後各量一次速度，GPU 不夠快時改測 CPU，留下較快的那個。分割的頻率依耗時自動調整（每 1–8 格跑一次）
- 衣服分析：從去背輪廓找出軀幹欄位（下擺為基準，排除貼著身體下垂的長袖）、肩線、腋下與袖口，估計肩關節位置
- 變形：48 欄網格。軀幹用雙線性對應（肩–髖四邊形），袖子／褲管用骨骼仿射變換；袖子與衣身只在袖籠處柔性混合，其他地方沿著輪廓切開，舉手時不會拉出長條殘影
- 光影：用相機畫面的「帶通」亮度比例（去掉布紋與雜訊，保留皺褶）乘上衣服顏色，只取對應部位的實際衣服；外加大範圍明暗、灰世界色溫、臉部亮度推算曝光、衣緣接觸陰影、感光雜訊
- 遮擋：頭髮與臉永遠在衣服前面；手臂只有在 3D 深度上位於身體前方（例如抱胸）時才會擋住衣服

**HD（Python 伺服器）**
- CatVTON（SD-1.5 inpainting UNet、移除 cross-attention、人與衣服在高度方向串接）
- 原版需要 DensePose + SCHP 產生遮罩（需要 detectron2，Mac 難以安裝），這裡改由瀏覽器用 MediaPipe 分割＋姿勢產生同等的「衣物無關遮罩」
- 適配 Apple GPU：UNet fp16、VAE fp32、自注意力分塊（限制記憶體）、fp16 出現 NaN 時自動改 fp32、DPM-Solver 以較少步數取樣
- 生成結果以羽化遮罩貼回完整鏡頭畫面

**驗證**
- `tools/test_engine_tiny.py`：用小型隨機權重比對官方 CatVTON 實作，權重載入索引、前向輸出、分塊注意力皆一致（差異 0）
- `tools/fake_hd_server.py`：不下載模型，測試 HD 流程（遮罩、上傳、進度、貼回、即時預覽切換）
- `http://127.0.0.1:8765/debug.html`：顯示每件衣服的自動錨點分析，以及套在合成骨架（A 字、舉手、側身、跨步）上的變形結果

## 6. 檔案結構

```
virtual-tryon/
├── setup.sh / run.sh            macOS、Linux：安裝、啟動
├── setup.command / run.command  macOS：雙擊版
├── setup.bat / run.bat          Windows
├── requirements.txt
├── server/
│   ├── app.py                   FastAPI：網頁、衣櫃 API、HD 工作佇列
│   ├── catvton_engine.py        CatVTON 推論（MPS/CUDA/CPU）
│   ├── garment_proc.py          抓圖、商品頁解析、去背
│   └── config.py                路徑與設定（可用環境變數覆寫）
├── web/
│   ├── index.html, styles.css
│   ├── debug.html               衣服分析除錯頁
│   └── js/
│       ├── main.js              介面與主迴圈
│       ├── tracker.js           MediaPipe 追蹤
│       ├── garment.js           衣服分析（平拍／模特兒照）
│       ├── rig.js               網格與骨架蒙皮
│       ├── renderer.js          WebGL2 合成與光影
│       ├── masks.js             HD 用的衣物無關遮罩
│       ├── hd.js                HD 拍照、輪詢、回饋即時預覽
│       └── editor.js            錨點校正
├── constraints/intel-mac.txt    Intel Mac 的版本鎖定
├── tools/                       install.py（安裝步驟）、selftest.py（自我檢測）、資產下載、測試工具
└── data/                        衣櫃、HD 結果（自動建立）；刪除的衣服移到 data/trash
```

## 7. 疑難排解

- **安裝失敗**：把終端機（或命令提示字元）最後幾十行訊息複製下來；修正後重新執行安裝程式即可，已下載的部分不會重抓
- **單獨重跑自我檢測**：macOS／Linux `.venv/bin/python tools/selftest.py`；Windows `.venv\Scripts\python.exe tools\selftest.py`
- **Mac 上的 `No matching distribution found for torch>=2.3`**（舊版安裝程式）：那是 Rosetta／Intel 版的 Python。新版改用 uv 下載原生 Python，不再發生
- **舊的 `.venv` 不相容**：安裝程式會把它改名成 `.venv.old-日期`（不會刪除），確認不需要後可自行刪掉
- **鏡頭打不開**：macOS 系統設定 → 隱私權與安全性 → 相機，允許 Chrome
- **HD 生成很慢**：先用「快速」；自我檢測的「PyTorch 裝置」那一行應顯示 `mps`（Mac）或 `cuda`（NVIDIA）
- **HD 結果整張黑掉**：程式偵測到 fp16 溢位會自動改用 fp32 重算
- **即時畫面卡頓**：畫面右下角會顯示「姿勢 / 分割」各花幾毫秒與使用 GPU 或 CPU；關掉其他使用鏡頭或大量 GPU 的分頁
- **更新程式後畫面沒變**：重新啟動 run.sh 後重新整理網頁即可（程式檔不會被瀏覽器長期快取）
- **衣服歪掉或太大**：用「尺寸」滑桿，或打開「錨點校正」
- **換連接埠**：`VTON_PORT=9000 bash run.sh`（Windows：`set VTON_PORT=9000` 後執行 `run.bat`）
- **特殊顯卡（例如 AMD ROCm）**：安裝前設定 `VTON_TORCH_INDEX`（PyTorch 套件庫網址）與 `VTON_TORCH_SPEC`（例如 `torch==2.6.0`）
- **Hugging Face 下載慢**：設定 `HF_ENDPOINT` 使用鏡像站

## 8. 版本與授權

目前版本見 `VERSION`。

- CatVTON 權重：CC BY-NC-SA 4.0（**僅限非商業用途**）
- Stable Diffusion inpainting：CreativeML OpenRAIL-M
- MediaPipe：Apache 2.0；rembg：MIT
- 網頁設定了 CSP，擋掉 MediaPipe 內建的使用量回報（odml.pa.googleapis.com）
