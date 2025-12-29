# Paipai Speech Emotion

本專案包含兩部分：

1. **前端錄音 + 情緒展示**（`demo/`）
2. **兩階段訓練 Pipeline**（Stage 1: CASIA，Stage 2: CHEAVD/自然情緒）

---

## 前端使用

1. 直接用瀏覽器開啟 `demo/index.html`
2. 點擊 **Start recording** / **Stop recording**
3. 點擊 **Analyze** 送到後端情緒模型

後端預設 API：`POST /api/analyze`，`multipart/form-data`，欄位 `file`。

---

## 訓練 Pipeline

訓練流程與資料格式已整理在：

- `docs/pipeline.md`

包含：
- CSV manifest 格式
- speaker-disjoint split
- Stage 1 / Stage 2 訓練
- 評估與推理指令

---

## 專案結構

```
paipai_speech_emo/
├── demo/                 # 前端錄音展示
├── configs/              # Stage 1 / Stage 2 設定檔
├── src/                  # dataset / model / training code
├── scripts/              # 訓練、評估、資料處理入口
├── data/                 # label map / split CSV
├── docs/                 # Pipeline 文件
├── checkpoints/          # 模型輸出
├── outputs/              # metrics / logs
└── README.md
```

---

## 依賴

```
pip install -r requirements.txt
```

---

- Stage 1：`python scripts/train_stage1.py`
- Stage 2：`python scripts/train_stage2.py`
- 主要文件：`docs/pipeline.md`
