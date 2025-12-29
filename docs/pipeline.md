# 訓練 Pipeline 說明（Stage 1 / Stage 2）

此目錄提供可直接交給訓練同事使用的最小可跑 pipeline。分兩階段：

- **Stage 1 (CASIA Emotional)**：跑通資料流、loss、prompt、head。
- **Stage 2 (CHEAVD / 自然情緒)**：混訓 + Domain Adaptation 提升泛化。

---

## 1. 資料格式（Manifest CSV）

所有資料集先轉成統一 CSV：

```csv
path,label,speaker,domain
casia/angry/001.wav,angry,spk01,0
casia/happy/002.wav,happy,spk02,0
```

欄位說明：
- `path`: 音訊路徑（相對或絕對）
- `label`: 情緒標籤（原始標籤）
- `speaker`: 說話人 ID（用於 speaker-disjoint split）
- `domain`: 資料來源 ID（CASIA=0, CHEAVD=1）


### 產生 CSV（資料夾模式）

```bash
python scripts/prepare_manifest.py \
  --mode folder \
  --input_dir /path/to/casia \
  --output_csv data/splits/casia/all.csv \
  --domain_id 0
```

### 產生 train/val/test split（speaker-disjoint）

```bash
python scripts/split_manifest.py \
  --input_csv data/splits/casia/all.csv \
  --output_dir data/splits/casia \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

---

## 2. Label Mapping

在 `data/label_maps/` 下提供統一映射表（可自行增補）：

- `casia_to_unified.json`
- `cheavd_to_unified.json`
- `unified.json`

如果資料集標籤不同，直接在 JSON 內映射到統一標籤即可。

---

## 3. Stage 1（CASIA）

設定檔：`configs/stage1_casia.yaml`

訓練：

```bash
python scripts/train_stage1.py
```

重點：
- 搭配 Prompt Tokens (`model.n_prompt`)
- 使用 Transformer + MLP Head
- Loss：CrossEntropy + label smoothing

輸出：
- `checkpoints/stage1/best.pt`
- `outputs/metrics/val_metrics.json`

---

## 4. Stage 2（CHEAVD / 自然情緒）

設定檔：`configs/stage2_cheavd.yaml`

訓練：

```bash
python scripts/train_stage2.py
```

重點：
- CASIA + CHEAVD 混訓
- Domain Adaptation（Gradient Reversal）
- 強化增強策略
 - 可用 `train_weights` / `train_domains` 控制混訓比例與 domain id

---

## 5. 評估

```bash
python scripts/evaluate.py \
  --config configs/stage2_cheavd.yaml \
  --checkpoint checkpoints/stage2/best.pt \
  --split test
```

---

## 6. 單檔推理（本地測試）

```bash
python scripts/predict.py \
  --checkpoint checkpoints/stage1/best.pt \
  --audio /path/to/test.wav \
  --device cpu
```

---

## 7. 你可以調整的重點

- `labels`: 統一類別順序（固定住，避免混亂）
- `data.sample_rate / n_mels / max_duration`
- `training.batch_size / lr / epochs`
- `training.domain_adaptation` 開關

---

## 8. 常見問題

- **找不到 GPU？** 會自動 fallback 到 CPU，但速度很慢。
- **Windows dataloader 卡住？** 設定檔內 `training.num_workers: 0`。
- **資料載入失敗？** 請確認 CSV 路徑與 `root_dir` 配置。
- **label 不一致？** 請更新 `label_maps/*.json`。
