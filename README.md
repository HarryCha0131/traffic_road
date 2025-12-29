## run_yolo.py 使用說明

此腳本用來一鍵跑 Ultralytics YOLO 訓練流程，並自動保存完整紀錄：

- 終端機即時顯示訓練進度，同時輸出 log 檔
- 保存系統資訊、套件版本、data.yaml 快照
- 複製 Ultralytics 產出的 artifacts（results.csv、plots、best.pt...）
- 將整個 run 資料夾打包成 zip，方便搬回本機

### 環境需求

- Python >= 3.12
- `ultralytics`（提供 `yolo` CLI）

安裝依賴（使用 pyproject）：

```bash
pip install -e .
```

### 基本用法

```bash
python run_yolo.py --data /path/to/data.yaml --model yolo11n.pt
```

或使用 `uv`：

```bash
uv run run_yolo.py --data /path/to/data.yaml --model yolo11n.pt
```

常見範例：

```bash
# 指定實驗標籤與輸出資料夾
python run_yolo.py --data data.yaml --model yolo11n.pt --tag baseline --project runs_local

# 加上驗證與推論
python run_yolo.py --data data.yaml --model yolo11n.pt --do_val --do_predict --predict_source images/

# 傳入額外的 Ultralytics 參數
python run_yolo.py --data data.yaml --model yolo11n.pt --extra "cos_lr=True lr0=0.01"
```

專案內也有提供範例腳本：

```bash
./start.sh
```

### 參數說明

必填：

- `--data`：data.yaml 路徑

常用：

- `--model`：權重檔名稱或路徑（預設 `yolo11n.pt`）
- `--task`：`detect` / `segment` / `pose`
- `--imgsz`：影像大小（預設 640）
- `--epochs`：訓練 epoch（預設 80）
- `--batch`：batch size（預設 16）
- `--device`：GPU 裝置或 `cpu`（預設 `0`）
- `--workers`：dataloader workers（預設 8）
- `--seed`：random seed（預設 42）

輸出與標記：

- `--project`：輸出資料夾根目錄（預設 `runs_local`）
- `--name`：自訂 run 名稱（不填則自動生成）
- `--tag`：實驗標籤（會加在檔名與資料夾前綴）
- `--extra`：額外傳給 Ultralytics CLI 的參數字串

額外流程：

- `--do_val`：訓練後執行 val（使用 best.pt）
- `--do_predict`：訓練後執行 predict（使用 best.pt）
- `--predict_source`：predict 影像資料夾路徑
- `--conf`：predict confidence 門檻（預設 0.25）

### 輸出結構（範例）

```
runs_local/
  baseline__yolo11n_detect_20240101_123000/
    meta/
      run_args.json
      system_info.json
      timing.json
      data.yaml
    logs/
      train.log
      val.log
      predict.log
    artifacts/
      train_run/
      val_run/
      predict_run/
    ultralytics_runs/
      train/
      val/
      predict/
  baseline__yolo11n_detect_20240101_123000.zip
```

### 備註

- `--tag` 會被自動清理成檔名安全的字元（空白改為 `_`）。
- 若 `best.pt` 不存在會改用 `last.pt`。
