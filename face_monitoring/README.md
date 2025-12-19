# Face Mesh - 姿勢守衛 (Posture Guard)

一個基於 MediaPipe 的智能姿勢監控系統，可以檢測頭部姿勢並提供即時反饋。

## 功能特點

- ✅ 即時頭部姿勢檢測（抬頭、低頭、前傾）
- ✅ 自動校準功能
- ✅ 系統通知提醒
- ✅ 系統托盤模式（可後台運行）
- ✅ 多線程攝像頭讀取（提升性能）
- ✅ 跨平台支持（macOS、Windows、Linux）

## 安裝

### 1. 安裝依賴

使用 `uv` 或 `pip` 安裝依賴：

```bash
# 使用 uv (推薦)
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 2. 確保模型文件存在

確保 `face_monitoring/face_landmarker.task` 文件存在於項目目錄中。

## 使用方法

### 標準模式（帶視窗）

直接運行主程序，會顯示 OpenCV 視窗：

**使用 uv (推薦)**：
```bash
uv run python -m face_monitoring.posture_guard_system_tray
```

**使用 python3**：
```bash
python3 -m face_monitoring.posture_guard_system_tray
```

或直接運行文件：

```bash
cd face_monitoring
uv run python posture_guard_system_tray.py
# 或
python3 posture_guard_system_tray.py
```

**操作說明：**
- 按 `C` 鍵：開始/重新校準姿勢基準
- 按 `ESC` 鍵：退出程序

### 系統托盤模式（後台運行）

使用 `--tray` 或 `-t` 參數啟動系統托盤模式：

**使用 uv (推薦)**：
```bash
uv run python -m face_monitoring.posture_guard_system_tray --tray
```

**使用 python3**：
```bash
python3 -m face_monitoring.posture_guard_system_tray --tray
```

**或使用簡寫**：
```bash
uv run python -m face_monitoring.posture_guard_system_tray -t
python3 -m face_monitoring.posture_guard_system_tray -t
```

**系統托盤功能：**
- 右鍵點擊系統托盤圖標可打開選單
- **顯示/隱藏視窗**：切換 OpenCV 視窗的顯示/隱藏
- **重新校準**：觸發姿勢重新校準
- **退出**：完全退出應用程式

**注意事項：**
- 在系統托盤模式下，即使隱藏視窗，監控功能仍在後台運行
- 姿勢不良時仍會收到系統通知提醒
- 按 `ESC` 鍵會隱藏視窗（而非退出程序）

## 配置參數

可以在 `PostureConfig` 類中調整以下參數：

```python
# 校準參數
calibration_frames: int = 60  # 校準所需的幀數

# 檢測閾值
pitch_up_threshold: float = -25.0      # 抬頭角度閾值（度）
pitch_down_threshold: float = 15.0     # 低頭角度閾值（度）
nose_offset_up_threshold: float = -30.0   # 抬頭鼻子偏移閾值（像素）
nose_offset_down_threshold: float = 30.0  # 低頭鼻子偏移閾值（像素）
z_forward_threshold: float = 0.05      # 前傾 Z 軸閾值

# 警告觸發條件
bad_posture_ratio: float = 0.8  # 姿勢不良比例閾值（80%）
history_size: int = 30           # 歷史記錄大小（約 1 秒，30 FPS）
```

## 系統要求

- Python >= 3.13
- 攝像頭設備
- macOS / Windows / Linux

## 依賴套件

- `mediapipe` - 臉部關鍵點檢測
- `opencv-python` - 影像處理和顯示
- `plyer` - 跨平台通知（備用）
- `pystray` - 系統托盤功能
- `pillow` - 圖像處理（用於托盤圖標）

## 故障排除

### 1. 無法打開攝像頭

- 檢查攝像頭是否被其他程序占用
- 嘗試更改 `camera_id` 參數（0, 1, 2...）

### 2. 系統托盤圖標不顯示

- 確保已安裝 `pystray` 和 `pillow`：
  ```bash
  pip install pystray pillow
  ```
- 在 macOS 上，可能需要授予終端機/IDE 系統權限

### 3. 通知不工作

- macOS：檢查系統偏好設定 > 通知與專注模式
- Windows：檢查通知設定
- Linux：確保桌面環境支持通知

### 4. 模型文件缺失

從 MediaPipe 官方下載 `face_landmarker.task` 文件並放置在 `face_monitoring/` 目錄下。

## 開發說明

### 項目結構

```
Face_Mesh/
├── face_monitoring/
│   ├── posture_guard_system_tray.py  # 主程序（含系統托盤功能）
│   ├── posture_guard_real.py         # 簡化版本
│   ├── posture_guard_multithread.py   # 多線程版本
│   ├── face_landmarker.task          # MediaPipe 模型文件
│   ├── head_pose.py                  # 頭部姿勢檢測
│   └── blinking.py                   # 眨眼檢測
├── main.py
├── pyproject.toml
└── README.md
```

### 核心類別說明

- `PostureConfig`: 配置參數類
- `PostureGuard`: 主監控類，整合所有功能
- `HeadPoseDetector`: 頭部姿勢檢測器
- `PostureCalibrator`: 姿勢校準器
- `PostureMonitor`: 姿勢評估器
- `NotificationManager`: 通知管理器
- `ThreadedCamera`: 多線程攝像頭讀取器
- `PostureTrayApp`: 系統托盤應用程式

## 授權

MIT License

