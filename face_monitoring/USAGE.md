# 姿勢守衛 - 快速使用指南

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 使用 uv (推薦)
uv sync

# 或使用 pip
pip install pystray pillow
```

### 2. 運行程序

#### 標準模式（帶視窗）

**使用 uv (推薦)**：
```bash
uv run python -m face_monitoring.posture_guard_system_tray
```

**使用 python3**：
```bash
python3 -m face_monitoring.posture_guard_system_tray
```

#### 系統托盤模式（後台運行）

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

## 📋 使用說明

### 標準模式操作

1. **首次使用**：程序啟動後會自動開始校準（約 2 秒）
2. **重新校準**：按 `C` 鍵重新校準姿勢基準
3. **退出程序**：按 `ESC` 鍵

### 系統托盤模式操作

1. **啟動後**：程序會在系統托盤顯示綠色圖標
2. **右鍵選單**：
   - **顯示/隱藏視窗**：切換 OpenCV 視窗顯示
   - **重新校準**：觸發姿勢重新校準
   - **退出**：完全退出程序
3. **鍵盤快捷鍵**（當視窗顯示時）：
   - `C` 鍵：重新校準
   - `ESC` 鍵：隱藏視窗（不退出程序）

## ⚙️ 功能說明

### 姿勢檢測

程序會檢測以下不良姿勢：
- **抬頭過高**：頭部向上傾斜過多
- **低頭過低**：頭部向下傾斜過多
- **前傾**：身體向前傾斜

### 通知提醒

當檢測到不良姿勢持續一段時間（約 1 秒）時，會發送系統通知提醒。

### 校準功能

校準功能會記錄您當前正確的坐姿作為基準，用於後續的姿勢判斷。

**建議校準步驟：**
1. 保持正確的坐姿
2. 直視前方
3. 按 `C` 鍵開始校準
4. 保持姿勢約 2 秒

## 🔧 常見問題

### Q: 系統托盤圖標不顯示？

A: 確保已安裝依賴：
```bash
pip install pystray pillow
```

在 macOS 上，可能需要授予終端機/IDE 系統權限。

### Q: 攝像頭無法打開？

A: 
- 檢查攝像頭是否被其他程序占用
- 確保已授予攝像頭權限（macOS 系統偏好設定 > 安全性與隱私權）

### Q: 通知不工作？

A:
- macOS：檢查系統偏好設定 > 通知與專注模式
- 確保程序有發送通知的權限

### Q: 如何調整檢測靈敏度？

A: 編輯 `posture_guard_system_tray.py` 中的 `PostureConfig` 類，調整閾值參數。

## 📝 注意事項

1. **首次使用**：建議先進行校準，確保檢測準確
2. **光線條件**：確保臉部光線充足，避免背光
3. **攝像頭位置**：建議攝像頭位於螢幕上方，與視線平行
4. **系統托盤模式**：即使隱藏視窗，監控功能仍在後台運行

## 🎯 最佳實踐

1. 啟動程序後立即進行校準
2. 使用系統托盤模式，讓程序在後台運行
3. 定期重新校準（特別是更換坐姿或位置後）
4. 根據個人情況調整檢測閾值

