# 快速啟動指南

## ⚠️ 重要提示

**項目需要 Python 3.13+**，如果您的系統 Python 版本較低，請使用 `uv` 運行（`uv` 會自動下載並管理正確的 Python 版本）。

## 🎯 最簡單的啟動方式

### 方法 1: 使用 uv (強烈推薦 ⭐)

`uv` 會自動管理 Python 版本和依賴，無需手動配置：

```bash
# 系統托盤模式
uv run python -m face_monitoring.posture_guard_system_tray --tray

# 標準模式
uv run python -m face_monitoring.posture_guard_system_tray
```

### 方法 2: 使用 python3 (需要 Python 3.13+)

**注意：** 此方法需要系統已安裝 Python 3.13 或更高版本。

```bash
# 系統托盤模式
python3 -m face_monitoring.posture_guard_system_tray --tray

# 標準模式
python3 -m face_monitoring.posture_guard_system_tray
```

## ⚠️ 常見問題解決

### 問題 1: `python` 命令找不到

**解決方案：**

1. **使用 `uv run` (最推薦)**
   ```bash
   uv run python -m face_monitoring.posture_guard_system_tray --tray
   ```
   `uv` 會自動處理 Python 版本和環境。

2. **使用 `python3` 代替 `python`**
   ```bash
   python3 -m face_monitoring.posture_guard_system_tray --tray
   ```
   **注意：** 需要 Python 3.13+

3. **創建別名 (可選)**
   在 `~/.zshrc` 中添加：
   ```bash
   alias python=python3
   ```
   然後執行：
   ```bash
   source ~/.zshrc
   ```

### 問題 2: Python 版本過低

如果系統 Python 版本低於 3.13，**必須使用 `uv`**：

```bash
# uv 會自動下載並使用 Python 3.13
uv run python -m face_monitoring.posture_guard_system_tray --tray
```

## 📝 完整步驟

1. **確保依賴已安裝**
   ```bash
   uv sync
   ```

2. **運行程序**
   ```bash
   uv run python -m face_monitoring.posture_guard_system_tray --tray
   ```

## 🔍 檢查環境

```bash
# 檢查 Python 版本
python3 --version

# 檢查 uv 是否安裝
uv --version

# 檢查依賴
uv pip list
```

