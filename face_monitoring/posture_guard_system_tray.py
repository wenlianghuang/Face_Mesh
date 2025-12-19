import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any
import queue
import threading
import time
import platform
import subprocess
import os
from pathlib import Path
try:
    from plyer import notification
except ImportError:
    notification = None


def get_model_path(relative_path: str = 'face_landmarker.task') -> str:
    """
    獲取模型文件的絕對路徑
    
    Args:
        relative_path: 相對於當前腳本文件的模型路徑
    
    Returns:
        模型文件的絕對路徑
    """
    # 獲取當前腳本所在目錄
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / relative_path
    
    # 如果文件存在，返回絕對路徑
    if model_path.exists():
        return str(model_path)
    
    # 如果不存在，嘗試相對於當前工作目錄
    cwd_path = Path.cwd() / relative_path
    if cwd_path.exists():
        return str(cwd_path)
    
    # 如果還是不存在，嘗試在 face_monitoring 目錄中查找
    face_monitoring_path = script_dir / relative_path
    if face_monitoring_path.exists():
        return str(face_monitoring_path)
    
    # 最後嘗試項目根目錄下的 face_monitoring 目錄
    project_root = script_dir.parent
    project_model_path = project_root / 'face_monitoring' / relative_path
    if project_model_path.exists():
        return str(project_model_path)
    
    # 如果都找不到，返回原始路徑（讓 MediaPipe 報錯）
    return relative_path


@dataclass
class PostureConfig:
    """姿勢監控配置參數"""
    # 模型路徑（會自動解析為絕對路徑）
    model_path: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """初始化後處理：自動解析模型路徑"""
        if self.model_path is None:
            self.model_path = get_model_path('face_landmarker.task')
        elif not os.path.isabs(self.model_path):
            # 如果是相對路徑，嘗試解析
            resolved = get_model_path(self.model_path)
            if os.path.exists(resolved):
                self.model_path = resolved
    
    # 校準參數
    calibration_frames: int = 60  # 校準所需的幀數
    
    # 檢測閾值
    pitch_up_threshold: float = -25.0  # 抬頭角度閾值（度）
    pitch_down_threshold: float = 15.0  # 低頭角度閾值（度）
    nose_offset_up_threshold: float = -30.0  # 抬頭鼻子偏移閾值（像素）
    nose_offset_down_threshold: float = 25.0  # 低頭鼻子偏移閾值（像素）
    z_forward_threshold: float = 0.05  # 前傾 Z 軸閾值
    
    # 警告觸發條件
    bad_posture_ratio: float = 0.8  # 姿勢不良比例閾值（80%）
    history_size: int = 30  # 歷史記錄大小（約 1 秒，30 FPS）
    
    # 視覺化參數
    window_name: str = 'Posture Guard v1.0'
    text_color_good: Tuple[int, int, int] = (0, 255, 0)
    text_color_bad: Tuple[int, int, int] = (0, 0, 255)

class ThreadedCamera:
    """多執行緒攝像頭讀取器"""
    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        self.frame_queue = queue.Queue(maxsize=1) # 只保留最新的一幀
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                return
            
            # 如果 queue 已滿，移除舊影格，放入最新影格
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def read(self) -> Optional[np.ndarray]:
        """非阻塞讀取最新幀"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        self.cap.release()

class NotificationManager:
    """系統通知管理（跨平台支持）"""
    def __init__(self, cooldown: int):
        self.last_notify_time = 0
        self.cooldown = cooldown
        self.platform = platform.system()  # 'Darwin', 'Windows', 'Linux'
        self._notification_available = self._check_notification_availability()
    
    def _check_notification_availability(self) -> bool:
        """檢查通知功能是否可用"""
        # 檢查 plyer 是否可用
        if notification is not None:
            return True
        
        # 在 macOS 上檢查 osascript 是否可用
        if self.platform == "Darwin":
            try:
                subprocess.run(
                    ["osascript", "-e", 'display notification "test"'],
                    check=True,
                    capture_output=True,
                    timeout=1
                )
                return True
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                return False
        
        return False
    
    def _send_via_osascript(self, title: str, message: str) -> bool:
        """使用 macOS 原生 osascript 發送通知（僅在 macOS 上使用）"""
        if self.platform != "Darwin":
            return False
        
        # 轉義特殊字符
        title_escaped = title.replace('"', '\\"')
        message_escaped = message.replace('"', '\\"')
        
        script = f'display notification "{message_escaped}" with title "{title_escaped}"'
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                timeout=2
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _send_via_plyer(self, title: str, message: str) -> bool:
        """使用 plyer 發送通知（跨平台）"""
        if notification is None:
            return False
        
        try:
            notification.notify(
                title=title,
                message=message,
                app_name="Posture Guard",
                timeout=5
            )
            return True
        except Exception:
            return False
    
    def send(self, title: str, message: str):
        """發送通知（自動選擇最佳方法）"""
        current_time = time.time()
        if current_time - self.last_notify_time < self.cooldown:
            return
        
        success = False
        
        # 策略 1: 在 macOS 上優先嘗試 osascript（更可靠）
        if self.platform == "Darwin":
            success = self._send_via_osascript(title, message)
        
        # 策略 2: 如果 osascript 失敗或不在 macOS，嘗試 plyer
        if not success:
            success = self._send_via_plyer(title, message)
        
        # 策略 3: 如果都失敗，回退到控制台輸出
        if success:
            self.last_notify_time = current_time
        else:
            print(f"[控制台通知] {title}: {message}")
class HeadPoseDetector:
    """頭部姿態檢測器"""
    
    # 標準 3D 臉部模型點
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # 鼻尖
        (0.0, -330.0, -65.0),        # 下巴
        (-225.0, 170.0, -135.0),     # 左眼左角
        (225.0, 170.0, -135.0),      # 右眼右角
        (-150.0, -150.0, -125.0),    # 左口角
        (150.0, -150.0, -125.0)      # 右口角
    ], dtype=np.float64)
    
    # 臉部關鍵點索引
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE = 33
    RIGHT_EYE = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    
    def __init__(self, model_path: str):
        """初始化 MediaPipe Face Landmarker"""
        # 確保路徑是絕對路徑或正確的相對路徑
        if not os.path.isabs(model_path):
            resolved_path = get_model_path(model_path)
            if os.path.exists(resolved_path):
                model_path = resolved_path
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """檢測臉部關鍵點"""
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)
        
        if result.face_landmarks:
            return result.face_landmarks[0]
        return None
    
    def calculate_head_pose(self, face_landmarks, w: int, h: int) -> dict:
        """
        計算頭部姿態
        
        Returns:
            dict: 包含 pitch_angle, z, nose_offset 的字典
        """
        # 獲取關鍵點座標
        nose_y = face_landmarks[self.NOSE_TIP].y * h
        left_eye_y = face_landmarks[self.LEFT_EYE].y * h
        right_eye_y = face_landmarks[self.RIGHT_EYE].y * h
        chin_y = face_landmarks[self.CHIN].y * h
        
        # 計算眼睛中心線的 Y 座標
        eye_center_y = (left_eye_y + right_eye_y) / 2
        
        # 計算鼻子相對於眼睛中心線的偏移（像素）
        nose_offset = nose_y - eye_center_y
        
        # 計算眼睛到下巴的距離作為參考長度
        face_height = chin_y - eye_center_y
        
        # 標準化偏移量並轉換為角度
        if face_height > 0:
            normalized_offset = nose_offset / face_height
            pitch_angle = np.degrees(np.arctan(normalized_offset * 3))
        else:
            pitch_angle = 0
        
        return {
            'pitch_angle': pitch_angle,
            'z': face_landmarks[self.NOSE_TIP].z,
            'nose_offset': nose_offset
        }


class PostureCalibrator:
    """姿勢校準器"""
    
    def __init__(self, config: PostureConfig):
        self.config = config
        self.is_calibrated = False
        self.calibrating_frames = 0
        self.baseline_pitch = 0.0
        self.baseline_z = 0.0
        self.baseline_nose_offset = 0.0
        self._accumulated_pitch = 0.0
        self._accumulated_z = 0.0
        self._accumulated_nose_offset = 0.0
    
    def start_calibration(self):
        """開始校準"""
        self.is_calibrated = False
        self.calibrating_frames = self.config.calibration_frames
        self._accumulated_pitch = 0.0
        self._accumulated_z = 0.0
        self._accumulated_nose_offset = 0.0
    
    def update(self, pitch: float, z: float, nose_offset: float) -> bool:
        """
        更新校準數據
        
        Returns:
            bool: 是否完成校準
        """
        if self.calibrating_frames > 0:
            self._accumulated_pitch += pitch
            self._accumulated_z += z
            self._accumulated_nose_offset += nose_offset
            self.calibrating_frames -= 1
            
            if self.calibrating_frames == 0:
                # 計算平均值
                frames_count = self.config.calibration_frames
                self.baseline_pitch = self._accumulated_pitch / frames_count
                self.baseline_z = self._accumulated_z / frames_count
                self.baseline_nose_offset = self._accumulated_nose_offset / frames_count
                self.is_calibrated = True
                return True
        return False
    
    def get_remaining_frames(self) -> int:
        """獲取剩餘校準幀數"""
        return self.calibrating_frames


class PostureMonitor:
    """姿勢監控器"""
    
    def __init__(self, config: PostureConfig):
        self.config = config
        self.posture_history = deque(maxlen=config.history_size)
    
    def evaluate(self, pitch_diff: float, nose_offset_diff: float, z_diff: float) -> dict:
        """
        評估當前姿勢
        
        Returns:
            dict: 包含 is_bad, is_looking_up, is_looking_down, is_leaning_forward 的字典
        """
        is_looking_up = (pitch_diff < self.config.pitch_up_threshold or 
                        nose_offset_diff < self.config.nose_offset_up_threshold)
        is_looking_down = (pitch_diff > self.config.pitch_down_threshold or 
                          nose_offset_diff > self.config.nose_offset_down_threshold)
        is_leaning_forward = z_diff > self.config.z_forward_threshold
        
        is_bad = is_looking_up or is_looking_down or is_leaning_forward
        self.posture_history.append(is_bad)
        
        return {
            'is_bad': is_bad,
            'is_looking_up': is_looking_up,
            'is_looking_down': is_looking_down,
            'is_leaning_forward': is_leaning_forward
        }
    
    def should_trigger_warning(self) -> bool:
        """判斷是否應該觸發警告"""
        if len(self.posture_history) == 0:
            return False
        bad_ratio = sum(self.posture_history) / len(self.posture_history)
        return bad_ratio > self.config.bad_posture_ratio
    
    def get_status_text(self, evaluation: dict) -> str:
        """獲取狀態文字"""
        if not self.should_trigger_warning():
            return "Good Posture"
        
        if evaluation['is_looking_down']:
            return "BAD POSTURE! HEAD TOO LOW"
        elif evaluation['is_looking_up']:
            return "BAD POSTURE! HEAD TOO HIGH"
        elif evaluation['is_leaning_forward']:
            return "BAD POSTURE! SIT STRAIGHT"
        else:
            return "BAD POSTURE!"


class Visualizer:
    """視覺化器"""
    
    def __init__(self, config: PostureConfig):
        self.config = config
    
    def draw_calibration_status(self, frame: np.ndarray, remaining_frames: int):
        """繪製校準狀態（增強版）"""
        h, w = frame.shape[:2]
        
        # 計算進度
        total_frames = self.config.calibration_frames
        progress = (total_frames - remaining_frames) / total_frames
        
        # 繪製半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # 繪製進度條背景
        bar_x, bar_y = w // 2 - 200, h // 2 + 50
        bar_w, bar_h = 400, 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        # 繪製進度條
        progress_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), (0, 255, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        
        # 繪製主要文字
        text = f"校準中... {remaining_frames} 幀"
        font_scale = 2.0
        thickness = 4
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2 - 20
        
        # 文字背景
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        
        # 文字
        cv2.putText(frame, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        
        # 進度百分比
        percent_text = f"{int(progress * 100)}%"
        percent_size = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        percent_x = (w - percent_size[0]) // 2
        cv2.putText(frame, percent_text, (percent_x, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # 提示文字
        hint_text = "請保持正確坐姿，直視前方"
        hint_size = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        hint_x = (w - hint_size[0]) // 2
        cv2.putText(frame, hint_text, (hint_x, h // 2 + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    def draw_posture_status(self, frame: np.ndarray, status_text: str, is_bad: bool):
        """繪製姿勢狀態（增強版）"""
        h, w = frame.shape[:2]
        color = self.config.text_color_bad if is_bad else self.config.text_color_good
        bg_color = (0, 0, 100) if is_bad else (0, 100, 0)  # 紅色或綠色背景
        
        # 繪製狀態背景條
        bar_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (0, 0), (w, bar_height), (255, 255, 255), 3)
        
        # 繪製主要狀態文字
        font_scale = 2.5
        thickness = 5
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = bar_height // 2 + text_size[1] // 2
        
        # 文字陰影效果
        cv2.putText(frame, status_text, (text_x + 3, text_y + 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, status_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # 如果姿勢不良，添加閃爍效果（通過邊框）
        if is_bad:
            # 繪製警告邊框
            border_thickness = 10
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_thickness)
            
            # 添加警告圖標文字
            warning_text = "⚠️"
            warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.putText(frame, warning_text, (w - warning_size[0] - 20, bar_height + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    def draw_no_face(self, frame: np.ndarray):
        """繪製未檢測到臉部的提示"""
        cv2.putText(frame, "No Face Detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    def draw_calibration_prompt(self, frame: np.ndarray):
        """繪製校準提示"""
        h, w = frame.shape[:2]
        cv2.putText(frame, "Press 'c' to Calibrate", (w-300, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)


class PostureGuard:
    """姿勢守衛主類"""
    
    def __init__(self, config: Optional[PostureConfig] = None, use_threaded_camera: bool = True, on_calibration_complete=None):
        self.config = config or PostureConfig()
        self.detector = HeadPoseDetector(self.config.model_path)
        self.calibrator = PostureCalibrator(self.config)
        self.monitor = PostureMonitor(self.config)
        self.visualizer = Visualizer(self.config)
        self.notification_manager = NotificationManager(cooldown=5)  # 5秒冷卻時間
        self.use_threaded_camera = use_threaded_camera
        self.cap = None
        self.threaded_camera = None
        # 狀態跟踪
        self.current_status = 'normal'  # 'normal', 'bad', 'calibrating'
        # 校準完成回調
        self.on_calibration_complete = on_calibration_complete
    
    def initialize_camera(self, camera_id: int = 0):
        """初始化攝像頭"""
        if self.use_threaded_camera:
            self.threaded_camera = ThreadedCamera(camera_id).start()
        else:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"無法打開攝像頭 {camera_id}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """處理單個幀"""
        h, w = frame.shape[:2]
        
        # 檢測臉部
        face_landmarks = self.detector.detect(frame)
        
        if face_landmarks is None:
            self.visualizer.draw_no_face(frame)
            self.visualizer.draw_calibration_prompt(frame)
            return frame
        
        # 計算頭部姿態
        pose_data = self.detector.calculate_head_pose(face_landmarks, w, h)
        
        # 校準邏輯
        if not self.calibrator.is_calibrated:
            remaining = self.calibrator.get_remaining_frames()
            self.current_status = 'calibrating'  # 更新狀態
            if remaining > 0:
                was_calibrating = remaining == self.config.calibration_frames
                self.calibrator.update(
                    pose_data['pitch_angle'],
                    pose_data['z'],
                    pose_data['nose_offset']
                )
                # 如果剛完成校準，打印提示並觸發回調
                if self.calibrator.is_calibrated and was_calibrating:
                    print("✅ 校準完成！現在開始監控姿勢")
                    self.current_status = 'normal'
                    # 觸發校準完成回調
                    if self.on_calibration_complete:
                        self.on_calibration_complete()
                self.visualizer.draw_calibration_status(frame, remaining)
            else:
                # 校準完成但還沒標記
                if self.calibrator.is_calibrated:
                    print("✅ 校準完成！現在開始監控姿勢")
                    self.current_status = 'normal'
                    # 觸發校準完成回調
                    if self.on_calibration_complete:
                        self.on_calibration_complete()
            self.visualizer.draw_calibration_prompt(frame)
            return frame
        
        # 監控邏輯
        pitch_diff = pose_data['pitch_angle'] - self.calibrator.baseline_pitch
        nose_offset_diff = pose_data['nose_offset'] - self.calibrator.baseline_nose_offset
        z_diff = self.calibrator.baseline_z - pose_data['z']
        
        evaluation = self.monitor.evaluate(pitch_diff, nose_offset_diff, z_diff)
        should_warn = self.monitor.should_trigger_warning()
        status_text = self.monitor.get_status_text(evaluation)
        
        # 更新狀態
        self.current_status = 'bad' if should_warn else 'normal'
        
        # 發送通知（如果需要）
        if should_warn:
            self.notification_manager.send("姿勢提醒", status_text)
        
        # 繪製姿勢狀態文字（增強版）
        self.visualizer.draw_posture_status(frame, status_text, should_warn)
        
        return frame
    
    def run(self):
        """運行主循環"""
        if self.cap is None and self.threaded_camera is None:
            self.initialize_camera()
        
        try:
            while True:
                # 讀取幀
                if self.use_threaded_camera:
                    frame = self.threaded_camera.read()
                    if frame is None:
                        time.sleep(0.01)  # 避免CPU空轉
                        continue
                else:
                    if not self.cap.isOpened():
                        break
                    success, frame = self.cap.read()
                    if not success:
                        break
                
                frame = self.process_frame(frame)
                cv2.imshow(self.config.window_name, frame)
                
                key = cv2.waitKey(5) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('c'):
                    self.calibrator.start_calibration()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理資源"""
        if self.threaded_camera:
            self.threaded_camera.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """主函數"""
    config = PostureConfig()
    guard = PostureGuard(config)
    guard.run()


class PostureTrayApp:
    """系統托盤應用程式"""
    
    def __init__(self):
        self.config = PostureConfig()
        # 設置校準完成回調
        self.guard = PostureGuard(
            self.config, 
            use_threaded_camera=True,
            on_calibration_complete=self.on_calibration_complete_callback
        )
        
        self.is_running = True
        self.show_window = False  # 初始時隱藏視窗，校準時顯示
        self.window_created = False  # 標記窗口是否已創建
        
        # 用於線程間傳遞處理好的幀
        self.frame_queue = queue.Queue(maxsize=2)  # 只保留最新的2幀
        self.window_lock = threading.Lock()  # 保護窗口操作的鎖
        
        # 建立系統工作列圖示
        try:
            import pystray
            from PIL import Image, ImageDraw
            
            self.pystray = pystray
            self.PIL_Image = Image
            self.PIL_ImageDraw = ImageDraw
            
            # 創建圖標圖片
            icon_image = self.create_icon_image('calibrating')
            if icon_image is None:
                raise ValueError("無法創建圖標圖片")
            
            print(f"✓ 圖標圖片已創建，尺寸: {icon_image.size}")
            
            # 初始圖標狀態為未校準（黃色）
            self.icon = pystray.Icon(
                "PostureGuard",
                icon_image,
                "姿勢守衛",
                menu=self.create_menu()
            )
            print("✓ 系統托盤圖標對象已創建")
        except ImportError as e:
            print(f"警告: pystray 或 PIL 未安裝，無法使用系統托盤功能")
            print(f"錯誤: {e}")
            print("請執行: pip install pystray pillow")
            self.icon = None
            self.pystray = None
            self.PIL_Image = None
            self.PIL_ImageDraw = None
        except Exception as e:
            print(f"✗ 創建系統托盤圖標時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            self.icon = None
    
    def create_icon_image(self, status='normal'):
        """
        建立圖示圖片，根據狀態顯示不同顏色
        
        Args:
            status: 'normal' (綠色), 'bad' (紅色), 'calibrating' (黃色)
        """
        if self.PIL_Image is None:
            return None
        width, height = 64, 64
        image = self.PIL_Image.new('RGB', (width, height), (255, 255, 255))
        dc = self.PIL_ImageDraw.Draw(image)
        
        # 根據狀態選擇顏色
        if status == 'bad':
            color = (255, 0, 0)  # 紅色 - 姿勢不良
        elif status == 'calibrating':
            color = (255, 255, 0)  # 黃色 - 校準中
        else:
            color = (0, 128, 0)  # 綠色 - 正常
        
        dc.ellipse((10, 10, 54, 54), fill=color)
        return image
    
    def update_icon_status(self, status='normal'):
        """更新系統托盤圖標狀態"""
        if self.icon is not None:
            try:
                self.icon.icon = self.create_icon_image(status)
            except:
                pass  # 忽略更新錯誤
    
    def create_menu(self):
        """建立右鍵選單"""
        if self.pystray is None:
            return None
        
        # 獲取當前狀態文字
        status_text = {
            'normal': '✅ 正常',
            'bad': '⚠️ 姿勢不良',
            'calibrating': '🔄 校準中'
        }.get(self.guard.current_status, '❓ 未知')
        
        calibrated_text = '✅ 已校準' if self.guard.calibrator.is_calibrated else '❌ 未校準'
        
        return self.pystray.Menu(
            self.pystray.MenuItem(f'狀態: {status_text}', lambda: None, enabled=False),
            self.pystray.MenuItem(f'校準: {calibrated_text}', lambda: None, enabled=False),
            self.pystray.MenuItem('---', lambda: None, enabled=False),
            self.pystray.MenuItem('顯示/隱藏視窗', self.toggle_window),
            self.pystray.MenuItem('重新校準', self.trigger_calibration),
            self.pystray.MenuItem('退出', self.on_quit)
        )
    
    def update_menu(self):
        """更新菜單（動態更新狀態）"""
        if self.icon is not None:
            try:
                self.icon.menu = self.create_menu()
            except:
                pass  # 忽略更新錯誤
    
    def toggle_window(self, icon=None, item=None):
        """切換視窗顯示/隱藏"""
        with self.window_lock:
            self.show_window = not self.show_window
            if self.show_window:
                print("顯示視窗")
                self.window_created = True  # 標記需要創建窗口
            else:
                print("隱藏視窗")
                try:
                    cv2.destroyAllWindows()
                    self.window_created = False
                except:
                    pass  # 忽略 OpenCV 錯誤
    
    def trigger_calibration(self, icon=None, item=None):
        """觸發重新校準"""
        print("觸發重新校準")
        with self.window_lock:
            self.show_window = True
            self.window_created = True  # 確保窗口會顯示
        self.guard.calibrator.start_calibration()
        print(f"校準已開始，需要 {self.config.calibration_frames} 幀")
        print("校準完成後窗口會自動隱藏，只通過系統托盤圖標顯示狀態")
    
    def on_calibration_complete_callback(self):
        """校準完成回調：自動隱藏窗口"""
        print("🔄 校準完成，自動隱藏窗口...")
        with self.window_lock:
            self.show_window = False
            self.window_created = False
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("✓ 窗口已隱藏，現在只通過系統托盤圖標顯示狀態")
        print("  - 綠色圓圈 = 姿勢正常")
        print("  - 紅色圓圈 = 姿勢不良")
        print("  - 黃色圓圈 = 校準中")
        print("  右鍵點擊圖標可以重新顯示窗口或重新校準")
    
    def on_quit(self, icon=None, item=None):
        """退出應用程式"""
        self.is_running = False
        if self.icon:
            self.icon.stop()
        self.guard.cleanup()
    
    def monitor_loop(self):
        """背景監控執行緒：處理影像與 AI 運算（不涉及 OpenCV GUI）"""
        try:
            print("正在初始化攝像頭...")
            self.guard.initialize_camera()  # 初始化攝像頭
            print("攝像頭初始化成功，開始監控")
            
            frame_count = 0
            while self.is_running:
                # 使用多線程攝像頭讀取
                if self.guard.use_threaded_camera:
                    frame = self.guard.threaded_camera.read()
                    if frame is None:
                        time.sleep(0.01)  # 避免CPU空轉
                        continue
                else:
                    if not self.guard.cap or not self.guard.cap.isOpened():
                        print("攝像頭已關閉")
                        break
                    success, frame = self.guard.cap.read()
                    if not success:
                        print("無法讀取攝像頭畫面")
                        break
                
                # 執行處理邏輯（不涉及 OpenCV GUI）
                frame = self.guard.process_frame(frame)
                
                # 更新系統托盤圖標狀態和菜單
                self.update_icon_status(self.guard.current_status)
                # 每10幀更新一次菜單（避免太頻繁）
                if frame_count % 10 == 0:
                    self.update_menu()
                
                # 每100幀打印一次狀態（用於調試）
                frame_count += 1
                if frame_count % 100 == 0:
                    calibrated = "已校準" if self.guard.calibrator.is_calibrated else "未校準"
                    status_text = {
                        'normal': '正常',
                        'bad': '姿勢不良',
                        'calibrating': '校準中'
                    }.get(self.guard.current_status, '未知')
                    print(f"監控運行中... (幀數: {frame_count}, 校準: {calibrated}, 狀態: {status_text})")
                
                # 將處理好的幀放入隊列（非阻塞）
                try:
                    # 如果隊列已滿，移除舊幀
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # 隊列滿了，跳過這一幀
                
                # 給其他線程一些時間
                time.sleep(0.01)
        except Exception as e:
            print(f"監控執行緒錯誤: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("監控執行緒結束，清理資源...")
            self.guard.cleanup()  # 資源清理
    
    def update_window(self):
        """在主線程中更新 OpenCV 窗口（必須在主線程中調用）"""
        try:
            with self.window_lock:
                should_show = self.show_window
                need_create = self.window_created
            
            if should_show:
                # 從隊列獲取處理好的幀（非阻塞）
                try:
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    # 如果隊列為空，創建一個黑色幀作為佔位符
                    if need_create:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "等待攝像頭畫面...", (50, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        return  # 沒有幀，跳過這次更新
                
                try:
                    # 創建或更新窗口（在主線程中執行）
                    cv2.imshow(self.config.window_name, frame)
                    with self.window_lock:
                        self.window_created = True
                    
                    # 更新系統托盤圖標狀態（在主線程中）
                    self.update_icon_status(self.guard.current_status)
                    
                    # 處理鍵盤輸入
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        print("鍵盤觸發校準")
                        self.guard.calibrator.start_calibration()
                    elif key == 27:  # 按下 ESC 隱藏視窗而非退出
                        with self.window_lock:
                            self.show_window = False
                            self.window_created = False
                        cv2.destroyAllWindows()
                        print("ESC 鍵：隱藏視窗")
                except Exception as e:
                    # 忽略 OpenCV 窗口相關錯誤
                    print(f"窗口顯示錯誤: {e}")
            else:
                # 窗口隱藏時，確保窗口已關閉
                if need_create:
                    try:
                        cv2.destroyAllWindows()
                        with self.window_lock:
                            self.window_created = False
                    except:
                        pass
        except Exception as e:
            print(f"更新窗口錯誤: {e}")
    
    def run(self):
        """啟動程式"""
        if self.icon is None:
            print("系統托盤功能不可用，使用標準模式運行")
            self.guard.run()
            return
        
        print("=" * 50)
        print("啟動系統托盤應用程式...")
        print("提示：右鍵點擊系統托盤圖標可以顯示/隱藏窗口或重新校準")
        print("注意：在 macOS 上，圖標會顯示在菜單欄右上角")
        print("=" * 50)
        print("如果看不到系統托盤圖標：")
        print("  1. 檢查菜單欄右上角（可能被其他圖標遮擋）")
        print("  2. 程序會自動顯示窗口，您可以直接使用窗口操作")
        print("  3. 按 'C' 鍵可以重新校準，按 ESC 鍵可以隱藏窗口")
        print("=" * 50)
        
        # 啟動背景監控執行緒（處理攝像頭和 AI）
        monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        monitor_thread.start()
        print("✓ 監控執行緒已啟動")
        
        # 等待一下讓監控線程初始化
        time.sleep(0.5)
        
        # 如果未校準，自動顯示窗口進行校準
        # 如果已校準，保持窗口隱藏
        if not self.guard.calibrator.is_calibrated:
            with self.window_lock:
                self.show_window = True
                self.window_created = True
            print("✓ 窗口已自動顯示（進行校準）")
            print("  校準完成後窗口會自動隱藏，只通過系統托盤圖標顯示狀態")
        else:
            print("✓ 已校準，窗口保持隱藏狀態")
            print("  只通過系統托盤圖標顯示狀態（綠色=正常，紅色=不良）")
        
        # 在 macOS 上，嘗試使用 run_detached（如果可用）
        # 否則使用後台線程（可能不會顯示圖標）
        icon_thread = None
        use_detached = False
        
        if hasattr(self.icon, 'run_detached'):
            try:
                print("嘗試使用 run_detached 模式...")
                self.icon.run_detached()
                use_detached = True
                print("✓ 系統托盤圖標已啟動（detached 模式）")
            except Exception as e:
                print(f"⚠️ run_detached 失敗: {e}，使用後台線程模式")
                use_detached = False
        
        if not use_detached:
            # 回退方案：在後台線程中運行系統托盤圖標
            def run_icon():
                try:
                    print("系統托盤圖標線程啟動...")
                    # 在 macOS 上，這可能不會顯示圖標，但至少不會崩潰
                    self.icon.run()
                except Exception as e:
                    print(f"✗ 系統托盤錯誤: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self.is_running = False
            
            icon_thread = threading.Thread(target=run_icon, daemon=False)
            icon_thread.start()
            print("✓ 系統托盤圖標線程已啟動（後台模式）")
            print("⚠️ 注意：在 macOS 上，後台線程模式可能不會顯示圖標")
            print("   如果看不到圖標，請檢查菜單欄右上角，或使用窗口模式")
        
        # 主線程處理 OpenCV 窗口（必須在主線程中）
        try:
            print("主線程開始處理 OpenCV 窗口...")
            while self.is_running:
                # 更新窗口（在主線程中執行）
                self.update_window()
                # 給其他線程一些時間
                time.sleep(0.033)  # 約 30 FPS
        except KeyboardInterrupt:
            print("收到中斷信號")
        finally:
            # 確保清理
            print("正在退出...")
            self.is_running = False
            
            # 等待圖標線程結束（如果使用後台線程）
            if icon_thread is not None and icon_thread.is_alive():
                if self.icon:
                    try:
                        self.icon.stop()
                    except:
                        pass
                icon_thread.join(timeout=1.0)
            elif use_detached and self.icon:
                try:
                    self.icon.stop()
                except:
                    pass
            
            time.sleep(0.2)  # 給線程一些時間清理
            self.guard.cleanup()
            try:
                cv2.destroyAllWindows()
            except:
                pass
            print("已退出")


def main_tray():
    """系統托盤模式主函數"""
    app = PostureTrayApp()
    app.run()


if __name__ == '__main__':
    import sys
    # 如果命令行參數包含 --tray，則使用系統托盤模式
    if '--tray' in sys.argv or '-t' in sys.argv:
        main_tray()
    else:
        main()

