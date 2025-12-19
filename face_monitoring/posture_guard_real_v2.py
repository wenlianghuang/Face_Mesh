import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Any


@dataclass
class PostureConfig:
    """姿勢與專注度監控配置參數"""
    # 模型路徑
    model_path: str = 'face_landmarker.task'
    
    # 校準參數
    calibration_frames: int = 60  # 校準所需的幀數
    
    # 檢測閾值 (角度單位)
    pitch_up_threshold: float = -12.0  # 抬頭角度閾值（度）
    pitch_down_threshold: float = 12.0  # 低頭角度閾值（度）
    yaw_threshold: float = 25.0  # 左右轉頭角度（分心）
    z_forward_threshold: float = 0.05  # 前傾 Z 軸閾值
    
    # 警告觸發條件
    bad_posture_ratio: float = 0.8  # 姿勢不良比例閾值（80%）
    history_size: int = 30  # 歷史記錄大小（約 1 秒，30 FPS）
    
    # 視覺化參數
    window_name: str = 'Posture & Attention Guard'
    text_color_good: Tuple[int, int, int] = (0, 255, 0)
    text_color_bad: Tuple[int, int, int] = (0, 0, 255)


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
        計算頭部姿態（使用 3D solvePnP 方法）
        
        Returns:
            dict: 包含 pitch, yaw, z, rvec, tvec, cam_matrix, dist_coeffs 的字典
        """
        # 提取 2D 關鍵點
        image_points = np.array([
            (face_landmarks[self.NOSE_TIP].x * w, face_landmarks[self.NOSE_TIP].y * h),
            (face_landmarks[self.CHIN].x * w, face_landmarks[self.CHIN].y * h),
            (face_landmarks[self.LEFT_EYE].x * w, face_landmarks[self.LEFT_EYE].y * h),
            (face_landmarks[self.RIGHT_EYE].x * w, face_landmarks[self.RIGHT_EYE].y * h),
            (face_landmarks[self.LEFT_MOUTH].x * w, face_landmarks[self.LEFT_MOUTH].y * h),
            (face_landmarks[self.RIGHT_MOUTH].x * w, face_landmarks[self.RIGHT_MOUTH].y * h)
        ], dtype=np.float64)

        # 設定相機矩陣 (假設無畸變)
        focal_length = w
        center = (w/2, h/2)
        cam_matrix = np.array([
            [focal_length, 0, center[0]], 
            [0, focal_length, center[1]], 
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        # SolvePnP 得到旋轉向量與位移向量
        _, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS, 
            image_points, 
            cam_matrix, 
            dist_coeffs
        )
        rmat, _ = cv2.Rodrigues(rvec)
        
        # 從旋轉矩陣提取歐拉角
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        
        return {
            'pitch': pitch,
            'yaw': yaw,
            'z': face_landmarks[self.NOSE_TIP].z,
            'rvec': rvec,
            'tvec': tvec,
            'cam_matrix': cam_matrix,
            'dist_coeffs': dist_coeffs
        }


class PostureCalibrator:
    """姿勢校準器"""
    
    def __init__(self, config: PostureConfig):
        self.config = config
        self.is_calibrated = False
        self.calibrating_frames = 0
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self.baseline_z = 0.0
        self._accumulated_pitch = 0.0
        self._accumulated_yaw = 0.0
        self._accumulated_z = 0.0
    
    def start_calibration(self):
        """開始校準"""
        self.is_calibrated = False
        self.calibrating_frames = self.config.calibration_frames
        self._accumulated_pitch = 0.0
        self._accumulated_yaw = 0.0
        self._accumulated_z = 0.0
    
    def update(self, pitch: float, yaw: float, z: float) -> bool:
        """
        更新校準數據
        
        Returns:
            bool: 是否完成校準
        """
        if self.calibrating_frames > 0:
            self._accumulated_pitch += pitch
            self._accumulated_yaw += yaw
            self._accumulated_z += z
            self.calibrating_frames -= 1
            
            if self.calibrating_frames == 0:
                # 計算平均值
                frames_count = self.config.calibration_frames
                self.baseline_pitch = self._accumulated_pitch / frames_count
                self.baseline_yaw = self._accumulated_yaw / frames_count
                self.baseline_z = self._accumulated_z / frames_count
                self.is_calibrated = True
                return True
        return False
    
    def get_remaining_frames(self) -> int:
        """獲取剩餘校準幀數"""
        return self.calibrating_frames


class PostureMonitor:
    """姿勢監控器（整合姿勢與轉頭監控）"""
    
    def __init__(self, config: PostureConfig):
        self.config = config
        self.posture_history = deque(maxlen=config.history_size)
    
    def evaluate(self, pitch_diff: float, yaw_diff: float, z_diff: float) -> dict:
        """
        評估當前姿勢
        
        Returns:
            dict: 包含 is_bad, is_looking_up, is_looking_down, is_leaning_forward, is_looking_away 的字典
        """
        is_looking_away = abs(yaw_diff) > self.config.yaw_threshold
        is_looking_down = pitch_diff > self.config.pitch_down_threshold
        is_looking_up = pitch_diff < self.config.pitch_up_threshold
        is_leaning_forward = z_diff > self.config.z_forward_threshold
        
        is_bad = is_looking_away or is_looking_down or is_looking_up or is_leaning_forward
        self.posture_history.append(is_bad)
        
        return {
            'is_bad': is_bad,
            'is_looking_up': is_looking_up,
            'is_looking_down': is_looking_down,
            'is_leaning_forward': is_leaning_forward,
            'is_looking_away': is_looking_away
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
        
        if evaluation['is_looking_away']:
            return "DISTRACTED! LOOKING AWAY"
        elif evaluation['is_looking_down']:
            return "HEAD TOO LOW"
        elif evaluation['is_looking_up']:
            return "HEAD TOO HIGH"
        elif evaluation['is_leaning_forward']:
            return "SIT STRAIGHT (FORWARD)"
        else:
            return "BAD POSTURE"


class Visualizer:
    """視覺化器"""
    
    def __init__(self, config: PostureConfig):
        self.config = config
    
    def draw_calibration_status(self, frame: np.ndarray, remaining_frames: int):
        """繪製校準狀態"""
        h, w = frame.shape[:2]
        text = f"Calibrating... {remaining_frames}"
        cv2.putText(frame, text, (w//2-100, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    def draw_posture_status(self, frame: np.ndarray, status_text: str, is_bad: bool):
        """繪製姿勢狀態（只顯示狀態文字）"""
        color = self.config.text_color_bad if is_bad else self.config.text_color_good
        font_scale = 1.5
        thickness = 3
        cv2.putText(frame, status_text, (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def draw_head_direction(self, frame: np.ndarray, face_landmarks, pose_data: dict, w: int, h: int, color: Tuple[int, int, int]):
        """繪製頭部朝向線（視覺化轉向）"""
        # 計算鼻子尖端在圖像中的位置
        nose_point = (int(face_landmarks[1].x * w), int(face_landmarks[1].y * h))
        
        # 投影一個 3D 點到 2D 圖像平面（沿著頭部朝向方向）
        (nose_end, _) = cv2.projectPoints(
            np.array([(0.0, 0.0, 500.0)]), 
            pose_data['rvec'], 
            pose_data['tvec'], 
            pose_data['cam_matrix'], 
            pose_data['dist_coeffs']
        )
        end_point = (int(nose_end[0][0][0]), int(nose_end[0][0][1]))
        
        # 繪製線條
        cv2.line(frame, nose_point, end_point, color, 2)
    
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
    
    def __init__(self, config: Optional[PostureConfig] = None):
        self.config = config or PostureConfig()
        self.detector = HeadPoseDetector(self.config.model_path)
        self.calibrator = PostureCalibrator(self.config)
        self.monitor = PostureMonitor(self.config)
        self.visualizer = Visualizer(self.config)
        self.cap = None
    
    def initialize_camera(self, camera_id: int = 0):
        """初始化攝像頭"""
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
            if remaining > 0:
                self.calibrator.update(
                    pose_data['pitch'],
                    pose_data['yaw'],
                    pose_data['z']
                )
                self.visualizer.draw_calibration_status(frame, remaining)
            self.visualizer.draw_calibration_prompt(frame)
            return frame
        
        # 監控邏輯
        pitch_diff = pose_data['pitch'] - self.calibrator.baseline_pitch
        yaw_diff = pose_data['yaw'] - self.calibrator.baseline_yaw
        z_diff = self.calibrator.baseline_z - pose_data['z']
        
        evaluation = self.monitor.evaluate(pitch_diff, yaw_diff, z_diff)
        should_warn = self.monitor.should_trigger_warning()
        status_text = self.monitor.get_status_text(evaluation)
        
        # 繪製姿勢狀態文字
        color = self.config.text_color_bad if should_warn else self.config.text_color_good
        self.visualizer.draw_posture_status(frame, status_text, should_warn)
        
        # 繪製頭部朝向線
        self.visualizer.draw_head_direction(frame, face_landmarks, pose_data, w, h, color)
        
        return frame
    
    def run(self):
        """運行主循環"""
        if self.cap is None:
            self.initialize_camera()
        
        try:
            while self.cap.isOpened():
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
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """主函數"""
    config = PostureConfig()
    guard = PostureGuard(config)
    guard.run()


if __name__ == '__main__':
    main()

