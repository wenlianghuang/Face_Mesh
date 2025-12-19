import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time

# 1. 初始化 MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 2. 定義標準 3D 臉部模型點 (引用自你的 head_pose.py)
model_points = np.array([
    (0.0, 0.0, 0.0),             # 鼻尖
    (0.0, -330.0, -65.0),        # 下巴
    (-225.0, 170.0, -135.0),     # 左眼左角
    (225.0, 170.0, -135.0),      # 右眼右角
    (-150.0, -150.0, -125.0),    # 左口角
    (150.0, -150.0, -125.0)      # 右口角
], dtype=np.float64)

# 3. 狀態變數與緩衝區
is_calibrated = False
calibrating_frames = 0
baseline_pitch = 0
baseline_z = 0
baseline_nose_offset = 0  # 基準鼻子偏移值
posture_history = deque(maxlen=60)  # 儲存最近約 2 秒的狀態 (30 FPS)

def get_head_pose(face_landmarks, w, h):
    """計算頭部姿態邏輯 - 使用鼻子相對於眼睛中心線的位置"""
    # 獲取關鍵點座標
    nose_y = face_landmarks[1].y * h  # 鼻尖 Y 座標
    left_eye_y = face_landmarks[33].y * h  # 左眼 Y 座標
    right_eye_y = face_landmarks[263].y * h  # 右眼 Y 座標
    chin_y = face_landmarks[152].y * h  # 下巴 Y 座標
    
    # 計算眼睛中心線的 Y 座標
    eye_center_y = (left_eye_y + right_eye_y) / 2
    
    # 計算鼻子相對於眼睛中心線的偏移（像素）
    # 正值表示鼻子在眼睛下方（正常/低頭），負值表示鼻子在眼睛上方（抬頭）
    nose_offset = nose_y - eye_center_y
    
    # 計算眼睛到下巴的距離作為參考長度（用於標準化）
    face_height = chin_y - eye_center_y
    
    # 標準化偏移量並轉換為角度
    # 使用反正切函數將偏移轉換為角度
    if face_height > 0:
        # 計算標準化偏移（相對於臉部高度）
        normalized_offset = nose_offset / face_height
        # 轉換為角度（約 -45 到 +45 度的範圍）
        # 使用更大的係數來放大角度變化，使其更敏感
        pitch_angle = np.degrees(np.arctan(normalized_offset * 3))
    else:
        pitch_angle = 0
    
    # 保留原有的 solvePnP 計算用於視覺化
    image_points = np.array([
        (face_landmarks[1].x * w, face_landmarks[1].y * h),
        (face_landmarks[152].x * w, face_landmarks[152].y * h),
        (face_landmarks[33].x * w, face_landmarks[33].y * h),
        (face_landmarks[263].x * w, face_landmarks[263].y * h),
        (face_landmarks[61].x * w, face_landmarks[61].y * h),
        (face_landmarks[291].x * w, face_landmarks[291].y * h)
    ], dtype=np.float64)

    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4,1))

    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    
    return pitch_angle, face_landmarks[1].z, rvec, tvec, camera_matrix, dist_coeffs, nose_offset, eye_center_y

# 4. 主迴圈
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]
        current_pitch, current_z, rvec, tvec, cam_mtx, dist, nose_offset, eye_center_y = get_head_pose(face_landmarks, w, h)

        # A. 校準邏輯
        if not is_calibrated and calibrating_frames > 0:
            baseline_pitch += current_pitch
            baseline_z += current_z
            baseline_nose_offset += nose_offset
            calibrating_frames -= 1
            cv2.putText(frame, f"Calibrating... {calibrating_frames}", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if calibrating_frames == 0:
                baseline_pitch /= 60
                baseline_z /= 60
                baseline_nose_offset /= 60
                is_calibrated = True
        
        # B. 監控邏輯
        elif is_calibrated:
            # 計算各種差異值
            pitch_diff = current_pitch - baseline_pitch
            nose_offset_diff = nose_offset - baseline_nose_offset  # 鼻子偏移的變化（像素）
            z_diff = baseline_z - current_z  # Z 越小代表離相機越近
            
            # 判斷標準：
            # 1. 抬頭：pitch_diff < -12 度 或 nose_offset_diff < -15 像素（鼻子向上移動）
            # 2. 低頭：pitch_diff > 10 度 或 nose_offset_diff > 20 像素（鼻子向下移動，使用更敏感的閾值）
            # 3. 前傾：z_diff > 0.05
            is_looking_up = pitch_diff < -12 or nose_offset_diff < -15
            is_looking_down = pitch_diff > 10 or nose_offset_diff > 20  # 低頭使用更敏感的閾值
            is_leaning_forward = z_diff > 0.05
            
            is_bad = is_looking_up or is_looking_down or is_leaning_forward
            posture_history.append(is_bad)
            
            # 視覺化數值
            color = (0, 255, 0)
            status_text = "Good Posture"
            
            # 觸發警告 (當最近 2 秒有 80% 時間姿勢不良)
            if sum(posture_history) / len(posture_history) > 0.8:
                color = (0, 0, 255)
                # 根據具體情況判斷是抬頭、低頭還是前傾
                if is_looking_down:
                    status_text = "BAD POSTURE! HEAD TOO LOW"
                elif is_looking_up:
                    status_text = "BAD POSTURE! HEAD TOO HIGH"
                elif is_leaning_forward:
                    status_text = "BAD POSTURE! SIT STRAIGHT"
                else:
                    status_text = "BAD POSTURE!"
            
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Pitch Diff: {pitch_diff:+.1f}°", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"Nose Offset: {nose_offset:+.1f}px", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"Nose Diff: {nose_offset_diff:+.1f}px", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"Z Diff: {z_diff:.3f}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # 視覺化：畫出眼睛中心線和鼻子位置
            eye_center_y_int = int(eye_center_y)
            nose_y = int(face_landmarks[1].y * h)
            nose_x = int(face_landmarks[1].x * w)
            cv2.line(frame, (0, eye_center_y_int), (w, eye_center_y_int), (255, 255, 0), 2)  # 眼睛中心線（黃色）
            cv2.circle(frame, (nose_x, nose_y), 8, color, -1)  # 鼻子位置（大圓點）
            cv2.circle(frame, (nose_x, nose_y), 8, (255, 255, 255), 2)  # 白色外圈

            # 畫出朝向線 (視覺化輔助)
            (nose_end, _) = cv2.projectPoints(np.array([(0.0, 0.0, 300.0)]), rvec, tvec, cam_mtx, dist)
            p1 = (nose_x, nose_y)
            p2 = (int(nose_end[0][0][0]), int(nose_end[0][0][1]))
            cv2.line(frame, p1, p2, color, 2)

    else:
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 操作提示
    if not is_calibrated:
        cv2.putText(frame, "Press 'c' to Calibrate", (w-300, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow('Posture Guard v1.0', frame)
    
    key = cv2.waitKey(5) & 0xFF
    if key == 27: break
    elif key == ord('c'):
        is_calibrated = False
        calibrating_frames = 60
        baseline_pitch = 0
        baseline_z = 0
        baseline_nose_offset = 0

cap.release()
cv2.destroyAllWindows()