import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. 初始化 MediaPipe Face Landmarker (Tasks API)
base_options = python.BaseOptions(model_asset_path='../face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 2. 定義標準 3D 臉部模型點 (World Coordinates)
# 這些是標準人臉的相對座標值
model_points = np.array([
    (0.0, 0.0, 0.0),             # 鼻尖 (Nose tip)
    (0.0, -330.0, -65.0),        # 下巴 (Chin)
    (-225.0, 170.0, -135.0),     # 左眼左角 (Left eye left corner)
    (225.0, 170.0, -135.0),      # 右眼右角 (Right eye right corner)
    (-150.0, -150.0, -125.0),    # 左口角 (Left mouth corner)
    (150.0, -150.0, -125.0)      # 右口角 (Right mouth corner)
], dtype=np.float64)

# 3. 開啟攝影機
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # 執行偵測
    result = detector.detect(mp_image)

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]
        
        # 4. 提取對應的 2D 影像點
        image_points = np.array([
            (face_landmarks[1].x * w, face_landmarks[1].y * h),     # 鼻尖
            (face_landmarks[152].x * w, face_landmarks[152].y * h), # 下巴
            (face_landmarks[33].x * w, face_landmarks[33].y * h),   # 左眼
            (face_landmarks[263].x * w, face_landmarks[263].y * h), # 右眼
            (face_landmarks[61].x * w, face_landmarks[61].y * h),   # 左口角
            (face_landmarks[291].x * w, face_landmarks[291].y * h)  # 右口角
        ], dtype=np.float64)

        # 5. 設定相機矩陣 (Camera Matrix) - 假設焦距等於影像寬度
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4,1)) # 假設無鏡頭畸變

        # 6. 求解 PnP (得到旋轉向量與位移向量)
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        # 7. 計算歐拉角 (Pitch, Yaw, Roll)
        rmat, _ = cv2.Rodrigues(rvec)
        
        # 從旋轉矩陣提取歐拉角 (ZYX 順序: Roll-Pitch-Yaw)
        # 使用標準的旋轉矩陣到歐拉角轉換公式
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(-rmat[2, 0], sy)  # 繞 Y 軸旋轉 (上下點頭)
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0])  # 繞 Z 軸旋轉 (左右轉頭)
            roll = np.arctan2(rmat[2, 1], rmat[2, 2])  # 繞 X 軸旋轉 (左右傾斜)
        else:
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = np.arctan2(-rmat[0, 1], rmat[1, 1])
            roll = 0
        
        # 轉換為度數
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        # 8. 視覺化：在鼻尖畫一條 3D 投影線 (指向前方)
        # 定義一個在 3D 空間中，從鼻子往外延伸 500 單位的點
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rvec, tvec, camera_matrix, dist_coeffs)
        
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        
        # 顯示角度數值
        cv2.putText(frame, f"Pitch: {int(pitch)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw  : {int(yaw)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll : {int(roll)}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Head Pose Estimation', frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()