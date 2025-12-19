import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

class GazeTracker:
    # 關鍵點位索引
    L_IRIS_CENTER = 468
    L_EYE_LEFT = 33
    L_EYE_RIGHT = 133
    L_EYE_TOP = 159
    L_EYE_BOTTOM = 145

    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_gaze_ratio(self, landmarks, w, h):
        """計算視線偏移比率"""
        # 1. 取得瞳孔與眼眶邊界座標
        pupil = landmarks[self.L_IRIS_CENTER]
        left = landmarks[self.L_EYE_LEFT]
        right = landmarks[self.L_EYE_RIGHT]
        
        # 2. 計算水平比率 (Horizontal Ratio)
        # 計算瞳孔到左眼角的距離 / 眼眶總寬度
        # 值越小代表越往右看 (靠近左眼角)，值越大代表越往左看 (靠近右眼角)
        total_width = math.dist([left.x, left.y], [right.x, right.y])
        pupil_dist = math.dist([pupil.x, pupil.y], [left.x, left.y])
        
        horizontal_ratio = pupil_dist / total_width
        return horizontal_ratio

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            h, w = frame.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.detector.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                
                # 計算比率
                ratio = self.get_gaze_ratio(landmarks, w, h)
                
                # 判定邏輯 (閥值可根據測試調整)
                if ratio < 0.42:
                    text = "Looking Right"
                elif ratio > 0.52:
                    text = "Looking Left"
                else:
                    text = "Looking Center"

                cv2.putText(frame, f"Ratio: {ratio:.2f}", (50, 50), 2, 1, (255, 255, 0), 2)
                cv2.putText(frame, text, (50, 100), 2, 1.5, (0, 255, 0), 3)

                # 畫出瞳孔位置方便觀察
                px, py = int(landmarks[self.L_IRIS_CENTER].x * w), int(landmarks[self.L_IRIS_CENTER].y * h)
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

            cv2.imshow("Gaze Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = GazeTracker('face_landmarker.task') # 確保模型路徑正確
    tracker.run()