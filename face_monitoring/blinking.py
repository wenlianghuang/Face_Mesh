import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import math
import os
from datetime import datetime
# 1. 初始化 MediaPipe Face Landmarker
# 降低檢測閾值以提高對側臉和部分臉部的檢測能力
base_options = python.BaseOptions(model_asset_path='../face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,  # 可以改為 2 來檢測多張臉，但這裡保持 1
    min_face_detection_confidence=0.3,  # 降低從 0.5 到 0.3，更容易檢測側臉
    min_face_presence_confidence=0.3,   # 降低從 0.5 到 0.3，提高對部分臉部的容忍度
    min_tracking_confidence=0.3         # 降低從 0.5 到 0.3，提高追蹤穩定性
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# 2. 定義臉部網格連接（簡化版 - 主要輪廓）
# MediaPipe Face Mesh 有 468 個地標點，這裡定義一些主要的連接線
FACE_CONNECTIONS = [
    # 臉部輪廓
    [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
    # 左眉毛
    [276, 283], [283, 282], [282, 295], [295, 285], [285, 336], [336, 296], [296, 334], [334, 293], [293, 300], [300, 276],
    # 右眉毛
    [46, 53], [53, 52], [52, 65], [65, 55], [55, 70], [70, 63], [63, 105], [105, 66], [66, 107], [107, 46],
    # 左眼
    [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133], [133, 173], [173, 157], [157, 158], [158, 159], [159, 160], [160, 161], [161, 246], [246, 33],
    # 右眼
    [263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381], [381, 382], [382, 362], [362, 398], [398, 384], [384, 385], [385, 386], [386, 387], [387, 388], [388, 466], [466, 263],
    # 鼻子
    [1, 2], [2, 5], [5, 4], [4, 6], [6, 19], [19, 20], [20, 94], [94, 2],
    # 嘴巴外圍
    [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 320], [320, 307], [307, 375], [375, 321], [321, 308], [308, 324], [324, 318], [318, 61],
    # 嘴巴內圍
    [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402], [402, 318], [318, 324], [324, 308], [308, 78]
]

# 3. 創建 blinking 文件夾和日誌文件
blinking_dir = "blinking"
os.makedirs(blinking_dir, exist_ok=True)

# 創建日誌文件，文件名包含時間戳
log_filename = os.path.join(blinking_dir, f"blink_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_file = open(log_filename, 'w', encoding='utf-8')
start_time = datetime.now()
log_file.write(f"=== 眨眼檢測開始 ===\n")
log_file.write(f"開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write(f"{'='*50}\n\n")
log_file.flush()
print(f"日誌文件已創建: {log_filename}")

# 4. 開啟攝影機
cap = cv2.VideoCapture(0)

def draw_landmarks(image, landmarks, connections):
    """繪製臉部地標和連接線"""
    h, w = image.shape[:2]
    
    # 繪製連接線
    for connection in connections:
        if connection[0] < len(landmarks) and connection[1] < len(landmarks):
            pt1 = (int(landmarks[connection[0]].x * w), int(landmarks[connection[0]].y * h))
            pt2 = (int(landmarks[connection[1]].x * w), int(landmarks[connection[1]].y * h))
            cv2.line(image, pt1, pt2, (0, 255, 0), 1)
    
    # 繪製地標點
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
def calculate_ear(landmarks, eye_indices, w, h):
    """計算眼睛寬高比 (EAR - Eye Aspect Ratio)
    返回 (ear_value, is_valid)
    is_valid 表示眼睛地標點是否完整可見
    """
    # 轉換為實際像素座標
    pts = []
    for idx in eye_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            # 檢查地標點是否在合理範圍內（避免完全不可見的情況）
            if 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
                pts.append((lm.x * w, lm.y * h))
            else:
                return (0.0, False)
        else:
            return (0.0, False)
    
    if len(pts) < 6:
        return (0.0, False)
    
    # EAR 計算公式（標準公式）：
    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    # 其中索引順序：[p1, p2, p3, p4, p5, p6]
    # 垂直距離 1 (p2-p6)
    v1 = math.dist(pts[1], pts[5])
    # 垂直距離 2 (p3-p5)
    v2 = math.dist(pts[2], pts[4])
    # 水平距離 (p1-p4)
    h_dist = math.dist(pts[0], pts[3])
    
    # 避免除以零
    if h_dist == 0:
        return (0.0, False)
    
    ear = (v1 + v2) / (2.0 * h_dist)
    return (ear, True)

# MediaPipe 臉部地標索引（恢復到原來能工作的版本）
# 左眼地標索引 (MediaPipe 順序)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
# 右眼地標索引
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 眨眼檢測參數
blink_count = 0           # 總眨眼次數
is_closed = False         # 當前眼睛是否處於閉合狀態
EAR_THRESHOLD = 0.18      # EAR 閾值（降低以提高敏感度，正常睜眼約 0.25-0.30）
EAR_SMOOTH_FRAMES = 2     # EAR 平滑幀數（減少平滑以更快反應）

# EAR 歷史記錄（用於平滑處理）
ear_history = []
max_history = 5  # 保留最近 5 幀的 EAR 值
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # 轉為 RGB（MediaPipe 需要 RGB，而 OpenCV 預設是 BGR）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 轉換為 MediaPipe Image 格式
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # 處理影像
    detection_result = face_landmarker.detect(mp_image)
    
    # 4. 畫出地標和檢測眨眼
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            h, w = image.shape[:2]
            
            # 計算左右眼的 EAR（現在返回 (ear_value, is_valid)）
            left_ear, left_valid = calculate_ear(face_landmarks, LEFT_EYE, w, h)
            right_ear, right_valid = calculate_ear(face_landmarks, RIGHT_EYE, w, h)
            
            # 改進：如果只有一隻眼睛可見（側臉情況），使用可見的那隻眼睛
            # 如果兩隻眼睛都不可見，跳過這一幀
            if not left_valid and not right_valid:
                # 兩隻眼睛都檢測不到，跳過
                continue
            elif not left_valid:
                # 只有右眼可見（左側臉）
                avg_ear = right_ear
                single_eye_mode = "右眼"
            elif not right_valid:
                # 只有左眼可見（右側臉）
                avg_ear = left_ear
                single_eye_mode = "左眼"
            else:
                # 兩隻眼睛都可見（正臉）
                avg_ear = (left_ear + right_ear) / 2.0
                single_eye_mode = "雙眼"
            
            # 將當前 EAR 加入歷史記錄
            ear_history.append(avg_ear)
            if len(ear_history) > max_history:
                ear_history.pop(0)
            
            # 計算平滑後的 EAR（使用最近幾幀的平均值）
            if len(ear_history) >= EAR_SMOOTH_FRAMES:
                smoothed_ear = sum(ear_history[-EAR_SMOOTH_FRAMES:]) / EAR_SMOOTH_FRAMES
            else:
                smoothed_ear = avg_ear  # 歷史記錄不足時使用當前值
            
            # 在畫面上顯示 EAR 值（方便調試）
            cv2.putText(image, f"EAR: {smoothed_ear:.3f} (raw: {avg_ear:.3f}) [{single_eye_mode}]", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if left_valid and right_valid:
                cv2.putText(image, f"Left: {left_ear:.3f} Right: {right_ear:.3f}", (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            elif left_valid:
                cv2.putText(image, f"Left: {left_ear:.3f} (Right: N/A)", (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            elif right_valid:
                cv2.putText(image, f"(Left: N/A) Right: {right_ear:.3f}", (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            cv2.putText(image, f"Threshold: {EAR_THRESHOLD:.2f}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 改進的眨眼檢測邏輯
            # 檢測完整的眨眼過程：睜眼 → 閉眼 → 睜眼
            # 使用原始 EAR 值而不是平滑值，以更快反應
            if avg_ear < EAR_THRESHOLD:
                # 眼睛閉合
                if not is_closed:
                    # 從睜眼狀態轉換到閉眼狀態
                    is_closed = True
                    cv2.putText(image, "EYES CLOSED", (50, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    if left_valid and right_valid:
                        eye_info = f"左: {left_ear:.4f}, 右: {right_ear:.4f}"
                    else:
                        eye_info = f"[{single_eye_mode}] {avg_ear:.4f}"
                    print(f"眼睛閉合 - EAR: {avg_ear:.4f} ({eye_info})")
            else:
                # 眼睛睜開
                if is_closed:
                    # 從閉眼狀態轉換到睜眼狀態 = 完成一次眨眼
                    blink_count += 1
                    blink_time = datetime.now()
                    elapsed_time = (blink_time - start_time).total_seconds()
                    if left_valid and right_valid:
                        eye_info = f"左眼: {left_ear:.4f} | 右眼: {right_ear:.4f}"
                    else:
                        eye_info = f"[{single_eye_mode}] EAR: {avg_ear:.4f}"
                    log_file.write(f"[眨眼 #{blink_count}] 時間: {blink_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | 經過時間: {elapsed_time:.2f} 秒 | EAR: {avg_ear:.4f} | {eye_info}\n")
                    log_file.flush()
                    print(f"✓ 眨眼 #{blink_count} 已記錄 - {blink_time.strftime('%H:%M:%S.%f')[:-3]} (EAR: {avg_ear:.4f})")
                    is_closed = False
                
                # 顯示眨眼計數
                cv2.putText(image, f"Blinks: {blink_count}", (10, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # 繪製所有地標點
            for landmark in face_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            
            # 繪製簡化的臉部輪廓（使用主要地標點）
            if len(face_landmarks) >= 468:
                # 臉部外圍輪廓（簡化）
                contour_points = [
                    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 
                    361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 
                    176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 
                    162, 21, 54, 103, 67, 109
                ]
                points = []
                for idx in contour_points:
                    if idx < len(face_landmarks):
                        x = int(face_landmarks[idx].x * w)
                        y = int(face_landmarks[idx].y * h)
                        points.append([x, y])
                
                if len(points) > 0:
                    pts = np.array(points, np.int32)
                    cv2.polylines(image, [pts], True, (0, 255, 0), 1)
    else:
        # 未檢測到臉部
        cv2.putText(image, "No Face Detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:  # 按 Esc 退出
        break

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()

# 記錄結束時間和統計信息
end_time = datetime.now()
total_duration = (end_time - start_time).total_seconds()
log_file.write(f"\n{'='*50}\n")
log_file.write(f"=== 眨眼檢測結束 ===\n")
log_file.write(f"結束時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write(f"總運行時間: {total_duration:.2f} 秒\n")
log_file.write(f"總眨眼次數: {blink_count}\n")
if total_duration > 0:
    log_file.write(f"平均眨眼頻率: {blink_count / (total_duration / 60):.2f} 次/分鐘\n")
log_file.close()
print(f"\n檢測結束！總共記錄 {blink_count} 次眨眼。")
print(f"日誌已保存至: {log_filename}")
