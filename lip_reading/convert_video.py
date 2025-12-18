import os
import cv2
import glob
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_mouth_roi(frame):
    """Xác định vùng miệng từ landmarks và trả về tọa độ"""
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None
    
    # Landmark indices cho vùng miệng (theo MediaPipe)
    mouth_landmarks = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146
    ]
    
    h, w = frame.shape[:2]
    landmarks = []
    for idx in mouth_landmarks:
        landmark = results.multi_face_landmarks[0].landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        landmarks.append((x, y))
    
    x_coords = [p[0] for p in landmarks]
    y_coords = [p[1] for p in landmarks]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)
    
    width = x2 - x1
    height = y2 - y1
    x1 = max(0, x1 - int(0.1 * width))
    x2 = min(w, x2 + int(0.1 * width))
    y1 = max(0, y1 - int(0.1 * height))
    y2 = min(h, y2 + int(0.1 * height))
    
    # Đảm bảo kích thước gần vuông (96x96)
    width = x2 - x1
    height = y2 - y1
    if width > height:
        diff = width - height
        y1 = max(0, y1 - diff // 2)
        y2 = min(h, y2 + diff // 2)
    else:
        diff = height - width
        x1 = max(0, x1 - diff // 2)
        x2 = min(w, x2 + diff // 2)
    
    return (y1, y2, x1, x2)

def extract_opencv_with_landmark(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    
    if not cap.isOpened():
        print(f"Error opening video file: {filename}")
        return None
    
    # Get total frames for progress
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Thử detect mặt trong 10 frame đầu
    roi = None
    for i in range(min(10, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        roi = get_mouth_roi(frame)
        if roi is not None:
            break

    if roi is None:
        print(f"Cannot detect face in {filename}")
        cap.release()
        return None
    
    y1, y2, x1, x2 = roi
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Quay lại đầu video

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        mouth_roi = frame[y1:y2, x1:x2]
        if mouth_roi.shape[0] == 0 or mouth_roi.shape[1] == 0:
            continue
            
        if mouth_roi.shape[0] != 96 or mouth_roi.shape[1] != 96:
            mouth_roi = cv2.resize(mouth_roi, (96, 96))
        
        video.append(mouth_roi)
        frame_count += 1

    cap.release()
    
    if len(video) == 0:
        print(f"No valid frames extracted from {filename}")
        return None
        
    return np.array(video)

def main():
    # Đường dẫn dữ liệu
    basedir = 'multi_speaker_data'
    basedir_to_save = 'processed_data'
    
    if not os.path.exists(basedir):
        print(f"Error: {basedir} directory not found!")
        print("Please create the raw_data directory and add your video files.")
        return
    
    filenames = glob.glob(os.path.join(basedir, '*', '*', '*.mp4'))
    filenames.extend(glob.glob(os.path.join(basedir, '*', '*', '*.avi')))
    filenames.extend(glob.glob(os.path.join(basedir, '*', '*', '*.mov')))
    
    if len(filenames) == 0:
        print(f"No video files found in {basedir}")
        print("Please add MP4, AVI, or MOV files to the raw_data directory.")
        return
    
    print(f"Found {len(filenames)} video files to process")
    
    success_count = 0
    error_count = 0
    
    for filename in tqdm(filenames, desc="Processing videos"):
        try:
            data = extract_opencv_with_landmark(filename)
            
            if data is None:
                error_count += 1
                continue
            
            parts = filename.split(os.sep)
            category = parts[-3]
            subset = parts[-2]
            video_name = parts[-1]
            # Remove extension
            video_name = os.path.splitext(video_name)[0]
            
            path_to_save = os.path.join(basedir_to_save, category, subset, f"{video_name}.npz")
            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
            
            np.savez_compressed(path_to_save, data=data)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            error_count += 1
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Processed data saved to: {basedir_to_save}")

if __name__ == "__main__":
    main()
