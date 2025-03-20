import os
import cv2
import dlib
import numpy as np
from skvideo.io import FFmpegReader


def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    video_reader = FFmpegReader(video_path)
    frame_number = 0

    for frame in video_reader:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Chuyển đổi sang định dạng OpenCV
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_number += 1

    print(f"Đã trích xuất {frame_number} khung hình vào thư mục {output_folder}")
    return frame_number


def detect_faces(input_folder, output_folder):
    detector = dlib.get_frontal_face_detector()
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            continue  

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 1:  
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)

    print(f"Quá trình phát hiện khuôn mặt hoàn tất. Ảnh có khuôn mặt đã lưu trong '{output_folder}'")


def crop_lips(input_folder, output_folder, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            lip_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(48, 68)])

            x_left, x_right = np.min(lip_points[:, 0]), np.max(lip_points[:, 0])
            y_top, y_bottom = np.min(lip_points[:, 1]), np.max(lip_points[:, 1])

            border = 15
            x_left_new = max(0, x_left - border)
            x_right_new = min(image.shape[1], x_right + border)
            y_top_new = max(0, y_top - border)
            y_bottom_new = min(image.shape[0], y_bottom + border)

            lip_crop = image[y_top_new:y_bottom_new, x_left_new:x_right_new]
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, lip_crop)

    print(f"Quá trình cắt môi hoàn tất. Ảnh môi đã lưu trong '{output_folder}'")


def data_augmentation(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        bright_img = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        dim_img = cv2.convertScaleAbs(image, alpha=0.5, beta=0)

        mirror_img = cv2.flip(image, 1)

        gauss_img = add_gaussian_noise(image)

        rotate_neg20 = rotate_image(image, -20)

        rotate_pos20 = rotate_image(image, 20)

        cv2.imwrite(os.path.join(output_folder, f"bright_{filename}"), bright_img)
        cv2.imwrite(os.path.join(output_folder, f"dim_{filename}"), dim_img)
        cv2.imwrite(os.path.join(output_folder, f"mirror_{filename}"), mirror_img)
        cv2.imwrite(os.path.join(output_folder, f"gauss_{filename}"), gauss_img)
        cv2.imwrite(os.path.join(output_folder, f"rotate_neg20_{filename}"), rotate_neg20)
        cv2.imwrite(os.path.join(output_folder, f"rotate_pos20_{filename}"), rotate_pos20)

    print(f"Quá trình data augmentation hoàn tất. Ảnh đã lưu trong '{output_folder}'")


def add_gaussian_noise(img, mean=0, var=0.008):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape).astype('uint8')
    noisy_img = cv2.add(img, gauss)
    return noisy_img


def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def main():
    video_path = "video.mp4"  
    frames_folder = "frames"  
    faces_folder = "faces_detected"  
    lips_folder = "lips_cropped"  
    augmented_folder = "augmented_data"  
    predictor_path = "shape_predictor_68_face_landmarks.dat"  

    #Trích xuất khung hình từ video
    extract_frames(video_path, frames_folder)

    #Phát hiện khuôn mặt
    detect_faces(frames_folder, faces_folder)

    #Cắt vùng môi
    crop_lips(faces_folder, lips_folder, predictor_path)

    #Data augmentation
    data_augmentation(lips_folder, augmented_folder)

if __name__ == "__main__":
    main()