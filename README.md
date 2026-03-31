# Dự án Tích hợp Video -> Lip Reading -> Nhận diện Khuôn mặt

Kho lưu trữ này chứa một pipeline tích hợp chạy theo thứ tự:
1) Inference lip reading trên video đầu vào
2) Xử lý frame (cắt khuôn mặt + tạo ảnh EXR grayscale + ảnh 3 kênh)
3) Tạo normal map
4) Chuyển normal map PNG -> EXR
5) Trích xuất embedding nhận diện khuôn mặt
6) So sánh embedding với “gallery” đã lưu (ngưỡng cố định)

Điểm vào chính là file `integrated_pipeline.py`.

## Yêu cầu (Requirements)

Bạn cần:
- Cài Python
- Các file mô hình/checkpoint phải tồn tại đúng vị trí tương đối mà chương trình đang tìm

### Python
- Python 3.x (dự án dùng PyTorch và TensorFlow; `tensorflow==2.16.1` có thể yêu cầu phiên bản Python tương thích).
- Nếu bạn cài bằng `py -m venv` thì đảm bảo `py` trỏ tới đúng Python.

## Cài đặt

Từ thư mục gốc dự án (`d:\doan`):

```powershell
cd D:\doan
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -U pip
py -m pip install -r requirements.txt
```

## Các file checkpoint/bộ mô hình cần có

`integrated_pipeline.py` sẽ kiểm tra các file sau theo đường dẫn tương đối (tính từ thư mục gốc dự án):

- `lip_reading/finetuneGRU_best.pt`
- `ps_sinh3d/checkpoint/final1.pth`
- `3d_face_recognition_magface/checkpoint/concat2/normapmal_albedo/best_cosine_auc_119.pth`
- `3d_face_recognition_magface/face_gallery_10people.pkl`

Ngoài ra pipeline cần input video/bộ mặt do bạn cung cấp:
- `--video`: đường dẫn file `.mp4`

Phần demo (tùy chọn):
- `lip_reading/test_video/*.mp4`

## Luồng hoạt động của Dự án (Integrated Pipeline)

`integrated_pipeline.py` chạy các bước:

1. Inference Lip Reading (`lip_reading_inference`)
   - Dò ROI vùng miệng bằng MediaPipe face mesh
   - Trích xuất 29 frame của ROI vùng miệng
   - Chạy model lip-reading và so sánh nhãn dự đoán với `--true_label`
   - Nếu lip reading không đúng, pipeline dừng sớm
2. Xử lý frame (`process_frame`)
   - Dò/cắt khuôn mặt trên frame đã lưu (frame index 2 trong code, tức là frame trích xuất thứ 2)
   - Tạo và lưu:
     - `temp_output.jpg`
     - `temp_grayscale.exr` (albedo/grayscale)
     - `temp_blue_channel.jpg`, `temp_green_channel.jpg`, `temp_red_channel.jpg`
3. Tạo Normal Map (`generate_normal_map`)
   - Chạy model normal map của `ps_sinh3d`
   - Xuất ra `temp_predicted_normal.png`
4. Chuyển Normal PNG -> EXR (`convert_normal_to_exr`)
   - Lưu `temp_predicted_normal.exr`
5. Nhận diện khuôn mặt (trích embedding) (`face_recognition_inference`)
   - Nạp grayscale EXR + normal EXR
   - Trích xuất vector embedding của khuôn mặt
6. So sánh với gallery (`compare_with_gallery`)
   - So embedding với `3d_face_recognition_magface/face_gallery_10people.pkl`
   - Dùng ngưỡng cosine similarity “nghiêm” (mặc định `0.93`)

Cuối cùng pipeline sẽ xóa các file tạm (`cleanup_temp_files`).

## Chạy Inference (1 video)

Ví dụ:

```powershell
cd D:\doan
py -X utf8 integrated_pipeline.py --video "D:\path\to\video.mp4" --true_label 0
```

Ghi chú:
- Dùng đúng `--true_label` (0-9) để khớp với nhãn trong dataset của bạn.
- Pipeline sẽ dùng GPU nếu có (`torch.cuda.is_available()`).
- Trên Windows có thể gặp lỗi encode liên quan tới emoji khi in log; dùng `py -X utf8` như ví dụ trên (hoặc đặt `PYTHONIOENCODING=utf-8`) sẽ giúp.

## Chạy Demo Suite (Nhiều test case)

`demo_pipeline.py` chạy nhiều kịch bản bằng cách gọi `integrated_pipeline.py` từ bên trong.

```powershell
cd D:\doan
py demo_pipeline.py
```

Demo suite kỳ vọng:
- Video test ở `lip_reading/test_video/*.mp4`
- Tất cả checkpoint/gallery bắt buộc như liệt kê ở phần trên

## Khắc phục sự cố (Common Issues)

1. Thiếu file checkpoint/gallery
   - Chương trình sẽ in ra file nào bị thiếu theo đường dẫn tương đối.
2. Không mở được video / không dò được khuôn mặt
   - Hãy dùng video rõ mặt hơn (có khuôn mặt xuất hiện).
3. Lỗi `UnicodeEncodeError` trên console Windows
   - Chạy với `py -X utf8 ...` như phần chạy inference.


