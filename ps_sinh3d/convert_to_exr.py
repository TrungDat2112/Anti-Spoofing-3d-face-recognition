import cv2
import numpy as np
import os
import glob
import OpenEXR
import Imath
import argparse

def png_to_exr(png_path, exr_path):
    """Chuyển đổi normal map từ PNG sang EXR"""
    
    print(f"Converting: {png_path} -> {exr_path}")
    
    # Đọc file PNG
    img_bgr = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: Cannot read {png_path}")
        return False
    
    # Chuyển từ BGR sang RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Chuyển từ uint8 [0,255] sang float32 [0,1]
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Chuyển từ [0,1] sang [-1,1] (normal map range)
    normal_map = img_float * 2.0 - 1.0
    
    # Normalize để đảm bảo normal vectors có độ dài = 1
    length = np.sqrt(np.sum(normal_map * normal_map, axis=2, keepdims=True))
    length = np.maximum(length, 1e-8)  # Tránh chia cho 0
    normal_map = normal_map / length
    
    height, width = normal_map.shape[:2]
    
    # Tạo header cho file EXR
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    
    # Lưu file EXR
    try:
        output = OpenEXR.OutputFile(exr_path, header)
        output.writePixels({
            'R': normal_map[:, :, 0].astype(np.float32).tobytes(),
            'G': normal_map[:, :, 1].astype(np.float32).tobytes(),
            'B': normal_map[:, :, 2].astype(np.float32).tobytes()
        })
        output.close()
        print(f"  ✅ Successfully saved: {exr_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error saving EXR: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert normal map PNG files to EXR format')
    parser.add_argument('--input_dir', default='results', help='Directory containing PNG files')
    parser.add_argument('--output_dir', default='results_exr', help='Directory to save EXR files')
    parser.add_argument('--pattern', default='*_predicted_normal.png', help='File pattern to match')
    
    args = parser.parse_args()
    
    print('=== PNG to EXR Converter ===')
    print(f'Input directory: {args.input_dir}')
    print(f'Output directory: {args.output_dir}')
    print(f'File pattern: {args.pattern}')
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tìm tất cả file PNG khớp với pattern
    search_pattern = os.path.join(args.input_dir, args.pattern)
    png_files = glob.glob(search_pattern)
    
    if not png_files:
        print(f"No files found matching pattern: {search_pattern}")
        return
    
    png_files.sort()
    print(f'Found {len(png_files)} PNG files to convert:')
    for png_file in png_files:
        print(f'  - {os.path.basename(png_file)}')
    
    print('=' * 50)
    
    # Chuyển đổi từng file
    successful = 0
    failed = 0
    
    for i, png_path in enumerate(png_files, 1):
        # Tạo tên file EXR
        base_name = os.path.basename(png_path)
        exr_name = base_name.replace('.png', '.exr')
        exr_path = os.path.join(args.output_dir, exr_name)
        
        print(f'\n[{i}/{len(png_files)}] Processing: {base_name}')
        
        if png_to_exr(png_path, exr_path):
            successful += 1
        else:
            failed += 1
    
    # Tổng kết
    print('\n' + '=' * 50)
    print('=== CONVERSION SUMMARY ===')
    print(f'Total files: {len(png_files)}')
    print(f'Successful: {successful}')
    print(f'Failed: {failed}')
    print(f'EXR files saved in: {args.output_dir}')
    
    if successful > 0:
        print('\nGenerated EXR files:')
        exr_files = glob.glob(os.path.join(args.output_dir, '*.exr'))
        exr_files.sort()
        for exr_file in exr_files:
            file_size = os.path.getsize(exr_file) / 1024  # KB
            print(f'  - {os.path.basename(exr_file)} ({file_size:.1f} KB)')
    
    print('=' * 50)

if __name__ == '__main__':
    main() 