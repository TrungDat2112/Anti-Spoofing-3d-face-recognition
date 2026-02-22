from __future__ import print_function, division
import torch
import cv2
import numpy as np
import os
from modules.builder.builder import builder
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_name', default='test_inference')
    parser.add_argument('--target', default='normal', choices=['normal', 'brdf', 'normal_and_brdf'])
    parser.add_argument('--checkpoint', default='checkpoint/final1.pth')
    parser.add_argument('--test_dir', default='test_data')
    parser.add_argument('--pixel_samples', type=int, default=2048)
    
    args = parser.parse_args()
    
    print(f'Starting inference session: {args.session_name}')
    print(f'Target: {args.target}')
    print(f'Checkpoint: {args.checkpoint}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    # Load model
    from modules.net.net import Net
    from modules.model.model_utils import loadmodel
    
    net = Net(args.pixel_samples, 'normal', device).to(device)
    net = torch.nn.DataParallel(net)
    net = loadmodel(net, args.checkpoint, strict=False)
    net.eval()
    
    print(f'Model loaded from {args.checkpoint}')
    
    # Load test images - support multiple formats
    test_images = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.exr']
    
    # Try different naming patterns and formats
    base_names = ['L1', 'L2', 'L3', 'L (1)', 'L (2)', 'L (3)']
    
    for base_name in base_names:
        found = False
        for ext in image_extensions:
            filename = base_name + ext
            # Try both in dat.data and data.exr directories
            for subdir in ['dat.data', 'data.exr', '']:
                if subdir:
                    img_path = os.path.join(args.test_dir, subdir, filename)
                else:
                    img_path = os.path.join(args.test_dir, filename)
                
                if os.path.exists(img_path):
                    print(f"Loading: {img_path}")
                    # Read image file
                    img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if img is not None:
                        # Convert to grayscale if needed
                        if len(img.shape) == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        test_images.append(img)
                        found = True
                        break
            if found:
                break
    
    if len(test_images) == 0:
        print("No valid test images found!")
        return
    
    print(f'Found {len(test_images)} test images')
    
    # Process images
    target_size = 256
    processed_images = []
    
    for img in test_images[:3]:  # Use up to 3 images
        # Resize to target size
        img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        
        # Normalize
        if img_resized.dtype == 'uint8':
            img_resized = img_resized.astype(np.float32) / 255.0
        elif img_resized.dtype == 'uint16':
            img_resized = img_resized.astype(np.float32) / 65535.0
        else:
            img_resized = img_resized.astype(np.float32)
        
        processed_images.append(img_resized)
    
    # Pad to 3 images if needed
    while len(processed_images) < 3:
        processed_images.append(processed_images[-1])
    
    # Convert grayscale to RGB (3-channel) format
    rgb_images = []
    for img in processed_images:
        # Convert grayscale to RGB by replicating the channel
        rgb_img = np.stack([img, img, img], axis=0)  # [3, H, W]
        rgb_images.append(rgb_img)
    
    # Stack images to create multi-channel input
    # The network expects [B, C, H, W, Nmax] format
    input_tensor = np.stack(rgb_images, axis=0)  # [3, 3, H, W] - 3 images, 3 channels each
    input_tensor = np.expand_dims(input_tensor, axis=0)  # [1, 3, 3, H, W] - batch=1
    input_tensor = np.transpose(input_tensor, (0, 2, 3, 4, 1))  # [B, C, H, W, Nmax] = [1, 3, H, W, 3]
    input_tensor = torch.from_numpy(input_tensor).float().to(device)
    
    # Create mask (all ones for now)
    mask = torch.ones(1, 1, target_size, target_size).to(device)
    
    # Create dummy normal (not used in inference)
    normal = torch.zeros(1, 3, target_size, target_size).to(device)
    
    # Number of images
    nImgArray = torch.tensor([3], dtype=torch.long).to(device)
    
    print(f'Input tensor shape: {input_tensor.shape}')
    print(f'Mask shape: {mask.shape}')
    print(f'Normal shape: {normal.shape}')
    print(f'nImgArray: {nImgArray.shape}')
    
    # Run inference
    with torch.no_grad():
        try:
            decoder_resolution = torch.ones(1, 1) * target_size
            canonical_resolution = torch.ones(1, 1) * 256
            
            pred_normal, loss, gt_normal, used_mask = net(
                input_tensor, mask, normal, nImgArray,
                decoder_resolution=decoder_resolution.to(device),
                canonical_resolution=canonical_resolution.to(device),
                mode_n='Test'
            )
            
            print("Inference successful!")
            print(f'Predicted normal shape: {pred_normal.shape}')
            
            # Convert output to numpy
            pred_normal_np = pred_normal.cpu().numpy()[0].transpose(1, 2, 0)  # [H, W, 3]
            
            # Normalize to [0, 1]
            pred_normal_np = (pred_normal_np + 1) / 2
            pred_normal_np = np.clip(pred_normal_np, 0, 1)
            
            # Save result
            output_path = 'predicted_normal.png'
            pred_normal_uint8 = (pred_normal_np * 255).astype(np.uint8)
            cv2.imwrite(output_path, cv2.cvtColor(pred_normal_uint8, cv2.COLOR_RGB2BGR))
            
            print(f'Result saved to {output_path}')
            
        except Exception as e:
            print(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main() 