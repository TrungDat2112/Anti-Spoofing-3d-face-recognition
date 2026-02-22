#!/usr/bin/env python3
"""
FINAL STANDARDIZED Face Recognition Inference
Phương pháp chuẩn hóa để đảm bảo kết quả nhất quán cho tất cả các lần chạy
"""

import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import OpenEXR
import Imath
import albumentations as A
from sklearn.metrics.pairwise import cosine_similarity

from going_modular.model.MTLFaceRecognition import MTLFaceRecognition
from going_modular.model.ConcatMTLFaceRecognition import ConcatMTLFaceRecognitionV2

def main():
    print("🔄 FINAL STANDARDIZED INFERENCE")
    print("="*60)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoint/concat2/normapmal_albedo/best_cosine_auc_119.pth"
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Get num_classes
    maglinear_shape = None
    for key, value in state_dict.items():
        if 'maglinear.weight' in key:
            maglinear_shape = value.shape
            break
    
    num_classes = maglinear_shape[0] if maglinear_shape else 512
    backbone1 = MTLFaceRecognition('miresnet18', num_classes)
    backbone2 = MTLFaceRecognition('miresnet18', num_classes)
    model = ConcatMTLFaceRecognitionV2(backbone1, backbone2, num_classes)
    
    # Load compatible weights
    model_dict = model.state_dict()
    compatible_dict = {}
    
    for key, checkpoint_param in state_dict.items():
        if key in model_dict:
            model_param = model_dict[key]
            if model_param.shape == checkpoint_param.shape:
                compatible_dict[key] = checkpoint_param
    
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {len(compatible_dict)}/{len(state_dict)} weights")
    
    # STANDARDIZED transforms
    transform = A.Compose([
        A.Resize(height=256, width=256)
    ], additional_targets={'image2': 'image'}, is_check_shapes=False)
    
    def load_exr_standardized(path):
        """STANDARDIZED EXR loading"""
        try:
            exr_file = OpenEXR.InputFile(str(path))
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            if 'Y' in header['channels']:
                y_channel = exr_file.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
                img = np.frombuffer(y_channel, dtype=np.float32).reshape(height, width)
            else:
                r_channel = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
                g_channel = exr_file.channel('G', Imath.PixelType(Imath.PixelType.FLOAT)) 
                b_channel = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
                
                r = np.frombuffer(r_channel, dtype=np.float32).reshape(height, width)
                g = np.frombuffer(g_channel, dtype=np.float32).reshape(height, width)
                b = np.frombuffer(b_channel, dtype=np.float32).reshape(height, width)
                
                img = np.stack([r, g, b], axis=2)
            
            exr_file.close()
            return img
        except:
            return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    def extract_embedding_standardized(albedo_path, normalmap_path):
        """STANDARDIZED embedding extraction"""
        # Load images
        albedo = load_exr_standardized(albedo_path)
        normalmap = load_exr_standardized(normalmap_path)
        
        # Process images
        def process_image(img):
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            return img.astype(np.float32)
        
        albedo = process_image(albedo)
        normalmap = process_image(normalmap)
        
        # Apply transforms
        transformed = transform(image=albedo, image2=normalmap)
        albedo = transformed['image']
        normalmap = transformed['image2']
        
        # Normalize
        if albedo.max() > 1.0:
            albedo = albedo / 255.0
        if normalmap.max() > 1.0:
            normalmap = normalmap / 255.0
        
        # Convert to tensors
        albedo_tensor = torch.from_numpy(albedo).permute(2, 0, 1).float()
        normalmap_tensor = torch.from_numpy(normalmap).permute(2, 0, 1).float()
        
        input_tensor = torch.stack([albedo_tensor, normalmap_tensor], dim=0).unsqueeze(0).to(device)
        
        # Extract embedding
        with torch.no_grad():
            try:
                results = model.get_result(input_tensor)
                embedding = results[0]
                return embedding.cpu().numpy().squeeze()
            except:
                outputs = model.forward(input_tensor)
                id_norm = outputs[-1]
                return id_norm.cpu().numpy().squeeze()
    
    # Find all pairs
    results_dir = Path("results")
    all_files = list(results_dir.glob("*.exr"))
    
    # Separate files
    albedo_files = []
    normal_files = []
    
    for file in all_files:
        if "_predicted_normal" in file.name:
            normal_files.append(file)
        elif "_grayscale" not in file.name:
            albedo_files.append(file)
    
    # Match pairs
    pairs = []
    for albedo_file in albedo_files:
        base_name = albedo_file.stem
        normal_file = None
        for nf in normal_files:
            if nf.name.startswith(base_name + "_predicted_normal"):
                normal_file = nf
                break
        
        if normal_file:
            pairs.append((albedo_file, normal_file, base_name))
    
    print(f"📊 Found {len(pairs)} pairs")
    
    # Extract embeddings
    embeddings = []
    valid_names = []
    
    for albedo_path, normal_path, name in pairs:
        print(f"Processing {name}...")
        try:
            embedding = extract_embedding_standardized(albedo_path, normal_path)
            if embedding is not None:
                embeddings.append(embedding.flatten())
                valid_names.append(name)
                print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    # Calculate similarity
    embeddings = np.array(embeddings)
    cosine_sim = cosine_similarity(embeddings)
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"FINAL_standardized_cosine_similarity_{len(valid_names)}people_{timestamp}.csv"
    
    cosine_df = pd.DataFrame(cosine_sim, index=valid_names, columns=valid_names)
    cosine_df.to_csv(filename)
    
    print(f"💾 Saved: {filename}")
    
    # Analysis
    print("\n" + "="*60)
    print("FINAL STANDARDIZED RESULTS")
    print("="*60)
    
    upper_triangle = np.triu(cosine_sim, k=1)
    max_indices = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
    max_similarity = cosine_sim[max_indices]
    
    print(f"\n🎯 Most Similar Pair:")
    print(f"   {valid_names[max_indices[0]]} ↔ {valid_names[max_indices[1]]}: {max_similarity:.4f}")
    
    # Top 5
    upper_flat = upper_triangle[upper_triangle > 0]
    sorted_indices = np.argsort(upper_flat)[::-1]
    
    print(f"\n🔥 Top 5 Most Similar Pairs:")
    count = 0
    for idx in sorted_indices[:5]:
        flat_indices = np.where(upper_triangle.flatten() == upper_flat[idx])
        for flat_idx in flat_indices[0]:
            if count < 5:
                i, j = np.unravel_index(flat_idx, upper_triangle.shape)
                similarity = cosine_sim[i, j]
                print(f"   {count+1}. {valid_names[i]} ↔ {valid_names[j]}: {similarity:.4f}")
                count += 1
    
    # Statistics
    diagonal_mask = np.eye(len(cosine_sim), dtype=bool)
    off_diagonal = cosine_sim[~diagonal_mask]
    
    print(f"\n📊 Statistics:")
    print(f"   Mean: {np.mean(off_diagonal):.4f}")
    print(f"   Std: {np.std(off_diagonal):.4f}")
    print(f"   Min: {np.min(off_diagonal):.4f}")
    print(f"   Max: {np.max(off_diagonal):.4f}")

if __name__ == "__main__":
    main() 