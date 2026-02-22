#!/usr/bin/env python3
"""
Create Face Gallery for Recognition
Tạo gallery đơn giản để lưu embeddings của 10 người làm cơ sở so sánh
"""

import torch
import cv2
import numpy as np
import pickle
from pathlib import Path
import OpenEXR
import Imath
import albumentations as A
from datetime import datetime

from going_modular.model.MTLFaceRecognition import MTLFaceRecognition
from going_modular.model.ConcatMTLFaceRecognition import ConcatMTLFaceRecognitionV2

class FaceGalleryCreator:
    def __init__(self, checkpoint_path="checkpoint/concat2/normapmal_albedo/best_cosine_auc_119.pth"):
        """Khởi tạo với phương pháp inference chuẩn hóa"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = self.load_model()
        
        # STANDARDIZED transforms - same as final_standardized_inference.py
        self.transform = A.Compose([
            A.Resize(height=256, width=256)
        ], additional_targets={'image2': 'image'}, is_check_shapes=False)
        
        print(f"✅ Model loaded on device: {self.device}")
    
    def load_model(self):
        """Load model - same as final_standardized_inference.py"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
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
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded: {len(compatible_dict)}/{len(state_dict)} weights")
        return model
    
    def load_exr_standardized(self, path):
        """STANDARDIZED EXR loading - same as final_standardized_inference.py"""
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
    
    def extract_embedding_standardized(self, albedo_path, normalmap_path):
        """STANDARDIZED embedding extraction - same as final_standardized_inference.py"""
        # Load images
        albedo = self.load_exr_standardized(albedo_path)
        normalmap = self.load_exr_standardized(normalmap_path)
        
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
        transformed = self.transform(image=albedo, image2=normalmap)
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
        
        input_tensor = torch.stack([albedo_tensor, normalmap_tensor], dim=0).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            try:
                results = self.model.get_result(input_tensor)
                embedding = results[0]
                return embedding.cpu().numpy().squeeze()
            except:
                outputs = self.model.forward(input_tensor)
                id_norm = outputs[-1]
                return id_norm.cpu().numpy().squeeze()

def create_gallery():
    print("🎯 CREATING FACE GALLERY")
    print("="*60)
    
    # Initialize
    gallery_creator = FaceGalleryCreator()
    
    # Find all pairs in results folder
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
    
    print(f"📊 Found {len(pairs)} pairs to process")
    
    # Create gallery dictionary
    gallery = {
        'embeddings': {},  # {person_name: embedding_array}
        'metadata': {
            'total_people': 0,
            'embedding_dim': None,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'method': 'standardized_inference',
            'model_checkpoint': gallery_creator.checkpoint_path
        },
        'file_paths': {}  # {person_name: (albedo_path, normal_path)}
    }
    
    # Process each person
    success_count = 0
    for albedo_path, normal_path, person_name in pairs:
        print(f"🔄 Processing {person_name}...")
        try:
            # Extract embedding using standardized method
            embedding = gallery_creator.extract_embedding_standardized(albedo_path, normal_path)
            
            if embedding is not None:
                gallery['embeddings'][person_name] = embedding.flatten()
                gallery['file_paths'][person_name] = (str(albedo_path), str(normal_path))
                success_count += 1
                print(f"✅ {person_name} - Embedding shape: {embedding.shape}")
                
                # Set embedding dimension if first success
                if gallery['metadata']['embedding_dim'] is None:
                    gallery['metadata']['embedding_dim'] = len(embedding.flatten())
            else:
                print(f"❌ {person_name} - Failed to extract embedding")
                
        except Exception as e:
            print(f"❌ {person_name} - Error: {e}")
    
    # Update metadata
    gallery['metadata']['total_people'] = success_count
    
    # Save gallery to pickle file
    gallery_filename = f"face_gallery_{success_count}people.pkl"
    with open(gallery_filename, 'wb') as f:
        pickle.dump(gallery, f)
    
    print(f"\n" + "="*60)
    print(f"GALLERY CREATED")
    print("="*60)
    print(f"✅ Successfully processed: {success_count}/{len(pairs)} people")
    print(f"💾 Gallery saved: {gallery_filename}")
    print(f"📏 Embedding dimension: {gallery['metadata']['embedding_dim']}")
    print(f"📋 People in gallery: {list(gallery['embeddings'].keys())}")
    
    return gallery_filename

def load_gallery(gallery_file="face_gallery_10people.pkl"):
    """Load gallery từ file"""
    try:
        with open(gallery_file, 'rb') as f:
            gallery = pickle.load(f)
        print(f"✅ Loaded gallery: {gallery_file}")
        print(f"   Total people: {gallery['metadata']['total_people']}")
        print(f"   Embedding dim: {gallery['metadata']['embedding_dim']}")
        print(f"   Created: {gallery['metadata']['created_at']}")
        return gallery
    except FileNotFoundError:
        print(f"❌ Gallery file not found: {gallery_file}")
        return None

def recognize_face(query_person, gallery, threshold=0.7):
    """Nhận diện khuôn mặt bằng gallery"""
    if query_person not in gallery['embeddings']:
        return f"❌ Person '{query_person}' not found in gallery"
    
    query_embedding = gallery['embeddings'][query_person]
    similarities = []
    
    # So sánh với tất cả người khác trong gallery
    for person_name, embedding in gallery['embeddings'].items():
        if person_name != query_person:
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((person_name, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Check threshold
    if similarities and similarities[0][1] >= threshold:
        best_match = similarities[0]
        return {
            'status': 'MATCH_FOUND',
            'query_person': query_person,
            'matched_person': best_match[0],
            'similarity': best_match[1],
            'confidence': 'HIGH' if best_match[1] > 0.8 else 'MEDIUM',
            'all_similarities': similarities[:3]  # Top 3
        }
    else:
        return {
            'status': 'NO_MATCH',
            'query_person': query_person,
            'best_candidate': similarities[0][0] if similarities else 'N/A',
            'similarity': similarities[0][1] if similarities else 0,
            'confidence': 'LOW',
            'all_similarities': similarities[:3]
        }

def main():
    # Create gallery
    gallery_file = create_gallery()
    
    # Demo recognition
    print(f"\n🔍 DEMO FACE RECOGNITION")
    print("-" * 40)
    
    # Load gallery
    gallery = load_gallery(gallery_file)
    if gallery is None:
        return
    
    # Test recognition với một vài người
    test_people = ['hoang', 'datv2', 'n1', 'n5']
    
    for person in test_people:
        if person in gallery['embeddings']:
            result = recognize_face(person, gallery, threshold=0.7)
            print(f"\n🎯 Testing: {person}")
            if isinstance(result, dict):
                print(f"   Status: {result['status']}")
                if result['status'] == 'MATCH_FOUND':
                    print(f"   Best match: {result['matched_person']} (similarity: {result['similarity']:.4f})")
                else:
                    print(f"   Best candidate: {result['best_candidate']} (similarity: {result['similarity']:.4f})")
                print(f"   Top 3 similarities:")
                for i, (name, sim) in enumerate(result['all_similarities'], 1):
                    print(f"      {i}. {name}: {sim:.4f}")
    
    print(f"\n🎉 Gallery ready for face recognition!")
    print(f"💡 Use: gallery = load_gallery('{gallery_file}')")
    print(f"💡 Then: result = recognize_face('person_name', gallery)")

if __name__ == "__main__":
    main() 