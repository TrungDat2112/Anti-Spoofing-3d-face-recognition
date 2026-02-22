#!/usr/bin/env python3
"""
Use Face Gallery for Recognition
Script đơn giản để sử dụng gallery đã tạo cho face recognition
"""

import pickle
import numpy as np

def load_gallery(gallery_file="face_gallery_10people.pkl"):
    """Load gallery từ file"""
    try:
        with open(gallery_file, 'rb') as f:
            gallery = pickle.load(f)
        print(f"✅ Loaded gallery: {gallery_file}")
        print(f"   📊 Total people: {gallery['metadata']['total_people']}")
        print(f"   📏 Embedding dim: {gallery['metadata']['embedding_dim']}")
        print(f"   📅 Created: {gallery['metadata']['created_at']}")
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
            'all_similarities': similarities[:5]  # Top 5
        }
    else:
        return {
            'status': 'NO_MATCH',
            'query_person': query_person,
            'best_candidate': similarities[0][0] if similarities else 'N/A',
            'similarity': similarities[0][1] if similarities else 0,
            'confidence': 'LOW',
            'all_similarities': similarities[:5]
        }

def find_most_similar_pair(gallery):
    """Tìm cặp người tương tự nhất trong gallery"""
    max_similarity = 0
    best_pair = None
    
    people = list(gallery['embeddings'].keys())
    
    for i in range(len(people)):
        for j in range(i+1, len(people)):
            person1, person2 = people[i], people[j]
            emb1 = gallery['embeddings'][person1]
            emb2 = gallery['embeddings'][person2]
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_pair = (person1, person2)
    
    return best_pair, max_similarity

def show_gallery_stats(gallery):
    """Hiển thị thống kê gallery"""
    print(f"\n📊 GALLERY STATISTICS")
    print("-" * 40)
    
    people = list(gallery['embeddings'].keys())
    print(f"👥 People: {people}")
    
    # Tính toán similarity statistics
    similarities = []
    for i in range(len(people)):
        for j in range(i+1, len(people)):
            emb1 = gallery['embeddings'][people[i]]
            emb2 = gallery['embeddings'][people[j]]
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(sim)
    
    similarities = np.array(similarities)
    
    print(f"🔢 Similarity Statistics:")
    print(f"   Mean: {np.mean(similarities):.4f}")
    print(f"   Std:  {np.std(similarities):.4f}")
    print(f"   Min:  {np.min(similarities):.4f}")
    print(f"   Max:  {np.max(similarities):.4f}")
    
    # Tìm cặp tương tự nhất
    best_pair, max_sim = find_most_similar_pair(gallery)
    print(f"🔥 Most similar pair: {best_pair[0]} ↔ {best_pair[1]} ({max_sim:.4f})")

def main():
    print("🎯 FACE GALLERY USAGE DEMO")
    print("="*60)
    
    # Load gallery
    gallery = load_gallery()
    if gallery is None:
        return
    
    # Show statistics
    show_gallery_stats(gallery)
    
    # Demo recognition với tất cả mọi người
    print(f"\n🔍 FACE RECOGNITION TESTS")
    print("-" * 40)
    
    people = list(gallery['embeddings'].keys())
    
    for person in people:
        result = recognize_face(person, gallery, threshold=0.7)
        if isinstance(result, dict):
            print(f"\n🎯 {person}:")
            print(f"   Status: {result['status']}")
            if result['status'] == 'MATCH_FOUND':
                print(f"   Best match: {result['matched_person']} (sim: {result['similarity']:.4f}, {result['confidence']})")
            else:
                print(f"   Best candidate: {result['best_candidate']} (sim: {result['similarity']:.4f})")
            
            print(f"   Top 3 similar:")
            for i, (name, sim) in enumerate(result['all_similarities'][:3], 1):
                print(f"      {i}. {name}: {sim:.4f}")
    
    print(f"\n" + "="*60)
    print(f"💡 USAGE EXAMPLES:")
    print(f"💡 gallery = load_gallery('face_gallery_10people.pkl')")
    print(f"💡 result = recognize_face('hoang', gallery, threshold=0.8)")
    print(f"💡 if result['status'] == 'MATCH_FOUND':")
    print(f"💡     print(f\"Found: {{result['matched_person']}} ({{result['similarity']:.4f}})\")")

if __name__ == "__main__":
    main() 