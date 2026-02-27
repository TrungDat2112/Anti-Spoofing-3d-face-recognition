#!/usr/bin/env python3
"""
DEMO SCRIPT FOR INTEGRATED PIPELINE
Script demo để test pipeline với nhiều video và scenarios khác nhau
"""

import os
import sys
import subprocess
import glob
from datetime import datetime

def run_pipeline(video_path, true_label, threshold=0.93):
    """Chạy pipeline với video và label cụ thể (threshold hard-coded 0.93)"""
    cmd = [
        sys.executable, 
        'integrated_pipeline.py',
        '--video', video_path,
        '--true_label', str(true_label)
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print("="*80)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout after 5 minutes"
    except Exception as e:
        return False, "", str(e)

def demo_basic_test():
    """Demo test cơ bản với 1 video"""
    print("🎯 DEMO 1: Basic Test")
    print("="*80)
    
    # Test với video có sẵn
    test_videos = glob.glob("lip_reading/test_video/*.mp4")
    
    if not test_videos:
        print("❌ Không tìm thấy video test trong lip_reading/test_video/")
        return
    
    # Lấy video đầu tiên
    test_video = test_videos[0]
    
    # Extract true label từ tên file (VD: 0a.mp4 -> 0)
    try:
        true_label = int(os.path.basename(test_video)[0])
    except:
        true_label = 0
    
    print(f"📹 Testing with: {test_video}")
    print(f"🏷️ True label: {true_label}")
    
    success, stdout, stderr = run_pipeline(test_video, true_label)
    
    if success:
        print("✅ DEMO 1 COMPLETED SUCCESSFULLY!")
        print("\nOutput:")
        print(stdout[-1000:])  # Last 1000 chars
    else:
        print("❌ DEMO 1 FAILED!")
        print("Error:", stderr)
    
    print("\n" + "="*80 + "\n")

def demo_multiple_videos():
    """Demo test với nhiều video"""
    print("🎯 DEMO 2: Multiple Videos Test")
    print("="*80)
    
    test_videos = glob.glob("lip_reading/test_video/*.mp4")[:3]  # Test 3 video đầu
    
    if not test_videos:
        print("❌ Không tìm thấy video test")
        return
    
    results = []
    
    for i, video in enumerate(test_videos, 1):
        print(f"\n📹 [{i}/{len(test_videos)}] Testing: {os.path.basename(video)}")
        
        # Extract true label
        try:
            true_label = int(os.path.basename(video)[0])
        except:
            true_label = 0
        
        success, stdout, stderr = run_pipeline(video, true_label)
        
        result = {
            'video': os.path.basename(video),
            'true_label': true_label,
            'success': success,
            'error': stderr if not success else None
        }
        results.append(result)
        
        if success:
            print(f"✅ {os.path.basename(video)}: SUCCESS")
        else:
            print(f"❌ {os.path.basename(video)}: FAILED - {stderr}")
    
    # Summary
    print("\n📊 DEMO 2 SUMMARY:")
    print("-" * 50)
    successful = sum(1 for r in results if r['success'])
    print(f"Total videos: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['video']} (label: {result['true_label']})")
    
    print("\n" + "="*80 + "\n")

def demo_threshold_comparison():
    """Demo test với strict threshold 0.93"""
    print("🎯 DEMO 3: Strict Threshold Test (0.93)")
    print("="*80)
    
    test_videos = glob.glob("lip_reading/test_video/*.mp4")
    
    if not test_videos:
        print("❌ Không tìm thấy video test")
        return
    
    test_video = test_videos[0]
    try:
        true_label = int(os.path.basename(test_video)[0])
    except:
        true_label = 0
    
    print(f"📹 Testing video: {os.path.basename(test_video)}")
    print(f"🏷️ True label: {true_label}")
    print(f"🎯 Using STRICT threshold: 0.93 (hard-coded)")
    
    print(f"\n🔍 Testing with strict threshold...")
    success, stdout, stderr = run_pipeline(test_video, true_label)
    
    if success:
        print(f"✅ Pipeline: SUCCESS")
        # Extract face recognition result từ stdout
        lines = stdout.split('\n')
        for line in lines:
            if 'MATCH_FOUND' in line or 'NO_MATCH' in line or 'Matched Person:' in line or 'Similarity:' in line or 'STRICT' in line:
                print(f"   {line.strip()}")
    else:
        print(f"❌ Pipeline: FAILED - {stderr}")
    
    print("\n" + "="*80 + "\n")

def demo_edge_cases():
    """Demo test các trường hợp edge cases"""
    print("🎯 DEMO 4: Edge Cases")
    print("="*80)
    
    # Test case 1: Sai true label (để lip reading fail)
    test_videos = glob.glob("lip_reading/test_video/*.mp4")
    
    if test_videos:
        test_video = test_videos[0]
        correct_label = int(os.path.basename(test_video)[0])
        wrong_label = (correct_label + 1) % 10  # Sai label
        
        print(f"📹 Test Case 1: Wrong True Label")
        print(f"Video: {os.path.basename(test_video)}")
        print(f"Correct label: {correct_label}, Using: {wrong_label}")
        
        success, stdout, stderr = run_pipeline(test_video, wrong_label)
        
        if not success or "LIP READING FAILED" in stdout:
            print("✅ Expected behavior: Lip reading failed, pipeline stopped")
        else:
            print("❌ Unexpected: Pipeline continued despite wrong label")
        
        print("\n" + "-" * 50 + "\n")
    
    # Test case 2: Video không tồn tại
    print("📹 Test Case 2: Non-existent Video")
    fake_video = "non_existent_video.mp4"
    
    success, stdout, stderr = run_pipeline(fake_video, 0)
    
    if not success:
        print("✅ Expected behavior: Failed to open non-existent video")
    else:
        print("❌ Unexpected: Pipeline worked with non-existent video")
    
    print("\n" + "="*80 + "\n")

def demo_performance_test():
    """Demo test performance với timing"""
    print("🎯 DEMO 5: Performance Test")
    print("="*80)
    
    test_videos = glob.glob("lip_reading/test_video/*.mp4")
    
    if not test_videos:
        print("❌ Không tìm thấy video test")
        return
    
    test_video = test_videos[0]
    try:
        true_label = int(os.path.basename(test_video)[0])
    except:
        true_label = 0
    
    print(f"📹 Performance test với: {os.path.basename(test_video)}")
    print(f"🏷️ True label: {true_label}")
    
    start_time = datetime.now()
    
    success, stdout, stderr = run_pipeline(test_video, true_label)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n⏱️ PERFORMANCE RESULTS:")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print(f"   Average time per step: {duration/9:.2f} seconds")
    
    if duration > 120:  # > 2 minutes
        print("⚠️ Pipeline took longer than expected (>2 min)")
    elif duration < 30:  # < 30 seconds
        print("🚀 Pipeline completed very quickly (<30 sec)")
    else:
        print("✅ Pipeline completed in reasonable time")
    
    print("\n" + "="*80 + "\n")

def check_prerequisites():
    """Kiểm tra prerequisites trước khi chạy demo"""
    print("🔍 CHECKING PREREQUISITES")
    print("="*80)
    
    required_files = [
        'integrated_pipeline.py',
        'lip_reading/finetuneGRU_best.pt',
        'ps_sinh3d/checkpoint/final1.pth',
        '3d_face_recognition_magface/checkpoint/concat2/normapmal_albedo/best_cosine_auc_119.pth',
        '3d_face_recognition_magface/face_gallery_10people.pkl'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    # Check test videos
    test_videos = glob.glob("lip_reading/test_video/*.mp4")
    if test_videos:
        print(f"✅ Found {len(test_videos)} test videos")
    else:
        print("❌ No test videos found in lip_reading/test_video/")
        missing_files.append("test videos")
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files/directories:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all files are present before running demos.")
        return False
    else:
        print(f"\n✅ All prerequisites satisfied!")
        return True

def main():
    print("🎮 INTEGRATED PIPELINE DEMO SUITE")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Exiting.")
        return
    
    print("\n")
    
    # Run demos
    try:
        demo_basic_test()
        demo_multiple_videos()
        demo_threshold_comparison()
        demo_edge_cases()
        demo_performance_test()
        
        print("🎉 ALL DEMOS COMPLETED!")
        print("="*80)
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\n⛔ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 