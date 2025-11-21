#!/usr/bin/env python3
"""
Real-Time Deepfake Detection System
Main detection script for identifying synthetic/deepfake images and videos
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from model import DeepfakeDetectionModel
from utils import preprocess_frame, postprocess_output


class DeepfakeDetector:
    """Main detector class for deepfake detection"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the deepfake detector
        
        Args:
            model_path: Path to pre-trained model weights
            device: Computation device (cuda/cpu)
        """
        self.device = device
        self.model = DeepfakeDetectionModel().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.threshold = 0.5
        
    def detect_image(self, image_path):
        """
        Detect deepfakes in a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_fake, confidence_score, detection_map)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        input_tensor = preprocess_frame(image, self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
        is_fake, confidence, detection_map = postprocess_output(output)
        return is_fake, confidence, detection_map
    
    def detect_video(self, video_path, output_path=None):
        """
        Detect deepfakes in video frames
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save output video
            
        Returns:
            Dictionary with detection statistics
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = 0
        fake_frames = 0
        confidences = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                is_fake, confidence, _ = self.detect_image(frame)
                frame_count += 1
                if is_fake:
                    fake_frames += 1
                confidences.append(confidence)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                
        cap.release()
        
        return {
            'total_frames': frame_count,
            'fake_frames': fake_frames,
            'average_confidence': np.mean(confidences),
            'is_deepfake': fake_frames / frame_count > 0.5 if frame_count > 0 else False
        }
    
    def set_threshold(self, threshold):
        """Set detection confidence threshold"""
        self.threshold = threshold


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deepfake Detection")
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    detector = DeepfakeDetector(args.model, args.device)
    
    if args.image:
        is_fake, conf, _ = detector.detect_image(args.image)
        print(f"Image: {args.image}")
        print(f"Is Deepfake: {is_fake}")
        print(f"Confidence: {conf:.4f}")
        
    elif args.video:
        results = detector.detect_video(args.video)
        print(f"Video Analysis Results: {results}")
