"""
Inference pipeline for video-audio emotion recognition.
Process new videos and predict emotions and congruence.
"""

import os
import sys
from pathlib import Path
import torch
import cv2
import librosa
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import DEVICE
from models.full_model import MultimodalEmotionModel


# Emotion mapping
EMOTION_NAMES = [
    'neutral', 'calm', 'happy', 'sad', 
    'angry', 'fearful', 'disgust', 'surprised'
]

CONGRUENCE_NAMES = ['incongruent', 'congruent']


class EmotionInference:
    """Inference pipeline for emotion recognition."""
    
    def __init__(
        self,
        checkpoint_path,
        num_emotions=8,
        max_frames=16,
        audio_sample_rate=16000,
        audio_max_length=5.0,
        video_size=(224, 224)
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            num_emotions: Number of emotion classes
            max_frames: Number of frames to extract
            audio_sample_rate: Audio sample rate
            audio_max_length: Max audio length in seconds
            video_size: Video frame size
        """
        self.num_emotions = num_emotions
        self.max_frames = max_frames
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_length = audio_max_length
        self.video_size = video_size
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = MultimodalEmotionModel(
            num_emotions=num_emotions,
            freeze_backbones=True
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Device: {DEVICE}")
    
    def extract_video_frames(self, video_path):
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tensor of shape (1, num_frames, 3, H, W)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frame sampling
        frame_interval = max(1, total_frames // self.max_frames)
        
        frames = []
        frame_idx = 0
        
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Resize frame
                frame = cv2.resize(frame, self.video_size)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                # Convert to tensor (H, W, C) -> (C, H, W)
                frame = torch.from_numpy(frame).permute(2, 0, 1)
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        # Pad or truncate to max_frames
        if len(frames) < self.max_frames:
            last_frame = frames[-1] if frames else torch.zeros((3, *self.video_size))
            while len(frames) < self.max_frames:
                frames.append(last_frame)
        else:
            frames = frames[:self.max_frames]
        
        # Stack frames: (num_frames, C, H, W)
        frames = torch.stack(frames)
        
        # Add batch dimension: (1, num_frames, C, H, W)
        frames = frames.unsqueeze(0)
        
        return frames
    
    def extract_audio(self, video_path):
        """
        Extract audio waveform from video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tensor of shape (1, num_samples)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract audio using librosa
        try:
            waveform, sample_rate = librosa.load(
                video_path,
                sr=self.audio_sample_rate,
                mono=True,
                duration=self.audio_max_length
            )
            
            # Convert to tensor
            waveform = torch.from_numpy(waveform).float()
            
            # Ensure fixed length
            max_samples = int(self.audio_sample_rate * self.audio_max_length)
            
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]
            elif waveform.shape[0] < max_samples:
                pad_length = max_samples - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            # Add batch dimension: (1, num_samples)
            waveform = waveform.unsqueeze(0)
            
            return waveform
        
        except Exception as e:
            print(f"Error extracting audio: {e}")
            # Return dummy audio
            max_samples = int(self.audio_sample_rate * self.audio_max_length)
            return torch.zeros((1, max_samples))
    
    def predict(self, video_path):
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with predictions
        """
        print(f"\nProcessing: {video_path}")
        
        # Extract features
        print("Extracting video frames...")
        video_frames = self.extract_video_frames(video_path)
        
        print("Extracting audio...")
        audio_waveform = self.extract_audio(video_path)
        
        # Move to device
        video_frames = video_frames.to(DEVICE)
        audio_waveform = audio_waveform.to(DEVICE)
        
        # Run inference
        print("Running inference...")
        with torch.no_grad():
            outputs = self.model(video_frames, audio_waveform)
        
        # Get predictions
        emotion_logits = outputs['emotion_logits']
        congruence_logits = outputs['congruence_logits']
        
        # Convert to probabilities
        emotion_probs = torch.softmax(emotion_logits, dim=1)[0]
        congruence_probs = torch.softmax(congruence_logits, dim=1)[0]
        
        # Get predicted classes
        emotion_pred = torch.argmax(emotion_logits, dim=1)[0].item()
        congruence_pred = torch.argmax(congruence_logits, dim=1)[0].item()
        
        # Prepare results
        results = {
            'video_path': video_path,
            'emotion': {
                'predicted': EMOTION_NAMES[emotion_pred],
                'confidence': emotion_probs[emotion_pred].item(),
                'probabilities': {
                    emotion: prob.item() 
                    for emotion, prob in zip(EMOTION_NAMES, emotion_probs)
                }
            },
            'congruence': {
                'predicted': CONGRUENCE_NAMES[congruence_pred],
                'confidence': congruence_probs[congruence_pred].item(),
                'probabilities': {
                    label: prob.item()
                    for label, prob in zip(CONGRUENCE_NAMES, congruence_probs)
                }
            }
        }
        
        return results
    
    def predict_batch(self, video_paths):
        """
        Run inference on multiple videos.
        
        Args:
            video_paths: List of video file paths
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for video_path in video_paths:
            try:
                result = self.predict(video_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append({
                    'video_path': video_path,
                    'error': str(e)
                })
        
        return results
    
    def print_results(self, results):
        """Print prediction results in a formatted way."""
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"\nVideo: {results['video_path']}")
        
        print(f"\nEmotion Prediction:")
        print(f"  Predicted: {results['emotion']['predicted'].upper()}")
        print(f"  Confidence: {results['emotion']['confidence']:.4f}")
        
        print(f"\n  All Probabilities:")
        sorted_emotions = sorted(
            results['emotion']['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for emotion, prob in sorted_emotions:
            bar = '#' * int(prob * 50)
            print(f"    {emotion:12s}: {prob:.4f} |{bar}")
        
        print(f"\nCongruence Prediction:")
        print(f"  Predicted: {results['congruence']['predicted'].upper()}")
        print(f"  Confidence: {results['congruence']['confidence']:.4f}")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run emotion inference on video")
    parser.add_argument("--checkpoint", type=str, 
                       default="./checkpoints/multimodal/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to video file or directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save results (JSON)")
    
    args = parser.parse_args()
    
    # Create inference pipeline
    pipeline = EmotionInference(checkpoint_path=args.checkpoint)
    
    # Check if video is a file or directory
    video_path = Path(args.video)
    
    if video_path.is_file():
        # Single video
        results = pipeline.predict(str(video_path))
        pipeline.print_results(results)
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nResults saved to {args.output}")
    
    elif video_path.is_dir():
        # Directory of videos
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.flv']:
            video_files.extend(video_path.glob(ext))
        
        if len(video_files) == 0:
            print(f"No video files found in {video_path}")
            exit(1)
        
        print(f"Found {len(video_files)} video files")
        
        # Process all videos
        results_list = pipeline.predict_batch([str(f) for f in video_files])
        
        # Print all results
        for results in results_list:
            if 'error' not in results:
                pipeline.print_results(results)
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results_list, f, indent=4)
            print(f"\nAll results saved to {args.output}")
    
    else:
        print(f"Error: {video_path} is not a valid file or directory")
        exit(1)