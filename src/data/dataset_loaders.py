"""
PyTorch Dataset classes for RAVDESS and CREMA-D.
Handles video frame extraction, audio loading, and multimodal batching.
"""

import os
import torch
import librosa
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import sys

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import config, IS_GPU_MODE


class VideoAudioEmotionDataset(Dataset):
    """
    Dataset for video-audio emotion recognition with congruence detection.
    """
    
    def __init__(
        self,
        metadata_path,
        max_frames=16,
        target_fps=None,
        audio_sample_rate=16000,
        audio_max_length=5.0,
        video_size=(224, 224),
        mode='both',  # 'video', 'audio', or 'both'
        cpu_mode=None
    ):
        """
        Args:
            metadata_path: Path to CSV file with metadata
            max_frames: Maximum number of frames to extract from video
            target_fps: Target FPS for frame extraction (None = use all frames)
            audio_sample_rate: Target sample rate for audio
            audio_max_length: Maximum audio length in seconds
            video_size: Target size for video frames (H, W)
            mode: 'video', 'audio', or 'both'
            cpu_mode: If True, use CPU-optimized settings
        """
        if cpu_mode is None:
            cpu_mode = not IS_GPU_MODE
        
        self.metadata = pd.read_csv(metadata_path)
        self.max_frames = config['max_frames'] if cpu_mode else max_frames
        self.target_fps = target_fps
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_length = audio_max_length
        self.video_size = video_size
        self.mode = mode
        
        print(f"Loaded metadata with {len(self.metadata)} samples")
        
        # Check if has_video column exists
        if 'has_video' not in self.metadata.columns:
            print("Warning: 'has_video' column not found. Assuming all samples have video.")
            self.metadata['has_video'] = True
        
        # Filter based on mode
        if mode == 'video':
            before = len(self.metadata)
            self.metadata = self.metadata[self.metadata['has_video'] == True]
            print(f"Filtered for video: {before} -> {len(self.metadata)} samples")
        elif mode == 'audio':
            # All samples have audio
            print(f"Audio mode: keeping all {len(self.metadata)} samples")
        elif mode == 'both':
            before = len(self.metadata)
            # For 'both' mode, if no video is available, use audio-only
            # But prefer samples with video
            has_video = self.metadata[self.metadata['has_video'] == True]
            if len(has_video) > 0:
                self.metadata = has_video
                print(f"Filtered for both (video+audio): {before} -> {len(self.metadata)} samples")
            else:
                print(f"No video samples found. Using audio-only mode with {len(self.metadata)} samples")
                self.mode = 'audio'
        
        self.metadata = self.metadata.reset_index(drop=True)
        
        print(f"Dataset initialized with {len(self.metadata)} samples")
        print(f"Mode: {self.mode}, Max frames: {self.max_frames}")
    
    def __len__(self):
        return len(self.metadata)
    
    def load_video(self, video_path):
        """
        Load video and extract frames.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tensor of shape (num_frames, C, H, W)
        """
        if not os.path.exists(video_path):
            # Return dummy frames if video doesn't exist
            frames = torch.zeros((self.max_frames, 3, *self.video_size))
            return frames
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            frames = torch.zeros((self.max_frames, 3, *self.video_size))
            return frames
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frame sampling
        if self.target_fps is not None:
            frame_interval = int(fps / self.target_fps)
        else:
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
            # Pad with last frame
            last_frame = frames[-1] if frames else torch.zeros((3, *self.video_size))
            while len(frames) < self.max_frames:
                frames.append(last_frame)
        else:
            frames = frames[:self.max_frames]
        
        # Stack frames: (num_frames, C, H, W)
        frames = torch.stack(frames)
        
        return frames
    
    def load_audio(self, audio_path):
        """
        Load audio waveform using librosa (more robust than torchaudio).
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tensor of shape (num_samples,)
        """
        if not os.path.exists(audio_path):
            # Return dummy audio if file doesn't exist
            audio_length = int(self.audio_sample_rate * self.audio_max_length)
            waveform = torch.zeros((audio_length,))
            return waveform
        
        try:
            # Load audio with librosa
            waveform, sample_rate = librosa.load(
                audio_path,
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
            
            return waveform
        
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            audio_length = int(self.audio_sample_rate * self.audio_max_length)
            waveform = torch.zeros((audio_length,))
            return waveform
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Returns:
            Dictionary with:
                - video: Tensor (num_frames, C, H, W) or None
                - audio: Tensor (num_samples,) or None
                - emotion_label: int
                - video_emotion_label: int
                - audio_emotion_label: int
                - congruence_label: int (0 or 1)
                - metadata: dict with additional info
        """
        row = self.metadata.iloc[idx]
        
        sample = {
            'video': None,
            'audio': None,
            'emotion_label': row['emotion_id'],
            'video_emotion_label': row['video_emotion_id'],
            'audio_emotion_label': row['audio_emotion_id'],
            'congruence_label': row['is_congruent'],
            'metadata': {
                'filename': row['filename'],
                'dataset': row['dataset'],
                'emotion': row['emotion']
            }
        }
        
        # Load video if needed
        if self.mode in ['video', 'both']:
            if row['has_video'] and pd.notna(row.get('video_filepath')):
                sample['video'] = self.load_video(row['video_filepath'])
            else:
                sample['video'] = torch.zeros((self.max_frames, 3, *self.video_size))
        
        # Load audio if needed
        if self.mode in ['audio', 'both']:
            sample['audio'] = self.load_audio(row['filepath'])
        
        return sample


def collate_fn(batch):
    """
    Custom collate function for batching.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched dictionary
    """
    batched = {
        'video': None,
        'audio': None,
        'emotion_label': torch.tensor([s['emotion_label'] for s in batch], dtype=torch.long),
        'video_emotion_label': torch.tensor([s['video_emotion_label'] for s in batch], dtype=torch.long),
        'audio_emotion_label': torch.tensor([s['audio_emotion_label'] for s in batch], dtype=torch.long),
        'congruence_label': torch.tensor([s['congruence_label'] for s in batch], dtype=torch.long),
        'metadata': [s['metadata'] for s in batch]
    }
    
    # Stack videos if present
    if batch[0]['video'] is not None:
        batched['video'] = torch.stack([s['video'] for s in batch])
    
    # Stack audio if present
    if batch[0]['audio'] is not None:
        batched['audio'] = torch.stack([s['audio'] for s in batch])
    
    return batched


def get_dataloaders(
    train_metadata_path,
    val_metadata_path,
    test_metadata_path,
    batch_size=None,
    num_workers=None,
    mode='both',
    cpu_mode=None
):
    """
    Create train, val, and test dataloaders.
    
    Args:
        train_metadata_path: Path to train metadata CSV
        val_metadata_path: Path to val metadata CSV
        test_metadata_path: Path to test metadata CSV
        batch_size: Batch size (auto-determined from config if None)
        num_workers: Number of workers (auto-determined from config if None)
        mode: 'video', 'audio', or 'both'
        cpu_mode: If True, use CPU-optimized settings
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if cpu_mode is None:
        cpu_mode = not IS_GPU_MODE
    
    if batch_size is None:
        batch_size = config['batch_size']
    
    if num_workers is None:
        num_workers = config['num_workers']
    
    print(f"\nCreating dataloaders...")
    print(f"Batch size: {batch_size}, Num workers: {num_workers}")
    
    # Create datasets
    train_dataset = VideoAudioEmotionDataset(
        train_metadata_path,
        mode=mode,
        cpu_mode=cpu_mode
    )
    
    val_dataset = VideoAudioEmotionDataset(
        val_metadata_path,
        mode=mode,
        cpu_mode=cpu_mode
    )
    
    test_dataset = VideoAudioEmotionDataset(
        test_metadata_path,
        mode=mode,
        cpu_mode=cpu_mode
    )
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty! Please check your metadata files and mode settings.")
    
    # Limit dataset size in CPU mode
    if cpu_mode:
        train_size = config.get('train_subset_size', 50)
        val_size = config.get('val_subset_size', 20)
        
        if train_size and len(train_dataset) > train_size:
            train_dataset.metadata = train_dataset.metadata.iloc[:train_size]
            print(f"CPU mode: Limited train set to {train_size} samples")
        
        if val_size and len(val_dataset) > val_size:
            val_dataset.metadata = val_dataset.metadata.iloc[:val_size]
            print(f"CPU mode: Limited val set to {val_size} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if not cpu_mode else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if not cpu_mode else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if not cpu_mode else False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test the dataset loaders."""
    
    print("Testing dataset loaders...")
    
    # Paths
    train_path = "./data/processed/train_metadata.csv"
    val_path = "./data/processed/val_metadata.csv"
    test_path = "./data/processed/test_metadata.csv"
    
    if not os.path.exists(train_path):
        print("Metadata not found. Please run preprocess.py first.")
        exit(1)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path,
        val_path,
        test_path,
        mode='both'
    )
    
    # Test one batch
    print("\nTesting one batch from train loader...")
    batch = next(iter(train_loader))
    
    print(f"Video shape: {batch['video'].shape if batch['video'] is not None else None}")
    print(f"Audio shape: {batch['audio'].shape if batch['audio'] is not None else None}")
    print(f"Emotion labels shape: {batch['emotion_label'].shape}")
    print(f"Congruence labels shape: {batch['congruence_label'].shape}")
    print(f"Sample emotions: {[m['emotion'] for m in batch['metadata']]}")
    
    print("\nDataset loaders test complete!")