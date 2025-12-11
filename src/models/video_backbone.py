"""
Video backbone using VideoMAE pretrained model.
Extracts visual features from video frames.
"""

import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEConfig, VideoMAEImageProcessor
import sys
from pathlib import Path

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE


class VideoBackbone(nn.Module):
    """
    Video feature extractor using VideoMAE.
    Model: MCG-NJU/videomae-base-short
    """
    
    def __init__(
        self,
        model_name="MCG-NJU/videomae-base-short",
        freeze_backbone=True,
        output_dim=768
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze_backbone: If True, freeze backbone weights
            output_dim: Dimension of output features
        """
        super(VideoBackbone, self).__init__()
        
        print(f"Loading VideoMAE model: {model_name}")
        
        # Load pretrained VideoMAE
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.output_dim = output_dim
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("VideoMAE backbone frozen")
        else:
            print("VideoMAE backbone trainable")
        
        # Get hidden size from config
        self.hidden_size = self.model.config.hidden_size
        
        print(f"VideoMAE loaded. Hidden size: {self.hidden_size}")
    
    def forward(self, video):
        """
        Forward pass through VideoMAE.
        
        Args:
            video: Tensor of shape (batch_size, num_frames, channels, height, width)
                   Values should be in range [0, 1]
        
        Returns:
            Tensor of shape (batch_size, hidden_size) - pooled video features
        """
        # VideoMAE expects pixel_values in shape: (batch_size, num_frames, num_channels, height, width)
        # and normalized to [0, 1] (which our data loader already does)
        
        batch_size = video.shape[0]
        
        # Forward through VideoMAE
        outputs = self.model(pixel_values=video)
        
        # Get the pooled output (CLS token representation)
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        
        # Use CLS token (first token) as video representation
        video_features = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        return video_features
    
    def get_output_dim(self):
        """Return output feature dimension."""
        return self.hidden_size


if __name__ == "__main__":
    """Test the video backbone."""
    
    print("Testing VideoBackbone...")
    
    # Create model
    model = VideoBackbone(freeze_backbone=True)
    model = model.to(DEVICE)
    model.eval()
    
    # Create dummy input matching the expected format
    # VideoMAE expects specific dimensions based on the model
    batch_size = 2
    num_frames = 16  # This must match the model's expected num_frames
    channels = 3
    height = 224
    width = 224
    
    dummy_video = torch.randn(batch_size, num_frames, channels, height, width)
    dummy_video = dummy_video.to(DEVICE)
    
    print(f"Input shape: {dummy_video.shape}")
    
    # Forward pass
    try:
        with torch.no_grad():
            features = model(dummy_video)
        
        print(f"Output features shape: {features.shape}")
        print(f"Output dim: {model.get_output_dim()}")
        print("\nVideoBackbone test complete!")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        print("\nThis might be due to frame count mismatch.")
        print("VideoMAE-base-short typically expects 16 frames.")