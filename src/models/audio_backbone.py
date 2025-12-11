"""
Audio backbone using Wav2Vec2 pretrained model.
Extracts audio features from waveforms.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
import sys
from pathlib import Path

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE


class AudioBackbone(nn.Module):
    """
    Audio feature extractor using Wav2Vec2.
    Model: wav2vec2-lg-xlsr-en-speech-emotion-recognition
    """
    
    def __init__(
        self,
        model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        freeze_backbone=True,
        output_dim=1024
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze_backbone: If True, freeze backbone weights
            output_dim: Dimension of output features
        """
        super(AudioBackbone, self).__init__()
        
        print(f"Loading Wav2Vec2 model: {model_name}")
        
        # Load pretrained Wav2Vec2
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.output_dim = output_dim
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Wav2Vec2 backbone frozen")
        else:
            print("Wav2Vec2 backbone trainable")
        
        # Get hidden size from config
        self.hidden_size = self.model.config.hidden_size
        
        print(f"Wav2Vec2 loaded. Hidden size: {self.hidden_size}")
    
    def forward(self, audio):
        """
        Forward pass through Wav2Vec2.
        
        Args:
            audio: Tensor of shape (batch_size, num_samples)
        
        Returns:
            Tensor of shape (batch_size, hidden_size) - pooled audio features
        """
        batch_size = audio.shape[0]
        
        # Forward through Wav2Vec2
        outputs = self.model(audio)
        
        # Get the last hidden state
        # Shape: (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        
        # Pool over time dimension (mean pooling)
        audio_features = torch.mean(last_hidden_state, dim=1)  # (batch_size, hidden_size)
        
        return audio_features
    
    def get_output_dim(self):
        """Return output feature dimension."""
        return self.hidden_size


if __name__ == "__main__":
    """Test the audio backbone."""
    
    print("Testing AudioBackbone...")
    
    # Create model
    model = AudioBackbone(freeze_backbone=True)
    model = model.to(DEVICE)
    model.eval()
    
    # Create dummy input
    # Shape: (batch_size, num_samples)
    batch_size = 2
    num_samples = 16000 * 5  # 5 seconds at 16kHz
    
    dummy_audio = torch.randn(batch_size, num_samples)
    dummy_audio = dummy_audio.to(DEVICE)
    
    print(f"Input shape: {dummy_audio.shape}")
    
    # Forward pass
    with torch.no_grad():
        features = model(dummy_audio)
    
    print(f"Output features shape: {features.shape}")
    print(f"Output dim: {model.get_output_dim()}")
    
    print("\nAudioBackbone test complete!")