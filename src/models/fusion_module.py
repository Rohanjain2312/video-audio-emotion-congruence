"""
Fusion module for combining video and audio features.
Includes emotion and congruence classification heads.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE


class FusionModule(nn.Module):
    """
    Fusion module using concatenation + MLP.
    
    Why concatenation + MLP:
    - Simple and effective baseline
    - Allows the model to learn arbitrary feature interactions
    - Computationally efficient
    - Works well when both modalities are informative
    """
    
    def __init__(
        self,
        video_dim=768,
        audio_dim=1024,
        fusion_dim=512,
        dropout=0.3
    ):
        """
        Args:
            video_dim: Dimension of video features
            audio_dim: Dimension of audio features
            fusion_dim: Dimension of fused features
            dropout: Dropout rate
        """
        super(FusionModule, self).__init__()
        
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.fusion_dim = fusion_dim
        
        # Concatenation-based fusion
        concat_dim = video_dim + audio_dim
        
        # MLP for fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        print(f"FusionModule initialized: {concat_dim} -> {fusion_dim}")
    
    def forward(self, video_features, audio_features):
        """
        Fuse video and audio features.
        
        Args:
            video_features: (batch_size, video_dim)
            audio_features: (batch_size, audio_dim)
        
        Returns:
            fused_features: (batch_size, fusion_dim)
        """
        # Concatenate features
        concat_features = torch.cat([video_features, audio_features], dim=1)
        
        # Pass through fusion MLP
        fused_features = self.fusion_mlp(concat_features)
        
        return fused_features
    
    def get_output_dim(self):
        """Return output feature dimension."""
        return self.fusion_dim


class EmotionClassifier(nn.Module):
    """
    Emotion classification head.
    Predicts emotion labels from features.
    """
    
    def __init__(
        self,
        input_dim=512,
        num_emotions=8,
        hidden_dim=256,
        dropout=0.3
    ):
        """
        Args:
            input_dim: Dimension of input features
            num_emotions: Number of emotion classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(EmotionClassifier, self).__init__()
        
        self.num_emotions = num_emotions
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions)
        )
        
        print(f"EmotionClassifier initialized: {input_dim} -> {num_emotions} emotions")
    
    def forward(self, features):
        """
        Classify emotion from features.
        
        Args:
            features: (batch_size, input_dim)
        
        Returns:
            logits: (batch_size, num_emotions)
        """
        logits = self.classifier(features)
        return logits


class CongruenceClassifier(nn.Module):
    """
    Congruence classification head.
    Predicts whether video and audio emotions match (congruent=1) or conflict (incongruent=0).
    """
    
    def __init__(
        self,
        input_dim=512,
        hidden_dim=128,
        dropout=0.3
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(CongruenceClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary: congruent vs incongruent
        )
        
        print(f"CongruenceClassifier initialized: {input_dim} -> 2 classes")
    
    def forward(self, features):
        """
        Classify congruence from features.
        
        Args:
            features: (batch_size, input_dim)
        
        Returns:
            logits: (batch_size, 2)
        """
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    """Test the fusion module and classifiers."""
    
    print("Testing FusionModule and Classifiers...")
    
    # Test fusion module
    video_dim = 768
    audio_dim = 1024
    fusion_dim = 512
    batch_size = 4
    
    fusion = FusionModule(video_dim, audio_dim, fusion_dim)
    fusion = fusion.to(DEVICE)
    
    # Create dummy features
    dummy_video = torch.randn(batch_size, video_dim).to(DEVICE)
    dummy_audio = torch.randn(batch_size, audio_dim).to(DEVICE)
    
    print(f"\nVideo features shape: {dummy_video.shape}")
    print(f"Audio features shape: {dummy_audio.shape}")
    
    # Test fusion
    fused = fusion(dummy_video, dummy_audio)
    print(f"Fused features shape: {fused.shape}")
    
    # Test emotion classifier
    num_emotions = 8
    emotion_classifier = EmotionClassifier(fusion_dim, num_emotions)
    emotion_classifier = emotion_classifier.to(DEVICE)
    
    emotion_logits = emotion_classifier(fused)
    print(f"\nEmotion logits shape: {emotion_logits.shape}")
    print(f"Predicted emotions: {torch.argmax(emotion_logits, dim=1)}")
    
    # Test congruence classifier
    congruence_classifier = CongruenceClassifier(fusion_dim)
    congruence_classifier = congruence_classifier.to(DEVICE)
    
    congruence_logits = congruence_classifier(fused)
    print(f"\nCongruence logits shape: {congruence_logits.shape}")
    print(f"Predicted congruence: {torch.argmax(congruence_logits, dim=1)}")
    
    print("\nFusion and Classifiers test complete!")