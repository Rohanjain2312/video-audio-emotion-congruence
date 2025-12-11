"""
Full multimodal model combining video and audio backbones,
fusion module, and classification heads.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE

# Import components
from models.video_backbone import VideoBackbone
from models.audio_backbone import AudioBackbone
from models.fusion_module import FusionModule, EmotionClassifier, CongruenceClassifier


class MultimodalEmotionModel(nn.Module):
    """
    Full multimodal model for video-audio emotion recognition and congruence detection.
    
    Architecture:
        1. Video Backbone (VideoMAE) -> video_features
        2. Audio Backbone (Wav2Vec2) -> audio_features
        3. Fusion Module -> fused_features
        4. Emotion Classifier -> emotion_logits
        5. Congruence Classifier -> congruence_logits
    """
    
    def __init__(
        self,
        num_emotions=8,
        video_model_name="MCG-NJU/videomae-base-short",
        audio_model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        freeze_backbones=True,
        fusion_dim=512,
        dropout=0.3
    ):
        """
        Args:
            num_emotions: Number of emotion classes
            video_model_name: HuggingFace model name for video
            audio_model_name: HuggingFace model name for audio
            freeze_backbones: If True, freeze pretrained backbones
            fusion_dim: Dimension of fused features
            dropout: Dropout rate
        """
        super(MultimodalEmotionModel, self).__init__()
        
        print("\nInitializing MultimodalEmotionModel...")
        
        # Video backbone
        self.video_backbone = VideoBackbone(
            model_name=video_model_name,
            freeze_backbone=freeze_backbones
        )
        video_dim = self.video_backbone.get_output_dim()
        
        # Audio backbone
        self.audio_backbone = AudioBackbone(
            model_name=audio_model_name,
            freeze_backbone=freeze_backbones
        )
        audio_dim = self.audio_backbone.get_output_dim()
        
        # Fusion module
        self.fusion = FusionModule(
            video_dim=video_dim,
            audio_dim=audio_dim,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # Emotion classifier (multiclass)
        self.emotion_classifier = EmotionClassifier(
            input_dim=fusion_dim,
            num_emotions=num_emotions,
            dropout=dropout
        )
        
        # Congruence classifier (binary)
        self.congruence_classifier = CongruenceClassifier(
            input_dim=fusion_dim,
            dropout=dropout
        )
        
        print("MultimodalEmotionModel initialized successfully!\n")
    
    def forward(self, video, audio):
        """
        Forward pass through the full model.
        
        Args:
            video: (batch_size, num_frames, channels, height, width)
            audio: (batch_size, num_samples)
        
        Returns:
            Dictionary with:
                - emotion_logits: (batch_size, num_emotions)
                - congruence_logits: (batch_size, 2)
                - video_features: (batch_size, video_dim)
                - audio_features: (batch_size, audio_dim)
                - fused_features: (batch_size, fusion_dim)
        """
        # Extract features from video
        video_features = self.video_backbone(video)
        
        # Extract features from audio
        audio_features = self.audio_backbone(audio)
        
        # Fuse features
        fused_features = self.fusion(video_features, audio_features)
        
        # Classify emotion
        emotion_logits = self.emotion_classifier(fused_features)
        
        # Classify congruence
        congruence_logits = self.congruence_classifier(fused_features)
        
        return {
            'emotion_logits': emotion_logits,
            'congruence_logits': congruence_logits,
            'video_features': video_features,
            'audio_features': audio_features,
            'fused_features': fused_features
        }
    
    def forward_video_only(self, video):
        """
        Forward pass using only video (for video-only baseline).
        
        Args:
            video: (batch_size, num_frames, channels, height, width)
        
        Returns:
            emotion_logits: (batch_size, num_emotions)
        """
        video_features = self.video_backbone(video)
        # Use emotion classifier directly on video features
        # Need a separate head for this, but for now we'll use fusion with zeros
        dummy_audio_features = torch.zeros_like(video_features[:, :self.audio_backbone.get_output_dim()]).to(video.device)
        fused_features = self.fusion(video_features, dummy_audio_features)
        emotion_logits = self.emotion_classifier(fused_features)
        return emotion_logits
    
    def forward_audio_only(self, audio):
        """
        Forward pass using only audio (for audio-only baseline).
        
        Args:
            audio: (batch_size, num_samples)
        
        Returns:
            emotion_logits: (batch_size, num_emotions)
        """
        audio_features = self.audio_backbone(audio)
        # Use emotion classifier directly on audio features
        # Need a separate head for this, but for now we'll use fusion with zeros
        dummy_video_features = torch.zeros_like(audio_features[:, :self.video_backbone.get_output_dim()]).to(audio.device)
        fused_features = self.fusion(dummy_video_features, audio_features)
        emotion_logits = self.emotion_classifier(fused_features)
        return emotion_logits


if __name__ == "__main__":
    """Test the full model."""
    
    print("Testing MultimodalEmotionModel...")
    
    # Create model
    model = MultimodalEmotionModel(
        num_emotions=8,
        freeze_backbones=True
    )
    model = model.to(DEVICE)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    num_frames = 16
    channels = 3
    height = 224
    width = 224
    num_samples = 16000 * 5  # 5 seconds
    
    dummy_video = torch.randn(batch_size, num_frames, channels, height, width).to(DEVICE)
    dummy_audio = torch.randn(batch_size, num_samples).to(DEVICE)
    
    print(f"\nInput shapes:")
    print(f"  Video: {dummy_video.shape}")
    print(f"  Audio: {dummy_audio.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_video, dummy_audio)
    
    print(f"\nOutput shapes:")
    print(f"  Emotion logits: {outputs['emotion_logits'].shape}")
    print(f"  Congruence logits: {outputs['congruence_logits'].shape}")
    print(f"  Video features: {outputs['video_features'].shape}")
    print(f"  Audio features: {outputs['audio_features'].shape}")
    print(f"  Fused features: {outputs['fused_features'].shape}")
    
    print(f"\nPredicted emotions: {torch.argmax(outputs['emotion_logits'], dim=1)}")
    print(f"Predicted congruence: {torch.argmax(outputs['congruence_logits'], dim=1)}")
    
    print("\nMultimodalEmotionModel test complete!")