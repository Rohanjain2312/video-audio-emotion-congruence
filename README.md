# Video-Audio Emotion Congruence Detector

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multimodal deep learning system that detects emotions from both video and audio, and determines whether they are congruent (matching) or incongruent (conflicting).

**Developed by:** [Rohan Jain](https://www.linkedin.com/in/jaroh23/) | University of Maryland - MS in Machine Learning

---

## 🎯 Project Overview

This system performs three key tasks:
1. **Visual Emotion Recognition**: Detect emotion from facial expressions in video
2. **Audio Emotion Recognition**: Detect emotion from speech in audio
3. **Congruence Detection**: Determine if video and audio emotions match or conflict

### Why Congruence Matters
In real-world scenarios, people's facial expressions don't always match their speech tone (e.g., saying "I'm fine" with a sad expression). Detecting this mismatch is crucial for applications in mental health, customer service, and human-computer interaction.

---

## 🏗️ Architecture
```
INPUT VIDEO → ┌─────────────┐
              │  VideoMAE   │ → Video Features (768-dim)
              └─────────────┘           ↓
                                        │
                                   ┌────▼────┐
                                   │ Fusion  │ → Fused Features (512-dim)
                                   │  (MLP)  │
                                   └────┬────┘
INPUT AUDIO → ┌─────────────┐           │
              │  Wav2Vec2   │ → Audio Features (1024-dim)
              └─────────────┘           ↓
                                        │
                        ┌───────────────┴───────────────┐
                        ↓                               ↓
              ┌─────────────────┐           ┌──────────────────┐
              │ Emotion Head    │           │ Congruence Head  │
              │ (8 classes)     │           │ (binary)         │
              └─────────────────┘           └──────────────────┘
                        ↓                               ↓
              Neutral, Calm, Happy,           Match / Conflict
              Sad, Angry, Fearful,
              Disgust, Surprised
```

### Model Components
- **Video Backbone**: [VideoMAE](https://huggingface.co/MCG-NJU/videomae-base-short) (768-dim features)
- **Audio Backbone**: [Wav2Vec2](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) (1024-dim features)
- **Fusion**: Concatenation + MLP (512-dim)
- **Total Parameters**: 403M (1.4M trainable - backbones frozen)

---

## 📊 Datasets

| Dataset | Videos | Emotions | Source |
|---------|--------|----------|--------|
| [RAVDESS](https://zenodo.org/record/1188976) | 1,440 | 8 classes | Ryerson Audio-Visual Database of Emotional Speech and Song |
| [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) | 7,442 | 6 classes | Crowd-sourced Emotional Multimodal Actors Dataset |

**Unified Emotions**: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.11+
CUDA-capable GPU (recommended for training)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Rohanjain2312/video-audio-emotion-congruence.git
cd video-audio-emotion-congruence
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download datasets**
```bash
# RAVDESS (auto-download)
python src/data/download_datasets.py --dataset ravdess

# CREMA-D (requires Kaggle API)
# Setup Kaggle credentials first: https://www.kaggle.com/docs/api
python src/data/download_datasets.py --dataset cremad
```

4. **Preprocess data**
```bash
python src/data/preprocess.py
```

5. **Train the model**
```bash
# Multimodal (video + audio)
python src/training/train_multimodal.py

# Or train baselines
python src/training/train_video_only.py
python src/training/train_audio_only.py
```

6. **Run inference**
```bash
# On a single video
python src/inference/inference_pipeline.py --video path/to/video.mp4 --checkpoint checkpoints/multimodal/best_model.pth

# Launch Gradio demo
python src/inference/gradio_app.py
```

---

## 📁 Project Structure
```
video-audio-emotion-congruence/
├── data/
│   ├── raw/                    # Downloaded datasets
│   ├── processed/              # Preprocessed metadata
│   └── samples/                # Sample videos for demo
├── src/
│   ├── config.py              # Global configuration
│   ├── data/
│   │   ├── download_datasets.py
│   │   ├── preprocess.py
│   │   └── dataset_loaders.py
│   ├── models/
│   │   ├── video_backbone.py
│   │   ├── audio_backbone.py
│   │   ├── fusion_module.py
│   │   └── full_model.py
│   ├── training/
│   │   ├── train_multimodal.py
│   │   ├── train_video_only.py
│   │   └── train_audio_only.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── compare_baselines.py
│   └── inference/
│       ├── inference_pipeline.py
│       └── gradio_app.py
├── checkpoints/               # Model checkpoints (not in repo)
├── outputs/                   # Logs and metrics
├── requirements.txt
└── README.md
```

---

## 🎮 Usage Examples

### Training
```python
# Train multimodal model
python src/training/train_multimodal.py \
    --data_dir ./data/processed \
    --checkpoint_dir ./checkpoints/multimodal \
    --num_epochs 20 \
    --batch_size 16

# Train with custom parameters
python src/training/train_multimodal.py \
    --num_epochs 30 \
    --learning_rate 5e-5 \
    --batch_size 32
```

### Evaluation
```python
# Evaluate trained model
python src/evaluation/evaluate.py \
    --checkpoint checkpoints/multimodal/best_model.pth \
    --model_type multimodal

# Compare all baselines
python src/evaluation/compare_baselines.py
```

### Inference
```python
# Single video
python src/inference/inference_pipeline.py \
    --checkpoint checkpoints/multimodal/best_model.pth \
    --video test_video.mp4 \
    --output results.json

# Batch processing
python src/inference/inference_pipeline.py \
    --checkpoint checkpoints/multimodal/best_model.pth \
    --video ./videos_folder/
```

### Gradio Demo
```python
# Launch interactive demo
python src/inference/gradio_app.py \
    --checkpoint checkpoints/multimodal/best_model.pth \
    --samples ./data/samples

# With public sharing
python src/inference/gradio_app.py --share
```

---

## 📈 Results

### Performance Metrics (CPU Mode - Limited Data)

| Model | Accuracy | Macro F1 | Weighted F1 | Congruence Acc |
|-------|----------|----------|-------------|----------------|
| Multimodal | 41.2% | 29.2% | 34.5% | 100% |
| Video Only | TBD | TBD | TBD | N/A |
| Audio Only | TBD | TBD | TBD | N/A |

*Note: These are preliminary results on limited CPU training. GPU training on full datasets expected to achieve 60-70% accuracy.*

### Expected Performance (Full Training)
- **Emotion Classification**: 60-70% accuracy (state-of-the-art: ~75%)
- **Congruence Detection**: 95%+ (limited by dataset - all samples congruent)

---

## 🔬 Technical Details

### CPU vs GPU Modes

The system automatically detects available hardware and adjusts:

**CPU Mode (Testing)**:
- 2 RAVDESS actors, 100 CREMA-D samples
- Batch size: 2
- Epochs: 2
- Training time: ~10 minutes

**GPU Mode (Full Training)**:
- All 24 RAVDESS actors, full CREMA-D
- Batch size: 16
- Epochs: 20
- Training time: 2-4 hours

### Design Decisions

1. **Frozen Backbones**: Pretrained VideoMAE and Wav2Vec2 weights are frozen to:
   - Reduce training time
   - Prevent overfitting on small datasets
   - Leverage robust pretrained features

2. **Concatenation Fusion**: Simple yet effective approach that:
   - Allows model to learn arbitrary feature interactions
   - Computationally efficient
   - Works well when both modalities are informative

3. **Multi-task Learning**: Joint training on emotion and congruence:
   - Shared representations improve both tasks
   - Congruence provides additional supervision signal

---

## 🎓 Training on Google Colab

For full GPU training, use Google Colab:

1. Upload this repository to Google Drive
2. Open a new Colab notebook
3. Mount Drive and navigate to project:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/video-audio-emotion-congruence
```

4. Install dependencies:
```bash
!pip install -r requirements.txt
```

5. Download datasets:
```bash
# Setup Kaggle credentials first
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download
!python src/data/download_datasets.py --dataset both --gpu_mode
!python src/data/download_videos.py --gpu_mode
```

6. Preprocess and train:
```bash
!python src/data/preprocess.py
!python src/training/train_multimodal.py --num_epochs 20 --batch_size 16
```

---

## 🤗 Deployment

### Hugging Face Spaces (Coming Soon)

The model will be deployed on Hugging Face Spaces with a Gradio interface.

**Demo Features**:
- Upload custom videos
- Select from sample videos
- Real-time emotion prediction
- Congruence visualization

---

## 🛠️ Development

### Adding New Features

1. **New Emotion Classes**: Modify `EMOTION_MAPPING` in `src/data/preprocess.py`
2. **Different Backbones**: Replace model names in `src/models/`
3. **Alternative Fusion**: Implement in `src/models/fusion_module.py`

### Testing
```bash
# Test individual components
python src/models/video_backbone.py
python src/models/audio_backbone.py
python src/models/fusion_module.py
python src/models/full_model.py

# Test data pipeline
python src/data/dataset_loaders.py
```

---

## 📝 Citation

If you use this code in your research, please cite:
```bibtex
@misc{jain2025videaudio,
  author = {Jain, Rohan},
  title = {Video-Audio Emotion Congruence Detector},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Rohanjain2312/video-audio-emotion-congruence}
}
```

---

## 📜 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **VideoMAE**: [MCG-NJU](https://github.com/MCG-NJU/VideoMAE)
- **Wav2Vec2**: [Hugging Face Transformers](https://huggingface.co/transformers/)
- **Datasets**: RAVDESS and CREMA-D research teams
- **University of Maryland**: MS in Machine Learning Program

---

## 📧 Contact

**Rohan Jain**
- LinkedIn: [linkedin.com/in/jaroh23](https://www.linkedin.com/in/jaroh23/)
- GitHub: [@Rohanjain2312](https://github.com/Rohanjain2312)
- Project: [Video-Audio Emotion Congruence](https://github.com/Rohanjain2312/video-audio-emotion-congruence)

---

## 🔮 Future Work

- [ ] Add attention-based fusion mechanisms
- [ ] Support for more datasets (IEMOCAP, MELD)
- [ ] Real-time webcam inference
- [ ] Fine-tune backbones for improved performance
- [ ] Multi-language support
- [ ] Temporal modeling for video sequences
- [ ] Explainability visualizations (attention maps, saliency)

---

**Built with PyTorch, Transformers, and Gradio** 🚀