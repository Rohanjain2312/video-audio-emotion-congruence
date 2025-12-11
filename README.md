# Video–Audio Emotion Congruence Detector

A multimodal deep learning system that predicts emotions from video and audio, and detects congruence between visual and auditory emotional signals.

## 🎯 Project Goal

- **Visual Emotion**: Predict emotion from video frames using VideoMAE
- **Audio Emotion**: Predict emotion from speech using Wav2Vec2
- **Fused Emotion**: Combine both modalities for overall emotion prediction
- **Congruence Detection**: Classify whether video and audio emotions match (congruent) or conflict (incongruent)

## 🏗️ Architecture

- **Video Backbone**: `MCG-NJU/videomae-base-short`
- **Audio Backbone**: `wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- **Fusion**: Concatenation + MLP (justification in docs)
- **Datasets**: RAVDESS + CREMA-D

## 📊 Datasets

| Dataset | Videos | Emotions |
|---------|--------|----------|
| RAVDESS | 1,440  | 8 classes |
| CREMA-D | 7,442  | 6 classes |

## 🚀 Quick Start

(Instructions coming soon)

## 📈 Results

(Metrics and comparisons coming soon)

## 🤗 Demo

Try the live demo on Hugging Face Spaces: (link coming soon)

## 📁 Repository Structure

(Structure coming soon)

## 🙏 Acknowledgments

- VideoMAE: [MCG-NJU](https://github.com/MCG-NJU/VideoMAE)
- Wav2Vec2: [Hugging Face](https://huggingface.co/)
- Datasets: RAVDESS, CREMA-D