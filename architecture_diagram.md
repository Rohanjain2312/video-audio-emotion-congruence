# Architecture Diagram
```
                        INPUT VIDEO CLIP
                               |
                    ┌──────────┴──────────┐
                    │                     │
              VIDEO FRAMES            AUDIO WAVEFORM
                    │                     │
                    ▼                     ▼
            ┌───────────────┐     ┌──────────────┐
            │   VideoMAE    │     │   Wav2Vec2   │
            │   (frozen)    │     │   (frozen)   │
            └───────┬───────┘     └──────┬───────┘
                    │                     │
             Video Embedding       Audio Embedding
                    │                     │
                    └──────────┬──────────┘
                               │
                        ┌──────▼──────┐
                        │   FUSION    │
                        │ (concat+MLP)│
                        └──────┬──────┘
                               │
                        Fused Embedding
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
         ┌──────────────────┐  ┌─────────────────┐
         │ Emotion Head     │  │ Congruence Head │
         │ (multiclass)     │  │ (binary)        │
         └────────┬─────────┘  └────────┬────────┘
                  │                     │
                  ▼                     ▼
          Visual/Audio/Fused      Match/Conflict
              Emotion
```