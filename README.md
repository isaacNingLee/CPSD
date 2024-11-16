
# DR-CPSD: Dream Replay with CLIP Projection Stable Diffusion

![Project Logo](resource\cpsd_block.png)

---

## Overview

**DR-CPSD** is a algorithm designed to address the limitations of generative replay in continual learning (CL) settings, particularly for image classification tasks. It leverages the power of **CLIP embeddings**, **Stable Diffusion models**, and **teacher-student distillation** to enhance stability and plasticity while minimizing distribution gaps and classification bias between real and synthetic images.

### Key Features
1. **Reduced Distribution Gap**: Utilised Maximum Mean Discrepancy (MMD) loss to minimize the divergence between real and synthetic image distributions. MMD loss is insipired by [Yuan et. al]( https://openreview.net/forum?id=svIdLLZpsA) 
2. **Real-Synthetic Isolation**: Ensures better classification performance by isolating training objectives for real and synthetic images.
3. **Single-Epoch Replay**: Combines Exponential Moving Average (EMA) initialization and Teacher-Student Distillation for faster convergence.

---

## Algorithm

### CPSD Embedding
CPSD embeddings are initialized using an **Identity Matrix** and trained for one epoch with real images to learn semantic and structural features. These embeddings are stored in a buffer and reused for synthetic image generation in subsequent tasks.

### Dream Replay
Dream Replay (DR) applies generative replay only after the classifier has been trained on the current task, mitigating the confusion caused by simultaneously training on real and synthetic samples.

### Objective Functions
### Objective Functions

- **CPSD Loss**:
  $$\mathcal{L}_{CPSD} = \mathcal{L}_{LDM} + \lambda \mathcal{L}_{MMD}$$

- **Real-Synthetic Isolation**:
  $$\mathcal{L}_{CE} = \mathcal{L}_{CE}(\hat{y}_{student}, \bar{y}) + \kappa \mathcal{L}_{CE}(\hat{y}_{student}, y)$$

- **Distillation Loss**:
  $$\mathcal{L}_{Distill} = \mathcal{L}_{MSE}(h_{student}, h_{old}) + \pi \mathcal{L}_{MSE}(h_{student}, h_{new})$$


---
## Results

- Faster convergence with single-epoch CPSD embedding training.
- Significant reduction in distribution gap between real and synthetic images.
- Improved task performance by balancing stability and plasticity.

---
