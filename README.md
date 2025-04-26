# Plant Disease Detection using Unified Self-Supervised Learning
---
This repository provides the implementation of the paper.

X. Huan, B. Chen, H. Zhou, "Learning Without Labels: A Unified Self-Supervised Approach for Plant Disease Detection" 

---
## Dataset
The dataset used in this study include:
- [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)

---       
## Getting Started
### Prerequisites
To run the project, ensure the following are installed:
- Python 3.10   
- Pytorch 2.2.0
- NVIDIA GPU with CUDA and CuDNN support (for training speedup)

### Running the Code

#### Example Directory Structure

## Setup Instructions

1. **Install Requirements**
   - Install the necessary Python libraries.
   - Prepare datasets and create the unlabeled dataset by running:
     ```bash
     datasets_library_setup.ipynb
     ```

2. **Pretraining**
   - Build and pretrain the SSL model with:
     ```bash
     BYOL_MIM_Contrastive_101_model.ipynb
     ```

3. **Downstream Fine-Tuning and Evaluation**
   - Fine-tune and evaluate on the PlantDoc dataset:
     ```bash
     plantDoc_downstream_classification.ipynb
     ```
   - Fine-tune and evaluate on the PlantVillage dataset:
     ```bash
     plantVillage_downstream_classification.ipynb
     ```

4. **Model Interpretability (Grad-CAM)**
   - Visualize important regions influencing predictions:
     - Apple Rust Leaf Example:
       ```bash
       Apple_rust_leaf_Grad_CAM.ipynb
       ```
     - Grape Leaf Black Rot Example:
       ```bash
       grape_leaf_black_rot_Grad_CAM.ipynb
       ```

5. **Feature Space Analysis (t-SNE)**
   - Visualize learned feature embeddings:
     ```bash
     plantVillage_t-SNE.ipynb
     ```

6. **Training and Validation Curves**
   - Plot training/validation accuracy and loss:
     ```bash
     plantVillage_training_val_plot.ipynb
     ```

---

## Project Overview

- **Backbone:** ResNet-101 pretrained on ImageNet
- **Self-Supervised Objectives:** 
  - BYOL for global semantic alignment
  - Masked Image Modeling (MIM) for local structure reconstruction
  - Contrastive Learning for instance-level discrimination
- **Loss Function:** Hybrid multi-objective loss combining BYOL, MIM, and InfoNCE objectives
