# Plant Disease Detection using Unified Self-Supervised Learning
---
This repository provides the implementation of the paper.

X. Huan, B. Chen, H. Zhou, "A Unified Self-Supervised Learning Framework for Plant Disease Detection" 

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
- Pytorch 2.1.0+
- NVIDIA GPU with CUDA and CuDNN support


## Files in src folder

1. **Install Requirements**
   - Prepare datasets and create the unlabeled dataset by running:
     ```bash
     datasets_library_setup.ipynb
     ```
2. **Pretraining**
   - Build and pretrain the SSL model with:
     ```bash
     BYOL_MIM_Contrastive_101_model.ipynb
     ```
   - Output file:
     ```bash
     byol_mim_contrastive_epoch85.pth
     ```
## Output: Pretrained Models
You can download the generated models here for the following programs:

- [byol_mim_contrastive_epoch85.pth](https://drive.google.com/file/d/1VubUCwEF2W9CgI_2SKQloLtDzkrfwSVC/view?usp=sharing): Self-supervised pretrained model.
- [plantdoc_best_finetuned_byol_mim_contrastive_epoch85.pth1.pth](https://drive.google.com/file/d/1LiMHRE4pDQ9-pbMNlwHE9YiuokctUNDn/view?usp=sharing): Fine-tuned model on PlantDoc.
- [plantvillage_best_finetuned_byol_mim_contrastive_epoch85.pth.pth](https://drive.google.com/file/d/1diQCL78KXzptXAlKZjXcF7GFH1KEu7R2/view?usp=sharing): Fine-tuned model on PlantVillage.
3. **Downstream Fine-Tuning and Evaluation**
   - Classification on the PlantDoc dataset:
     ```bash
     plantDoc_downstream_classification.ipynb
     ```
     - Input file:
     ```bash
     byol_mim_contrastive_epoch85.pth
     ```
     - Output file:
     ```bash
     plantdoc_best_finetuned_byol_mim_contrastive_epoch85.pth1.pth
     ```
   - Classification on the PlantVillage dataset:
     ```bash
     plantVillage_downstream_classification_plot.ipynb
     ```
     - Input file:
     ```bash
     byol_mim_contrastive_epoch85.pth
     ```
     - Output file:
     ```bash
     plantvillage_best_finetuned_byol_mim_contrastive_epoch85.pth.pth
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
     - Input file:
       ```bash
       plantdoc_best_finetuned_byol_mim_contrastive_epoch85.pth1.pth
       ```

5. **Feature Space Analysis (t-SNE)**
   - Visualize learned feature embeddings:
     ```bash
     plantVillage_t-SNE.ipynb
     ```
     - Input file:
     ```bash
     plantvillage_best_finetuned_byol_mim_contrastive_epoch85.pth.pth
     ```

