# Satellite Image Classification: A Comparative Study

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## Overview
This repository contains a comprehensive comparative study on **Satellite Image Classification** using the **EuroSAT dataset**. The primary objective is to evaluate and compare the performance of various Deep Learning architectures and training paradigms across different data availability regimes, including **Full Data, Few-Shot (5-shot, 10-shot), and Zero-Shot** scenarios.

By exploring different methodologies—ranging from custom Convolutional Neural Networks (CNNs) to Self-Supervised Learning and Vision Transformers—this project aims to highlight the trade-offs between model complexity, data efficiency, and classification accuracy in the domain of Earth Observation.

## Dataset: EuroSAT
The study uses the **EuroSAT** dataset, a widely recognized benchmark for land use and land cover classification based on **Sentinel-2 satellite imagery**. 

**Key Dataset Details:**
* **Total Images:** 27,000 labeled and geo-referenced image patches.
* **Image Size & Resolution:** 64x64 pixels with a spatial resolution of 10 meters per pixel.
* **Spectral Bands:** While Sentinel-2 captures 13 multi-spectral bands, this study primarily focuses on the **RGB** optical bands.
* **Classes:** The dataset is categorized into 10 distinct land cover classes:
  *AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.*

To prevent data leakage, the dataset is strictly partitioned into distinct train, validation, and test splits. The few-shot sets (5-shot and 10-shot) are created as strict subsets of the training set to evaluate data-constrained performance accurately.

## Frameworks & Libraries
The implementation heavily relies on the modern Python Deep Learning ecosystem:
- **PyTorch & Torchvision**: For building model architectures, training loops, and complex data augmentations.
- **Hugging Face Transformers**: For deploying Vision Transformers (ViT) and OpenAI's CLIP model.
- **PEFT (Parameter-Efficient Fine-Tuning)**: For implementing LoRA (Low-Rank Adaptation).
- **Scikit-Learn**: For evaluation metrics, t-SNE feature visualizations, and confusion matrices.
- **Torchinfo & Torchview**: For generating detailed model summaries and architectural graphs.

## Models & Techniques Implemented
This study systematically implements and evaluates the following approaches:

### 1. Custom CNN & Baseline Transfer Learning
- **Custom CNN**: A convolutional baseline built from scratch to establish a lower bound.
- **Transfer Learning**: Evaluating pre-trained **ResNet50** and **EfficientNet-B0** models, leveraging ImageNet weights and fine-tuning their classifier heads for satellite imagery.

### 2. Vision Transformers (ViT)
- Fully fine-tuning a pre-trained **ViT-Base (`google/vit-base-patch16-224`)** for image classification, serving as the high-performance benchmark.

### 3. Zero-Shot Learning with CLIP
- Utilizing OpenAI's **CLIP** model for zero-shot classification.
- Implementing **Prompt Ensembling** using domain-specific prompts (e.g., *"an aerial view of {} terrain"*) to significantly improve zero-shot accuracy compared to generic prompts.

### 4. Self-Supervised Learning (SimCLR)
- Pre-training a **ResNet** backbone using **SimCLR** (contrastive learning with NT-Xent loss) on unlabeled data.
- Heavy augmentations (RandomResizedCrop, ColorJitter, GaussianBlur) are applied to force the model to ignore trivial pixel details and learn robust, domain-specific representations (e.g., shape and texture).

### 5. Parameter-Efficient Fine-Tuning (LoRA)
- Applying **LoRA (Low-Rank Adaptation)** to a frozen Vision Transformer.
- This technique adds small, trainable rank decomposition matrices into the transformer layers, drastically reducing compute requirements while maintaining high accuracy.

## Comparative Results

### Table 1: Performance Across Data Regimes

| Method                   | Full Data (%) | 10-shot (%) | 5-shot (%) | Zero-shot (%) | Training Cost (min)               |
| :----------------------- | :------------ | :---------- | :--------- | :------------ | :-------------------------------- |
| **Baseline CNN** | 89.51         | N/A         | N/A        | N/A           | 2.38                              |
| **ViT (Full FT)** | 98.47         | 86.27       | 83.31      | N/A           | 76.52                             |
| **Prototype Classifier** | N/A           | 76.35       | 73.58      | N/A           | Very Low (Instant)                |
| **CLIP Zero-Shot** | N/A           | N/A         | N/A        | 42.57         | Zero (Inference only)             |
| **SimCLR (linear eval)** | 82.57         | N/A         | N/A        | N/A           | ~60 (pre-train) + 3.55            |
| **SimCLR (fine-tuned)** | 94.42         | 59.38       | 58.15      | N/A           | ~60 (pre-train) + 6.64 + 0.67 + 1 |
| **LoRA (ViT)** | 98.62         | N/A         | N/A        | N/A           | 103.27                            |

### Table 2: Transfer Learning vs. Fine-Tuning Strategies (Full Data)

| Model               | Strategy | Test Accuracy (%) | Training Time (minutes) | Trainable Parameters |
| :------------------ | :------- | :---------------- | :---------------------- | :------------------- |
| **ResNet50** | Full FT  | 96.12             | 8.33                    | 23,528,522           |
| **ResNet50** | Frozen   | 86.86             | 2.10                    | 20,490               |
| **EfficientNet-B0** | Full FT  | 96.37             | 3.16                    | 4,020,358            |
| **EfficientNet-B0** | Frozen   | 76.79             | 1.78                    | 12,810               |
| **ViT-B/16** | Full FT  | 98.47             | 76.52                   | 85,806,346           |
| **ViT-B/16** | Frozen   | 95.80             | 76.83                   | 7,690                |

### Key Observations

* **The Accuracy Ceiling:** Vision Transformers represent the state-of-the-art for this dataset. Both **ViT-B/16 (Full FT)** and **LoRA (ViT)** hit an impressive ~98.5% accuracy. However, this comes at a steep computational cost (~76–103 minutes) compared to CNN baselines.
* **Full Fine-Tuning vs. Frozen Backbones:** Across all architectures, unfreezing and training all parameters (Full FT) yields significantly higher accuracy than just training a frozen classification head. Interestingly, a *frozen* **ViT-B/16** (95.80%) still massively outperforms a *frozen* **ResNet50** (86.86%) and is highly competitive with fully fine-tuned CNNs, proving the incredible representational power of pre-trained transformer embeddings.
* **The Efficiency Sweet Spot:** **EfficientNet-B0 (Full FT)** offers arguably the best performance-to-cost ratio. It achieves an excellent 96.37% test accuracy in just 3.16 minutes, requiring only ~4 million trainable parameters compared to ResNet50's ~23.5 million or ViT's ~85 million.
* **Parameter Efficiency with LoRA:** Adding **LoRA** to the ViT model matches the top accuracy of full fine-tuning (98.62%) while bypassing the need to update all 85.8 million parameters. This demonstrates how modern PEFT methods can democratize transformer training on consumer hardware.
* **Few-Shot & Zero-Shot Capabilities:** In data-starved scenarios, **ViT (Full FT)** remains highly robust (83.31% on 5-shot). For faster, parameter-free evaluation, the **Prototype Classifier** yields solid results instantly (~73.5% on 5-shot). Without *any* training data, **CLIP Zero-Shot** manages a respectable 42.57% baseline using purely semantic text-image alignment.
