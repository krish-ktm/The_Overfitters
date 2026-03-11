# Project Abstract

**Project Title:** U-Net: Implementation, Reproduction, and Empirical Analysis of Skip Connections for Biomedical Image Segmentation

**Team:** The Overfitters (Ramyashree, Ashwini Vitekar, Krishkumar Patel)
**Course:** Machine Learning Project Submission (CS 6140)
**Instructor:** Prof. Smruthi Mukund

## Objective and Goal of the Project
The primary goal of this project is to implement, reproduce, and empirically analyze the U-Net architecture proposed by Ronneberger, Fischer, and Brox (2015) for biomedical image segmentation.

We will build U-Net from scratch in PyTorch and evaluate it on two publicly available biomedical datasets:
* Kaggle 2018 Data Science Bowl nuclei segmentation dataset
* DRIVE retinal vessel segmentation dataset

We will conduct a systematic ablation study to isolate the contribution of key architectural components. Specifically, we will:
* Remove and modify skip connections
* Vary encoder depth
* Compare concatenation vs. additive skip strategies
* Analyze gradient flow and convergence behavior
* Visualize intermediate feature maps
* Test performance under limited training data (10%, 25%, 50%, 100%)

A plain FCN baseline (encoder-decoder without skip connections) will serve as our comparison point. All models will be evaluated using the following metrics:
* Dice coefficient
* Intersection over Union (IoU)
* Pixel accuracy
* Precision
* Recall

## Motivation and Significance
Biomedical image segmentation is a critical task in healthcare, requiring automated delineation of structures such as cells, nuclei, and blood vessels from microscopy and medical images. 

U-Net remains one of the most widely used architectures for this task, yet most implementations simply train the model and report accuracy without deeply investigating why its design choices work. 

Our motivation for choosing this paper is twofold:
* To deeply understand how U-Net's architectural innovations — the symmetric encoder-decoder structure and skip connections — improve segmentation performance.
* To conduct empirical analysis through ablation experiments and controlled comparisons.

By systematically modifying and evaluating architectural components, we aim to provide a clearer understanding of which elements contribute most to U-Net's success in biomedical segmentation tasks.
