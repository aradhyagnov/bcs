# AXIAL: Attention-based eXplainability for Interpretable Alzheimer's Diagnosis

## 🧠 Introduction

### Problem Statement
Accurate early diagnosis of Alzheimer’s Disease (AD) remains a clinical challenge. Traditional deep learning approaches using 3D MRI are powerful but lack interpretability, hindering their adoption in real-world settings.

### Project Goal
This project introduces a preprocessing and classification pipeline based on the AXIAL framework to:
- Enable interpretable AD diagnosis from 3D MRI data.
- Generate 3D attention maps localizing disease-affected brain regions.

---


## ⚙️ Methodology


The AXIAL pipeline introduces a diagnosis and explainability framework that processes 3D MRI brain scans as a sequence of 2D slices, applying attention mechanisms to identify both diagnosis and region-specific explanations. This enables precise and interpretable Alzheimer’s Disease classification.

---

### Preprocessing Steps

High-quality preprocessing is essential to ensure reliable deep learning performance. The original AXIAL study employed the following steps:

1. **Bias Field Correction (Skipped)**  
   We skipped this step because N3 bias field correction had already been applied in our dataset.

2. **Affine Registration to MNI152 Space**  
   To ensure consistency across different subjects’ MRI scans, each image is aligned to a common anatomical reference frame using affine registration. This  process is carried out by employing the Symmetric Normalization (SyN) algorithm—an efficient and accurate registration method. The target template used for this alignment is the ICBM 2009c nonlinear symmetric brain atlas, which serves as a standardized anatomical space. Registering all images to this template allows for meaningful voxel-wise comparisons and consistent spatial correspondence across subjects. This step is critical for enabling the model to learn generalized spatial patterns related to Alzheimer’s Disease rather than subject-specific anatomical variations.

3. **Skull Stripping (Brain Extraction)**  
  To isolate the brain from surrounding anatomical structures, a skull stripping procedure is applied to each MRI volume. BET works by identifying and removing non-brain tissues such as the skull, scalp, neck, and cerebrospinal fluid. This step is essential for eliminating irrelevant regions that could introduce noise or bias into the model’s learning process. By focusing exclusively on brain tissue, the model can better detect structural changes and patterns that are meaningful for diagnosing Alzheimer’s Disease

---

### Model Architecture

#### 1. Feature Extraction Module

- **Backbone**: Pretrained 2D CNNs (e.g., VGG16) are applied to each 2D slice.
- **Preprocessing**: Each 1-channel brain slice is resized to 224x224. Since the backbone expects 3-channel input, input filters are summed across channels for efficiency.
- **Output**: Generates feature vectors from each slice using shared convolution weights.

#### 2. Attention XAI Fusion Module

- **Aim**: Learn the relative importance of each slice in the overall 3D MRI volume.
- **Mechanism**:
  - A fully connected layer outputs unnormalized attention weights.
  - A softmax function normalizes the weights.
  - Final feature vector = weighted sum of all slice feature vectors.
- **Benefit**: Enables the model to focus on diagnostically relevant slices, enabling both prediction and explainability.

#### 3. Diagnosis Module

- Takes the fused feature vector and performs binary classification using a fully connected head layer with softmax activation.
- **Outputs**: Probability distribution for diagnostic classes (e.g., AD vs CN).

#### 4. XAI Attention Map Generation

- Separate diagnosis+attention networks are trained for each slicing plane: axial, sagittal, coronal.
- Attention scores from all three planes are combined to generate a 3D attention heatmap
- The attention map is min-max normalized.

#### 5. Brain Region Quantification

- **Overlay** :
The 3D attention map is overlaid onto the subject’s MRI scan, which has been normalized to the MNI152 template. This ensures anatomical consistency across subjects.
- **Atlas Mapping**:
A brain atlas is used to map each activated voxel in the attention map to a specific anatomical brain region, allowing interpretation of model focus in clinically meaningful terms.
- **Statistics Computed per Region**:
-Mean attention score: Average importance assigned to the region by the model.
-Maximum and minimum attention values: Highlight the most and least activated points within the region.
-Standard deviation: Measures variability in attention distribution across the region.
-Overlap volume (Vr): Total number of voxels in a region that intersect with the attention map.
-Percentage of region activated (Pr): Indicates how much of the region is highlighted as relevant by the model, helping to assess diagnostic saliency.

---

### Transfer Learning Strategy

The pipeline employs **Double Transfer Learning** for sMCI vs pMCI prediction:
1. Train on AD vs CN (large data, more distinguishable).
2. Fine-tune on sMCI vs pMCI using the previously trained model.
3. This hierarchical strategy improves generalization and focuses learning on subtle morphological changes.



### Tools & Libraries
- PyTorch
- ANTs, FSL
- Matplotlib, Seaborn (for visualizations)
- Nibabel, Nilearn (for neuroimaging I/O)

### Parameter Choices
- Backbone: VGG16 (best performance)

Justification: These configurations were found optimal through ablation experiments and cross-validation.

---
