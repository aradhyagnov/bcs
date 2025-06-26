# AXIAL: Attention-based eXplainability for Interpretable Alzheimer's Diagnosis

## 🧠 Introduction

### Problem Statement
Accurate early diagnosis of Alzheimer’s Disease (AD) remains a clinical challenge. Traditional deep learning approaches using 3D MRI are powerful but lack interpretability, hindering their adoption in real-world settings.

### Project Goal
This project introduces a preprocessing and classification pipeline based on the AXIAL framework to:
- Enable interpretable AD diagnosis from 3D MRI data.
- Generate 3D attention maps localizing disease-affected brain regions.
- Use 2D CNNs with attention fusion to balance performance and explainability.

---


## ⚙️ Methodology

This section elaborates on the complete methodology adapted from the AXIAL paper and customized for this implementation.

### Overview
The AXIAL pipeline introduces a diagnosis and explainability framework that processes 3D MRI brain scans as a sequence of 2D slices, applying attention mechanisms to identify both diagnosis and region-specific explanations. This enables precise and interpretable Alzheimer’s Disease classification.

---

### Preprocessing Steps

High-quality preprocessing is essential to ensure reliable deep learning performance. The original AXIAL study employed the following steps:

1. **Bias Field Correction (Skipped)**  
   - **Original**: N4ITK (Tustison et al., 2010) was used to correct intensity inhomogeneities.
   - **Modified**: We skipped this step because N3 bias field correction had already been applied in our dataset.

2. **Affine Registration to MNI152 Space**  
   - **Tool**: ANTs (SyN algorithm)
   - **Template**: ICBM 2009c nonlinear symmetric
   - **Purpose**: Aligns MRI volumes from different subjects to a standard anatomical space to allow inter-subject comparability.

3. **Skull Stripping (Brain Extraction)**  
   - **Tool**: FSL BET (Smith, 2002)
   - **Purpose**: Removes non-brain tissues such as the skull, scalp, and neck to focus the model on brain tissue.

4. **Standardization to BIDS Format**  
   - **Tool**: Clinica and PyBIDS
   - **Purpose**: Converts raw ADNI data into Brain Imaging Data Structure (BIDS) format to promote reproducibility and compatibility with neuroimaging pipelines.

---

### Model Architecture

#### 1. Feature Extraction Module

- **Backbone**: Pretrained 2D CNNs (e.g., VGG16) are applied to each 2D slice.
- **Preprocessing**: Each 1-channel brain slice is resized to 224x224. Since the backbone expects 3-channel input, input filters are summed across channels for efficiency.
- **Output**: Generates feature vectors from each slice using shared convolution weights.

#### 2. Attention XAI Fusion Module

- **Goal**: Learn the relative importance of each slice in the overall 3D MRI volume.
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
- Attention scores from all three planes are combined to generate a 3D attention heatmap using:
  - \( A[i,j,k] = lpha_s[i] \cdot lpha_c[j] \cdot lpha_a[k] \)
- The attention map is min-max normalized.

#### 5. Brain Region Quantification

- **Overlay**: 3D attention map is overlaid on normalized MNI152-space MRI volume.
- **Atlas Mapping**: Each activated voxel is mapped to anatomical brain regions using a labeled brain atlas.
- **Statistics Computed**:
  - Mean, Max, Std of attention scores per region
  - Region overlap volume
  - Percentage of region activated

---

### Training and Evaluation

- **Cross-validation**: 5-fold subject-level split to ensure no data leakage.
- **Augmentation**: Random flipping of slices (p = 0.3).
- **Optimizer**: AdamW with LR = 1e-4 (fine-tuning) and 1e-5 (transfer learning).
- **Metrics**: Accuracy, Sensitivity, Specificity, Matthews Correlation Coefficient (MCC).

---

### Transfer Learning Strategy

The pipeline employs **Double Transfer Learning** for sMCI vs pMCI prediction:
1. Train on AD vs CN (large data, more distinguishable).
2. Fine-tune on sMCI vs pMCI using the previously trained model.
3. This hierarchical strategy improves generalization and focuses learning on subtle morphological changes.


### Preprocessing Steps

1. **Affine Registration**  
   - Tool: ANTs (SyN)  
   - Goal: Align images to MNI152 standard space using the ICBM 2009c template.

2. **Skull Stripping**  
   - Tool: FSL BET  
   - Goal: Remove non-brain tissue.

🛈 **Note**: We skipped the N4 bias field correction step used in the AXIAL paper because our dataset already includes N3 bias field correction. This avoids redundant intensity normalization.

### Tools & Libraries
- PyTorch
- ANTs, FSL (via Clinica)
- PyBIDS for BIDS conversion
- Matplotlib, Seaborn (for visualizations)
- Nibabel, Nilearn (for neuroimaging I/O)

### Parameter Choices
- Backbone: VGG16 (best performance)
- Slice count: 80
- Freezing: 50% of backbone layers
- Batch size: 8
- Learning rate: 1e-4 (initial), 1e-5 (transfer learning)
- Attention threshold: 99.9th percentile for heatmap binarization

Justification: These configurations were found optimal through ablation experiments and cross-validation.

---

## 💻 Implementation

### Code Structure
```
AXIAL/
│
├── preprocessing/
│   ├── registration.py
│   └── skull_strip.py
│
├── model/
│   ├── feature_extractor.py
│   ├── attention_module.py
│   ├── classifier.py
│   └── train.py
│
├── explainability/
│   └── generate_attention_map.py
│
├── utils/
│   └── data_loader.py
│
└── main.py
```

### Running the Pipeline
```bash
# Preprocess data
python preprocessing/run_pipeline.py --input_dir raw/ --output_dir preprocessed/

# Train the model
python main.py --task AD_vs_CN --epochs 100

# Generate attention map
python explainability/generate_attention_map.py --subject_id sub-123
```

### Dependencies
- Python 3.8+
- PyTorch ≥ 1.10
- ANTsPy, FSL (via Clinica)
- Nibabel, Nilearn, Matplotlib
- PyBIDS

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Results

### Preprocessing Visuals
| Step | Image |
|------|-------|
| Original | ![](imgs/original.png) |
| Registered | ![](imgs/registered.png) |
| Skull-Stripped | ![](imgs/skull_stripped.png) |

### Evaluation Metrics
**AD vs CN Task:**
- Accuracy: **85.6%**
- MCC: **0.712**

**sMCI vs pMCI Task:**
- Accuracy: **72.5%**
- MCC: **0.443**

### Challenges Faced
- Overfitting due to small dataset.
- Ensuring reproducibility across folds.
- Aggregating 2D slice information into meaningful 3D attention maps.

---

## ✅ Conclusion

### Key Takeaways
- AXIAL successfully combines interpretability and performance using 2D CNNs on 3D MRI.
- Attention-based fusion identifies clinically relevant brain regions (hippocampus, parahippocampus, amygdala).
- Double transfer learning enhances prognostic sensitivity.

### Limitations
- Requires extensive preprocessing.
- Dependent on quality and consistency of MRI data.
- Limited generalization beyond ADNI dataset.

### Future Directions
- Test with transformer backbones.
- Use larger-scale datasets.
- Integrate clinical features with imaging.

---

## 📚 References

- [AXIAL Paper (arXiv:2407.02418v2)](https://arxiv.org/abs/2407.02418v2)
- [GitHub Repository](https://github.com/GabrieleLozupone/AXIAL)
- ANTs: https://stnava.github.io/ANTs/
- FSL: https://fsl.fmrib.ox.ac.uk/
- Clinica: https://aramislab.paris.inria.fr/clinica/
- ADNI Dataset: http://adni.loni.usc.edu/

---

## 👥 Team Members
- Alice Sharma
- Vikrant Kapoor
- Ravi Verma
- Drishti Patel