# 🧠 MRI Data Visualization - Week 2 Assignment

## 📌 Objective

The goal of this assignment is to develop a deep understanding of how to read, parse, and visualize medical imaging data stored in **NIfTI** (`.nii`) and **DICOM** (`.dcm`) formats using Python.

---

## 📁 Repository Structure

```
repo_root/
├── week2/
│   ├── read_viz.ipynb          # Main notebook with core answers
│   ├── enhanced_read_viz.ipynb # Bonus: Interactive & 3D visualizations
│   ├── report.md               # Report merged here
│   ├── Sample_Data/
│   │   ├── sub-65304_ses-1_acq-t1csmp2ragesag06mmUNIDEN_T1w.nii
│   │   ├── 0002.DCM
│   │   └── axial_slices.gif    # GIF of axial slices (auto-generated)
```

---

## ✅ Completed Tasks

### 1. **Reading and Loading Files**
- Used `nibabel` to load `.nii` files
- Used `pydicom` to load `.dcm` files
- Included example code and visual outputs

### 2. **Inspecting Metadata**
- NIfTI: Shape, voxel size, affine matrix, header
- DICOM: Patient Name, Study Date, Modality, Pixel Spacing, Slice Thickness

### 3. **Stacking DICOM Slices**
- Demonstrated how to sort slices using `InstanceNumber` or `ImagePositionPatient`
- Used NumPy to stack slices into 3D arrays

### 4. **Visualizing Anatomical Planes**
- Displayed static axial, sagittal, and coronal slices using `matplotlib`
- Interactive visualization with `ipywidgets` and `interact`
- Exported axial slice stack as animated `.gif`

### 5. **Image Orientation**
- Interpreted NIfTI orientation using the affine matrix
- Explained DICOM orientation using metadata fields

### 6. **DICOM vs NIfTI Comparison**
| Feature       | DICOM                          | NIfTI                          |
|---------------|--------------------------------|--------------------------------|
| Use Case      | Clinical practice              | Research                       |
| Structure     | One file per slice             | One file per volume            |
| Metadata      | Extensive (tags)               | Compact header                 |
| Compatibility | Medical software, PACS         | Neuroimaging tools             |
| Ease of Use   | Complex handling, needs sorting| Easier to load and visualize   |

---

## 🧪 Description of Approach

1. **Load and inspect** individual files for format-specific structure.
2. **Extract metadata** using `header` (NIfTI) and DICOM tags (Pydicom).
3. **Visualize slices** in all three planes using both static and interactive plots.
4. **Compare formats** by organizing observations in code and markdown.
5. **Enhance with extras** like interactive widgets, histograms, 3D rendering, and GIF export.

---

## 🖼️ Screenshots of Visualizations

(Screenshots should be added manually by the user if this is rendered on GitHub.)

- Axial Slice Example
- Coronal Slice
- Sagittal View
- Interactive Viewer Widget
- 3D Volume Plot
- Voxel Intensity Histogram
- Axial GIF (Generated)

---

## 🔧 Preprocessing & Assumptions

- Only one `.dcm` file was available; assumed others would be similarly structured if added.
- Used `.get()` to prevent crashes if tags like `PixelSpacing` were missing.
- No normalization or resampling was done, since the focus was visualization.
- Image orientation is inferred without changing voxel order.

---

## 🔍 Observations & Challenges

### Challenges:
- DICOM metadata is inconsistent across datasets (e.g., missing PixelSpacing)
- Sorting DICOM slices without a full series is difficult
- Image orientation tags require careful interpretation

### Observations:
- NIfTI is far simpler for quick research workflows
- DICOM is rich in metadata but harder to manipulate in code
- Interactive tools (like widgets and 3D rendering) significantly improve user experience

---

## ✨ Enhancements (Bonus Features)

- 📊 **Voxel Intensity Histogram**
- 🔄 **Interactive Slice Viewer** with dropdown & slider
- 🌀 **3D Volume Rendering** using `plotly`
- 🖼️ **Animated GIF** export of axial slices

---

## 📅 Deadline

📌 Submitted before: **31/05/2025**, **EOD**

---

## 🧠 Learning Outcome

This project strengthened my understanding of:
- Medical image formats
- Python-based data visualization
- Working with real-world, metadata-rich data in 3D
