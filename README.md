# 🛢️ AI-Driven Oil Spill Detection System

An end-to-end deep learning–based system for **oil spill detection, segmentation, and visualization**
using **CNN classification** and **U-Net segmentation**, designed primarily for **SAR and satellite imagery**.

This project provides:
- Oil spill detection
- Clear visual overlays
- Area distribution analysis
- A professional, interactive Streamlit-based UI

---

## 📌 Project Highlights

- Two-stage deep learning pipeline (CNN → U-Net)
- Land–Water separation to reduce false positives
- High-contrast oil spill overlays with boundary outlines
- Oil spill area distribution using pie charts
- Timestamped analysis & downloadable results
- Interactive UI with toggle controls

---

## 🧠 System Architecture Overview
![WorkFlow Architecture](https://github.com/Krish1908/oil-spill/blob/main/Workflow%20Architecture.png)


---

## 🔍 Detection Workflow

### 1️⃣ CNN Classification
- Determines whether an oil spill is present
- Prevents unnecessary segmentation when no spill exists

### 2️⃣ U-Net Segmentation
- Performs pixel-level oil spill detection
- Outputs a probability mask

### 3️⃣ Post-Processing
- Thresholding to convert probabilities into binary mask
- Morphological cleanup to remove noise
- Area filtering to remove small false regions
- Land–water masking to suppress land-based false positives

---

## 🎨 Visualization & Interpretation

### Color & Mask Legend

| Visual Element | Meaning |
|---------------|--------|
| 🟥 Red Fill | Oil spill region |
| ⬜ White Mask | Oil pixels (segmentation output) |
| ⬛ Black Mask | Clean water / non-oil |
| ⭕ Boundary Outline | Oil spill boundary (toggleable) |

### Boundary Outline Options
- **Black outline** → Day-time SAR images
- **White outline** → Night-time SAR images

---

## 📊 Oil Spill Area Analysis

- A pie chart displays:
  - **Oil spill area (%)**
  - **Clean water area (%)**
- Timestamp shown below the chart
- Used for quick situational assessment

---

## 🖼️ Output Samples

The system produces:
- Original image
- Segmentation mask
- Overlay image with boundary
- Combined downloadable output:
  - Overlay image
  - Oil spill distribution pie chart
  - Timestamp

---

## 📥 Downloaded Output Details

- Includes:
  - Oil spill overlay image
  - Pie chart for oil vs clean water
  - Timestamp
- File name format: `oil_spill_result_DD-MM-YYYY_HH-MM-SS.png`

---

## 🛠️ Technology Stack

### Frontend
- Streamlit

### Deep Learning
- TensorFlow / Keras

### Image Processing
- OpenCV
- NumPy

### Visualization
- Matplotlib

### Models Used
- CNN (Binary Classification)
- U-Net (Segmentation)

---

## 📂 Dataset Source
This project uses a publicly available, research-grade satellite dataset for oil spill detection and segmentation.
[Zenodo Repository](https://zenodo.org/records/10555314)

---

## ▶️ How to Run the Application

### Install Dependencies

`pip install -r requirements.txt`

### Run the Streamlit App

`streamlit run src/app.py`

### Open in Browser

`http://localhost:8501`

---


## 🧪 Supported Image Types
- SAR images (recommended)
- Satellite images
- Image formats: `.jpg, .jpeg, .png`
⚠️ Model performance is optimized for SAR-like water textures.
