# PCA-CNN Hybrid Model for Side Face Recognition

This project explores a novel approach to **side face recognition** using a hybrid of **Principal Component Analysis (PCA)** for dimensionality reduction and **Convolutional Neural Networks (CNNs)** for feature extraction and classification. By integrating PCA as a preprocessing step, we achieve high accuracy while significantly reducing the model size compared to traditional CNN-based methods.

---

## Current Status
The implementation and results discussed in this project are part of an ongoing research effort. The code is not publicly available at this time, as the research paper based on this work is under review for publication.

---

## Motivation

Side face recognition is a challenging task in computer vision due to variations in facial profiles and occlusions. This research investigates the effectiveness of PCA for reducing the dimensionality of high-resolution side face images, followed by a CNN model for accurate classification. 

The aim is to enhance computational efficiency while maintaining high performance, making this approach suitable for resource-constrained environments.

---

## Key Features

- **Hybrid PCA-CNN Approach:** 
  - PCA reduces image dimensions while retaining essential features.
  - CNN classifies reconstructed images from PCA-transformed data.
  
- **Improved Training Efficiency:**
  - Reduced model size by **90%** compared to a standard CNN model.
  
- **High Accuracy:**
  - Achieved **94.6% test accuracy** with the hybrid model.

- **Baseline Comparison:**
  - Performance tested without PCA preprocessing for insights on model adaptability.

---

## Dataset

The dataset includes **side face images** from **21 individuals**, with **50 augmented samples per person**. Images are resized to **480x480 pixels** for consistency.

---

## Workflow

1. **Preprocessing:**
   - Input side face images are resized and normalized.
   - PCA is applied for dimensionality reduction, retaining the top 10 components for the analysis.
   - Reconstructed images from PCA are used as inputs for the CNN.

2. **CNN Architecture:**
   - Custom CNN model designed for classification.
   - Trained using reconstructed images with categorical cross-entropy loss.

3. **Evaluation:**
   - Validation and test sets are used to measure accuracy, model size, and training time.
   - Baseline model trained directly on original images (without PCA).

---

## Results

| Approach       | Accuracy | Model Size Reduction | Training Time |  
|----------------|----------|----------------------|---------------|  
| PCA-CNN Hybrid | 94.6%    | 90% smaller          | 20% less      |  
| CNN Only       | 99       | 165mb                | Higher        |  

---

## Installation and Usage

### Prerequisites

- Python 3.8+
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/hybrid-pca-cnn-side-face.git
   cd hybrid-pca-cnn-side-face
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python train_model.py
   ```

4. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```

---

## Insights and Applications

- **Dimensionality Reduction:** PCA effectively reduces redundant information in high-dimensional images.
- **Training Efficiency:** Hybrid models like this are ideal for environments with limited computational resources.
- **Model Optimization:** With a 90% smaller model size, this approach is suitable for deployment on edge devices.
- **Real-World Use Cases:** Surveillance systems, profile-based biometric authentication, and more.

---

## Future Work

- Explore advanced dimensionality reduction techniques.
- Integrate real-time side face recognition pipelines.
- Test with larger and more diverse datasets.

---

## Acknowledgments

- **Mentor:** Girish Kumar for guidance and invaluable feedback throughout this research.  
- **Dataset Augmentation:** Special thanks to libraries like OpenCV for efficient image augmentation.  
- **Peers:** Gratitude to all who contributed ideas and support during this research.

---
## Citation
If you find this work relevant, please cite it once the research paper is published. Citation details will be added here post-publication.

---

## Contact

For any inquiries or discussions about this project, feel free to contact me via email: `jay7080dev@gmail.com`.
