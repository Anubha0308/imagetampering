# Image Forgery Detection Using Error Level Analysis (ELA) and CNN

This project builds an automated deep learning system to detect whether a digital image is **Authentic** or **Tampered** (forged/edited). Instead of relying on the human eye—which can easily be fooled by modern photo-editing tools—this pipeline uncovers invisible JPEG compression artifacts and analyzes them using a lightweight Convolutional Neural Network (CNN), 2D Fast Fourier Transform (FFT), and Grad-CAM explainability.

---

## 1. Dataset Loading

**Dataset Source:** We used the benchmark **CASIA 2.0** image tampering dataset, which organizes images into two folders: Authentic (`Au`) and Tampered (`Tp`).

**Filtering and Cleaning:** During loading, ground-truth binary mask files (`_gt.png`, `_gt.bmp`) and unsupported `.tif` files were skipped so the model only trains on clean RGB images. After filtering, we loaded a total of **9,501 images**:
* **Authentic Images (`Class 0`):** 7,437 images (78.3% of the dataset).
* **Tampered Images (`Class 1`):** 2,064 images (21.7% of the dataset).

---

## 2. Error Level Analysis (ELA) of Images

**What is ELA?** When a JPEG image is saved, the entire image is compressed at a uniform quality level. However, if someone copies, pastes, or edits a region onto that image and re-saves it, the newly edited region will have a **different compression error level** compared to the original background.

**How We Applied It:**
1. Every loaded image was temporarily re-saved at a fixed **90% JPEG quality**.
2. We calculated the pixel-by-pixel difference between the original image and the re-saved image, then scaled up the brightness of those differences so they become visible.
3. Finally, all ELA images were resized to **128 × 128 × 3** and normalized (`0` to `1`) before being fed into the neural network.

**What It Shows:** Authentic images produce a uniform, dark, grainy ELA pattern across the entire frame, whereas tampered images reveal bright, localized error residuals around the edited or pasted boundaries.

---

## 3. Splitting of Dataset

To fairly train and evaluate our model without data leakage, the **9,501 preprocessed ELA images** were divided using a stratified **80% / 10% / 10%** split:

* **Training Set (80% – 7,600 images):** Used by the CNN to learn patterns that separate authentic compression noise from tampering artifacts.
* **Validation Set (10% – 950 images):** Used after every training epoch to monitor generalization, save the best weights (`ModelCheckpoint`), and stop training early if validation loss stopped improving (`EarlyStopping`).
* **Test Set (10% – 951 images):** Kept completely unseen during training (744 Authentic and 207 Tampered images) to measure final real-world performance.

---

## 4. CNN Model Architecture & Training

We designed a lightweight **Sequential Convolutional Neural Network (CNN)** tailored for 128 × 128 × 3 ELA inputs:

* **Feature Extraction Blocks:** Three convolutional layers with increasing filter sizes (**32 → 64 → 128** filters, `3×3` kernel, ReLU activation). Each block includes **Batch Normalization** (to stabilize learning) and **Max Pooling** (`2×2` to downsample spatial dimensions and capture broader patterns).
* **Classification Head:** The extracted feature maps are passed through a `Flatten` layer, a `Dense(128)` fully connected layer, a **Dropout (0.5)** layer (which randomly turns off 50% of neurons during training to prevent overfitting), and a final `Dense(1)` **Sigmoid** output neuron that predicts a probability between `0` (Authentic) and `1` (Tampered).
* **Training Setup:** Trained using the **Adam optimizer** (`learning rate = 1e-4`) and **Binary Cross-Entropy** loss. Validation accuracy peaked at **95.89% at Epoch 10** (saved as `best_model.h5`), and `EarlyStopping` halted training at Epoch 16 to avoid overfitting.

---

## 5. Model Metrics and Results

Evaluated on the **951 unseen test images** using the best saved model weights (`best_model.h5`):

* **Test Accuracy:** The model achieved an accuracy of **94.95%** on the unseen test set. This indicates that the model correctly classified approximately **95 out of every 100 images** (903 out of 951) in the test set as either authentic or tampered.
* **Test Precision (Tampered):** The precision for the 'Tampered' class is **87.32%**. This means that when the model predicted an image was tampered, it was correct about 87.32% of the time (keeping false alarms low).
* **Test Recall (Tampered):** The recall for the 'Tampered' class is **89.86%**. This indicates that the model successfully identified 89.86% of all actual tampered images in the test set.
* **Test F1-Score (Tampered):** The F1-score for the 'Tampered' class is **0.89**, showing a strong balance between precision and recall despite the class imbalance.
* **Test ROC-AUC:** The ROC-AUC score is **98.62%** (`0.99` on the ROC plot). This is a very high score, suggesting that the model has a strong ability to distinguish between authentic and tampered images across all probability thresholds.

### Confusion Matrix Breakdown
The confusion matrix provides a detailed breakdown of correct and incorrect classifications across the 951 test samples:
* **True Negatives (717):** Authentic images that were correctly identified as authentic.
* **True Positives (186):** Tampered images that were correctly identified as tampered.
* **False Positives (27):** Authentic images that were mistakenly flagged as tampered (a low 3.63% false alarm rate).
* **False Negatives (21):** Tampered images that slipped past the detector as authentic (a 10.14% miss rate).

---

## 6. 2D Fast Fourier Transform (FFT) Analysis

**What is FFT?** The 2D Fast Fourier Transform converts an image from the **spatial domain** (normal pixel grid) into the **frequency domain**, showing how rapidly pixel intensities change across the image.

**Why We Used FFT Here:**
* Subtle image manipulations can alter periodic JPEG compression patterns in ways that are easier to verify in frequency spectra than in pixel space alone.
* **Authentic ELA FFT:** Displays a smooth, evenly distributed noise pattern extending out to the edges, confirming uniform JPEG compression across the whole image.
* **Tampered ELA FFT:** Displays a noticeably brighter yellow-green cluster concentrated at the center (low frequencies) and sharper crosshairs, proving that pasting or editing a region disrupted the natural compression periodicity.

---

## 7. Grad-CAM Explainability

**What is Grad-CAM?** **Grad-CAM (Gradient-weighted Class Activation Mapping)** acts like a visual X-ray for a CNN. It inspects the gradients flowing into the final convolutional layer (`conv2d_2`) and produces a color heatmap showing **which regions of the image the model focused on** to make its decision.

**Why We Used Grad-CAM Here:**
1. **To Build Trust Beyond Accuracy:** High accuracy alone does not prove a model is working properly—a CNN could potentially "cheat" by memorizing background textures. Grad-CAM opens the black box to verify what the model actually sees.
2. **Validating Authentic Predictions (`Score: 0.00`):** On authentic test images, the Grad-CAM heatmap remains faint and diffusely spread out, confirming that no normal object falsely triggered a tampering alert.
3. **Diagnosing Tampered Predictions (`Score: 1.00`):** On spliced images, while ELA cleanly isolates the pasted object and the CNN predicts `Tampered (1.00)`, Grad-CAM shows activation spreading across the surrounding background contrast due to the `Flatten()` layer. This insight directly guides our future improvement of replacing `Flatten()` with `GlobalAveragePooling2D()` for tighter boundary localization.

---

## Overall Summary

The lightweight CNN model, trained using Error Level Analysis (ELA) images, demonstrates strong performance on the CASIA 2.0 test set with **94.95% accuracy** and a **98.62% ROC-AUC score**. Combining spatial ELA preprocessing with **2D FFT frequency analysis** and **Grad-CAM visualizations** provides both reliable forgery detection and transparent insight into how the model makes its predictions.