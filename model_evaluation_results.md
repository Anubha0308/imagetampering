# Model Evaluation & Visual Results

## 1. What is the ROC-AUC Score & What Does the Graph Represent?

### Simple Meaning of ROC-AUC
* **ROC (Receiver Operating Characteristic) Curve:** Normally, our model uses a `0.5` probability cutoff to decide if an image is Authentic (`< 0.5`) or Tampered (`>= 0.5`). The ROC curve tests our model across **every possible cutoff threshold** from `0.0` to `1.0`.
  * **Y-Axis (True Positive Rate / Recall):** Out of all *actual tampered* images, how many did we catch? (Higher is better).
  * **X-Axis (False Positive Rate):** Out of all *actual authentic* images, how many did we falsely accuse of being tampered? (Lower is better).
* **AUC (Area Under the Curve) Score (`0.99` / `98.62%`):** This single number summarizes the entire curve between `0.0` and `1.0`.
  * **In practical terms:** If you randomly pick **one tampered image** and **one authentic image**, our model will assign a higher tampering probability to the tampered image **98.62% of the time**.

### What Our ROC Graph Tells Us
* **The Dashed Blue Diagonal Line (`AUC = 0.50`):** Represents a model that is just guessing randomly (like flipping a coin).
* **The Solid Orange Curve (`AUC = 0.99`):** Rises almost straight up to `1.0` on the left and hugs the **top-left corner**. This proves our model can catch almost all tampered images while keeping false alarms near zero, regardless of where we set the decision threshold.

---

## 2. Training & Validation Learning Curves (Accuracy & Loss Graphs)

These two side-by-side graphs track how well the model learned across **16 training epochs** (Epochs `0` to `15`).

### Left Graph: Model Accuracy
* **Epochs `0` to `1` (Warm-up Phase):** Training accuracy (blue) starts rising immediately, while validation accuracy (orange) stays flat at **78.3%**. Because 78.3% of the dataset is Authentic, the model briefly guessed "Authentic" for everything at the very beginning.
* **Epochs `2` to `9` (Rapid Learning):** Validation accuracy jumps sharply to **90%** and climbs steadily to its **peak of 95.89% at Epoch 9** (10th epoch).
* **Epochs `10` to `15` (Overfitting Begins):** Training accuracy keeps climbing toward **99%**, while validation accuracy starts wobbling and dips at Epoch 14.

### Right Graph: Model Loss
* **Early Spike & Drop:** Validation loss (orange) spikes at Epoch 1 (`1.37`) while weights are adjusting, then drops rapidly to a low, stable plateau (`~0.13`) between Epochs `4` and `9`.
* **Why `ModelCheckpoint` & `EarlyStopping` Saved Us:** After Epoch 9, training loss keeps falling while validation loss creeps upward and spikes at Epoch 14 (`0.72`)—a classic sign of **overfitting** (memorizing training images). Because we used `ModelCheckpoint` and `EarlyStopping`, our pipeline automatically saved the **best weights from the peak epoch** and stopped training early before performance degraded.

---

## 3. Confusion Matrix (Unseen Test Set of 951 Images)

The confusion matrix shows the exact count of right and wrong predictions on the **951 test images** (744 Authentic + 207 Tampered):

| | **Predicted: Authentic** | **Predicted: Tampered** | **What It Means** |
| :--- | :---: | :---: | :--- |
| **Actual: Authentic (744)** | **717** *(True Negatives)* | **27** *(False Positives)* | **96.4%** of authentic photos were cleared; only **3.6%** were false alarms. |
| **Actual: Tampered (207)** | **21** *(False Negatives)* | **186** *(True Positives)* | **89.9%** of fake photos were caught; only **10.1%** slipped past. |

### Key Takeaways to Remember
* **903 out of 951 images** (`717 + 186`) were classified completely accurately (**94.95% Test Accuracy**).
* **Only 48 total mistakes** (`27 + 21`) occurred across the entire unseen test set.
* Even though tampered images made up only ~22% of the data, the model still caught **186 out of 207** forgeries (**89.86% Recall**) with **87.32% Precision**.

---

## 4. Sample ELA Predictions Grid (5 Test Images)

The 5-panel visualization displays actual **Error Level Analysis (ELA)** inputs passed into the trained model alongside their ground-truth labels (`True`) and model confidence scores (`Pred`):

* **Panels 1, 2, 3, and 5 (`True: Authentic` → `Pred: Authentic (0.00)`):**
  * The ELA maps show uniform, faint compression noise across the entire image with no isolated high-error patches.
  * The model outputs a probability of **`0.00`**, showing 100% confidence that these images are genuine.
* **Panel 4 (`True: Tampered` → `Pred: Tampered (1.00)`):**
  * Notice the distinct bluish/purplish compression error cluster on the left side of the dark ELA map where an object was manipulated.
  * The model detects this compression mismatch and outputs a probability of **`1.00`**, confidently flagging the image as tampered.

---

## Quick One-Line Summary for Presentation
> *"Across 951 unseen test images, our model achieved **94.95% accuracy** and a **0.99 ROC-AUC score**—correctly classifying 903 images, keeping false alarms down to 3.6%, and assigning confident `0.00` and `1.00` probabilities on real ELA test samples."*
