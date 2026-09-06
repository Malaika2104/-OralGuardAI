# 🦷 OralGuard AI

A deep learning pipeline for classifying oral cancer (OSCC) vs. Normal tissue from histopathology images, using three CNN/Transformer architectures and a final ensemble model.

## 📌 Overview

OralGuard AI distinguishes **Oral Squamous Cell Carcinoma (OSCC)** from **Normal** tissue in histopathology images at two magnification levels — **100x** and **400x**. The project trains three independent architectures, compares their performance, and combines them into a weighted ensemble.

## 📂 Dataset

- **Source**: Histopathological Imaging Database for Oral Cancer Analysis — Mendeley Data (dataset ID: `ftmp4cvtmb`, v2)
- **Structure**:
  - First Set → 100x magnification (Normal / OSCC)
  - Second Set → 400x magnification (Normal / OSCC)
- The raw dataset is **not included in this repo** (~3GB) — download it separately and update the paths in the notebooks, or mount it from Google Drive as the notebooks do.

## 🗂️ Repository Structure

| Notebook | Description |
|---|---|
| `01_Oral_Cancer_Dataset.ipynb` | Downloads dataset from source, extracts and verifies folder structure |
| `02_Oral_Cancer_Dataset_EDA.ipynb` | Exploratory data analysis — class counts, image dimensions, corruption checks, pixel/brightness statistics |
| `03_OralGuard_Phase1_Training.ipynb` | Trains EfficientNet-B0 (100x & 400x) |
| `04_OralGuard_Phase2_Training.ipynb` | Trains VGG16 (Batch Norm) (100x & 400x) |
| `05_OralGuard_Phase3_Training.ipynb` | Trains Vision Transformer (ViT-B/16) with warmup LR scheduler |
| `06_OralGuard_Phase4_Training.ipynb` | Builds and evaluates the ensemble (majority vote, averaging, weighted voting) |

## 🏗️ Pipeline

1. **Data prep**: 80/10/10 train/val/test split, stratified by class, with augmentation and class weighting to handle imbalance
2. **Phase 1 — EfficientNet-B0**: baseline CNN model
3. **Phase 2 — VGG16 (BN)**: second CNN for comparison
4. **Phase 3 — ViT-B/16**: transformer-based model with warmup scheduling
5. **Phase 4 — Ensemble**: combines all three models' predictions to find the best-performing strategy per magnification

## 📊 Results

| Model | 100x Accuracy | 400x Accuracy |
|---|---|---|
| EfficientNet-B0 | ~83% | ~81% |
| VGG16 | ~87–89% | ~71–79% |
| ViT-B/16 | ~85% | ~81% |
| Best Ensemble | ~85% | ~81% |

ViT-B/16 was the most consistent performer across both magnifications.

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

**Requirements**: `torch`, `torchvision`, `timm`, `efficientnet_pytorch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `Pillow`, `tqdm`

## ▶️ How to Run

1. Open the notebooks in Google Colab
2. Mount your Google Drive (each notebook does this automatically)
3. Run notebooks in order: `01 → 02 → 03 → 04 → 05 → 06`
4. Trained model weights (`.pth`) and result summaries are saved to Drive under `oralguard_models/Phase-1` through `Phase-4`

## 💾 Model Weights

Trained model checkpoints (`.pth` files) are not stored in this repo due to size. [Add your Drive/Hugging Face link here if you're hosting them externally.]

## 🚀 Future Work

- Grad-CAM visualization for model interpretability
- REST API for deployment
- Clinical validation and testing

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author

Malaika — [GitHub](https://github.com/Malaika2104)
