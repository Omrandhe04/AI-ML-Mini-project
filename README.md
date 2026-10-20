# AI-ML-Mini-project

## 🩻 RadVision AI: Chest X-Ray CAD & Heatmap Localization System

An end-to-end Computer-Aided Diagnosis (CAD) and Explainable AI (XAI) clinical workstation designed for thoracic radiograph screening, multi-label pathology detection, and automated Grad-CAM anatomical heatmap localization.

---

### 🌟 Key Highlights & Features

- **Multi-Pathology Detection**: Evaluates frontal radiographs for 10 key thoracic conditions:
  - Pneumonia
  - Cardiomegaly
  - Atelectasis
  - Pleural Effusion
  - Infiltration
  - Pneumothorax
  - Consolidation
  - Pulmonary Edema
  - Mass / Nodule
  - Normal / Unremarkable
- **Explainable AI (Grad-CAM Heatmaps)**: Generates localized attention maps highlighting affected anatomical zones (e.g., bilateral basilar opacities, cardiac silhouette enlargement, blunted costophrenic angles).
- **Interactive Medical UI**: Modern dark-mode clinical interface featuring:
  - Side-by-side radiograph vs. heatmap comparative slider
  - Real-time heatmap opacity, brightness, and contrast controls
  - Built-in one-click sample radiographs (Normal, Cardiomegaly, Pneumonia)
  - Pre-screening risk indicators (Normal, Inconclusive, Critical)
- **Clinical Report Generation**: Generates automated, structured diagnostic reports with patient metadata, clinical impressions, localized findings, and printable formatting.
- **Audit & History Log**: Persists scan history, findings, and metadata in a local SQLite database (`radiology.db`).
- **CheXNet / DenseNet-121 Architecture**: Includes PyTorch training pipeline script (`train_densenet121_chexnet.py`) aligned with Stanford CheXNet research benchmarks.

---

### 📁 Repository Structure

```text
├── app.py                             # Flask backend application & REST API
├── ai_engine.py                       # Computer vision inference, Grad-CAM heatmap engine
├── database.py                        # SQLite database management & patient record queries
├── train_densenet121_chexnet.py       # PyTorch DenseNet-121 CheXNet multi-label training pipeline
├── cxr_radiology_project_blueprint.md # Complete project research blueprint and math formulation
├── requirements.txt                   # Python dependencies
├── run_server.bat                     # Windows one-click launcher
├── templates/
│   ├── index.html                     # Primary radiologist clinical console UI
│   └── report_print.html              # Formal printable clinical radiology report
└── static/
    ├── css/style.css                  # Custom dark-mode medical glassmorphic design system
    ├── js/app.js                      # Client-side interactive viewer & async API controllers
    ├── sample_images/                 # Built-in verified sample radiographs for testing
    ├── uploads/                       # CXR upload directory
    └── heatmaps/                      # Generated Grad-CAM visualization heatmaps
```

---

### 🚀 Getting Started

#### 1. Clone the Repository
```bash
git clone https://github.com/Omrandhe04/AI-ML-Mini-project.git
cd AI-ML-Mini-project
```

#### 2. Install Dependencies
Make sure you have Python 3.9+ installed:
```bash
pip install -r requirements.txt
```

#### 3. Run the Application
Launch via batch script:
```bash
run_server.bat
```
Or run directly with Python:
```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

---

### 🧪 Training & Deep Learning Model Specs

- **Model Backbone**: DenseNet-121 with Global Average Pooling (GAP) and Sigmoid multi-label output layer.
- **Explainability**: Gradient-weighted Class Activation Mapping (Grad-CAM) targeting feature extraction layer `features.denseblock4.denselayer16.conv2`.
- **Loss Function**: Weighted Binary Cross-Entropy (BCE) to handle class imbalance across thoracic abnormalities.
- **Reference Standard**: CheXNet: *Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning* (Stanford ML Group).

---

### 🛡️ Disclaimer
This software is intended for educational, research, and computer-assisted screening demonstration purposes only. It should not be used as a standalone diagnostic medical device.
