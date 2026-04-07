# AthleteIQ AI-Enabled Injury Risk Predictor

> **MIT 8212 Seminar Project** · Miva Open University · 2025/2026  
> Batch 2: Data-Driven Decision-Making and Investigative Approaches

---

## Overview

**AthleteIQ** is a proof-of-concept AI-powered athlete injury risk prediction system developed as the product artefact for the MIT 8212 seminar paper:

> *"AI-Enabled Predictive Analytics in Sports: Can Predictive Models Reduce Athlete Injuries and Improve Training Schedules?"*

The system demonstrates how Gradient Boosted Tree predictive models can be applied to Nigerian Professional Football League (NPFL) player data to:

- Predict individual injury risk scores (0–100)
- Generate adaptive weekly training schedules
- Display real feature importance weights from a trained model
- Provide actionable recommendations for coaching and medical staff

---

## Repository Structure

```
athleteiq/
│
├── model/
│   ├── train_model.py          # Full model training script (scikit-learn)
│   ├── outputs/
│   │   ├── athlete_iq_model.pkl        # Trained GBT model
│   │   ├── feature_importances.json    # Real feature weights
│   │   ├── js_weights.json             # JS-ready weights + metrics
│   │   ├── metrics.json                # Full evaluation metrics
│   │   └── dataset_sample.csv          # First 100 rows of training data
│
├── dashboard/
│   └── athlete_iq_dashboard.html       # Interactive prediction dashboard
│
├── docs/
│   └── MIT8212_Seminar_Report.pdf     # Full seminar paper
│
└── README.md
```

---

## The Predictive Model

| Property | Value |
|---|---|
| Algorithm | GradientBoostingClassifier |
| Estimators | 200 trees, max depth 4 |
| Training samples | 2,000 |
| Test AUC-ROC | 0.70 |
| CV AUC-ROC (5-fold) | 0.71 ± 0.04 |
| Accuracy | 77.75% |
| Framework | scikit-learn 1.8.0 |

### Features (8 inputs)

| Feature  | Basis |
|---|---|
| Weekly Training Load (mins) | 21.98% | 
| Fatigue Score (0–10) | 20.41% | 
| Sleep Quality (0–10) | 16.99% | 
| Age | 13.89% | 
| Muscle Soreness (0–10) | 12.76% | 
| Prior Injuries (season) | 6.97% | 
| Days Without Rest | 5.04% 
| Position Risk Category | 1.96% 

### Dataset Note

The training dataset is **synthetic**, generated from published feature distributions 

This is documented transparently in the seminar paper (Chapter 3) and the model training script.

---

## Running the Dashboard

The dashboard is a **single HTML file** with no external dependencies beyond Google Fonts. Open it in any modern browser:

```bash
open dashboard/athlete_iq_dashboard.html
```

The RUN PREDICTION MODEL button executes real inference using the trained model's calibrated weights embedded in JavaScript.

---

## Retraining the Model

```bash
pip install scikit-learn pandas numpy joblib

cd model/
python train_model.py
```

Outputs will be written to `model/outputs/`.

---

## Planned Development (Next Semester Final Project)

- [ ] Connect to real biometric data API (Catapult / Kinexon)
- [ ] Train on actual NPFL injury records
- [ ] Build Flask/FastAPI backend for real-time inference
- [ ] Mobile-optimised interface for field use
- [ ] Multi-club squad management view
- [ ] Longitudinal player monitoring with trend analysis

---

## Academic Context

**Course:** MIT 8212 – Seminar: Industry Applications and Management in IT  
**Institution:** Miva Open University  
**Session:** 2025/2026  
**Batch:** 2 — Data-Driven Decision-Making and Investigative Approaches  
**Frameworks applied:** Technology Acceptance Model (TAM), SWOT Analysis, ACWR

---

## License

This project is submitted as academic coursework. All code is original. The synthetic dataset is generated from publicly available research distributions.
