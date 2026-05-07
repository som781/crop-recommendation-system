# crop-recommendation-system

An ML-driven crop suggestion system that takes a farmer's location, pulls historical and real-time soil/weather data, and recommends suitable crops. Combines tabular ML (crop selection from soil + climate) with a CNN (plant disease detection from leaf images).

> 🏆 **AgriHackathon Winner** — Dr. N.G.P. Institute of Tech.
> 🥉 **Top 30 / 4000** — Hack Of PI, IIT Kanpur (Semifinalist)
> 📄 **Published** — IRJMETS, e-ISSN: 2582-5208 · [Paper PDF](https://www.irjmets.com/uploadedfiles/paper/volume3/issue_5_may_2021/10895/1628083439.pdf)

---

## What it does

```
   farmer's location (GPS)
            │
            ├──► historical soil/weather (tn.data.gov.in)
            ├──► live weather (API)
            ▼
   feature vector: N, P, K, pH, rainfall, temperature, humidity, soil-type
            │
            ▼
   crop recommendation model  → ranked list of suitable crops
            │
   (parallel)
            ▼
   leaf image upload  → ResNet plant-disease classifier (~99% acc on test split)
```

Two ML problems, one product:

1. **Crop recommendation** — given soil chemistry + climate, classify which crop is most suitable. Models compared in `Phase 1/`: Decision Tree, Naive Bayes, Random Forest, XGBoost.
2. **Plant disease detection** — given a leaf photo, classify healthy vs. diseased. ResNet-based CNN, served via TensorFlow Lite for mobile deployment (`Static Website/The-Farming-Company/notebooks/`).

## Tech stack

- **ML:** scikit-learn (Decision Tree, NB, Random Forest, XGBoost), PyTorch / ResNet for vision, TensorFlow Lite for deployment
- **Backend:** Flask
- **Frontend / mobile:** static web app + TF Lite mobile model
- **Storage:** Firebase Firestore
- **Data sources:** tn.data.gov.in (historical), weather API (live)

## Project layout

```
Phase 1/                    crop-recommendation models — 4 classifiers compared
   ├── Crop Recommendation.ipynb
   ├── DecisionTree.pkl, NBClassifier.pkl, RandomForest.pkl, XGBoost.pkl
   └── Crop_recommendation.csv
Phase 2/                    end-to-end notebook + soil/place lookup tables
   ├── Automated_Crop_Recommendation_System.ipynb
   ├── ACSS PAPER.pdf
   └── crops.csv, cropSoil.csv, placeSoil.csv, …
Phase 3/                    final deck / writeup (VyaVas.pdf)
Static Website/             Flask app + plant-disease ResNet (TF Lite)
   └── The-Farming-Company/
        ├── Data-raw/                    raw districtwise yield data
        ├── notebooks/                    plant-disease-classification ResNet (99.2% acc)
        └── models/                       trained model artifacts
Hackathons/                 pitch deck + demo recording (AgriHackathon, HACK OF PI)
```

## Recognition

- **Winner**, AgriHackathon — Dr. N.G.P. Institute of Tech.
- **Semifinalist** (Top 30 of 4000), HACK OF PI — IIT Kanpur
- **Published** in IRJMETS (e-ISSN 2582-5208) — [paper](https://www.irjmets.com/uploadedfiles/paper/volume3/issue_5_may_2021/10895/1628083439.pdf)

## Notes

This repository contains the full set of artifacts from a multi-phase project (research → notebooks → static deployment). It's preserved as the work-of-record rather than a clean library.
