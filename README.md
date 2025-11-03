# StaySure Customer Churn Prediction Portfolio

**StaySure ** is a demo portfolio and Streamlit web app for Customer Churn Prediction.
This repository contains:
- Streamlit app (`streamlit_app.py`) to get churn predictions (loads `models/churn_model.pkl` if present)
- `train_model.py` — script to train a model using the Telco Customer Churn CSV placed into `data/`
- Notebooks placeholder for exploratory analysis
- `requirements.txt` listing Python packages needed

## Quick Start

1. Clone or unzip this folder.
2. Put the Telco dataset CSV (`Telco-Customer-Churn.csv`) into the `data/` folder.
   You can download the dataset from Kaggle: *Telco Customer Churn*.
3. (Optional) Train a model:
   ```bash
   python train_model.py
   ```
   This will read `data/Telco-Customer-Churn.csv`, train a RandomForest classifier, and save `models/churn_model.pkl`.

4. Run the Streamlit app:
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
5. The app will load `models/churn_model.pkl`. If it's missing, the app shows instructions to train the model.

## Folder structure

```
StaySure-Portfolio/
├── app/
│   └── streamlit_app.py
├── assets/
│   └── logo.png (placeholder)
├── data/
│   └── Telco-Customer-Churn.csv (you should add this)
├── models/
│   └── churn_model.pkl (generated after training)
├── notebooks/
│   └── churn_analysis.ipynb
├── train_model.py
├── requirements.txt
└── README.md
```

## Notes
- This package intentionally keeps the model-training step offline (you must provide the dataset).
- The Streamlit UI includes a stylish layout and simple motion-like effects using CSS and Streamlit components.
- Customize colors, logo, and assets in the `assets/` folder.

If you want, I can also generate a ready-trained model (if you provide the dataset or allow me to fetch it).