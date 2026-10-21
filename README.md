# Customer Churn Prediction: ANN (PyTorch) vs ML Baselines

Telecom customers ke churn ki prediction: ek PyTorch ANN aur classical ML models ka muqabla, aur ek interactive Streamlit dashboard jisme threshold slider aur live prediction hai.

**Live demo:** (Streamlit link yahan lagao)

## Problem
Churn karne wale customers ko pehle se pehchanna, taake unhein retention offer diya ja sake. Churn miss karna aksar flag galat karne se mehnga hota hai, isliye sirf accuracy nahi, recall aur precision ka balance dekha gaya.

## Dataset
[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 20 features, lagbhag 26.5% churn (imbalanced).

## Approach
1. **EDA aur cleaning:** `TotalCharges` ke blank values fix kiye, `customerID` drop kiya
2. **Preprocessing:** one-hot encoding, StandardScaler (sirf train par fit, data leakage se bachne ke liye), stratified 80/20 split
3. **Baselines:** Logistic Regression, Random Forest, XGBoost
4. **ANN (PyTorch):** 2 hidden layers (64, 32), ReLU, Dropout 0.3, Adam, class weighting (`pos_weight`), early stopping
5. **Threshold tuning:** 0.2 se 0.8 tak precision/recall trade-off ka analysis
6. **Dashboard:** Streamlit app

## Results (test set, 1,409 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.807 | 0.660 | 0.561 | 0.607 | 0.842 |
| Random Forest | 0.790 | 0.629 | 0.508 | 0.562 | 0.826 |
| XGBoost | 0.799 | 0.651 | 0.524 | 0.581 | 0.844 |
| ANN (threshold 0.5) | 0.731 | 0.496 | 0.834 | 0.622 | 0.845 |

### ANN threshold analysis

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.425 | 0.928 | 0.583 |
| 0.50 | 0.496 | 0.834 | 0.622 |
| 0.60 | 0.547 | 0.741 | 0.630 |
| 0.70 | 0.613 | 0.636 | 0.625 |
| 0.80 | 0.696 | 0.441 | 0.540 |

## Key findings
- **ROC-AUC me sab models lagbhag barabar hain (~0.83-0.85).** Is tabular dataset par deep learning ka koi khaas faida nahi mila.
- ANN ka high recall (0.83) **class weighting aur threshold** ki wajah se hai, architecture ki wajah se nahi.
- Threshold 0.55-0.70 ke beech F1 lagbhag flat hai (~0.63): ek balanced zone hai, koi ek jadui number nahi.
- Month-to-month contract, kam tenure aur zyada monthly charges wale customers zyada churn karte hain.
- **Business ke liye:** sasta retention offer ho to low threshold (high recall), mehnga offer ho to high threshold (high precision).

## Limitations
- ANN ki early stopping test set par ki gayi, isliye result thora optimistic ho sakta hai. Agla step: alag validation set.
- ANN ke scores calibrated probabilities nahi, "risk score" hain (class weighting ki wajah se).
- `TotalCharges` ke 11 missing values (tenure = 0 wale naye customers) ko 0 se fill kiya gaya.

## Dashboard
3 pages:
1. **Data Overview:** churn rate aur key EDA charts
2. **Model Comparison:** sab models ki metrics
3. **Live Prediction:** threshold slider (live precision/recall) aur single-customer risk score

![dashboard](screenshots/dashboard.png)

## Project structure
```
churn-prediction-ann/
├── app/app.py          # Streamlit dashboard
├── data/               # dataset
├── models/             # trained ANN, scaler, baselines
├── notebooks/          # EDA aur training
├── requirements.txt
└── README.md
```

## Run locally
```bash
git clone https://github.com/hannanali9620/churn-prediction-ann.git
cd churn-prediction-ann
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Tech stack
Python, PyTorch, scikit-learn, XGBoost, Streamlit, pandas, seaborn

## Future work
- Alag validation set aur cross-validation
- SHAP se prediction explanations
- Business impact calculator (retention cost vs customer value)
- Batch CSV prediction