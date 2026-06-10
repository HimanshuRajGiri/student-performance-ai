# 🎓 AI-Powered Student Performance Prediction System

## 📌 Project Overview
End-to-end ML project predicting student outcomes using 25,000+ synthetic student records with 44 features.

**Three Prediction Systems:**
| Module | Type | Target |
|--------|------|--------|
| Placement Prediction | Classification | Placed / Not Placed |
| Salary Forecasting | Regression | Expected LPA |
| Burnout Risk Detection | Classification | High / Low Risk |

## 🏗️ Project Structure
```
student-performance-ai/
├── 01_data_preparation.ipynb    ← Data cleaning & feature engineering
├── 02_eda_visualisation.ipynb   ← 6 detailed EDA charts
├── 03a_placement_model.ipynb    ← Placement classifier (XGBoost, RF, LR)
├── 03b_salary_model.ipynb       ← Salary regressor (XGBoost, RF, Linear)
├── 03c_burnout_model.ipynb      ← Burnout classifier (XGBoost, RF, GB)
├── 04_model_evaluation.ipynb    ← Final evaluation dashboard
├── app.py                       ← Streamlit live dashboard
├── data/                        ← Processed CSVs (auto-created)
├── models/                      ← Saved .pkl models (auto-created)
├── outputs/                     ← Saved charts (auto-created)
└── requirements.txt
```

## 🚀 How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Place Dataset
Put your CSV file in this folder:
```
student_placement_career_success_dataset.csv
```

### Step 3: Run Notebooks in Order
1. `01_data_preparation.ipynb`
2. `02_eda_visualisation.ipynb`
3. `03a_placement_model.ipynb`
4. `03b_salary_model.ipynb`
5. `03c_burnout_model.ipynb`
6. `04_model_evaluation.ipynb`

### Step 4: Launch Dashboard
```bash
streamlit run app.py
```
## 🛠️ Tech Stack
- **Python** | **Pandas** | **NumPy**
- **Scikit-learn** | **XGBoost**
- **Matplotlib** | **Seaborn**
- **Streamlit** (deployment)
- **Jupyter Notebook**

## 📊 Dataset
- Source: Kaggle — Student Placement & Career Success Dataset 2026
- Records: 25,000 synthetic student profiles
- Features: 44 columns (academic, coding, lifestyle, career)
- Note: Synthetic dataset mimicking real Indian engineering college trends

---
*Made for resume project portfolio | 2025-26*
