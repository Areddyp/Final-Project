# 🧠 Workplace Stress Level Analyzer

This project is a **Streamlit-based web app** that predicts and analyzes **employee stress levels** using machine learning. It enables HR professionals and team leaders to identify potential stress risks and take proactive measures to support employee well-being.



## 🚀 Features

- Predicts stress level: **Low**, **Medium**, or **High**
- Accepts user input via an intuitive form
- Displays prediction confidence using progress bars
- Offers stress management recommendations
- Uses a trained machine learning model (Random Forest)
- Easy SHAP-based model explainability (optional)
- Designed for deployment with **Streamlit**

---

## 📊 Inputs Collected

- Age
- Gender
- Job Role
- Work Location (Remote, Hybrid, Onsite)
- Weekly Hours Worked
- Work-Life Balance Rating

---

## 🧠 How It Works

1. User fills out the form with employee data
2. The model preprocesses the data using a saved pipeline
3. A trained classifier predicts the stress level
4. Probabilities for each stress category are shown
5. Actionable recommendations are displayed

---

## 🛠️ Technologies Used

- Python 3.10+
- Streamlit
- scikit-learn
- pandas
- joblib / pickle
- SHAP (optional for model interpretability)

---

To run the `final_project.ipynb` notebook for data analysis:

### 1. Clone the Repository

```bash
git clone https://github.com/Areddyp/Final-Project.git
cd Final-Project

Install Dependencies

pip install -r requirements.txt

Launch Jupyter Notebook

jupyter notebook
Then open final_project.ipynb and run all cells (Kernel > Restart & Run All).

###  What You'll Get
Data preprocessing and EDA

Stress level classification using Random Forest

Feature importance analysis

SHAP visualizations for model interpretability

Streamlit app (app.py) for dashboard deployment


