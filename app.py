import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Mental Health Dashboard", layout="centered")
st.title("🧠 Mental Health Model Dashboard")

# Navigation
option = st.selectbox("Choose an option:", ["📈 View Model Summary", "🧪 Predict Employee Stress"])

# ------------------------ 1. VIEW MODEL SUMMARY ------------------------
if option == "📈 View Model Summary":
    st.subheader("📊 Model Performance Summary")
    try:
        summary = pd.read_csv("saved_models/model_performance_summary.csv")
        st.dataframe(summary)

        model_choice = st.selectbox("Choose model to view predictions", summary["Model"].unique())
        pred_file = f"saved_models/{model_choice.replace(' ', '_').lower()}_predictions.csv"
        preds = pd.read_csv(pred_file)

        st.subheader(f"📋 Predictions - {model_choice}")
        st.dataframe(preds.head())

        actual = preds['Actual']
        predicted = preds['Predicted']
        cm = confusion_matrix(actual, predicted)

        st.subheader("🔢 Confusion Matrix")
        fig = plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)

        if model_choice == "Random Forest":
            st.subheader("🌲 Feature Importance")
            st.image("saved_models/random_forest_feature_importance.png")

    except FileNotFoundError as e:
        st.error(f"Missing file: {e.filename}")

# ------------------------ 2. PREDICT EMPLOYEE STRESS ------------------------
elif option == "🧪 Predict Employee Stress":
    st.subheader("👤 Enter Employee Details for Stress Prediction")

    with st.form("input_form"):
        age = st.number_input("Age", 18, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        job_role = st.selectbox("Job Role", ["Software Engineer", "Manager", "Data Analyst"])
        work_location = st.selectbox("Work Location", ["Remote", "Hybrid", "Onsite"])
        hours_worked = st.slider("Weekly Hours", 20, 80, 40)
        work_life = st.slider("Work-Life Balance", 1, 5, 3)
        submitted = st.form_submit_button("Analyze Stress")

    if submitted:
        try:
            input_data = {
                'Age': age,
                'Gender': gender,
                'Job_Role': job_role,
                'Work_Location': work_location,
                'Hours_Worked_Per_Week': hours_worked,
                'Work_Life_Balance_Rating': work_life,
                'Company_Support_for_Remote_Work': 3,
                'Industry': 'Tech',
                'Mental_Health_Condition': 'None'
            }

            df = pd.DataFrame([input_data])

            # Load pipeline
            pipeline = joblib.load("saved_models/random_forest_model.joblib")
            processed = pipeline['preprocessor'].transform(df)

            model = pipeline['model']
            pred = model.predict(processed)[0]
            proba = model.predict_proba(processed)[0]

            # Display Prediction
            st.subheader("🔍 Analysis Outcome")
            st.metric("Stress Level", pred)
            st.write("**Confidence Distribution:**")
            for cls, p in zip(model.classes_, proba):
                st.progress(p, text=f"{cls}: {p * 100:.1f}%")

            # Recommendations
            st.write("**Management Plan**")
            if pred == "High":
                st.error("""
                - Immediate workload review  
                - Mandatory counseling  
                - Flexible hours activation  
                """)
            elif pred == "Medium":
                st.warning("""
                - Weekly check-ins  
                - Stress management resources  
                - Workload monitoring  
                """)
            else:
                st.success("""
                - Maintain current programs  
                - Encourage PTO  
                - Regular wellness checks  
                """)

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
