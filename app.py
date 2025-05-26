# 14. Employee Stress Prediction Dashboard (Error Suppressed)
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import warnings

def main():
    warnings.filterwarnings("ignore")
    
    try:
        pipeline = joblib.load('employee_stress_analysis_pipeline.pkl')
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return

    with st.sidebar:
        st.title("Employee Stress Analysis")
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
            # Simplified input structure
            input_data = {
                'Age': age,
                'Gender': gender,
                'Job_Role': job_role,
                'Work_Location': work_location,
                'Hours_Worked_Per_Week': hours_worked,
                'Work_Life_Balance_Rating': work_life,
                'Company_Support_for_Remote_Work': 3,  # Default value
                'Industry': 'Tech',
                'Mental_Health_Condition': 'None'
            }
            
            df = pd.DataFrame([input_data])
            processed = pipeline['preprocessor'].transform(df)
            
            model = pipeline['model']
            pred = model.predict(processed)[0]
            proba = model.predict_proba(processed)[0]
            
            # Display core results
            st.subheader("Analysis Outcome")
            st.metric("Stress Level", pred)
            st.write("**Confidence Distribution:**")
            for cls, p in zip(model.classes_, proba):
                st.progress(p, text=f"{cls}: {p*100:.1f}%")

            # Safe SHAP handling
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(processed.toarray())
                
                if isinstance(shap_vals, list):
                    class_idx = list(model.classes_).index(pred)
                    sv = shap_vals[class_idx][0]
                else:
                    sv = shap_vals[0]
                
                features = [f.split("__")[-1] for f in 
                          pipeline['preprocessor'].get_feature_names_out()]
                
                impact_df = pd.DataFrame({
                    'Factor': features,
                    'Influence': sv
                }).nlargest(5, 'Influence', key=abs)
                
                st.write("**Key Influencers:**")
                st.bar_chart(impact_df.set_index('Factor'))

            except Exception as shap_error:
                st.warning("Done now check Recommendations below")

            # Show recommendations
            st.write("**Management Plan**")
            if pred == "High":
                st.error("""
                - Immediate workload review
                - Mandatory counseling
                - Flexible hours activation""")
            elif pred == "Medium":
                st.warning("""
                - Weekly check-ins
                - Stress management resources
                - Workload monitoring""")
            else:
                st.success("""
                - Maintain current programs
                - Encourage PTO
                - Regular wellness checks""")

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
