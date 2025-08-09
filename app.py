import streamlit as st
import pandas as pd
import pickle
import base64
import os

# Function to create a download link for a DataFrame as a CSV file
def get_binary_file_downloader_html(df):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()
  href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
  return href

st.title("Heart Disease Model")
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    gender = st.selectbox("Gender", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numeric
    gender = 0 if gender == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # IMPORTANT: Match training column names exactly
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = ['DTC.pkl', 'LogisticR.pkl', 'RFC.pkl', 'SVM.pkl']

def predict_heart_disease(data):
    preds = []
    for modelname in modelnames:
        if not os.path.exists(modelname) or os.path.getsize(modelname) == 0:
            preds.append(["Error: Model file missing"])
            continue
        with open(modelname, 'rb') as f:
            model = pickle.load(f)
        preds.append(model.predict(data))
    return preds

# Submit button
if st.button("Submit"):
    st.subheader("Results......")
    st.markdown("--------------------------")

    result = predict_heart_disease(input_data)
    for i in range(len(result)):
        st.subheader(algonames[i])
        if isinstance(result[i], list) and "Error" in result[i][0]:
            st.error(result[i][0])
        elif result[i][0] == 0:
            st.write("No Heart Disease is Predicted")
        else:
            st.write("Heart Disease Detected")
        st.markdown("--------------------------")

with tab2:
    st.title("Upload CSV")
    st.subheader("Instructions to note before uploading the Files")
    st.info("""The file must contain the Same attribute as\n 
            Age → Age of the patient (in years)\n
            Gender → 0 = Male, 1 = Female (or as per model encoding)\n
            ChestPainType → Encoded type of chest pain (Typical, Atypical, Non-Anginal, Asymptomatic)\n
            RestingBP → Resting blood pressure (mm Hg)\n
            Cholesterol → Serum cholesterol level (mg/dl)\n
            FastingBS → 0 = ≤120 mg/dl, 1 = >120 mg/dl\n
            RestingECG → Encoded ECG results\n
            MaxHR → Maximum heart rate achieved\n
            ExerciseAngina → 0 = No, 1 = Yes\n
            Oldpeak → ST depression induced by exercise relative to rest\n
            ST_Slope → Encoded slope of the peak exercise ST segment""")
    uploaded_file = st.file_uploader("Upload a CSV file" , type = ["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open("LogisticR.pkl" , "rb"))

        expected_coulmns = ["Age" , "Gender" , "ChestPainType", "RestingBP" , "Cholesterol" , "FastingBS" , "RestingECG" , "MaxHR" , "ExerciseAngina" , "Oldpeak" , "ST_Slope"]

        if set(expected_coulmns).issubset(input_data):
            input_data['Prediction LR'] = ''
            
            for i in range(len(input_data)):
                arr = input_data.iloc[i, :-1].values
                input_data['Prediction LR'][i] = model.predict([arr])[0]
            input_data.to_csv('PredictedHeartLR.csv')
        else:
            st.warning("Please make sure the uploaded CSV file has the correct columns")    

        # Display the predictions
        st.subheader("Predictions:")
        st.write(input_data)


        # Create a button to download the updated CSV file
        st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)

    else:
        st.info("Upload a CSV file to get Predictions")


with tab3:
    import plotly.express as px

    data = {'Decision Trees': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine' : 84.22}
    Models = list(data.keys())
    Accuracies = list(data.values())

    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])

    fig = px.bar(df,y='Accuracies',x='Models')
    st.plotly_chart(fig)