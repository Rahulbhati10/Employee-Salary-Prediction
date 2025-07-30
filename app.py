import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 18, 65, 30, key="age")
# education = st.sidebar.selectbox("Education Level", [
#     "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
# ],key ="education")
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
], key="workclass")
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=50000,key ="fnlwgt")
education_num = st.sidebar.slider("Educational Number", 1, 16, 10,key ="education_num")
marital_status = st.sidebar.selectbox("Marital Status", [
    "Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
], key="marital_status")
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
],key ="occupation")
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40,key="hours_per_week")
relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
], key="relationship")
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
], key="race")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key="gender")
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0,key="capital_gain")
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=5000, value=0,key="capital_loss")
# experience = st.sidebar.slider("Years of Experience", 0, 40, 5,key="experience")
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "England", "China",
    "Cuba", "Iran", "Jamaica", "Vietnam", "Italy", "Poland", "France", "Japan", "Greece", "South",
    "Puerto-Rico", "Other"
],key="native_country")
workclass_map = {'Private': 4, 'Self-emp-not-inc': 5, 'Self-emp-inc': 3, 
                 'Federal-gov': 0, 'Local-gov': 1, 'State-gov': 6, 'Without-pay': 7, 'Never-worked': 2, 'Others': 8}

gender_map = {'Male': 1, 'Female': 0}
race_map = {'White': 4, 'Black': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3}
relationship_map = {'Wife': 5, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 0}
marital_status_map = {'Never-married': 2, 'Married-civ-spouse': 1, 'Divorced': 0, 
                      'Separated': 3, 'Widowed': 5, 'Married-spouse-absent': 4, 'Married-AF-spouse': 6}
occupation_map = {'Tech-support': 12, 'Craft-repair': 4, 'Other-service': 10, 'Sales': 11,
                  'Exec-managerial': 5, 'Prof-specialty': 9, 'Handlers-cleaners': 6, 
                  'Machine-op-inspct': 7, 'Adm-clerical': 0, 'Farming-fishing': 3,
                  'Transport-moving': 13, 'Priv-house-serv': 8, 'Protective-serv': 14, 'Armed-Forces': 1}
native_country_map = {'United-States': 39, 'Mexico': 24, 'Philippines': 28, 'Germany': 12, 'Canada': 6, 
                      'India': 17, 'England': 9, 'China': 7, 'Cuba': 8, 'Iran': 18, 'Jamaica': 19, 
                      'Vietnam': 38, 'Italy': 20, 'Poland': 27, 'France': 13, 'Japan': 21, 'Greece': 16, 
                      'South': 35, 'Puerto-Rico': 30, 'Other': 100}

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    # 'education': [education],
    'fnlwgt': [fnlwgt],
    'educational-num': [education_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    # 'experience': [experience],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    input_df['workclass'] = input_df['workclass'].map(workclass_map)
    input_df['gender'] = input_df['gender'].map(gender_map)
    input_df['race'] = input_df['race'].map(race_map)
    input_df['relationship'] = input_df['relationship'].map(relationship_map)
    input_df['marital-status'] = input_df['marital-status'].map(marital_status_map)
    input_df['occupation'] = input_df['occupation'].map(occupation_map)
    input_df['native-country'] = input_df['native-country'].map(native_country_map)
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

